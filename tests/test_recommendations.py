"""Tests for the three recommendation models."""
import numpy as np
import pandas as pd
import pytest

from recommendation.content_model import ContentRecommender
from recommendation.hybrid_model import DEFAULT_WEIGHTS, default_preferences, hybrid_recommend
from recommendation.popularity_model import compute_weighted_rating, popularity_recommendations


# ---------- popularity ----------

def test_weighted_rating_penalises_few_reviews():
    df = pd.DataFrame({"rating": [5.0, 4.6, 3.0], "review_count": [2, 5000, 1000]})
    scored, params = compute_weighted_rating(df)
    assert scored.loc[0, "weighted_rating"] < scored.loc[1, "weighted_rating"]   # 5.0 from 2 reviews loses
    # the score always lies between the product's own rating and the overall mean
    for _, row in scored.iterrows():
        low, high = sorted([row["rating"], params["mean_rating"]])
        assert low - 1e-9 <= row["weighted_rating"] <= high + 1e-9


def test_weighted_rating_formula_by_hand():
    df = pd.DataFrame({"rating": [5.0, 3.0], "review_count": [100, 100]})
    scored, params = compute_weighted_rating(df)
    c, m = params["mean_rating"], params["min_votes"]
    assert (c, m) == (4.0, 100.0)
    assert scored.loc[0, "weighted_rating"] == pytest.approx(100 / 200 * 5.0 + 100 / 200 * 4.0)


def test_weighted_rating_handles_zero_reviews():
    df = pd.DataFrame({"rating": [4.0, 5.0], "review_count": [0, 0]})
    scored, _ = compute_weighted_rating(df)
    assert scored["weighted_rating"].tolist() == pytest.approx([4.5, 4.5])


def test_popularity_recommendations_sorted_and_filtered(products):
    result, _ = popularity_recommendations(products, 5, "Books")
    assert len(result) == 5 and (result["category"] == "Books").all()
    assert result["rank"].tolist() == [1, 2, 3, 4, 5]
    assert (np.diff(result["weighted_rating"].to_numpy()) <= 0).all()
    assert popularity_recommendations(products, 5, "No such category")[0].empty


# ---------- content-based ----------

def test_similar_products_are_valid(products, content_model):
    similar = content_model.similar_products("P001", 5)
    assert len(similar) == 5
    assert "P001" not in similar["product_id"].tolist()
    assert similar["similarity"].between(0, 1.000001).all()
    assert (np.diff(similar["similarity"].to_numpy()) <= 0).all()
    # a Samsung phone should have other phones among its nearest neighbours
    assert (similar["category"] == "Mobiles").sum() >= 3


def test_query_similarity_finds_relevant_products(content_model):
    top = content_model.recommend_for_text("lightweight running shoes", 3)
    assert (top["category"] == "Footwear").sum() >= 2
    assert top["similarity"].iloc[0] > 0


def test_query_edge_cases(products, content_model):
    assert content_model.query_similarity("").sum() == 0
    assert content_model.query_similarity("qqqzzz xxyyww").sum() == 0
    with pytest.raises(KeyError):
        content_model.similar_products("NOPE")
    with pytest.raises(RuntimeError):
        ContentRecommender().similar_products("P001")
    with pytest.raises(ValueError):
        ContentRecommender().fit(products.iloc[0:0])


def test_category_agreement_is_a_fraction(content_model):
    value = content_model.category_agreement(5)
    assert 0.0 <= value <= 1.0


def test_content_model_works_on_a_new_small_dataset():
    new = pd.DataFrame({
        "product_id": ["A", "B", "C"], "product_name": ["Red Kettle", "Blue Kettle", "Green Novel"],
        "brand": ["X", "X", "Y"], "category": ["Home", "Home", "Books"], "price": [10, 12, 5],
        "rating": [4.0, 4.5, 3.5], "review_count": [5, 50, 500],
        "description": ["electric kettle", "electric kettle fast", "story book"],
        "features": ["steel", "steel; fast boil", "paperback"], "popularity": [10, 20, 30]})
    model = ContentRecommender().fit(new)
    assert model.similar_products("A", 1)["product_id"].iloc[0] == "B"


# ---------- hybrid ----------

EXAMPLE_PREFS = dict(category="Mobiles", min_budget=10000, max_budget=30000, brands=["Samsung", "Motorola"],
                     min_rating=4.0, purpose="gaming and daily use", required_features=["5G", "120Hz"])


def test_hybrid_respects_hard_filters_and_is_sorted(products, content_model):
    outcome = hybrid_recommend(products, EXAMPLE_PREFS, content_model, n=10)
    results = outcome["results"]
    assert not results.empty
    assert (results["category"] == "Mobiles").all()
    assert results["price"].between(10000, 30000).all() and (results["rating"] >= 4.0).all()
    assert (np.diff(results["match_score"].to_numpy()) <= 0).all()
    assert results["match_score"].between(0, 100).all()
    assert results["rank"].tolist() == list(range(1, len(results) + 1))
    assert results["explanation"].str.len().min() > 20


def test_hybrid_score_equals_weighted_components(products, content_model):
    outcome = hybrid_recommend(products, EXAMPLE_PREFS, content_model)
    row = outcome["results"].iloc[0]
    expected = sum(row[f"{name}_score"] * weight for name, weight in outcome["used_weights"].items()) * 100
    assert row["match_score"] == pytest.approx(expected, abs=0.06)       # rounded to 1 decimal
    assert sum(outcome["used_weights"].values()) == pytest.approx(1.0)


def test_scores_are_not_random_and_depend_on_input(products, content_model):
    first = hybrid_recommend(products, EXAMPLE_PREFS, content_model)["results"]
    second = hybrid_recommend(products, EXAMPLE_PREFS, content_model)["results"]
    pd.testing.assert_frame_equal(first, second)                          # deterministic
    other = hybrid_recommend(products, {**EXAMPLE_PREFS, "brands": ["Redmi"]}, content_model)["results"]
    assert first["product_id"].tolist() != other["product_id"].tolist()
    assert other.iloc[0]["brand"] == "Redmi"                              # preferred brand now ranks first


def test_missing_preferences_drop_components(products, content_model):
    outcome = hybrid_recommend(products, {"category": "Books"}, content_model)
    assert set(outcome["used_weights"]) == {"rating", "popularity"}
    assert sum(outcome["used_weights"].values()) == pytest.approx(1.0)


def test_strict_brands_only_returns_selected_brands(products, content_model):
    prefs = {**EXAMPLE_PREFS, "strict_brands": True}
    results = hybrid_recommend(products, prefs, content_model)["results"]
    assert set(results["brand"]) <= {"Samsung", "Motorola"}


def test_empty_results_explain_what_to_change(products, content_model):
    outcome = hybrid_recommend(products, {"category": "Books", "min_budget": 5000, "max_budget": 9000}, content_model)
    assert outcome["results"].empty
    assert any("budget" in tip.lower() for tip in outcome["diagnostics"])

    outcome = hybrid_recommend(products, {"category": "Books", "min_rating": 4.95}, content_model)
    assert outcome["results"].empty
    assert any("rating" in tip.lower() for tip in outcome["diagnostics"])

    outcome = hybrid_recommend(products, {"category": "Nonexistent"}, content_model)
    assert outcome["results"].empty and outcome["diagnostics"]


def test_invalid_input_is_rejected(products, content_model):
    with pytest.raises(ValueError):
        hybrid_recommend(products, {"min_budget": 500, "max_budget": 100}, content_model)
    with pytest.raises(ValueError):
        hybrid_recommend(products, {"min_rating": 7}, content_model)
    with pytest.raises(ValueError):
        hybrid_recommend(products, {}, content_model, weights={k: 0 for k in DEFAULT_WEIGHTS})
    with pytest.raises(ValueError):
        hybrid_recommend(products, {}, content_model, weights={**DEFAULT_WEIGHTS, "rating": -1})


def test_weights_change_the_ranking(products, content_model):
    rating_only = {"preference": 0, "content": 0, "rating": 1, "popularity": 0}
    popularity_only = {"preference": 0, "content": 0, "rating": 0, "popularity": 1}
    by_rating = hybrid_recommend(products, {"category": "Mobiles"}, content_model, weights=rating_only)["results"]
    by_reviews = hybrid_recommend(products, {"category": "Mobiles"}, content_model, weights=popularity_only)["results"]
    assert by_reviews.iloc[0]["review_count"] == products.loc[products["category"] == "Mobiles", "review_count"].max()
    assert by_rating.iloc[0]["product_id"] != by_reviews.iloc[0]["product_id"]


def test_default_preferences_return_everything(products, content_model):
    outcome = hybrid_recommend(products, default_preferences(), content_model, n=500)
    assert outcome["candidates"] == len(products) == len(outcome["results"])


# ---------- charts that describe recommendations ----------

def test_recommendation_charts_render(products, content_model):
    from analysis import visualizations as viz
    outcome = hybrid_recommend(products, EXAMPLE_PREFS, content_model)
    results = outcome["results"]
    for figure in (viz.match_score_chart(results),
                   viz.score_breakdown_chart(results, outcome["used_weights"]),
                   viz.recommended_scatter_chart(products, results, results.iloc[0]["product_id"]),
                   viz.compare_selected_chart(products, results.iloc[0])):
        assert viz.figure_to_png_bytes(figure).startswith(b"\x89PNG")


def test_breakdown_bars_add_up_to_the_match_score(products, content_model):
    outcome = hybrid_recommend(products, {"category": "Books"}, content_model)       # only 2 components used
    results = outcome["results"]
    totals = sum(results[f"{name}_score"].fillna(0) * weight * 100 for name, weight in outcome["used_weights"].items())
    assert totals.round(1).tolist() == pytest.approx(results["match_score"].tolist(), abs=0.06)
