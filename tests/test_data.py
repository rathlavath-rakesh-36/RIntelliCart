"""Tests for loading, cleaning, NumPy statistics, Pandas analysis, charts and helpers."""
import io

import numpy as np
import pandas as pd
import pytest

from analysis import visualizations as viz
from analysis.numpy_analysis import (min_max_scale, price_statistics, rating_distribution,
                                     rating_statistics, weighted_score)
from analysis.pandas_analysis import (brand_summary, category_summary, dashboard_metrics, filter_products,
                                      lowest_rated, sort_products, top_rated)
from data.data_cleaning import REQUIRED_COLUMNS, DatasetError, clean_products
from data.data_loader import load_products
from utils.helpers import format_inr, get_feature_options, product_card_html, star_string


# ---------- loading ----------

def test_demo_dataset_loads(products):
    assert len(products) >= 100
    assert list(products.columns) == REQUIRED_COLUMNS
    assert products["category"].nunique() == 10
    assert not products.isna().any().any()
    assert products["product_id"].is_unique
    assert products["rating"].between(0, 5).all()


def test_sample_csv_matches_products_csv():
    main, _ = load_products()
    sample, _ = load_products("data/sample_products.csv")
    pd.testing.assert_frame_equal(main, sample)


# ---------- cleaning ----------

def _row(**overrides):
    base = {"product_id": "P1", "product_name": "Phone", "brand": "A", "category": "Mobiles", "price": 100,
            "rating": 4.0, "review_count": 10, "description": "d", "features": "f", "popularity": 50}
    base.update(overrides)
    return base


def test_missing_column_gives_clear_error():
    frame = pd.DataFrame([_row()]).drop(columns=["rating", "brand"])
    with pytest.raises(DatasetError) as error:
        clean_products(frame)
    assert "rating" in str(error.value) and "brand" in str(error.value)


def test_duplicates_and_missing_values_are_handled():
    frame = pd.DataFrame([
        _row(),
        _row(),                                             # exact duplicate
        _row(product_id="P2", product_name="Tablet", review_count=None, popularity=None),
        _row(product_id="P3", product_name="No price", price=None),   # must be dropped
        _row(product_id="P4", product_name="Bad", rating=9.5),        # rating clipped to 5
    ])
    clean, report = clean_products(frame)
    assert report["duplicates_removed"] == 1
    assert report["rows_dropped_missing_values"] == 1
    assert report["review_counts_filled_with_zero"] == 1
    assert len(clean) == 3
    assert not clean.isna().any().any()
    assert clean["rating"].max() == 5.0
    assert clean.loc[clean["product_id"] == "P2", "review_count"].iloc[0] == 0


def test_text_is_stripped_and_numbers_coerced():
    frame = pd.DataFrame([_row(product_name="  Phone  ", price="1,000")])  # unparseable price -> dropped
    with pytest.raises(DatasetError):
        clean_products(frame)
    clean, _ = clean_products(pd.DataFrame([_row(product_name="  Phone  ", price="1000")]))
    assert clean.loc[0, "product_name"] == "Phone"
    assert clean.loc[0, "price"] == 1000


def test_uploaded_file_object_is_validated():
    good = io.StringIO(pd.DataFrame([_row()]).to_csv(index=False))
    df, _ = load_products(good)
    assert len(df) == 1
    with pytest.raises(DatasetError):
        load_products(io.StringIO(""))
    with pytest.raises(DatasetError):
        load_products(io.StringIO("a,b\n1,2\n"))
    with pytest.raises(DatasetError):
        load_products("does_not_exist.csv")


# ---------- NumPy ----------

def test_rating_statistics_against_hand_calculation():
    stats = rating_statistics([3.0, 4.0, 5.0, 4.0], [10, 20, 10, 60])
    assert stats["average"] == pytest.approx(4.0)
    assert stats["median"] == pytest.approx(4.0)
    assert stats["maximum"] == 5.0 and stats["minimum"] == 3.0
    assert stats["std_dev"] == pytest.approx(np.sqrt(0.5))
    assert stats["total_reviews"] == 100
    assert stats["weighted_average"] == pytest.approx((30 + 80 + 50 + 240) / 100)
    with pytest.raises(ValueError):
        rating_statistics([])


def test_price_statistics_and_distribution():
    prices = price_statistics([100, 200, 300, 400])
    assert prices["average"] == 250 and prices["median"] == 250
    labels, counts = rating_distribution([2.5, 3.2, 3.9, 4.0, 4.9, 5.0])
    assert counts.tolist() == [1, 1, 1, 1, 2]
    assert len(labels) == len(counts)


def test_min_max_scale_and_weighted_score():
    assert min_max_scale([10, 20, 30]).tolist() == [0.0, 0.5, 1.0]
    assert min_max_scale([5, 5, 5]).tolist() == [0.0, 0.0, 0.0]
    scores = weighted_score([[1.0, 0.0], [0.5, 0.5]], [3, 1])      # weights rescaled to 0.75 / 0.25
    assert scores.tolist() == pytest.approx([0.75, 0.5])
    with pytest.raises(ValueError):
        weighted_score([[1, 1]], [0, 0])


# ---------- Pandas ----------

def test_filters_search_and_sorting(products):
    samsung = filter_products(products, search="SAMSUNG")
    assert len(samsung) > 0 and samsung["product_name"].str.lower().str.contains("samsung").all()

    mobiles = filter_products(products, categories=["Mobiles"], price_range=(10000, 30000), min_rating=4.0)
    assert len(mobiles) > 0
    assert (mobiles["category"] == "Mobiles").all()
    assert mobiles["price"].between(10000, 30000).all() and (mobiles["rating"] >= 4.0).all()

    assert filter_products(products, search="zzzz-not-a-product").empty
    assert len(filter_products(products)) == len(products)           # no filters = everything

    cheap_first = sort_products(products, "Price: Low to High")["price"].to_numpy()
    assert (np.diff(cheap_first) >= 0).all()
    best_first = sort_products(products, "Rating: High to Low")["rating"].to_numpy()
    assert (np.diff(best_first) <= 0).all()
    popular_first = sort_products(products, "Popularity: High to Low")["popularity"].to_numpy()
    assert (np.diff(popular_first) <= 0).all()


def test_grouping_and_rankings(products):
    categories = category_summary(products)
    assert categories["products"].sum() == len(products)
    manual = products.loc[products["category"] == "Books", "rating"].mean()
    assert categories.set_index("category").loc["Books", "average_rating"] == pytest.approx(manual)

    brands = brand_summary(products, min_products=2)
    assert (brands["products"] >= 2).all()

    assert top_rated(products, 3)["rating"].iloc[0] == products["rating"].max()
    assert lowest_rated(products, 3)["rating"].iloc[0] == products["rating"].min()

    metrics = dashboard_metrics(products)
    assert metrics["total_products"] == len(products)
    assert metrics["average_rating"] == pytest.approx(products["rating"].mean())
    assert metrics["top_product_rating"] == products["rating"].max()


# ---------- charts ----------

CHART_FUNCTIONS = ["top_rated_chart", "category_count_chart", "brand_rating_chart", "rating_histogram",
                   "price_vs_rating_chart", "category_rating_chart", "reviews_by_product_chart",
                   "brand_count_chart", "category_share_pie"]


@pytest.mark.parametrize("name", CHART_FUNCTIONS)
def test_every_chart_renders_to_png(products, name):
    figure = getattr(viz, name)(products)
    png = viz.figure_to_png_bytes(figure)
    assert png.startswith(b"\x89PNG") and len(png) > 2000


def test_charts_work_on_a_single_category_and_tiny_data(products):
    small = products[products["category"] == "Books"].head(1)
    for name in CHART_FUNCTIONS:
        assert viz.figure_to_png_bytes(getattr(viz, name)(small))


# ---------- helpers ----------

def test_format_helpers(products):
    assert format_inr(999) == "₹999"
    assert format_inr(18999) == "₹18,999"
    assert format_inr(1234567) == "₹12,34,567"
    assert star_string(4.4) == "★★★★☆"
    assert "Samsung" in product_card_html(products[products["brand"] == "Samsung"].iloc[0])
    assert "5G" in get_feature_options(products, "Mobiles")


def test_card_html_escapes_untrusted_text(products):
    row = products.iloc[0].copy()
    row["product_name"] = "<script>alert(1)</script>"
    html = product_card_html(row)
    assert "<script>" not in html and "&lt;script&gt;" in html


# ---------- product cards and images ----------

def test_card_html_has_no_blank_lines_or_indentation_for_every_product(products):
    """Markdown ends an HTML block at a blank line, which used to print raw <div> text."""
    for _, row in products.iterrows():
        for kwargs in ({}, {"rank": 1, "match_score": 80.5, "explanation": "because", "selected": True}):
            html = product_card_html(row, **kwargs)
            assert "\n" not in html
            assert html.startswith("<div class=\"ric-card") and html.endswith("</div>")
            assert html.count("<div") == html.count("</div>")


def test_every_category_gets_a_drawn_image_and_url_overrides_it(products):
    from utils.helpers import product_image_uri
    for category in products["category"].unique():
        row = products[products["category"] == category].iloc[0]
        assert product_image_uri(row).startswith("data:image/svg+xml")
    row = products.iloc[0].copy()
    row["image_url"] = "https://example.com/photo.jpg"
    assert product_image_uri(row) == "https://example.com/photo.jpg"
    row["image_url"] = "javascript:alert(1)"          # unsafe addresses are ignored
    assert product_image_uri(row).startswith("data:image/svg+xml")


def test_optional_image_url_column_survives_cleaning():
    frame = pd.DataFrame([_row(image_url="https://example.com/a.jpg"), _row(product_id="P2", product_name="B")])
    clean, _ = clean_products(frame)
    assert "image_url" in clean.columns and clean["image_url"].tolist() == ["https://example.com/a.jpg", ""]
    assert "image_url" not in clean_products(pd.DataFrame([_row()]))[0].columns
