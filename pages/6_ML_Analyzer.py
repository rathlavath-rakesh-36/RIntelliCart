"""Page 6: Machine Learning Analyzer - try and understand the three recommendation models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from recommendation.hybrid_model import COMPONENT_LABELS, DEFAULT_WEIGHTS, hybrid_recommend
from recommendation.popularity_model import popularity_recommendations
from utils.helpers import format_inr, get_feature_options
from utils.ui import demo_notice, empty_state, get_content_model, get_dataset, page_header, setup_page

setup_page("ML Analyzer", "🤖")
page_header("🤖 Machine Learning Analyzer", "Three recommendation models. Every score is calculated from the data.")
demo_notice()

df = get_dataset()
model_name = st.radio("Choose a model", ["1 · Popularity-based", "2 · Content-based (TF-IDF)", "3 · Hybrid"],
                      horizontal=True, key="ml_model")

# =====================================================================
if model_name.startswith("1"):
    st.markdown("""
**How it works.** A 5.0 rating from 2 reviews is not as trustworthy as a 4.6 from 5,000 reviews.
The weighted rating pulls products with few reviews towards the overall average:

`WR = v/(v+m) × R + m/(v+m) × C`  —  R = product rating, v = its reviews, C = average rating of all products,
m = minimum reviews (here the 60th percentile of review counts, computed with NumPy).
""")
    col_a, col_b = st.columns(2)
    category = col_a.selectbox("Category", ["All"] + sorted(df["category"].unique()), key="ml_pop_cat")
    count = col_b.slider("Number of products", 3, 20, 10, key="ml_pop_n")
    result, params = popularity_recommendations(df, count, category)
    if result.empty:
        empty_state("Nothing to show", "This category has no products.")
    else:
        st.dataframe(result[["rank", "product_name", "brand", "category", "rating", "review_count",
                             "weighted_rating"]].round({"weighted_rating": 3}),
                     hide_index=True)
        top = result.iloc[0]
        v, m, c = float(top["review_count"]), params["min_votes"], params["mean_rating"]
        st.markdown(f"**Worked example — rank 1: {top['product_name']}**")
        st.code(f"R = {top['rating']:.2f}   v = {v:,.0f}   C = {c:.3f}   m = {m:,.0f}\n"
                f"WR = {v:,.0f}/({v:,.0f}+{m:,.0f}) × {top['rating']:.2f} + {m:,.0f}/({v:,.0f}+{m:,.0f}) × {c:.3f}"
                f" = {top['weighted_rating']:.3f}")

# =====================================================================
elif model_name.startswith("2"):
    st.markdown("""
**How it works.** Each product's name, brand, category, description and features are joined into one text.
**TF-IDF** turns the text into numbers (rare, descriptive words get more weight). **Cosine similarity**
then measures how close two texts are: 1 = very similar, 0 = nothing in common.
""")
    model = get_content_model()
    mode = st.radio("What do you want to do?", ["Find products similar to a product", "Match my own text"],
                    horizontal=True, key="ml_content_mode")
    if mode.startswith("Find"):
        labels = {f"{row.product_name} ({row.product_id})": row.product_id for row in df.itertuples()}
        choice = st.selectbox("Pick a product", list(labels), key="ml_content_product")
        count = st.slider("Number of similar products", 3, 10, 5, key="ml_content_n")
        similar = model.similar_products(labels[choice], count)
        st.dataframe(similar[["product_name", "brand", "category", "price", "similarity"]]
                     .assign(price=lambda t: t["price"].map(format_inr)).round({"similarity": 3}),
                     hide_index=True)
        terms = model.top_terms(labels[choice])
        st.caption("Highest TF-IDF words for the chosen product: " + ", ".join(f"{w} ({s:.2f})" for w, s in terms))
    else:
        text = st.text_input("Describe what you want", placeholder="e.g. lightweight running shoes for the gym",
                             key="ml_content_text")
        if not text.strip():
            empty_state("Type something", "Describe the product you are looking for.")
        else:
            matches = model.recommend_for_text(text, 10)
            if matches["similarity"].max() == 0:
                empty_state("No matching words", "None of your words appear in the product texts. Try other words.")
            else:
                st.dataframe(matches[["rank", "product_name", "brand", "category", "price", "similarity"]]
                             .assign(price=lambda t: t["price"].map(format_inr)).round({"similarity": 3}),
                             hide_index=True)

# =====================================================================
else:
    st.markdown("""
**How it works.** After hard filters (category, budget, rating), each product gets four scores between 0 and 1:
**preference match** (brand + required features), **content similarity** (TF-IDF cosine with your purpose text),
**weighted rating** and **review popularity**. They are combined with weights:

`match score = w1×preference + w2×content + w3×rating + w4×popularity`  (weights rescaled to add up to 1)
""")
    st.markdown("**Weights** (change them to see how the ranking changes)")
    w_cols = st.columns(4)
    weights = {name: w_cols[i].slider(COMPONENT_LABELS[name], 0.0, 1.0, DEFAULT_WEIGHTS[name], 0.05,
                                      key=f"ml_w_{name}")
               for i, name in enumerate(DEFAULT_WEIGHTS)}

    c1, c2, c3 = st.columns(3)
    category = c1.selectbox("Category", ["All"] + sorted(df["category"].unique()), key="ml_h_cat")
    subset = df if category == "All" else df[df["category"] == category]
    low, high = int(subset["price"].min()), int(subset["price"].max())
    budget = c1.slider("Budget (₹)", low, max(high, low + 1), (low, high), key=f"ml_h_budget_{category}")
    brands = c2.multiselect("Preferred brands", sorted(subset["brand"].unique()), key=f"ml_h_brands_{category}")
    min_rating = c2.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1, key="ml_h_rating")
    purpose = c3.text_input("Purpose / wishes", placeholder="e.g. gaming and daily use", key="ml_h_purpose")
    features = c3.multiselect("Required features", get_feature_options(df, category), key=f"ml_h_feat_{category}")

    prefs = {"category": category, "min_budget": budget[0], "max_budget": budget[1], "brands": brands,
             "min_rating": min_rating, "purpose": purpose, "required_features": features}
    try:
        outcome = hybrid_recommend(df, prefs, get_content_model(), weights=weights, n=10)
    except ValueError as error:
        st.error(f"❌ {error}")
        st.stop()

    results = outcome["results"]
    if results.empty:
        st.warning("No products match these filters.")
        for tip in outcome["diagnostics"]:
            st.markdown(f"- {tip}")
    else:
        st.dataframe(results[["rank", "product_name", "brand", "price", "rating", "match_score"]]
                     .assign(price=lambda t: t["price"].map(format_inr)),
                     hide_index=True)
        top = results.iloc[0]
        breakdown = pd.DataFrame([
            {"Component": COMPONENT_LABELS[name], "Score (0-1)": round(float(top[f"{name}_score"]), 3),
             "Weight used": round(weight, 3), "Contribution": round(float(top[f"{name}_score"]) * weight, 4)}
            for name, weight in outcome["used_weights"].items()])
        st.markdown(f"**Worked example — rank 1: {top['product_name']}**")
        st.dataframe(breakdown, hide_index=True)
        st.code(f"match score = sum of contributions × 100 = {breakdown['Contribution'].sum() * 100:.1f}"
                f"   (shown score: {top['match_score']:.1f})")
        st.caption("Components you gave no information for are left out and the other weights are rescaled.")

# =====================================================================
st.markdown("---")
st.markdown("### 🔬 About evaluation")
st.info("""
**These models cannot be given an accuracy score with this data.** Measuring recommendation accuracy needs
ground truth — for example which products real customers clicked, bought or rated, or a set of products
experts marked as correct matches. The demo dataset has only product information, no customer interactions,
so no accuracy, precision or recall numbers are shown. Accuracy figures would be invented.
""")
if st.button("Run text-similarity sanity check", key="ml_sanity"):
    agreement = get_content_model().category_agreement(5)
    if agreement is None:
        st.warning("Need at least two products.")
    else:
        st.metric("Top-5 similar products in the same category", f"{agreement:.0%}")
        st.caption("A sanity check, not an accuracy metric: products that are similar by text should usually "
                   "belong to the same category.")
