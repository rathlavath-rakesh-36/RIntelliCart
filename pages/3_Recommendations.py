"""Page 3: Intelligent Recommendations - customer preferences in, ranked and explained products out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from analysis import visualizations as viz
from data.data_loader import dataset_to_csv_bytes
from recommendation.hybrid_model import COMPONENT_LABELS, hybrid_recommend
from utils.helpers import card_grid_html, format_inr, get_feature_options, product_card_html
from utils.ui import (demo_notice, empty_state, get_content_model, get_dataset, page_header, select_product,
                      setup_page, show_figure)

setup_page("Recommendations", "✨")
page_header("✨ Intelligent Recommendations", "Tell us what you need — we rank the best matching products and explain why.")
demo_notice()

df = get_dataset()

# Each purpose is turned into words that can be compared with product text (TF-IDF).
PURPOSES = {
    "Gaming": "gaming", "Daily use": "daily use", "Work / office": "office work",
    "Study / students": "study students", "Photography": "photography camera",
    "Fitness / workouts": "gym workout fitness", "Travel": "travel", "Gifting": "gift",
    "Festive / party": "festive party", "Home cooking": "cooking home kitchen",
}

st.markdown("#### 1 · Your preferences")
# Category sits outside the form so the brand and feature lists can update immediately.
category = st.selectbox("Product category", ["All"] + sorted(df["category"].unique()), key="rec_category")
subset = df if category == "All" else df[df["category"] == category]
low_price, high_price = int(subset["price"].min()), int(subset["price"].max())
brand_options = sorted(subset["brand"].unique())
feature_options = get_feature_options(df, category)

with st.form("preferences_form"):
    budget = st.slider("Budget range (₹)", low_price, max(high_price, low_price + 1), (low_price, high_price),
                       step=max(1, (high_price - low_price) // 200 or 1), key=f"rec_budget_{category}",
                       help="Minimum and maximum amount you want to spend.")
    col_a, col_b = st.columns(2)
    brands = col_a.multiselect("Preferred brands (optional)", brand_options, key=f"rec_brands_{category}")
    strict = col_a.checkbox("Show preferred brands only", key="rec_strict",
                            help="Off: other brands may appear but preferred brands score higher.")
    min_rating = col_b.slider("Minimum rating", 0.0, 5.0, 4.0, step=0.1, key="rec_min_rating")
    top_n = col_b.slider("How many recommendations?", 3, 20, 6, key="rec_top_n")
    purposes = st.multiselect("Purpose (optional)", list(PURPOSES), key="rec_purposes",
                              help="Compared with each product's description using TF-IDF.")
    extra_text = st.text_input("Anything else? (optional)", placeholder="e.g. lightweight, long battery",
                               key="rec_extra")
    features = st.multiselect("Required features (optional)", feature_options, key=f"rec_features_{category}",
                              help="Products containing more of these features score higher.")
    submitted = st.form_submit_button("✨ Get recommendations", type="primary")

if submitted:
    purpose_text = " ".join([PURPOSES[p] for p in purposes] + [extra_text.strip()]).strip()
    prefs = {"category": category, "min_budget": budget[0], "max_budget": budget[1], "brands": brands,
             "min_rating": min_rating, "purpose": purpose_text, "required_features": features,
             "strict_brands": strict}
    try:
        with st.spinner("Analysing products..."):
            outcome = hybrid_recommend(df, prefs, get_content_model(), n=top_n)
        st.session_state["rec_outcome"] = (outcome, prefs)
    except ValueError as error:           # for example: max budget below min budget
        st.session_state.pop("rec_outcome", None)
        st.error(f"❌ {error}")

saved = st.session_state.get("rec_outcome")
st.markdown("#### 2 · Recommendations")
if saved is None:
    empty_state("No recommendations yet", "Choose your preferences above and press <b>Get recommendations</b>.")
    st.stop()

outcome, prefs = saved
results = outcome["results"]

if results.empty:
    st.warning("😕 No products match all of your preferences.")
    for tip in outcome["diagnostics"]:
        st.markdown(f"- {tip}")
    st.info("Tip: widen the budget, lower the minimum rating, or choose a different category.")
    st.stop()

st.success(f"✅ {outcome['candidates']} product(s) passed your filters. Showing the top {len(results)}.")

selected_id = st.session_state.get("rec_selected")
st.caption("Click **Select** on a product to see its details and comparison charts on the Dashboard.")
per_row = 3
for start in range(0, len(results), per_row):
    columns = st.columns(per_row)
    for column, (_, row) in zip(columns, results.iloc[start:start + per_row].iterrows()):
        with column:
            st.markdown(product_card_html(row, rank=row["rank"], match_score=row["match_score"],
                                          explanation=row["explanation"],
                                          selected=row["product_id"] == selected_id),
                        unsafe_allow_html=True)
            st.button("✔ Selected" if row["product_id"] == selected_id else "📌 Select this product",
                      key=f"rec_pick_{row['product_id']}", on_click=select_product,
                      args=(row["product_id"],), disabled=row["product_id"] == selected_id)

if selected_id is not None and selected_id in set(results["product_id"]):
    st.success("✅ Product selected. Open the Dashboard to see its details and charts.")
st.page_link("pages/1_Dashboard.py", label="➡ Open Dashboard with my recommendations", icon="📊")

st.markdown("#### 3 · Why these products?")
chart_left, chart_right = st.columns(2)
with chart_left:
    show_figure(viz.match_score_chart(results), "match_scores", "rec_dl_score")
with chart_right:
    show_figure(viz.score_breakdown_chart(results, outcome["used_weights"]), "score_breakdown", "rec_dl_break")

with st.expander("📋 Table view and score breakdown"):
    table = results[["rank", "product_name", "brand", "category", "price", "rating", "review_count",
                     "match_score", "preference_score", "content_score", "rating_score", "popularity_score"]].copy()
    table["price"] = table["price"].map(format_inr)
    st.dataframe(table.round(3), hide_index=True)
    st.caption("Each score is between 0 and 1. A blank score means you gave no information for it, so it was "
               "left out of the calculation.")
    weights_text = " · ".join(f"{COMPONENT_LABELS[name]}: {value:.0%}"
                              for name, value in outcome["used_weights"].items())
    st.markdown(f"**Weights used:** {weights_text}")

st.download_button("⬇ Download recommendations (CSV)",
                   data=dataset_to_csv_bytes(results),
                   file_name="recommendations.csv", mime="text/csv")
