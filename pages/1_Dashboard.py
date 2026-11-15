"""Page 1: Dashboard - YOUR recommendations first, then the overview of the whole dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from analysis import visualizations as viz
from analysis.pandas_analysis import dashboard_metrics
from utils.helpers import card_grid_html, format_inr, metric_card_html, product_card_html
from utils.ui import (demo_notice, get_content_model, get_dataset, page_header, recommended_results,
                      setup_page, show_figure)

setup_page("Dashboard", "📊")
page_header("📊 Dashboard", "Your recommendations and an overview of the product dataset.")
demo_notice()

df = get_dataset()

# =====================================================================
# SECTION 1: everything about the customer's recommendations
# =====================================================================
st.markdown("### ✨ Your recommendations")
recommended = recommended_results()
if recommended is None:
    st.info("You have no recommendations yet. Open **Recommendations** in the sidebar, enter your preferences "
            "and press *Get recommendations*. Your products and their charts will appear here.")
    st.page_link("pages/3_Recommendations.py", label="Go to Recommendations", icon="✨")
else:
    outcome, prefs = st.session_state["rec_outcome"]
    selected_id = st.session_state.get("rec_selected")
    if selected_id not in set(recommended["product_id"]):
        selected_id = None          # the old selection is not part of the new recommendations

    best = recommended.iloc[0]
    cards = st.columns(4)
    cards[0].markdown(metric_card_html("Recommended products", len(recommended),
                                       f"{outcome['candidates']} passed your filters"), unsafe_allow_html=True)
    cards[1].markdown(metric_card_html("Best match", best["product_name"],
                                       f"Score {best['match_score']:.1f} / 100"), unsafe_allow_html=True)
    cards[2].markdown(metric_card_html("Price range", f"{format_inr(recommended['price'].min())} – "
                                       f"{format_inr(recommended['price'].max())}"), unsafe_allow_html=True)
    cards[3].markdown(metric_card_html("Average rating", f"{recommended['rating'].mean():.2f} / 5",
                                       f"dataset average {df['rating'].mean():.2f}"), unsafe_allow_html=True)
    st.write("")

    # ---- the product the customer clicked on the Recommendations page
    if selected_id is None:
        st.caption("Tip: press **Select this product** on the Recommendations page to see a product in detail here.")
    else:
        product = recommended[recommended["product_id"] == selected_id].iloc[0]
        st.markdown("#### 📌 Your selected product")
        left, right = st.columns([1, 2])
        with left:
            st.markdown(card_grid_html([product_card_html(product, rank=product["rank"],
                                                          match_score=product["match_score"],
                                                          explanation=product["explanation"], selected=True)]),
                        unsafe_allow_html=True)
        with right:
            show_figure(viz.compare_selected_chart(df, product), "selected_vs_average", "dash_sel_cmp")
            category_rank = int((df[df["category"] == product["category"]]["rating"] > product["rating"]).sum()) + 1
            st.markdown(f"Rated **{product['rating']:.1f}**, which puts it at rank **{category_rank}** of "
                        f"{int((df['category'] == product['category']).sum())} products in "
                        f"*{product['category']}* (by rating).")
        st.markdown("**Similar products** (TF-IDF + cosine similarity)")
        similar = get_content_model().similar_products(selected_id, 3)
        st.markdown(card_grid_html([product_card_html(row) for _, row in similar.iterrows()]),
                    unsafe_allow_html=True)

    st.markdown("#### All recommended products")
    st.markdown(card_grid_html([product_card_html(row, rank=row["rank"], match_score=row["match_score"],
                                                  selected=row["product_id"] == selected_id)
                                for _, row in recommended.iterrows()]), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        show_figure(viz.match_score_chart(recommended), "match_scores", "dash_rec_score")
    with right:
        show_figure(viz.recommended_scatter_chart(df, recommended, selected_id), "recommended_price_vs_rating",
                    "dash_rec_scatter")
    show_figure(viz.score_breakdown_chart(recommended, outcome["used_weights"]), "score_breakdown", "dash_rec_break")

# =====================================================================
# SECTION 2: overview of the whole dataset
# =====================================================================
st.markdown("---")
st.markdown("### 🗂 Dataset overview")
metrics = dashboard_metrics(df)   # Pandas + NumPy calculations

row1 = st.columns(3)
row1[0].markdown(metric_card_html("Total products", f"{metrics['total_products']:,}"), unsafe_allow_html=True)
row1[1].markdown(metric_card_html("Product categories", metrics["total_categories"]), unsafe_allow_html=True)
row1[2].markdown(metric_card_html("Brands", metrics["total_brands"]), unsafe_allow_html=True)
st.write("")
row2 = st.columns(3)
row2[0].markdown(metric_card_html("Average rating", f"{metrics['average_rating']:.2f} / 5"), unsafe_allow_html=True)
row2[1].markdown(metric_card_html("Highest-rated product", metrics["top_product_name"],
                                  f"Rating {metrics['top_product_rating']:.1f} (ties: most reviews)"),
                 unsafe_allow_html=True)
row2[2].markdown(metric_card_html("Most popular category", metrics["popular_category"],
                                  f"{metrics['popular_category_reviews']:,} total reviews"),
                 unsafe_allow_html=True)
st.write("")

left, right = st.columns(2)
with left:
    show_figure(viz.category_count_chart(df), "category_product_count", "dash_cat")
with right:
    show_figure(viz.brand_count_chart(df), "brand_product_count", "dash_brand")
left, right = st.columns(2)
with left:
    show_figure(viz.rating_histogram(df), "rating_distribution", "dash_hist")
with right:
    show_figure(viz.top_rated_chart(df), "top_10_rated_products", "dash_top")
