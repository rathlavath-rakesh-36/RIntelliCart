"""Page 5: Data Visualizations - the seven required Matplotlib charts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from analysis import visualizations as viz
from utils.ui import demo_notice, focus_on_recommendations, get_dataset, page_header, setup_page, show_figure

setup_page("Visualizations", "📈")
page_header("📈 Data Visualizations", "Seven charts drawn with Matplotlib from the active dataset.")
demo_notice()

df, focused = focus_on_recommendations(get_dataset(), "ml_focus_viz")

# Optional category filter: charts redraw using only the chosen categories.
# Drop remembered categories that no longer exist (for example after switching to "recommended only").
if "viz_categories" in st.session_state:
    st.session_state["viz_categories"] = [c for c in st.session_state["viz_categories"]
                                          if c in set(df["category"])]
chosen = st.multiselect("Limit charts to these categories (leave empty for all)",
                        sorted(df["category"].unique()), key="viz_categories")
data = df[df["category"].isin(chosen)] if chosen else df
if data.empty:
    st.info("No products to draw.")
    st.stop()
st.caption(f"Charts use {len(data)} product(s).")

CHARTS = [
    ("1 · Top 10 rated", lambda d: viz.top_rated_chart(d), "top_10_rated_products",
     "Products with the highest rating; ties are broken by number of reviews."),
    ("2 · Category count", lambda d: viz.category_count_chart(d), "category_product_count",
     "How many products each category contains."),
    ("3 · Brand ratings", lambda d: viz.brand_rating_chart(d), "brand_average_rating",
     "Average rating of brands that have at least two products (all brands for tiny datasets)."),
    ("4 · Rating distribution", lambda d: viz.rating_histogram(d), "rating_distribution",
     "Histogram of ratings with the average and median marked."),
    ("5 · Price vs rating", lambda d: viz.price_vs_rating_chart(d), "price_vs_rating",
     "Each dot is a product. Larger dots have more reviews. Price uses a log scale."),
    ("6 · Category ratings", lambda d: viz.category_rating_chart(d), "category_average_rating",
     "Average rating per category compared with the overall average."),
    ("7 · Reviews by product", lambda d: viz.reviews_by_product_chart(d), "reviews_by_product",
     "The ten products with the most reviews."),
]

tabs = st.tabs([name for name, *_ in CHARTS])
for index, (tab, (name, builder, filename, note)) in enumerate(zip(tabs, CHARTS)):
    with tab:
        st.caption(note)
        with st.spinner("Drawing chart..."):
            show_figure(builder(data), filename, f"viz_dl_{index}")

with st.expander("Extra: share of reviews by category (pie chart)"):
    show_figure(viz.category_share_pie(data), "review_share_by_category", "viz_pie")
