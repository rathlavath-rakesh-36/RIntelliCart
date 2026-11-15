"""Page 4: Rating Analyzer - NumPy statistics and Pandas rankings of historical ratings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from analysis import visualizations as viz
from analysis.numpy_analysis import compare_with_average, rating_distribution, rating_statistics
from analysis.pandas_analysis import (brand_summary, category_summary, lowest_rated, most_popular,
                                      most_reviewed, top_rated)
from utils.helpers import metric_card_html
from utils.ui import demo_notice, focus_on_recommendations, get_dataset, page_header, setup_page, show_figure

setup_page("Rating Analyzer", "⭐")
page_header("⭐ Rating Analyzer", "What do historical ratings say about products, categories and brands?")
demo_notice()

df, focused = focus_on_recommendations(get_dataset(), "ml_focus_rating")
if focused:
    st.info(f"Showing statistics for your {len(df)} recommended products only.")
# NumPy works on arrays: we pull the columns out of the DataFrame with .to_numpy().
stats = rating_statistics(df["rating"].to_numpy(), df["review_count"].to_numpy())

st.markdown("#### Rating statistics (NumPy)")
row1 = st.columns(4)
row1[0].markdown(metric_card_html("Average rating", f"{stats['average']:.2f}", "np.mean"), unsafe_allow_html=True)
row1[1].markdown(metric_card_html("Median rating", f"{stats['median']:.2f}", "np.median"), unsafe_allow_html=True)
row1[2].markdown(metric_card_html("Highest rating", f"{stats['maximum']:.1f}", "np.max"), unsafe_allow_html=True)
row1[3].markdown(metric_card_html("Lowest rating", f"{stats['minimum']:.1f}", "np.min"), unsafe_allow_html=True)
st.write("")
row2 = st.columns(4)
row2[0].markdown(metric_card_html("Standard deviation", f"{stats['std_dev']:.3f}", "np.std — how spread out ratings are"),
                 unsafe_allow_html=True)
row2[1].markdown(metric_card_html("Total reviews", f"{stats['total_reviews']:,}", "np.sum"), unsafe_allow_html=True)
row2[2].markdown(metric_card_html("Review-weighted average", f"{stats['weighted_average']:.2f}", "np.average with weights"),
                 unsafe_allow_html=True)
above = int(np.sum(compare_with_average(df["rating"].to_numpy()) > 0))
row2[3].markdown(metric_card_html("Above-average products", f"{above} of {stats['count']}", "rating − mean > 0"),
                 unsafe_allow_html=True)

st.markdown("#### Group analysis (Pandas)")
tab_cat, tab_brand = st.tabs(["By category", "By brand"])
with tab_cat:
    summary = category_summary(df).sort_values("average_rating", ascending=False)
    st.dataframe(summary.round(2), hide_index=True)
with tab_brand:
    max_products = int(df.groupby("brand").size().max())
    min_products = st.slider("Only brands with at least this many products", 1, max(2, max_products), 1,
                             key="ml_min_brand_products")
    brands = brand_summary(df, min_products)
    if brands.empty:
        st.info("No brand has that many products. Lower the number above.")
    else:
        st.dataframe(brands.round(2), hide_index=True)
        best = brands.iloc[0]
        st.success(f"🏆 Best-rated brand: **{best['brand']}** — average rating {best['average_rating']:.2f} "
                   f"across {int(best['products'])} product(s).")

st.markdown("#### Product rankings")
count = st.slider("How many products to list", 3, 20, 5, key="ml_rank_count")
columns = ["product_name", "brand", "category", "rating", "review_count", "popularity"]
tabs = st.tabs(["Highest rated", "Lowest rated", "Most reviewed", "Popular"])
for tab, frame in zip(tabs, [top_rated(df, count), lowest_rated(df, count),
                             most_reviewed(df, count), most_popular(df, count)]):
    with tab:
        st.dataframe(frame[columns], hide_index=True)
st.caption("Ties in rating are broken by review count. 'Popular' uses the dataset's popularity column.")

st.markdown("#### Charts (Matplotlib)")
left, right = st.columns(2)
with left:
    show_figure(viz.rating_histogram(df), "rating_distribution", "ra_hist")
    labels, counts = rating_distribution(df["rating"].to_numpy())
    st.dataframe({"Rating band": labels, "Products": counts.tolist()}, hide_index=True,
                 )
with right:
    show_figure(viz.category_rating_chart(df), "category_average_rating", "ra_cat")
show_figure(viz.brand_rating_chart(df), "brand_average_rating", "ra_brand")
