"""Page 2: Product Explorer - search, filter, sort and browse product cards."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from analysis.pandas_analysis import SORT_OPTIONS, filter_products, sort_products
from data.data_loader import dataset_to_csv_bytes
from utils.helpers import card_grid_html, format_inr, product_card_html
from utils.ui import demo_notice, empty_state, focus_on_recommendations, get_dataset, page_header, setup_page

setup_page("Product Explorer", "🔎")
page_header("🔎 Product Explorer", "Search, filter and sort the product catalogue.")
demo_notice()

df = get_dataset()


def reset_filters():
    """Delete every explorer widget value so all filters return to their defaults."""
    for key in [k for k in st.session_state if k.startswith("exp_")]:
        del st.session_state[key]


price_min, price_max = int(df["price"].min()), int(df["price"].max())

# ---------- filter controls ----------
top = st.columns([3, 2, 2])
search = top[0].text_input("Search by product name", placeholder="e.g. samsung, running shoes",
                           key="exp_search")
categories = top[1].multiselect("Category", sorted(df["category"].unique()), key="exp_categories")
brand_pool = sorted(df[df["category"].isin(categories)]["brand"].unique()) if categories \
    else sorted(df["brand"].unique())
# Keep only brands that still exist after the category selection changed.
if "exp_brands" in st.session_state:
    st.session_state["exp_brands"] = [b for b in st.session_state["exp_brands"] if b in brand_pool]
brands = top[2].multiselect("Brand", brand_pool, key="exp_brands")

bottom = st.columns([3, 2, 2, 1])
price_range = bottom[0].slider("Price range (₹)", price_min, price_max, (price_min, price_max),
                               step=max(1, (price_max - price_min) // 200), key="exp_price")
min_rating = bottom[1].slider("Minimum rating", 0.0, 5.0, 0.0, step=0.1, key="exp_rating")
sort_label = bottom[2].selectbox("Sort by", list(SORT_OPTIONS), key="exp_sort")
bottom[3].write("")
bottom[3].button("↺ Reset", on_click=reset_filters)

# ---------- apply filters and sorting (Pandas) ----------
base_df, _ = focus_on_recommendations(df, "exp_focus")
filtered = filter_products(base_df, search, categories, brands, price_range, min_rating)
filtered = sort_products(filtered, sort_label)

info_col, download_col = st.columns([4, 1])
info_col.markdown(f"**{len(filtered)}** of {len(df)} products match "
                  f"· price {format_inr(price_range[0])} – {format_inr(price_range[1])}")
if not filtered.empty:
    download_col.download_button("⬇ Export results", data=dataset_to_csv_bytes(filtered),
                                 file_name="filtered_products.csv", mime="text/csv")

if filtered.empty:
    empty_state("No products match these filters",
                "Try a different search word, widen the price range or lower the minimum rating, "
                "or press <b>Reset</b>.")
    st.stop()

# ---------- pagination ----------
size_col, page_col = st.columns([1, 1])
page_size = size_col.selectbox("Products per page", [12, 24, 48], key="exp_page_size")
total_pages = max(1, math.ceil(len(filtered) / page_size))
# If filters shrank the results, a remembered page number may now be too high.
if st.session_state.get("exp_page", 1) > total_pages:
    st.session_state["exp_page"] = total_pages
page = page_col.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="exp_page")
start = (int(page) - 1) * page_size
visible = filtered.iloc[start:start + page_size]

st.markdown(card_grid_html([product_card_html(row, selected=row["product_id"] == st.session_state.get("rec_selected")) for _, row in visible.iterrows()]),
            unsafe_allow_html=True)
st.caption(f"Showing {start + 1}–{start + len(visible)} of {len(filtered)} · page {int(page)} of {total_pages}")
