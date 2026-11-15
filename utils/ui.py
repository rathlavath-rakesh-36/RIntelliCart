"""Streamlit helpers shared by every page: theme, data loading, charts, empty states."""
import pandas as pd
import streamlit as st

from analysis.visualizations import figure_to_png_bytes
from data.data_cleaning import DatasetError
from data.data_loader import DEMO_DATA_NOTICE, load_products
from recommendation.content_model import ContentRecommender

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1, h2, h3 {color: #1E1B4B; letter-spacing: -0.01em;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%);}
[data-testid="stSidebar"] * {color: #E0E7FF;}
.ric-hero {background: linear-gradient(120deg, #1E1B4B 0%, #4F46E5 60%, #7C3AED 100%);
  color: white; padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
  box-shadow: 0 8px 24px rgba(79,70,229,.25);}
.ric-hero h1 {color: white; margin: 0; font-size: 2rem;}
.ric-hero p {margin: .3rem 0 0 0; color: #E0E7FF; font-size: 1.02rem;}
.ric-metric {background: white; border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
  box-shadow: 0 2px 10px rgba(30,27,75,.08); border-top: 4px solid #4F46E5;}
.ric-metric-label {font-size: .78rem; text-transform: uppercase; color: #6B7280; letter-spacing: .05em;}
.ric-metric-value {font-size: 1.45rem; font-weight: 700; color: #1E1B4B; line-height: 1.25; margin-top: .2rem;}
.ric-metric-note {font-size: .8rem; color: #6B7280; margin-top: .15rem;}
.ric-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 1rem; margin: .6rem 0 1rem;}
.ric-card {background: white; border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: 0 2px 12px rgba(30,27,75,.09);
  border: 1px solid #EEF2FF; display: flex; flex-direction: column; gap: .3rem;}
.ric-card:hover {box-shadow: 0 6px 20px rgba(79,70,229,.2);}
.ric-img {width: 100%; height: 150px; object-fit: cover; border-radius: 12px; margin-bottom: .3rem; background: #EEF2FF;}
.ric-card-selected {border: 2px solid #7C3AED; box-shadow: 0 6px 20px rgba(124,58,237,.25);}
.ric-picked {background: #7C3AED; color: white; border-radius: 8px; padding: .1rem .5rem; font-weight: 700;}
.ric-card-top {display: flex; gap: .5rem; align-items: center; font-size: .75rem;}
.ric-rank {background: #4F46E5; color: white; border-radius: 8px; padding: .1rem .5rem; font-weight: 700;}
.ric-id {color: #6B7280;}
.ric-cat {margin-left: auto; background: #EEF2FF; color: #4338CA; border-radius: 999px; padding: .1rem .6rem; font-weight: 600;}
.ric-name {font-weight: 700; color: #1F2937; font-size: 1.02rem; line-height: 1.3;}
.ric-brand {color: #6B7280; font-size: .85rem;}
.ric-price {font-size: 1.3rem; font-weight: 800; color: #1E1B4B;}
.ric-rating {font-size: .88rem; color: #374151;}
.ric-stars {color: #F59E0B; letter-spacing: 1px;}
.ric-score {background: #F5F3FF; color: #5B21B6; border-radius: 8px; padding: .25rem .6rem; font-size: .85rem; width: fit-content;}
.ric-desc {font-size: .86rem; color: #4B5563;}
.ric-chips {display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .2rem;}
.ric-chip {background: #F3F4F6; color: #374151; border-radius: 999px; padding: .1rem .55rem; font-size: .74rem;}
.ric-why {background: #F8FAFC; border-left: 3px solid #7C3AED; padding: .4rem .6rem; font-size: .82rem; color: #374151; border-radius: 4px;}
.ric-empty {text-align: center; padding: 2.2rem 1rem; background: white; border: 2px dashed #C7D2FE;
  border-radius: 16px; color: #4B5563;}
.ric-empty b {color: #1E1B4B; font-size: 1.1rem;}
.ric-note {background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; border-radius: 10px;
  padding: .55rem .8rem; font-size: .85rem; margin: .4rem 0 1rem;}
footer {visibility: hidden;}
</style>
"""


def setup_page(title, icon="🛒"):
    """Call first on every page: page config, theme CSS, data and sidebar."""
    st.set_page_config(page_title=f"{title} · RIntelliCart", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_dataset()
    render_sidebar()


def page_header(title, subtitle):
    st.markdown(f'<div class="ric-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def demo_notice():
    """Remind users that the built-in data is demonstration data."""
    if st.session_state.get("dataset_name") == "Built-in demo dataset":
        st.markdown(f'<div class="ric-note">ℹ️ {DEMO_DATA_NOTICE}</div>', unsafe_allow_html=True)


def empty_state(title, message):
    st.markdown(f'<div class="ric-empty"><b>{title}</b><br>{message}</div>', unsafe_allow_html=True)


def _dataset_id(df):
    """A number that changes whenever the data changes (used to reset filters and models)."""
    return int(pd.util.hash_pandas_object(df, index=False).sum() % (10 ** 12))


def set_dataset(df, report, name):
    """Store a cleaned dataset in session state and drop anything built from the old one."""
    st.session_state["df"] = df
    st.session_state["report"] = report
    st.session_state["dataset_name"] = name
    st.session_state["dataset_id"] = _dataset_id(df)
    st.session_state.pop("content_model", None)
    for key in [k for k in st.session_state if k.startswith(("exp_", "rec_", "ml_"))]:
        del st.session_state[key]


def ensure_dataset():
    """Load the built-in demo dataset the first time the app runs."""
    if "df" not in st.session_state:
        try:
            with st.spinner("Loading product data..."):
                df, report = load_products()
            set_dataset(df, report, "Built-in demo dataset")
        except DatasetError as error:
            st.error(f"Could not load the built-in dataset: {error}")
            st.stop()


def get_dataset():
    return st.session_state["df"]


def get_content_model():
    """Fit the TF-IDF model once per dataset and keep it in session state."""
    cached = st.session_state.get("content_model")
    if cached is None or cached[0] != st.session_state["dataset_id"]:
        with st.spinner("Training the TF-IDF content model..."):
            model = ContentRecommender().fit(get_dataset())
        st.session_state["content_model"] = (st.session_state["dataset_id"], model)
        return model
    return cached[1]


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛒 RIntelliCart")
        st.caption("Discover Better Products with Intelligent Recommendations.")
        df = st.session_state["df"]
        st.markdown("---")
        st.markdown(f"**Dataset:** {st.session_state['dataset_name']}")
        st.markdown(f"{len(df)} products · {df['category'].nunique()} categories")
        st.caption("Upload or download datasets on the Home page.")


def recommended_results():
    """The DataFrame of the customer's latest recommendations, or None if there are none yet."""
    saved = st.session_state.get("rec_outcome")
    if saved is None or saved[0]["results"].empty:
        return None
    return saved[0]["results"]


def select_product(product_id):
    """Remember which recommended product the customer clicked (used by the Dashboard)."""
    st.session_state["rec_selected"] = product_id


def focus_on_recommendations(df, key):
    """Show a switch on analysis pages: use only the recommended products.

    Returns (dataframe_to_use, is_focused). When there are no recommendations yet the
    switch is not shown and the whole dataset is returned.
    """
    recommended = recommended_results()
    if recommended is None:
        st.caption("💡 Get recommendations first to focus this page on your recommended products.")
        return df, False
    if st.toggle(f"⭐ Show only my {len(recommended)} recommended products", key=key):
        return df[df["product_id"].isin(recommended["product_id"])], True
    return df, False


def show_figure(fig, filename, key):
    """Show a Matplotlib figure and offer a PNG download."""
    st.pyplot(fig)
    st.download_button("⬇ Download chart (PNG)", data=figure_to_png_bytes(fig),
                       file_name=f"{filename}.png", mime="image/png", key=key)
