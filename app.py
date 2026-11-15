"""RIntelliCart - Intelligent Product Recommendation Analyzer (Home page).

Run with:   streamlit run app.py
The other pages (Dashboard, Explorer, ...) appear automatically in the sidebar
because they live in the pages/ folder.
"""
import streamlit as st

from analysis.pandas_analysis import column_summary, dataset_overview
from data.data_cleaning import REQUIRED_COLUMNS, DatasetError
from data.data_loader import DEFAULT_DATASET_PATH, dataset_to_csv_bytes, load_products
from utils.helpers import metric_card_html
from utils.ui import demo_notice, empty_state, get_dataset, page_header, set_dataset, setup_page

setup_page("Home", "🛒")
page_header("🛒 RIntelliCart", "Intelligent Product Recommendation Analyzer — "
            "Discover Better Products with Intelligent Recommendations.")

# Show a message saved before a rerun (for example after loading a dataset).
flash = st.session_state.pop("flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

demo_notice()
df = get_dataset()

st.markdown("""
**RIntelliCart** analyses historical product ratings, prices, brands and features, learns what a
customer is looking for, and recommends the most suitable products — with a clear explanation of
*why* each product was chosen. Every number comes from real calculations on the dataset.

| Library | Where it is used in this project |
|---|---|
| **Pandas** | Loading, cleaning, filtering, sorting, grouping |
| **NumPy** | Rating statistics, weighted scores, scaling |
| **Matplotlib** | Every chart in the app |
| **scikit-learn** | TF-IDF + cosine similarity recommendation model |
""")

cols = st.columns(4)
overview = dataset_overview(df)
cards = [("Products", f"{overview['rows']:,}", "in the active dataset"),
         ("Columns", overview["columns"], "per product"),
         ("Missing values", overview["missing_values"], "after cleaning"),
         ("Duplicate IDs", overview["duplicate_ids"], "after cleaning")]
for col, (label, value, note) in zip(cols, cards):
    col.markdown(metric_card_html(label, value, note), unsafe_allow_html=True)

st.markdown("### 📂 Dataset")
tab_overview, tab_upload, tab_download = st.tabs(["Data overview", "Upload your own CSV", "Download"])

with tab_overview:
    st.caption(f"Active dataset: **{st.session_state['dataset_name']}**")
    st.dataframe(df, hide_index=True, height=320)
    left, right = st.columns(2)
    with left:
        st.markdown("**Columns**")
        st.dataframe(column_summary(df), hide_index=True)
    with right:
        st.markdown("**Cleaning report**")
        report = st.session_state["report"]
        st.dataframe(
            {"Step": ["Rows read", "Rows kept", "Rows dropped (missing key values)",
                      "Review counts filled with 0", "Duplicates removed",
                      "Rows with invalid values fixed", "Popularity values filled"],
             "Count": list(report.values())},
            hide_index=True)
    st.markdown("**Numeric summary** (Pandas `describe()`)")
    st.dataframe(df[["price", "rating", "review_count", "popularity"]].describe().round(2),
                 )

with tab_upload:
    st.markdown("Upload a CSV with these columns: `" + "`, `".join(REQUIRED_COLUMNS) + "`.")
    st.caption("Put several features in one cell separated by semicolons, for example `5G; 120Hz display`.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="upload_file")
    if st.button("Validate and use this dataset", type="primary", disabled=uploaded is None):
        try:
            with st.spinner("Validating and cleaning your file..."):
                new_df, new_report = load_products(uploaded)
        except DatasetError as error:
            st.error(f"❌ {error}")
        except Exception as error:  # unexpected problem: still show a friendly message
            st.error(f"❌ Something went wrong while reading the file: {error}")
        else:
            set_dataset(new_df, new_report, f"Uploaded: {uploaded.name}")
            st.session_state["flash"] = ("success", f"✅ Loaded {len(new_df)} products from {uploaded.name}.")
            st.rerun()
    if st.session_state["dataset_name"] != "Built-in demo dataset":
        if st.button("↩ Go back to the built-in demo dataset"):
            demo_df, demo_report = load_products(DEFAULT_DATASET_PATH)
            set_dataset(demo_df, demo_report, "Built-in demo dataset")
            st.session_state["flash"] = ("info", "Switched back to the demo dataset.")
            st.rerun()

with tab_download:
    st.markdown("Download the active (cleaned) dataset, or use it as a template for your own file.")
    st.download_button("⬇ Download dataset (CSV)", data=dataset_to_csv_bytes(df),
                       file_name="rintellicart_products.csv", mime="text/csv")

st.markdown("### 🧭 Where to go next")
st.markdown("""
Use the sidebar to open: **Dashboard** · **Product Explorer** · **Recommendations** ·
**Rating Analyzer** · **Visualizations** · **ML Analyzer**.
""")
if df.empty:
    empty_state("No data", "Upload a CSV to get started.")
