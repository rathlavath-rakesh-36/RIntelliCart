# 🛒 RIntelliCart — Intelligent Product Recommendation Analyzer

*Discover Better Products with Intelligent Recommendations.*

A B.Tech CSE (AI/ML) mini-project that shows how **NumPy, Pandas, Matplotlib and scikit-learn** work
together in one real application, presented as a Streamlit shopping-analytics dashboard.

> ⚠️ **The built-in dataset (`products.csv`) is DEMONSTRATION data.** The 110 products, prices, ratings and
> review counts were created for this project. They are **not live marketplace information**, not verified
> values, and there are **no real customer reviews**. Brand names appear only so the demo feels realistic.
> You can upload your own CSV at any time.

## Problem statement
Online shoppers face thousands of products. Sorting only by price or star rating is misleading (a 5.0 rating from
two reviews is not reliable) and ignores personal needs such as budget, brand, and features.
The project builds a system that analyses historical ratings and customer preferences and recommends suitable
products **with an explanation of why**.

## Objectives
1. Clean and analyse a product dataset with Pandas and NumPy.
2. Visualise ratings, prices, brands and categories with Matplotlib.
3. Build popularity-based, content-based (TF-IDF + cosine similarity) and hybrid recommenders with scikit-learn/NumPy.
4. Let a customer enter preferences and receive ranked, explained recommendations calculated from real data.
5. Present everything in an interactive Streamlit web app that accepts new datasets.

## Technologies
| Technology | Purpose |
|---|---|
| Python 3.10+ | Main language |
| NumPy | Rating statistics, scaling, weighted scores |
| Pandas | Loading, cleaning, filtering, sorting, grouping |
| Matplotlib | All charts |
| scikit-learn | `TfidfVectorizer`, `cosine_similarity` |
| Streamlit | Web interface |
| CSV | Dataset format |

## Features
- **Home** – branding, data overview, cleaning report, dataset **upload with validation**, dataset **download**
- **Everything is linked:** press *Select this product* on a recommendation and the **Dashboard** shows your recommended products, the selected product's details, comparison with category averages, similar products, and match-score / score-breakdown / price-vs-rating charts. Product Explorer, Rating Analyzer and Visualizations have a **"Show only my recommended products"** switch.
- **Product pictures** – every card has an image. Without a photo URL a category illustration (drawn by the app, *not* a real product photo) is used; add an optional `image_url` column with web addresses of photos to your CSV to show real photos.
- **Dashboard** – your recommendations first, then KPI cards + 4 charts for the whole dataset
- **Product Explorer** – product cards; search; category/brand/price/rating filters; 5 sort options; pagination; reset; CSV export
- **Recommendations** – preference form → ranked products with match score, score breakdown and plain-English explanation; helpful message when nothing matches
- **Rating Analyzer** – mean, median, max, min, standard deviation, total reviews, weighted average, category/brand tables, rankings
- **Visualizations** – the 7 required Matplotlib charts, each downloadable as PNG, with optional category filter
- **ML Analyzer** – try the 3 models, adjust hybrid weights, see worked calculations, read the evaluation limitation

## Folder structure
```
RIntelliCart/
├── app.py                      # Home page (run this)
├── products.csv                # Demonstration dataset (110 products)
├── requirements.txt
├── README.md
├── .streamlit/config.toml      # Theme colours
├── data/
│   ├── data_loader.py          # read CSV (path or upload)
│   ├── data_cleaning.py        # validation + cleaning (Pandas)
│   └── sample_products.csv     # copy of the demo dataset / upload template
├── analysis/
│   ├── numpy_analysis.py       # NumPy statistics and scoring helpers
│   ├── pandas_analysis.py      # filters, sorting, groupby, rankings
│   └── visualizations.py       # reusable Matplotlib chart functions
├── recommendation/
│   ├── popularity_model.py     # Model 1: weighted rating
│   ├── content_model.py        # Model 2: TF-IDF + cosine similarity
│   └── hybrid_model.py         # Model 3: weighted hybrid ranking
├── pages/                      # Streamlit pages 1-6 (appear in the sidebar)
├── utils/
│   ├── helpers.py              # formatting + card HTML (no Streamlit)
│   └── ui.py                   # theme, session state, shared UI pieces
├── tests/                      # pytest tests
└── docs/                       # report, architecture, viva, presentation, beginner guide, charts
```

## Installation
```bash
cd RIntelliCart
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run
```bash
streamlit run app.py              # opens http://localhost:8501
python -m pytest -q               # run the tests
```
Run both commands from inside the `RIntelliCart` folder.

## Dataset
`products.csv` has these columns: `product_id, product_name, brand, category, price, rating, review_count,
description, features, popularity` (optional extra column: `image_url`). Features are separated by `;`.
There are 110 products in 10 categories (Mobiles, Laptops, Electronics, Men's Fashion, Women's Fashion,
Footwear, Home and Kitchen, Beauty, Books, Sports and Fitness), 11 each. **Smartphones are in the
"Mobiles" category**; "Electronics" holds audio, TVs, tablets, cameras and accessories.
`popularity` (0–100) is a demonstration index: `min(100, 18·log10(review_count) + 8·(rating − 3))`.

**Upload your own data** on the Home page → *Upload your own CSV*. The file is validated: a clear error names any
missing required column; rows without id/name/price/rating are dropped, duplicates are removed, ratings are clipped
to 0–5, and a cleaning report shows exactly what changed.

## Recommendation algorithms (short version)
1. **Popularity (weighted rating):** `WR = v/(v+m)·R + m/(v+m)·C` — products with few reviews are pulled toward the average rating `C`.
2. **Content-based:** product text → TF-IDF vectors → cosine similarity with a product or with the customer's wish text.
3. **Hybrid:** hard filters (category, budget, rating) → four scores in 0–1
   (preference, content similarity, weighted rating, review popularity) → weighted sum
   (default weights 0.30 / 0.30 / 0.25 / 0.15, adjustable) → match score out of 100 + explanation.
   Components with no customer input are dropped and weights are rescaled.

See `docs/beginner_guide.md` for a worked example with real numbers.

## Evaluation limitation (important)
There are **no customer interactions** (clicks, purchases, individual ratings) in the data, so recommendation
accuracy, precision or recall **cannot be measured, and none are reported**. The ML page offers one clearly-labelled
*sanity check* (do text-similar products share a category?) which is not an accuracy metric.

## Tests
`python -m pytest -q` runs 46 tests covering data loading, cleaning, validation, NumPy maths, Pandas filtering and
sorting, all chart functions, and the three recommenders (including empty-result and invalid-input cases).
The Streamlit pages themselves are not covered by pytest; check them by running the app.
