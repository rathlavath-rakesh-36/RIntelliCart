# Beginner's Guide to RIntelliCart

Use this to understand and explain your project during evaluation. Every module is explained the same way:
**What / Why / Library / Logic / Output / Demo / Viva.**

---

## The big picture in one paragraph
A CSV of products is **loaded and cleaned (Pandas)**. **NumPy** calculates statistics and scores. **Matplotlib**
draws charts. **scikit-learn** turns product text into numbers (TF-IDF) and compares texts (cosine similarity).
The hybrid recommender mixes all of that into one **match score**. **Streamlit** shows it as a website.

| Library | Example of where it is used (file → function) |
|---|---|
| Pandas | `data_cleaning.clean_products`, `pandas_analysis.filter_products`, `category_summary` |
| NumPy | `numpy_analysis.rating_statistics`, `min_max_scale`, `weighted_score`; `popularity_model` (percentile) |
| Matplotlib | `visualizations.top_rated_chart`, `rating_histogram`, `price_vs_rating_chart` … |
| scikit-learn | `content_model.ContentRecommender` (`TfidfVectorizer`, `cosine_similarity`) |

---

## Module 1 — Data loading and cleaning (`data/`)
- **What:** reads a CSV, checks required columns, fixes problems.
- **Why:** real data is messy; bad rows would break statistics and recommendations.
- **Library:** Pandas (`read_csv`, `to_numeric`, `drop_duplicates`, `fillna`, `clip`).
- **Logic:** (1) check columns → (2) strip spaces → (3) convert numbers → (4) drop rows missing id/name/price/rating →
  (5) fill review_count with 0 → (6) remove duplicates → (7) clip rating to 0–5 → (8) fill popularity.
- **Output:** a clean DataFrame + a report (rows kept, duplicates removed …).
- **Demo:** Home page → *Data overview* tab. Upload a CSV without a `rating` column to show the clear error.
- **Viva:** Why `errors="coerce"`? Why drop duplicates? What does `fillna` do?

## Module 2 — NumPy analysis (`analysis/numpy_analysis.py`)
- **What:** mean, median, max, min, standard deviation, weighted average, histogram bins, min-max scaling, weighted score.
- **Why:** NumPy works on whole arrays at once — faster and shorter than loops.
- **Logic of the weighted score:** `scores = component_matrix @ (weights / weights.sum())` — a matrix times a vector gives the weighted sum for every product at once.
- **Demo:** Rating Analyzer page shows each statistic with its NumPy function name.
- **Viva:** Difference between `np.mean` and `np.average`? What is standard deviation?

## Module 3 — Pandas analysis (`analysis/pandas_analysis.py`)
- **What:** dashboard numbers, group summaries, rankings, explorer filters and sorting.
- **Logic of filtering:** each filter makes a True/False *mask*; masks are combined with `&`; `df[mask]` keeps the True rows.
- **Demo:** Product Explorer → search "samsung", set price ₹10,000–₹30,000, rating ≥ 4, sort by price.
- **Viva:** What is `groupby`? What does `agg` do? What is a boolean mask?

## Module 4 — Matplotlib charts (`analysis/visualizations.py`)
- **What:** reusable functions; each takes a DataFrame and returns a figure.
- **Why:** one function per chart avoids repeated code and the same chart is reused on several pages.
- **Library:** Matplotlib (`barh`, `hist`, `scatter`, `pie`, titles, labels, legends).
- **Demo:** Visualizations page, 7 tabs; use the *Limit charts to categories* filter and watch the charts redraw.
- **Viva:** Why a log axis for price? When do you use a histogram vs a bar chart?

## Module 5 — Popularity model (`recommendation/popularity_model.py`)
**Problem:** product A: 5.0 from 2 reviews; product B: 4.6 from 5,000 reviews. Which is better? B, obviously.

**Weighted rating:** `WR = v/(v+m)·R + m/(v+m)·C`

Small example: overall average `C = 4.0`, `m = 100` reviews.
- Product with `R = 5.0`, `v = 100` → `WR = 100/200·5.0 + 100/200·4.0 = 4.5`
- Product with `R = 5.0`, `v = 2` → `WR = 2/102·5.0 + 100/102·4.0 ≈ 4.02` (pulled strongly to the average)

In the demo data `C = 4.20` and `m ≈ 6,492` (60th percentile of review counts). Rank 1 is the *Logitech Wireless Mouse*:
rating 4.5, 31,200 reviews → WR 4.448.

## Module 6 — Content-based model (`recommendation/content_model.py`)
- **TF-IDF:** gives each word a weight. *Term Frequency* = how often it appears in this product's text.
  *Inverse Document Frequency* = how rare it is across all products. "gaming" (rare, descriptive) scores higher than
  "and" (everywhere; also removed as a stop word).
- **Cosine similarity:** the angle between two vectors. 1 = same direction, 0 = nothing in common.
- **Steps:** build one text per product → `fit_transform` → vectors; for a wish text → `transform` → `cosine_similarity`
  against all products → highest values first.
- **Demo:** ML Analyzer → model 2 → pick *Samsung Galaxy A-Series* → similar products are mostly other phones.
  Then "Match my own text": type `lightweight running shoes`.
- **Viva:** Why not just count words? What is a stop word? Why can cosine similarity be 0?

## Module 7 — Hybrid model (`recommendation/hybrid_model.py`) — the heart of the project
**Example from the demo data** — Mobiles, budget ₹10,000–₹30,000, brands Samsung/Motorola, rating ≥ 4.0,
purpose "gaming and daily use", features "5G" and "120Hz":

1. **Hard filters** keep 8 products.
2. **Four scores (0–1)** for the winner, *Motorola G-Series 5G 6GB/128GB* (₹14,999):

| Component | Score | How it was calculated | Weight |
|---|---|---|---|
| Preference | 1.00 | brand is preferred (1) and both features found (1) → average 1.00 | 0.30 |
| Content | 0.80 | TF-IDF cosine similarity of "gaming daily use 5G 120Hz" ÷ best candidate's similarity | 0.30 |
| Rating | 0.84 | weighted rating 4.2 ÷ 5 | 0.25 |
| Popularity | 0.50 | log(reviews) scaled 0–1 over the dataset | 0.15 |

3. **Weighted sum:** `0.30·1.00 + 0.30·0.80 + 0.25·0.84 + 0.15·0.50 = 0.825` → **match score 82.5 / 100**
4. **Explanation** is generated from these same numbers ("brand Motorola is one of your preferred brands; has 2 of 2
   required features; rated 4.2/5 from 6,720 reviews; …").

The second place, *Samsung Galaxy A-Series 5G*, scores 80.3 because it has the brand but only one of the two
features (preference 0.75), even though its text similarity is higher.

**If you give no brand or feature,** the preference component is dropped and the other weights are rescaled so the
product is not punished for something you never asked.

**Demo:** Recommendations page with the preferences above, then ML Analyzer → Hybrid → move the weight sliders
(for example set Popularity to 1 and the others to 0) and watch the ranking change.

## Module 8 — Streamlit app (`app.py`, `pages/`, `utils/ui.py`)
- **Pages:** every file in `pages/` becomes a sidebar page automatically; `app.py` is Home.
- **Session state:** `st.session_state` remembers the active dataset, trained model and recommendation results
  between clicks (Streamlit re-runs the whole script on every interaction).
- **Dataset flow:** upload → `load_products` (validate + clean) → `set_dataset` → every page uses `get_dataset()`.
- **Viva:** Why does Streamlit re-run the script? What is session state for?

---

## What the project does NOT do (say this honestly in the viva)
- It does **not** use live marketplace data; the dataset is demonstration data.
- It does **not** report accuracy. Without customer interaction data (who bought or clicked what), accuracy,
  precision and recall cannot be measured. The "category agreement" on the ML page is only a sanity check.
- The hybrid weights (0.30/0.30/0.25/0.15) are design choices you can change, not values learned from data.

## Suggested 5-minute demo order
1. Home → data overview, upload error demo (missing column).
2. Dashboard → numbers and charts.
3. Product Explorer → search + filters + sort + reset.
4. Recommendations → the Samsung/Motorola example; then an impossible budget to show the helpful message.
5. ML Analyzer → three models, weights, worked example, evaluation limitation.
