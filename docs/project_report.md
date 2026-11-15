# RIntelliCart — Intelligent Product Recommendation Analyzer
### B.Tech CSE (AI/ML) Mini-Project Report

*Student name:* ____________  *Roll no.:* ____________  *Guide:* ____________  *Academic year:* ____________

---

## 1. Abstract
RIntelliCart is a Python mini-project that analyses historical product ratings, prices, brands, categories and
features and recommends suitable products to a customer according to stated preferences. Data handling uses Pandas,
statistics and score calculations use NumPy, charts use Matplotlib, and the recommendation models use scikit-learn
(TF-IDF vectorisation and cosine similarity). Three models are implemented: a popularity model with a weighted rating,
a content-based model, and a hybrid model that combines preference matching, content similarity, rating and review
popularity into one explainable match score. The system is delivered as a six-page Streamlit application with dataset
upload and validation. It is demonstrated on a 110-product **demonstration dataset** created for the project; the data
is not live marketplace information.

## 2. Introduction
E-commerce sites list thousands of products per category. Customers have different budgets, preferred brands and needs.
Recommendation systems help by filtering and ranking products. This project builds a small, transparent recommender so
that every step — cleaning, analysis, scoring, ranking — can be understood and explained.

## 3. Problem Statement
Sorting products only by price or by star rating is unreliable: ratings based on very few reviews are noisy, and
generic rankings ignore a customer's budget, brand preference and required features. There is a need for a system
that combines rating quality, review volume and personal preferences, and explains each recommendation.

## 4. Existing System
- Manual browsing with category, price and rating filters.
- Ranking by average star rating alone (ignores number of reviews).
- Ranking by "best selling" or "most reviewed" (ignores fit with the customer's needs).
- Large commercial recommenders that need big interaction datasets and are not transparent to students.

## 5. Proposed System
RIntelliCart cleans a product dataset, computes rating statistics, ranks products with a review-aware weighted rating,
compares customer wish text with product text using TF-IDF and cosine similarity, and combines everything in a
weighted hybrid score. Each recommendation is accompanied by an explanation generated from the actual numbers.
Users can upload their own CSV, which is validated and cleaned before use.

## 6. Objectives
1. Load, validate and clean a product dataset (Pandas).
2. Compute rating and price statistics (NumPy).
3. Visualise the data (Matplotlib).
4. Implement popularity-based, content-based and hybrid recommenders (scikit-learn, NumPy).
5. Provide an interactive, professional web interface (Streamlit).
6. Be honest about data and evaluation limits.

## 7. Software Requirements
| Item | Requirement |
|---|---|
| Operating system | Windows 10/11, macOS or Linux |
| Python | 3.10 or later |
| Libraries | numpy, pandas, matplotlib, scikit-learn, streamlit, pytest (see `requirements.txt`) |
| Browser | Any modern browser (Chrome, Edge, Firefox) |
| Editor (optional) | VS Code or PyCharm |

## 8. Hardware Requirements
| Item | Minimum |
|---|---|
| Processor | Dual-core, 2 GHz |
| RAM | 4 GB |
| Storage | 500 MB free |
| Display | 1366 × 768 or higher (the interface also works on a phone-sized screen) |

## 9. Technologies Used
Python (language), NumPy (numerical arrays), Pandas (tabular data), Matplotlib (charts), scikit-learn
(`TfidfVectorizer`, `cosine_similarity`), Streamlit (web interface), CSV (data format), pytest (testing).

## 10. System Architecture
Layered design (see `architecture.md` for the diagram):
**Data layer** (`data/`) → **Analysis layer** (`analysis/`) → **Recommendation layer** (`recommendation/`) →
**Presentation layer** (`app.py`, `pages/`, `utils/`). The dataset is held in Streamlit session state so all pages share
the same cleaned data.

## 11. Dataset Description
`products.csv` — **demonstration data**, 110 products, 10 categories (11 each), 88 distinct brands.

| Column | Meaning |
|---|---|
| product_id | Unique id (P001 …) |
| product_name, brand, category | Descriptive text |
| price | Price in ₹ (illustrative) |
| rating | Average rating 0–5 (illustrative) |
| review_count | Number of reviews (illustrative count; no review text exists) |
| description, features | Text used by the content model; features separated by `;` |
| popularity | Demonstration index `min(100, 18·log10(reviews) + 8·(rating − 3))` |

Summary statistics computed from the file: average rating 4.20, median 4.2, standard deviation 0.222, minimum 3.6,
maximum 4.6, total reviews 839,410, review-weighted average rating 4.221.
The data was written for this project; it is not scraped, not live, and not verified against any marketplace.

## 12. Implementation
- **Cleaning** (`data_cleaning.py`): column validation, whitespace stripping, numeric conversion, removal of rows missing
  id/name/price/rating, filling review counts with 0, duplicate removal, rating clipping, popularity fallback, a cleaning report.
- **NumPy** (`numpy_analysis.py`): `mean`, `median`, `std`, `max`, `min`, `average` (weighted), `histogram`, `percentile`,
  min-max scaling, matrix-vector weighted score.
- **Pandas** (`pandas_analysis.py`): boolean-mask filtering, sorting, `groupby` + `agg`, rankings.
- **Matplotlib** (`visualizations.py`): bar charts, histogram, scatter plot (log price axis), pie chart; all return figures.
- **scikit-learn** (`content_model.py`): `ContentRecommender` class with `fit`, `similar_products`, `query_similarity`.
- **Streamlit**: Home plus six pages with session state, a preference form, spinners, success/error/empty-state messages.

## 13. Algorithms
### 13.1 Weighted rating (popularity model)
`WR = v/(v+m)·R + m/(v+m)·C` where R = product rating, v = its review count, C = mean rating of all products,
m = 60th percentile of review counts (6,492 in the demo data).

### 13.2 TF-IDF + cosine similarity (content model)
1. Text per product = name + brand + category + description + features (brand and category repeated for extra weight).
2. `TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)` creates sparse vectors.
3. `cosine(a, b) = (a·b) / (‖a‖‖b‖)`; for a customer text the same vectoriser transforms the text and similarity is
   computed against every product.

### 13.3 Hybrid ranking
1. Hard filters: category, budget range, minimum rating (optionally preferred brands only).
2. Components in [0, 1]: preference (brand match and share of required features), content similarity (divided by the best
   candidate's similarity), weighted rating ÷ 5, popularity (log of review count, min-max scaled).
3. `match score = 100 × Σ(wᵢ·componentᵢ) / Σwᵢ`, default weights 0.30 / 0.30 / 0.25 / 0.15.
   Components without customer input are dropped and the remaining weights rescaled.
4. Products are sorted by match score; the explanation is built from the same values.

## 14. Results
Values below were produced by running the code on the demonstration dataset.

- **Highest-rated products:** Samsung Galaxy S-Series Flagship and Lenovo ThinkPad-Class Ultraportable (rating 4.6).
- **Top by weighted rating:** Logitech Wireless Mouse (4.448), Nivea Soft Moisturising Cream (4.438), Jockey Cotton Innerwear (4.420).
- **Category averages:** highest Electronics (4.28), lowest Women's Fashion (4.11).
- **Example recommendation** (Mobiles, ₹10,000–₹30,000, Samsung/Motorola preferred, rating ≥ 4.0, "gaming and daily use",
  features 5G and 120Hz): 8 products pass the filters; rank 1 is *Motorola G-Series 5G 6GB/128GB* with match score 82.5,
  rank 2 *Samsung Galaxy A-Series 5G* with 80.3, rank 3 *Motorola Edge 5G* with 76.6.
- **Edge cases handled:** impossible budget → message that explains price ranges; rating too high → message with the best available rating; invalid input → validation error.
- **Content-model sanity check:** for 88.5 % of (product, neighbour) pairs among the top-5 text-similar products, the neighbour is in the same category. This is *only* a sanity check.
- **Testing:** 41 pytest tests pass (data, cleaning, NumPy, Pandas, charts, three recommenders).

**No accuracy, precision or recall is reported.** These need ground-truth data about real customer behaviour, which the
demonstration dataset does not contain.

## 15. Screenshots
*Insert screenshots of the running application here (take them after running `streamlit run app.py`):*

| # | Screen | Figure |
|---|---|---|
| 1 | Home page and data overview | Figure 1 — ____ |
| 2 | Dashboard | Figure 2 — ____ |
| 3 | Product Explorer with filters | Figure 3 — ____ |
| 4 | Recommendations (Samsung/Motorola example) | Figure 4 — ____ |
| 5 | Rating Analyzer | Figure 5 — ____ |
| 6 | Visualizations | Figure 6 — ____ |
| 7 | ML Analyzer (hybrid worked example) | Figure 7 — ____ |

The seven Matplotlib charts generated by the code are saved in `docs/charts/` and can be used in this report.

## 16. Limitations
- Demonstration data, not live; 110 products only.
- No customer interaction data, so recommendation accuracy cannot be evaluated.
- Hybrid weights are chosen by design, not learned.
- TF-IDF matches words, not meaning (synonyms such as "phone" and "mobile" are not linked).
- Feature matching is simple text matching.
- No user accounts or personalised history; no real-time prices.

## 17. Future Enhancements
- Collect real user ratings/clicks and evaluate with precision@k, recall@k.
- Learn the hybrid weights from data.
- Collaborative filtering (user-item matrix) once interaction data exists.
- Word embeddings for meaning-based text similarity.
- Connect to a verified product database or an official data source.
- User accounts and saved wishlists.

## 18. Conclusion
RIntelliCart shows how NumPy, Pandas, Matplotlib and scikit-learn combine in one practical application. It cleans and
analyses product data, visualises it, and produces transparent, explainable recommendations from real calculations
rather than hard-coded or random values. Its limitations — demonstration data and no ground truth for evaluation — are
stated openly and point to clear next steps.
