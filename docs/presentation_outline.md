# RIntelliCart — Presentation Outline (12 slides, about 10 minutes)

**Slide 1 — Title**
RIntelliCart: Intelligent Product Recommendation Analyzer · *Discover Better Products with Intelligent Recommendations.*
Your name, roll number, guide, department, date.

**Slide 2 — Introduction**
- Too many products, different needs. Recommendation systems help.
- Project: analyse ratings + preferences → ranked, explained recommendations.
- Built with NumPy, Pandas, Matplotlib, scikit-learn, Streamlit.

**Slide 3 — Problem Statement**
- Star ratings alone mislead (5.0 from 2 reviews vs 4.6 from 5,000).
- Filters ignore budget, brand and feature needs together.
- Need: transparent recommendations that explain *why*.

**Slide 4 — Objectives**
Clean and analyse data · visualise · three recommenders · interactive app · be honest about limits.

**Slide 5 — Technologies**
Table: Python · NumPy (statistics, scoring) · Pandas (cleaning, grouping) · Matplotlib (charts) · scikit-learn (TF-IDF, cosine) · Streamlit (web).

**Slide 6 — System Architecture**
Use the layer diagram from `architecture.md`: Data → Analysis → Recommendation → Presentation. One line on session state.

**Slide 7 — Dataset**
110 demonstration products · 10 categories · 10 columns · **demo data, not live marketplace data**.
Show the cleaning report and the upload-validation error message.

**Slide 8 — Recommendation Algorithm**
- Popularity: weighted rating formula.
- Content: TF-IDF + cosine similarity.
- Hybrid: filters → 4 scores → weighted sum → match score.
- Mini example: 0.30·1.00 + 0.30·0.80 + 0.25·0.84 + 0.15·0.50 = 0.825 → 82.5.

**Slide 9 — Screenshots / Live Demo**
Dashboard · Explorer · Recommendations · Rating Analyzer · Visualizations · ML Analyzer.
Demo order: Dashboard → Explorer filters → Samsung/Motorola recommendation → impossible budget → weight sliders.

**Slide 10 — Results**
- 110 products, average rating 4.20, std 0.222, 839,410 reviews.
- Example recommendation: Motorola G-Series 5G ranked first (82.5).
- 41 automated tests pass.
- Charts from `docs/charts/`.

**Slide 11 — Limitations and Future Scope**
Limitations: demo data, no interaction data → no accuracy metric, hand-set weights, word matching only.
Future: real interaction data + precision@k, learned weights, collaborative filtering, embeddings, verified data source.

**Slide 12 — Conclusion and Q&A**
The four libraries work together to give transparent, calculated recommendations. Thank you — questions?
