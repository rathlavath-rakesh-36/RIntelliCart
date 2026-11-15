# RIntelliCart Architecture

## Layers

```
┌───────────────────────────────────────────────────────────────┐
│ PRESENTATION  (Streamlit)                                     │
│ app.py (Home) · pages/1..6 · utils/ui.py · utils/helpers.py   │
└───────────────┬───────────────────────────────────────────────┘
                │ calls
┌───────────────▼───────────────────────────────────────────────┐
│ RECOMMENDATION  (scikit-learn + NumPy)                        │
│ popularity_model · content_model (TF-IDF, cosine) · hybrid    │
└───────────────┬───────────────────────────────────────────────┘
                │ uses
┌───────────────▼───────────────────────────────────────────────┐
│ ANALYSIS                                                      │
│ numpy_analysis · pandas_analysis · visualizations (Matplotlib)│
└───────────────┬───────────────────────────────────────────────┘
                │ reads cleaned DataFrame
┌───────────────▼───────────────────────────────────────────────┐
│ DATA  (Pandas)                                                │
│ data_loader · data_cleaning · products.csv / uploaded CSV     │
└───────────────────────────────────────────────────────────────┘
```

Rule: a lower layer never imports from a higher one. `utils/helpers.py` and everything below `utils/ui.py` do not
import Streamlit, which is why they can be tested with plain pytest.

## Data flow
```
CSV (demo or upload)
   → data_loader.load_products → data_cleaning.clean_products   (validate, clean, report)
   → st.session_state["df"]                                      (shared by all pages)
        ├─ pandas_analysis / numpy_analysis → tables, KPI cards
        ├─ visualizations                    → Matplotlib figures → st.pyplot
        └─ ContentRecommender.fit            → TF-IDF matrix (cached per dataset)
              └─ hybrid_recommend(prefs)     → ranked, explained products
```

## Hybrid recommendation flow
```
customer preferences ──► validate ──► hard filters (category, budget, rating[, brands])
                                          │ none left? → diagnose_no_results → helpful message
                                          ▼
                      ┌──────────── component scores (0-1) ────────────┐
                      │ preference   brand match + required features    │
                      │ content      TF-IDF cosine / best candidate     │
                      │ rating       weighted rating / 5                │
                      │ popularity   scaled log(review count)           │
                      └────────────────────┬───────────────────────────┘
                                           ▼
                 drop unused components → rescale weights → NumPy weighted sum
                                           ▼
                    sort → rank → explanation text → table + cards
```

## Session state keys
| Key | Content |
|---|---|
| `df`, `report`, `dataset_name`, `dataset_id` | Active cleaned dataset and its cleaning report |
| `content_model` | `(dataset_id, ContentRecommender)` — retrained only when the dataset changes |
| `exp_*` | Product Explorer filter widgets (cleared by *Reset*) |
| `rec_*` | Recommendation form widgets and the last `rec_outcome` |
| `ml_*` | ML Analyzer and Rating Analyzer widgets |

Loading a new dataset (`set_dataset`) clears all derived state so no page shows stale results.

## Design decisions
- **Streamlit multipage app**: no separate frontend or server; each file in `pages/` is a page.
- **Figure objects, not pyplot**: `matplotlib.figure.Figure` is used directly, which avoids global state and memory
  leaks when charts are redrawn on every interaction.
- **Pure functions for logic**: recommenders take DataFrames and return DataFrames/dicts; pages only collect input and display results.
- **Brand preference is "soft" by default** (a score boost); a checkbox makes it strict.
- **Honest evaluation**: no accuracy metrics are produced because no ground truth exists.
