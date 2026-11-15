"""Model 3: Hybrid recommendation (preferences + content + rating + popularity).

How a product gets its Match Score
----------------------------------
Step 1 - HARD FILTERS (a product must pass these to be considered)
    category, minimum budget, maximum budget, minimum rating
    (and "preferred brands only" if the customer switches that on)

Step 2 - FOUR SCORE COMPONENTS, each between 0 and 1
    preference  = average of  [brand match (1 or 0)]  and  [share of required features found]
    content     = TF-IDF cosine similarity between the customer's purpose/feature text
                  and the product text, divided by the best similarity among candidates
    rating      = weighted rating (popularity model) / 5
    popularity  = log(review_count) scaled to 0-1 over the whole dataset

Step 3 - WEIGHTED SUM (NumPy)
    match_score = w1*preference + w2*content + w3*rating + w4*popularity
    Weights are rescaled to add up to 1. A component the customer gave no
    information for (e.g. no brand or feature chosen) is left out and the
    remaining weights are rescaled, so nobody is rewarded or punished for it.

Step 4 - Sort by match_score. The explanation text is built from the same numbers.
"""
import numpy as np

from analysis.numpy_analysis import min_max_scale, weighted_score
from recommendation.popularity_model import compute_weighted_rating

DEFAULT_WEIGHTS = {"preference": 0.30, "content": 0.30, "rating": 0.25, "popularity": 0.15}
COMPONENT_LABELS = {
    "preference": "Preference match",
    "content": "Content similarity",
    "rating": "Weighted rating",
    "popularity": "Review popularity",
}


def default_preferences():
    """A blank set of preferences (nothing filtered)."""
    return {
        "category": "All", "min_budget": 0, "max_budget": None, "brands": [],
        "min_rating": 0.0, "purpose": "", "required_features": [], "strict_brands": False,
    }


def validate_preferences(prefs):
    """Check customer input; raise ValueError with a friendly message if it is invalid."""
    low, high = prefs.get("min_budget") or 0, prefs.get("max_budget")
    if low < 0:
        raise ValueError("Minimum budget cannot be negative.")
    if high is not None and high < low:
        raise ValueError("Maximum budget must be greater than or equal to the minimum budget.")
    if not 0 <= prefs.get("min_rating", 0) <= 5:
        raise ValueError("Minimum rating must be between 0 and 5.")


def validate_weights(weights):
    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative.")
    if sum(weights.values()) <= 0:
        raise ValueError("At least one weight must be greater than zero.")


def _feature_matches(df, required_features):
    """Return (match_fraction array, list of matched-feature lists), one entry per product."""
    searchable = (df["product_name"] + " " + df["features"] + " " + df["description"]).str.lower()
    matched = [[f for f in required_features if f.lower() in text] for text in searchable]
    fraction = np.array([len(m) / len(required_features) for m in matched])
    return fraction, matched


def _hard_filter(df, prefs):
    """Apply the hard filters and return the boolean mask of products that pass."""
    mask = np.ones(len(df), dtype=bool)
    if prefs["category"] not in (None, "All"):
        mask &= (df["category"] == prefs["category"]).to_numpy()
    mask &= (df["price"] >= (prefs["min_budget"] or 0)).to_numpy()
    if prefs["max_budget"] is not None:
        mask &= (df["price"] <= prefs["max_budget"]).to_numpy()
    mask &= (df["rating"] >= prefs["min_rating"]).to_numpy()
    if prefs["strict_brands"] and prefs["brands"]:
        mask &= df["brand"].isin(prefs["brands"]).to_numpy()
    return mask


def diagnose_no_results(df, prefs):
    """Explain which criteria are too strict (used when nothing matches)."""
    tips = []
    if prefs["category"] not in (None, "All"):
        in_category = df[df["category"] == prefs["category"]]
        tips.append(f"{len(in_category)} product(s) exist in the category '{prefs['category']}'.")
        if in_category.empty:
            return tips
    else:
        in_category = df
    low, high = prefs["min_budget"] or 0, prefs["max_budget"]
    in_budget = in_category[(in_category["price"] >= low) &
                            ((in_category["price"] <= high) if high is not None else True)]
    if in_budget.empty:
        cheapest, dearest = in_category["price"].min(), in_category["price"].max()
        tips.append(f"No product fits the budget. Prices here range from ₹{cheapest:,.0f} to ₹{dearest:,.0f}. "
                    "Try widening the budget.")
        return tips
    tips.append(f"{len(in_budget)} product(s) fit the budget.")
    good_rating = in_budget[in_budget["rating"] >= prefs["min_rating"]]
    if good_rating.empty:
        tips.append(f"None of them has a rating of {prefs['min_rating']:.1f} or more "
                    f"(the best is {in_budget['rating'].max():.1f}). Try lowering the minimum rating.")
        return tips
    tips.append(f"{len(good_rating)} of them have a rating of {prefs['min_rating']:.1f} or more.")
    if prefs["strict_brands"] and prefs["brands"]:
        tips.append("None of them belongs to the selected brand(s). "
                    "Try switching off 'Show preferred brands only' or adding more brands.")
    return tips


def _build_explanation(row, prefs, matched_features, component_values, used):
    """Write a plain-English reason from the real numbers of one product."""
    reasons = []
    if prefs["category"] not in (None, "All"):
        reasons.append(f"belongs to {row['category']}")
    reasons.append(f"price ₹{row['price']:,.0f} is inside your budget")
    if prefs["brands"]:
        if row["brand"] in prefs["brands"]:
            reasons.append(f"brand {row['brand']} is one of your preferred brands")
        else:
            reasons.append(f"brand {row['brand']} is not in your preferred list")
    if prefs["required_features"]:
        total = len(prefs["required_features"])
        if matched_features:
            reasons.append(f"has {len(matched_features)} of {total} required features ({', '.join(matched_features)})")
        else:
            reasons.append(f"has none of the {total} required features")
    reasons.append(f"rated {row['rating']:.1f}/5 from {row['review_count']:,} reviews")
    if "content" in used:
        reasons.append(f"text similarity to your purpose is {component_values['content']:.2f}")
    return "This product " + "; ".join(reasons) + "."


def hybrid_recommend(df, prefs, content_model, weights=None, n=10):
    """Rank products for a customer. Returns a dict:

        results      DataFrame sorted by match_score (empty if nothing matches)
        used_weights the rescaled weights that were actually applied
        diagnostics  list of helpful sentences (filled when results are empty)
        candidates   how many products passed the hard filters
    """
    prefs = {**default_preferences(), **prefs}
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    validate_preferences(prefs)
    validate_weights(weights)

    # Popularity values are scaled on the FULL dataset so they stay comparable.
    scored_all, _ = compute_weighted_rating(df)
    popularity_all = min_max_scale(np.log1p(df["review_count"].to_numpy(dtype=float)))

    mask = _hard_filter(df, prefs)
    if not mask.any():
        return {"results": df.iloc[0:0], "used_weights": {}, "candidates": 0,
                "diagnostics": diagnose_no_results(df, prefs)}

    candidates = scored_all[mask].copy()
    positions = np.flatnonzero(mask)

    # --- component 1: preference match (brand + features)
    preference_parts, matched_lists = [], [[] for _ in range(len(candidates))]
    if prefs["brands"]:
        preference_parts.append(candidates["brand"].isin(prefs["brands"]).to_numpy().astype(float))
    if prefs["required_features"]:
        fraction, matched_lists = _feature_matches(candidates, prefs["required_features"])
        preference_parts.append(fraction)
    preference = np.mean(preference_parts, axis=0) if preference_parts else None

    # --- component 2: content similarity (TF-IDF + cosine), relative to the best candidate
    query_text = (prefs["purpose"] + " " + " ".join(prefs["required_features"])).strip()
    similarity_all = content_model.query_similarity(query_text)[positions]
    content = similarity_all / similarity_all.max() if similarity_all.max() > 0 else None

    # --- components 3 and 4: quality and popularity
    rating = np.clip(candidates["weighted_rating"].to_numpy() / 5.0, 0, 1)
    popularity = popularity_all[positions]

    # Keep only the components we actually have information for.
    available = {"preference": preference, "content": content, "rating": rating, "popularity": popularity}
    used = [name for name, values in available.items() if values is not None and weights.get(name, 0) > 0]
    if not used:
        raise ValueError("All active components have zero weight. Increase at least one weight.")
    matrix = np.column_stack([available[name] for name in used])
    weight_vector = np.array([weights[name] for name in used])
    final = weighted_score(matrix, weight_vector)
    used_weights = dict(zip(used, (weight_vector / weight_vector.sum()).tolist()))

    for name in ("preference", "content", "rating", "popularity"):
        values = available[name]
        candidates[f"{name}_score"] = np.nan if values is None else values
    candidates["match_score"] = np.round(final * 100, 1)       # shown as 0-100
    candidates["matched_features"] = [", ".join(m) for m in matched_lists]

    candidates = candidates.sort_values(["match_score", "weighted_rating"], ascending=False).head(n)
    explanations = []
    for _, row in candidates.iterrows():
        values = {name: row[f"{name}_score"] for name in used}
        matched = [f for f in row["matched_features"].split(", ") if f]
        explanations.append(_build_explanation(row, prefs, matched, values, used))
    candidates["explanation"] = explanations
    candidates = candidates.reset_index(drop=True)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    return {"results": candidates, "used_weights": used_weights,
            "candidates": int(mask.sum()), "diagnostics": []}
