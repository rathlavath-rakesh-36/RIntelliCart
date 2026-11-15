"""Model 1: Popularity-based recommendation with a weighted rating.

Problem: a product with a 5.0 rating from 2 reviews should not beat a product
with a 4.6 rating from 5,000 reviews.

Solution (the IMDb weighted-rating formula):

    WR = (v / (v + m)) * R  +  (m / (v + m)) * C

    R = the product's own average rating
    v = the product's number of reviews
    C = the average rating of ALL products
    m = minimum reviews needed to be taken seriously
        (here: the 60th percentile of review counts, computed with NumPy)

Products with few reviews are pulled towards the overall average C.
Products with many reviews keep a score close to their own rating R.
"""
import numpy as np


def compute_weighted_rating(df, percentile=60):
    """Add a 'weighted_rating' column and return (new_df, parameters)."""
    ratings = df["rating"].to_numpy(dtype=float)
    votes = df["review_count"].to_numpy(dtype=float)
    mean_rating = float(np.mean(ratings))                    # C
    min_votes = float(np.percentile(votes, percentile))      # m
    denominator = votes + min_votes
    # If both v and m are 0 the formula would divide by zero: use the plain mean then.
    safe = np.where(denominator == 0, 1.0, denominator)
    weighted = np.where(denominator == 0, mean_rating,
                        (votes / safe) * ratings + (min_votes / safe) * mean_rating)
    result = df.copy()
    result["weighted_rating"] = weighted
    return result, {"mean_rating": mean_rating, "min_votes": min_votes, "percentile": percentile}


def popularity_recommendations(df, n=10, category=None, percentile=60):
    """Return the top n products ranked by weighted rating (optionally inside one category).

    The weighted rating is always calculated on the FULL dataset so scores stay
    comparable, then we filter by category.
    """
    scored, params = compute_weighted_rating(df, percentile)
    if category and category != "All":
        scored = scored[scored["category"] == category]
    scored = scored.sort_values(["weighted_rating", "review_count"], ascending=False).head(n).reset_index(drop=True)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1))
    return scored, params
