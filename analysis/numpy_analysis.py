"""NumPy calculations used across the project.

NumPy works on fast arrays of numbers. Here we turn DataFrame columns into
arrays (with .to_numpy()) and compute statistics and scores with them.
"""
import numpy as np


def rating_statistics(ratings, review_counts=None):
    """Return basic statistics for an array of ratings.

    If review_counts is given we also compute the review-weighted average
    (a product with 10,000 reviews counts more than one with 10).
    """
    ratings = np.asarray(ratings, dtype=float)
    if ratings.size == 0:
        raise ValueError("No ratings available.")
    stats = {
        "average": float(np.mean(ratings)),
        "maximum": float(np.max(ratings)),
        "minimum": float(np.min(ratings)),
        "median": float(np.median(ratings)),
        "std_dev": float(np.std(ratings)),   # population standard deviation
        "count": int(ratings.size),
    }
    if review_counts is not None:
        counts = np.asarray(review_counts, dtype=float)
        stats["total_reviews"] = int(np.sum(counts))
        stats["weighted_average"] = float(np.average(ratings, weights=counts)) if counts.sum() > 0 else stats["average"]
    return stats


def price_statistics(prices):
    """Return basic statistics for an array of prices."""
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        raise ValueError("No prices available.")
    return {
        "average": float(np.mean(prices)),
        "median": float(np.median(prices)),
        "minimum": float(np.min(prices)),
        "maximum": float(np.max(prices)),
        "q1": float(np.percentile(prices, 25)),
        "q3": float(np.percentile(prices, 75)),
    }


def compare_with_average(ratings):
    """Compare each rating with the overall average.

    Returns an array: positive = above average, negative = below average.
    """
    ratings = np.asarray(ratings, dtype=float)
    return ratings - np.mean(ratings)


def rating_distribution(ratings):
    """Count ratings in bins: <3, 3-3.5, 3.5-4, 4-4.5, 4.5-5 (uses np.histogram)."""
    edges = np.array([0, 3.0, 3.5, 4.0, 4.5, 5.01])
    counts, _ = np.histogram(np.asarray(ratings, dtype=float), bins=edges)
    labels = ["Below 3.0", "3.0 - 3.4", "3.5 - 3.9", "4.0 - 4.4", "4.5 - 5.0"]
    return labels, counts


def min_max_scale(values):
    """Scale numbers to the range 0..1:  (x - min) / (max - min).

    If all values are equal we return zeros (avoids dividing by zero).
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    spread = values.max() - values.min()
    if spread == 0:
        return np.zeros_like(values)
    return (values - values.min()) / spread


def weighted_score(component_matrix, weights):
    """Combine score components with weights (used by the hybrid recommender).

    component_matrix: 2-D array, one row per product, one column per component,
                      every value between 0 and 1.
    weights:          1-D array with one weight per column.
    Weights are re-scaled to add up to 1, so the result is also between 0 and 1.
    """
    component_matrix = np.asarray(component_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("At least one weight must be greater than zero.")
    return component_matrix @ (weights / total)   # matrix x vector = weighted sum per row
