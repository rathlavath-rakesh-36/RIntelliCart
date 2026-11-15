"""Pandas analysis: overview, grouping, filtering, sorting and rankings."""
import numpy as np
import pandas as pd

SORT_OPTIONS = {
    "Default order": None,
    "Price: Low to High": ("price", True),
    "Price: High to Low": ("price", False),
    "Rating: High to Low": ("rating", False),
    "Popularity: High to Low": ("popularity", False),
    "Most reviewed": ("review_count", False),
}


# ---------- dataset overview ----------

def dataset_overview(df):
    """Facts shown in the 'data overview' section."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_ids": int(df["product_id"].duplicated().sum()),
    }


def column_summary(df):
    """One row per column: data type, missing values and unique values."""
    return pd.DataFrame({
        "column": df.columns,
        "data_type": [str(t) for t in df.dtypes],
        "missing": df.isna().sum().values,
        "unique_values": df.nunique().values,
    })


# ---------- headline numbers for the dashboard ----------

def dashboard_metrics(df):
    """Headline values for the dashboard cards (all computed from the data)."""
    # Highest-rated product: best rating, ties broken by number of reviews.
    top = df.sort_values(["rating", "review_count"], ascending=False).iloc[0]
    # Most popular category = the category with the most total reviews.
    reviews_by_category = df.groupby("category")["review_count"].sum()
    return {
        "total_products": len(df),
        "total_categories": df["category"].nunique(),
        "total_brands": df["brand"].nunique(),
        "average_rating": float(np.mean(df["rating"].to_numpy())),
        "top_product_name": top["product_name"],
        "top_product_rating": float(top["rating"]),
        "popular_category": reviews_by_category.idxmax(),
        "popular_category_reviews": int(reviews_by_category.max()),
    }


# ---------- grouping and aggregation ----------

def category_summary(df):
    """Per-category product count, average rating, total reviews and average price."""
    summary = df.groupby("category").agg(
        products=("product_id", "count"),
        average_rating=("rating", "mean"),
        total_reviews=("review_count", "sum"),
        average_price=("price", "mean"),
    ).reset_index()
    return summary.sort_values("products", ascending=False).reset_index(drop=True)


def brand_summary(df, min_products=1):
    """Per-brand product count and ratings. Brands with fewer products than min_products are hidden."""
    summary = df.groupby("brand").agg(
        products=("product_id", "count"),
        average_rating=("rating", "mean"),
        total_reviews=("review_count", "sum"),
    ).reset_index()
    summary = summary[summary["products"] >= min_products]
    return summary.sort_values(["average_rating", "total_reviews"], ascending=False).reset_index(drop=True)


# ---------- rankings ----------

def top_rated(df, n=10):
    """Highest rating first; ties are broken by more reviews."""
    return df.sort_values(["rating", "review_count"], ascending=False).head(n).reset_index(drop=True)


def lowest_rated(df, n=10):
    """Lowest rating first; ties are broken by more reviews (more reliable low score)."""
    return df.sort_values(["rating", "review_count"], ascending=[True, False]).head(n).reset_index(drop=True)


def most_reviewed(df, n=10):
    return df.sort_values("review_count", ascending=False).head(n).reset_index(drop=True)


def most_popular(df, n=10):
    return df.sort_values("popularity", ascending=False).head(n).reset_index(drop=True)


# ---------- filtering and sorting (Product Explorer) ----------

def filter_products(df, search="", categories=None, brands=None,
                    price_range=None, min_rating=0.0):
    """Apply all explorer filters with boolean masks. Empty selections mean 'no filter'."""
    mask = pd.Series(True, index=df.index)
    search = (search or "").strip().lower()
    if search:
        mask &= df["product_name"].str.lower().str.contains(search, regex=False)
    if categories:
        mask &= df["category"].isin(categories)
    if brands:
        mask &= df["brand"].isin(brands)
    if price_range is not None:
        low, high = price_range
        mask &= df["price"].between(low, high)
    mask &= df["rating"] >= min_rating
    return df[mask]


def sort_products(df, sort_label):
    """Sort using one of the labels in SORT_OPTIONS."""
    rule = SORT_OPTIONS.get(sort_label)
    if rule is None:
        return df
    column, ascending = rule
    return df.sort_values(column, ascending=ascending, kind="stable")
