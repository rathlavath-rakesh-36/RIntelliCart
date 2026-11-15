"""Data validation and cleaning (Pandas).

Why this file exists:
    Real-world CSV files are messy. Before any analysis or machine learning,
    we check the columns and clean the data so every other module can trust it.
"""
import numpy as np
import pandas as pd

# Columns every product file must contain.
REQUIRED_COLUMNS = [
    "product_id", "product_name", "brand", "category", "price",
    "rating", "review_count", "description", "features", "popularity",
]
OPTIONAL_COLUMNS = ["image_url"]   # optional: web address of a product photo
TEXT_COLUMNS = ["product_id", "product_name", "brand", "category", "description", "features"]
NUMBER_COLUMNS = ["price", "rating", "review_count", "popularity"]


class DatasetError(ValueError):
    """Raised when a dataset cannot be used (for example, a missing column)."""


def validate_columns(df):
    """Raise DatasetError with a clear message if a required column is missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Required columns are: " + ", ".join(REQUIRED_COLUMNS) + "."
        )


def clean_products(df):
    """Clean a product DataFrame and return (clean_df, report).

    Steps (each one is counted in the report so the user can see what changed):
      1. Validate that required columns exist.
      2. Keep only the required columns and strip extra spaces from text.
      3. Convert number columns to numbers (bad values become NaN).
      4. Drop rows with no usable product_id / name / price / rating.
      5. Fill remaining gaps (review_count -> 0, text -> "").
      6. Remove duplicate products.
      7. Clip rating to 0-5 and make sure price is positive.
      8. Fill a missing popularity value with a simple rating x reviews index.
    """
    validate_columns(df)
    rows_in = len(df)
    keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    data = df[keep].copy()

    # Step 2: text cleanup
    for col in TEXT_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in data.columns]:
        data[col] = data[col].astype("string").str.strip()
    data = data.replace({"": pd.NA})

    # Step 3: numeric conversion
    for col in NUMBER_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Step 4: rows that cannot be analysed
    must_have = ["product_id", "product_name", "price", "rating"]
    missing_before = int(data[must_have].isna().any(axis=1).sum())
    data = data.dropna(subset=must_have)

    # Step 5: fill the gaps that are safe to fill
    filled_reviews = int(data["review_count"].isna().sum())
    data["review_count"] = data["review_count"].fillna(0)
    for col in ["brand", "category", "description", "features"]:
        data[col] = data[col].fillna("Unknown" if col in ("brand", "category") else "")

    # Step 6: duplicates (same id, or same name + brand)
    before = len(data)
    data = data.drop_duplicates(subset="product_id", keep="first")
    data = data.drop_duplicates(subset=["product_name", "brand"], keep="first")
    duplicates_removed = before - len(data)

    # Step 7: impossible values
    invalid_value_rows = int(((data["rating"] < 0) | (data["rating"] > 5) | (data["price"] <= 0)).sum())
    data["rating"] = data["rating"].clip(0, 5)
    data = data[data["price"] > 0]
    data["review_count"] = data["review_count"].clip(lower=0).round().astype(int)

    # Step 8: popularity fallback
    missing_popularity = int(data["popularity"].isna().sum())
    # Same simple index used for the demo data: grows with reviews and rating, max 100.
    fallback = np.minimum(
        100.0,
        18 * np.log10(data["review_count"].clip(lower=1)) + 8 * (data["rating"] - 3),
    )
    data["popularity"] = data["popularity"].fillna(fallback).round(1)

    for col in TEXT_COLUMNS:
        data[col] = data[col].astype(str)
    if "image_url" in data.columns:
        data["image_url"] = data["image_url"].fillna("").astype(str)
    data = data.reset_index(drop=True)

    if data.empty:
        raise DatasetError("No usable rows are left after cleaning. Please check the file contents.")

    report = {
        "rows_in": rows_in,
        "rows_out": len(data),
        "rows_dropped_missing_values": missing_before,
        "review_counts_filled_with_zero": filled_reviews,
        "duplicates_removed": duplicates_removed,
        "rows_with_invalid_values": invalid_value_rows,
        "popularity_values_filled": missing_popularity,
    }
    return data, report
