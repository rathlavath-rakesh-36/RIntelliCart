"""Loading product data from CSV files (Pandas).

The same function works for the built-in demo file and for a file the user uploads
in the app (Streamlit gives us a file-like object, which pandas can read directly).
"""
from pathlib import Path

import pandas as pd

from data.data_cleaning import DatasetError, clean_products

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "products.csv"

DEMO_DATA_NOTICE = (
    "The built-in dataset is DEMONSTRATION data created for this mini-project. "
    "Prices, ratings and review counts are illustrative, not live marketplace "
    "information, and there are no real customer reviews in it."
)


def load_products(source=DEFAULT_DATASET_PATH):
    """Read a CSV (path or uploaded file), validate and clean it.

    Returns (clean_dataframe, cleaning_report). Raises DatasetError with a
    readable message if the file cannot be used.
    """
    try:
        raw = pd.read_csv(source)
    except pd.errors.EmptyDataError as error:
        raise DatasetError("The file is empty.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DatasetError(f"The file could not be read as a CSV: {error}") from error
    except FileNotFoundError as error:
        raise DatasetError(f"File not found: {source}") from error
    return clean_products(raw)


def dataset_to_csv_bytes(df):
    """Convert a DataFrame to CSV bytes for the download button."""
    return df.to_csv(index=False).encode("utf-8")
