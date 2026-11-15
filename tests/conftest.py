"""Pytest setup: make the project root importable and share a clean demo dataset."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.data_loader import load_products  # noqa: E402
from recommendation.content_model import ContentRecommender  # noqa: E402


@pytest.fixture(scope="session")
def products():
    df, _ = load_products()
    return df


@pytest.fixture(scope="session")
def content_model(products):
    return ContentRecommender().fit(products)
