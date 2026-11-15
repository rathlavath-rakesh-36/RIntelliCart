"""Model 2: Content-based recommendation (TF-IDF + cosine similarity, scikit-learn).

Idea: describe every product as one piece of text, turn the text into numbers,
and compare products (or a customer's wish text) by how similar those numbers are.

Step by step
    1. Build one text per product: name + brand + category + description + features.
    2. TF-IDF turns each text into a vector. Words that are common in ONE product
       but rare across all products get a high weight (e.g. "gaming"),
       and words found everywhere (e.g. "the") get a low weight.
    3. Cosine similarity measures the angle between two vectors:
       1.0 = same direction (very similar text), 0.0 = nothing in common.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_product_text(df):
    """Join the descriptive columns of every product into one lowercase text."""
    # Brand and category are repeated so they carry a bit more weight than other words.
    parts = (df["product_name"] + " " + df["brand"] + " " + df["brand"] + " "
             + df["category"] + " " + df["category"] + " "
             + df["description"] + " " + df["features"].str.replace(";", " "))
    return parts.str.lower()


class ContentRecommender:
    """Reusable content model: call fit(df) once, then use the other methods.

    It works with any dataset that has passed the cleaning step, including
    newly uploaded files.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = None
        self.products = None

    def fit(self, df):
        if df.empty:
            raise ValueError("Cannot fit the content model on an empty dataset.")
        self.products = df.reset_index(drop=True)
        self.matrix = self.vectorizer.fit_transform(build_product_text(self.products))
        return self

    def _check_fitted(self):
        if self.matrix is None:
            raise RuntimeError("Call fit(df) before using the model.")

    def similar_products(self, product_id, n=5):
        """Products most similar to the given product (the product itself is excluded)."""
        self._check_fitted()
        matches = np.flatnonzero(self.products["product_id"].to_numpy() == product_id)
        if matches.size == 0:
            raise KeyError(f"Unknown product_id: {product_id}")
        position = int(matches[0])
        scores = cosine_similarity(self.matrix[position], self.matrix).ravel()
        scores[position] = -1                      # never recommend the product to itself
        best = np.argsort(scores)[::-1][:n]
        result = self.products.iloc[best].copy()
        result["similarity"] = scores[best]
        return result.reset_index(drop=True)

    def query_similarity(self, query_text):
        """Cosine similarity between free text (customer wishes) and every product.

        Returns an array with one value per product (0 if no words match).
        """
        self._check_fitted()
        query_text = (query_text or "").strip().lower()
        if not query_text:
            return np.zeros(len(self.products))
        query_vector = self.vectorizer.transform([query_text])
        if query_vector.nnz == 0:                  # none of the words exist in the vocabulary
            return np.zeros(len(self.products))
        return cosine_similarity(query_vector, self.matrix).ravel()

    def recommend_for_text(self, query_text, n=10):
        """Rank products by similarity to a customer's preference text."""
        scores = self.query_similarity(query_text)
        best = np.argsort(scores)[::-1][:n]
        result = self.products.iloc[best].copy()
        result["similarity"] = scores[best]
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result.reset_index(drop=True)

    def category_agreement(self, k=5):
        """SANITY CHECK (not an accuracy metric): how often are a product's k most
        similar products in the same category as the product itself?

        We have no real customer interaction data, so we cannot measure true
        recommendation accuracy. This only checks that the text similarity is sensible.
        """
        self._check_fitted()
        if len(self.products) < 2:
            return None
        similarity = cosine_similarity(self.matrix)
        np.fill_diagonal(similarity, -1)            # a product is not its own neighbour
        k = min(k, len(self.products) - 1)
        neighbours = np.argsort(similarity, axis=1)[:, ::-1][:, :k]
        categories = self.products["category"].to_numpy()
        same = categories[neighbours] == categories[:, None]
        return float(same.mean())

    def top_terms(self, product_id, n=8):
        """The words with the highest TF-IDF weight for one product (for explanations)."""
        self._check_fitted()
        position = int(np.flatnonzero(self.products["product_id"].to_numpy() == product_id)[0])
        row = self.matrix[position].toarray().ravel()
        names = self.vectorizer.get_feature_names_out()
        best = np.argsort(row)[::-1][:n]
        return [(names[i], float(row[i])) for i in best if row[i] > 0]
