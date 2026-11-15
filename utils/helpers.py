"""Small helper functions that do NOT depend on Streamlit (easy to test).

They format numbers and build the HTML used for product cards and metric cards.
All text that comes from the dataset is escaped so odd characters in an
uploaded CSV cannot break the page.
"""
from html import escape
from urllib.parse import quote

import pandas as pd


def format_inr(value):
    """Format a number as Indian rupees with Indian digit grouping: 1234567 -> ₹12,34,567."""
    number = int(round(float(value)))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"{sign}₹{digits}"


def star_string(rating):
    """Return a 5-star text such as ★★★★☆ for a rating."""
    filled = int(round(max(0.0, min(5.0, float(rating)))))
    return "★" * filled + "☆" * (5 - filled)


def split_features(features_text):
    """'5G; 120Hz display' -> ['5G', '120Hz display']"""
    return [part.strip() for part in str(features_text).split(";") if part.strip()]


def get_feature_options(df, category=None, limit=40):
    """Most common feature phrases (optionally inside one category) for the feature picker."""
    subset = df if category in (None, "All") else df[df["category"] == category]
    counts = pd.Series([f for text in subset["features"] for f in split_features(text)]).value_counts()
    return counts.head(limit).index.tolist()


def metric_card_html(label, value, note=""):
    note_html = f'<div class="ric-metric-note">{escape(str(note))}</div>' if note else ""
    return (f'<div class="ric-metric"><div class="ric-metric-label">{escape(str(label))}</div>'
            f'<div class="ric-metric-value">{escape(str(value))}</div>{note_html}</div>')


# Colour + emoji per category, used to draw a product picture when no photo URL is given.
CATEGORY_STYLE = {
    "Mobiles": ("#4F46E5", "#818CF8", "📱"),
    "Laptops": ("#1E1B4B", "#4F46E5", "💻"),
    "Electronics": ("#7C3AED", "#A78BFA", "🎧"),
    "Men's Fashion": ("#0F766E", "#5EEAD4", "👔"),
    "Women's Fashion": ("#BE185D", "#F9A8D4", "👗"),
    "Footwear": ("#B45309", "#FCD34D", "👟"),
    "Home and Kitchen": ("#15803D", "#86EFAC", "🍳"),
    "Beauty": ("#C026D3", "#F0ABFC", "💄"),
    "Books": ("#1D4ED8", "#93C5FD", "📚"),
    "Sports and Fitness": ("#DC2626", "#FCA5A5", "🏏"),
}
DEFAULT_STYLE = ("#4F46E5", "#A78BFA", "🛍️")


def product_image_uri(row):
    """Return an image address for a product.

    If the dataset has an 'image_url' column with a web address, that photo is used.
    Otherwise we draw a small category picture (an SVG) so every card has an image.
    These drawn pictures are illustrations, NOT photos of the real product.
    """
    url = str(row.get("image_url", "") if hasattr(row, "get") else "").strip()
    if url.lower().startswith(("http://", "https://")):
        return url
    dark, light, icon = CATEGORY_STYLE.get(str(row["category"]), DEFAULT_STYLE)
    brand = escape(str(row["brand"])[:18])
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 170'>"
           f"<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
           f"<stop offset='0' stop-color='{dark}'/><stop offset='1' stop-color='{light}'/></linearGradient></defs>"
           f"<rect width='300' height='170' fill='url(#g)'/>"
           f"<circle cx='250' cy='30' r='60' fill='white' opacity='0.12'/>"
           f"<text x='150' y='95' font-size='64' text-anchor='middle'>{icon}</text>"
           f"<text x='150' y='148' font-size='15' font-family='Arial' font-weight='bold' fill='white' "
           f"text-anchor='middle' opacity='0.95'>{brand}</text></svg>")
    return "data:image/svg+xml;utf8," + quote(svg)


def product_card_html(row, rank=None, match_score=None, explanation=None, selected=False):
    """HTML for one product card (Explorer, Recommendations and Dashboard).

    IMPORTANT: the HTML is joined into ONE line with no blank lines or indentation.
    Streamlit's Markdown ends an HTML block at a blank line and treats indented text
    as code, which used to show raw <div> text on the page.
    """
    chips = "".join(f'<span class="ric-chip">{escape(f)}</span>' for f in split_features(row["features"]))
    badge = f'<span class="ric-rank">#{int(rank)}</span>' if rank is not None else ""
    score = (f'<div class="ric-score">Match score <b>{float(match_score):.1f}</b>/100</div>'
             if match_score is not None else "")
    why = f'<div class="ric-why"><b>Why:</b> {escape(str(explanation))}</div>' if explanation else ""
    picked = '<span class="ric-picked">✔ Selected</span>' if selected else ""
    parts = [
        f'<div class="ric-card{" ric-card-selected" if selected else ""}">',
        f'<img class="ric-img" alt="{escape(str(row["product_name"]))}" src="{escape(product_image_uri(row), quote=True)}"/>',
        f'<div class="ric-card-top">{badge}<span class="ric-id">{escape(str(row["product_id"]))}</span>',
        f'{picked}<span class="ric-cat">{escape(str(row["category"]))}</span></div>',
        f'<div class="ric-name">{escape(str(row["product_name"]))}</div>',
        f'<div class="ric-brand">by {escape(str(row["brand"]))}</div>',
        f'<div class="ric-price">{format_inr(row["price"])}</div>',
        f'<div class="ric-rating"><span class="ric-stars">{star_string(row["rating"])}</span> ',
        f'{float(row["rating"]):.1f} · {int(row["review_count"]):,} reviews</div>',
        score,
        f'<div class="ric-desc">{escape(str(row["description"]))}</div>',
        f'<div class="ric-chips">{chips}</div>',
        why,
        '</div>',
    ]
    return "".join(parts)


def card_grid_html(cards):
    """Wrap card HTML strings in a responsive grid (auto-fits phones and desktops)."""
    return '<div class="ric-grid">' + "".join(cards) + "</div>"
