"""Reusable Matplotlib chart functions.

Every function takes a DataFrame (real data), builds a Figure and returns it.
We use matplotlib's Figure class directly (not pyplot) so charts are safe to
create many times inside a web app. Streamlit shows them with st.pyplot(fig).
"""
import io

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Brand colours: navy, indigo, violet on a light background.
NAVY = "#1E1B4B"
INDIGO = "#4F46E5"
VIOLET = "#7C3AED"
LIGHT_INDIGO = "#C7D2FE"
CHARCOAL = "#1F2937"
GRID = "#E5E7EB"
MUTED = "#6B7280"


def _new_axes(width=8, height=4.6):
    """Create a figure with a clean, light style."""
    fig = Figure(figsize=(width, height), dpi=110)
    ax = fig.subplots()
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(xlabel, color=CHARCOAL, fontsize=10)
    ax.set_ylabel(ylabel, color=CHARCOAL, fontsize=10)


def _short(text, limit=34):
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _horizontal_bars(labels, values, title, xlabel, ylabel, color=INDIGO,
                     value_format="{:.2f}", xlim=None, reference=None, reference_label=None):
    """Generic horizontal bar chart: largest value at the top."""
    fig, ax = _new_axes(8, max(3.2, 0.42 * len(labels) + 1.4))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=color, height=0.62)
    ax.set_yticks(positions)
    ax.set_yticklabels([_short(str(label)) for label in labels], color=CHARCOAL)
    ax.invert_yaxis()
    for pos, value in zip(positions, values):
        ax.text(value, pos, " " + value_format.format(value), va="center", fontsize=8.5, color=CHARCOAL)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, float(np.max(values)) * 1.15)
    if reference is not None:
        ax.axvline(reference, color=VIOLET, linestyle="--", linewidth=1.4, label=reference_label)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    _style(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


# ---------- the seven required charts ----------

def top_rated_chart(df, n=10):
    """1. Top N highly rated products (ties broken by review count)."""
    top = df.sort_values(["rating", "review_count"], ascending=False).head(n)
    return _horizontal_bars(top["product_name"], top["rating"].to_numpy(),
                            f"Top {len(top)} highly rated products", "Rating (out of 5)", "Product",
                            xlim=(max(0, float(top["rating"].min()) - 0.6), 5.0),
                            reference=float(df["rating"].mean()),
                            reference_label=f"Dataset average ({df['rating'].mean():.2f})")


def category_count_chart(df):
    """2. Category-wise product count."""
    counts = df["category"].value_counts()
    return _horizontal_bars(counts.index, counts.to_numpy(), "Products per category",
                            "Number of products", "Category", value_format="{:.0f}")


def brand_rating_chart(df, n=10, min_products=2):
    """3. Brand-wise average rating (brands with at least min_products products)."""
    grouped = df.groupby("brand").agg(avg=("rating", "mean"), products=("product_id", "count"))
    grouped = grouped[grouped["products"] >= min_products].sort_values("avg", ascending=False).head(n)
    if grouped.empty:  # small datasets: fall back to all brands
        grouped = df.groupby("brand").agg(avg=("rating", "mean"), products=("product_id", "count")) \
            .sort_values("avg", ascending=False).head(n)
        note = "all brands"
    else:
        note = f"brands with {min_products}+ products"
    return _horizontal_bars(grouped.index, grouped["avg"].to_numpy(),
                            f"Average rating by brand ({note})", "Average rating", "Brand",
                            xlim=(max(0, float(grouped["avg"].min()) - 0.6), 5.0),
                            reference=float(df["rating"].mean()),
                            reference_label=f"Dataset average ({df['rating'].mean():.2f})")


def rating_histogram(df):
    """4. Product rating distribution (histogram)."""
    fig, ax = _new_axes(8, 4.2)
    ratings = df["rating"].to_numpy()
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.hist(ratings, bins=np.arange(3.0, 5.21, 0.2), color=INDIGO, edgecolor="white", linewidth=1.5)
    ax.axvline(np.mean(ratings), color=VIOLET, linestyle="--", linewidth=1.4,
               label=f"Average ({np.mean(ratings):.2f})")
    ax.axvline(np.median(ratings), color=NAVY, linestyle=":", linewidth=1.6,
               label=f"Median ({np.median(ratings):.2f})")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Distribution of product ratings", "Rating", "Number of products")
    fig.tight_layout()
    return fig


def price_vs_rating_chart(df):
    """5. Price versus rating (scatter, log price axis because prices span a wide range)."""
    fig, ax = _new_axes(8, 4.6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    sizes = 20 + 60 * (np.log10(df["review_count"].clip(lower=1)) / np.log10(df["review_count"].max() + 1))
    ax.scatter(df["price"], df["rating"], s=sizes, color=INDIGO, alpha=0.65,
               edgecolor="white", linewidth=0.8, label="Product (bigger dot = more reviews)")
    ax.axhline(df["rating"].mean(), color=VIOLET, linestyle="--", linewidth=1.3,
               label=f"Average rating ({df['rating'].mean():.2f})")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda value, _: f"₹{value:,.0f}")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "Price versus rating", "Price (log scale)", "Rating")
    fig.tight_layout()
    return fig


def category_rating_chart(df):
    """6. Category-wise average rating."""
    avg = df.groupby("category")["rating"].mean().sort_values(ascending=False)
    return _horizontal_bars(avg.index, avg.to_numpy(), "Average rating by category",
                            "Average rating", "Category",
                            xlim=(max(0, float(avg.min()) - 0.6), 5.0),
                            reference=float(df["rating"].mean()),
                            reference_label=f"Dataset average ({df['rating'].mean():.2f})")


def reviews_by_product_chart(df, n=10):
    """7. Number of reviews by product (top N)."""
    top = df.sort_values("review_count", ascending=False).head(n)
    return _horizontal_bars(top["product_name"], top["review_count"].to_numpy(),
                            f"Top {len(top)} products by number of reviews", "Number of reviews",
                            "Product", color=VIOLET, value_format="{:,.0f}")


# ---------- extra charts used on the dashboard ----------

def brand_count_chart(df, n=10):
    """Brands with the most products."""
    counts = df["brand"].value_counts().head(n)
    return _horizontal_bars(counts.index, counts.to_numpy(), f"Top {len(counts)} brands by number of products",
                            "Number of products", "Brand", value_format="{:.0f}")


def category_share_pie(df):
    """Share of total reviews by category (pie chart)."""
    shares = df.groupby("category")["review_count"].sum().sort_values(ascending=False)
    fig = Figure(figsize=(6.5, 4.6), dpi=110)
    ax = fig.subplots()
    # Show the top 6 categories; fold the rest into "Other" so the pie stays readable.
    if len(shares) > 6:
        other = pd.Series({"Other": shares.iloc[6:].sum()})
        shares = pd.concat([shares.head(6), other])
    palette = [NAVY, INDIGO, VIOLET, "#818CF8", "#A78BFA", LIGHT_INDIGO, "#CBD5E1"]
    _, _, percent_labels = ax.pie(
        shares.to_numpy(), labels=shares.index, colors=palette[: len(shares)],
        autopct="%1.0f%%", startangle=90, counterclock=False, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9, "color": CHARCOAL})
    # White text on the three dark slices so the percentages stay readable.
    for index, label in enumerate(percent_labels):
        if index < 3:
            label.set_color("white")
    ax.set_title("Share of reviews by category", loc="left", fontsize=12, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig):
    """Convert a figure to PNG bytes (for the 'download chart' button)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


# ---------- charts that describe the customer's RECOMMENDATIONS ----------

SCORE_COLORS = {"preference": NAVY, "content": INDIGO, "rating": VIOLET, "popularity": "#A5B4FC"}
SCORE_LABELS = {"preference": "Preference match", "content": "Content similarity",
                "rating": "Weighted rating", "popularity": "Review popularity"}


def match_score_chart(results):
    """Match score (0-100) of each recommended product."""
    return _horizontal_bars(results["product_name"], results["match_score"].to_numpy(),
                            "Match score of your recommendations", "Match score (out of 100)",
                            "Product", value_format="{:.1f}", xlim=(0, 100))


def score_breakdown_chart(results, used_weights):
    """Stacked bars: how much each component adds to a product's match score.

    contribution = component score (0-1) x weight x 100, so the bars add up to the match score.
    """
    fig, ax = _new_axes(8, max(3.2, 0.5 * len(results) + 1.6))
    positions = np.arange(len(results))
    left = np.zeros(len(results))
    for name, weight in used_weights.items():
        values = results[f"{name}_score"].fillna(0).to_numpy(dtype=float) * weight * 100
        ax.barh(positions, values, left=left, height=0.62, color=SCORE_COLORS[name],
                edgecolor="white", linewidth=1.2, label=f"{SCORE_LABELS[name]} (weight {weight:.0%})")
        left = left + values
    ax.set_yticks(positions)
    ax.set_yticklabels([_short(str(n)) for n in results["product_name"]], color=CHARCOAL)
    ax.invert_yaxis()
    for pos, total in zip(positions, left):
        ax.text(total, pos, f" {total:.1f}", va="center", fontsize=8.5, color=CHARCOAL)
    ax.set_xlim(0, max(100.0, float(left.max()) * 1.1))
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    _style(ax, "What makes up each match score", "Points added to the match score", "Product")
    fig.tight_layout()
    return fig


def recommended_scatter_chart(df, results, selected_id=None):
    """Price vs rating of the WHOLE dataset (grey) with the recommended products highlighted."""
    fig, ax = _new_axes(8, 4.6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    others = df[~df["product_id"].isin(results["product_id"])]
    ax.scatter(others["price"], others["rating"], s=28, color="#CBD5E1", alpha=0.8,
               edgecolor="white", linewidth=0.6, label="Other products")
    ax.scatter(results["price"], results["rating"], s=90, color=INDIGO, edgecolor="white",
               linewidth=1.2, label="Recommended", zorder=3)
    offsets = [(6, 7), (-20, -13), (6, -13), (-20, 7)]     # stagger labels so close points do not overlap
    for index, (_, row) in enumerate(results.iterrows()):
        ax.annotate(f"#{int(row['rank'])}", (row["price"], row["rating"]), xytext=offsets[index % 4],
                    textcoords="offset points", fontsize=8.5, color=NAVY, fontweight="bold")
    if selected_id is not None:
        chosen = df[df["product_id"] == selected_id]
        ax.scatter(chosen["price"], chosen["rating"], s=240, facecolor="none", edgecolor=VIOLET,
                   linewidth=2.4, label="Selected", zorder=4)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda value, _: f"₹{value:,.0f}")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "Where your recommendations sit in the catalogue", "Price (log scale)", "Rating")
    fig.tight_layout()
    return fig


def compare_selected_chart(df, product):
    """Selected product vs the average of its category and of all products (rating and price index)."""
    category_df = df[df["category"] == product["category"]]
    labels = ["Rating", "Price (₹ thousands)"]
    selected = [float(product["rating"]), float(product["price"]) / 1000]
    category_avg = [float(category_df["rating"].mean()), float(category_df["price"].mean()) / 1000]
    overall_avg = [float(df["rating"].mean()), float(df["price"].mean()) / 1000]
    fig = Figure(figsize=(8, 3.6), dpi=110)
    axes = fig.subplots(1, 2)
    for ax, label, sel, cat, allv in zip(axes, labels, selected, category_avg, overall_avg):
        bars = ax.bar(["Selected", "Category avg", "All avg"], [sel, cat, allv],
                      color=[VIOLET, INDIGO, LIGHT_INDIGO], width=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9, color=CHARCOAL)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold", color=NAVY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_ylim(0, max(sel, cat, allv) * 1.2)
    fig.suptitle(f"{_short(str(product['product_name']), 50)} compared with averages", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig
