#!/usr/bin/env python3
"""Stock clustering by return and volatility using K-means.

Clusters stocks by annualized return and volatility to identify
investment opportunities. Reads pre-computed metrics from the ticker skill cache.

Delegates all data fetching to the /ticker skill. This skill focuses on analysis.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, vq

# Ticker skill cache location (well-known path)
TICKER_CACHE_FILE = Path.home() / ".cache" / "ticker" / "tickers.json"


class StockMetrics(NamedTuple):
    ticker: str
    returns: float
    volatility: float
    cluster: int


def _find_ticker_script() -> Path:
    """Find the ticker.py script path."""
    # Try relative to this script first
    this_dir = Path(__file__).parent
    ticker_script = this_dir.parent.parent / "ticker" / "scripts" / "ticker.py"
    if ticker_script.exists():
        return ticker_script

    # Try ~/.claude/skills/ticker
    home_ticker = Path.home() / ".claude" / "skills" / "ticker" / "scripts" / "ticker.py"
    if home_ticker.exists():
        return home_ticker

    raise FileNotFoundError("ticker.py script not found. Install the ticker skill first.")


def fetch_ticker_info_with_metrics(tickers: list[str], verbose: bool = True) -> dict[str, dict]:
    """Fetch company info and metrics for tickers using the ticker skill.

    Calls ticker skill with --refresh-metrics to ensure metrics are calculated.
    Returns dict of ticker -> {"name": str, "sector": str, "industry": str}
    """
    if not tickers:
        return {}

    if verbose:
        print(f"Fetching info and metrics for {len(tickers)} tickers...", file=sys.stderr)

    try:
        ticker_script = _find_ticker_script()
        result = subprocess.run(
            [sys.executable, str(ticker_script), ",".join(tickers), "--refresh-metrics"],
            capture_output=True,
            text=True,
            timeout=300,  # Longer timeout for metrics calculation
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Warning: ticker script failed: {result.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print("Warning: ticker skill not found", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("Warning: ticker lookup timed out", file=sys.stderr)
    except json.JSONDecodeError:
        print("Warning: could not parse ticker response", file=sys.stderr)

    # Return empty info on failure
    return {t: {"name": t, "sector": "", "industry": ""} for t in tickers}


def read_metrics_from_cache(tickers: list[str]) -> pd.DataFrame:
    """Read returns and volatility from the ticker cache.

    Args:
        tickers: List of ticker symbols to read

    Returns:
        DataFrame with columns [Ticker, Returns, Volatility] for tickers that have metrics
    """
    if not TICKER_CACHE_FILE.exists():
        return pd.DataFrame(columns=["Ticker", "Returns", "Volatility"])

    try:
        with open(TICKER_CACHE_FILE) as f:
            cache = json.load(f)
    except (json.JSONDecodeError, IOError):
        return pd.DataFrame(columns=["Ticker", "Returns", "Volatility"])

    rows = []
    for ticker in tickers:
        ticker_upper = ticker.upper()
        if ticker_upper in cache:
            data = cache[ticker_upper]
            returns = data.get("returns")
            volatility = data.get("volatility")
            if returns is not None and volatility is not None:
                rows.append({
                    "Ticker": ticker_upper,
                    "Returns": returns,
                    "Volatility": volatility,
                })

    return pd.DataFrame(rows)


def fetch_index_tickers_with_metrics(index: str, verbose: bool = True) -> tuple[list[str], dict[str, dict]]:
    """Fetch tickers and metrics from a market index via the ticker skill.

    Calls ticker skill with --index and --refresh-metrics to fetch tickers
    and ensure their metrics are calculated.

    Args:
        index: Index name - 'sp500', 'nasdaq100', or 'dow'
        verbose: Print progress messages

    Returns:
        Tuple of (list of ticker symbols, dict of ticker info)
    """
    if verbose:
        print(f"Fetching {index} tickers and metrics via ticker skill...", file=sys.stderr)

    try:
        ticker_script = _find_ticker_script()
        result = subprocess.run(
            [sys.executable, str(ticker_script), "--index", index, "--refresh-metrics"],
            capture_output=True,
            text=True,
            timeout=600,  # Longer timeout for index with metrics
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            tickers = list(data.keys())
            if verbose:
                print(f"  Retrieved {len(tickers)} tickers with metrics", file=sys.stderr)
            return tickers, data
        else:
            print(f"Error: ticker script failed: {result.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print("Error: ticker skill not found. Install it first.", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("Error: ticker lookup timed out", file=sys.stderr)
    except json.JSONDecodeError:
        print("Error: could not parse ticker response", file=sys.stderr)

    return [], {}


def find_elbow(data: np.ndarray, k_range: range) -> list[float]:
    """Calculate distortions for elbow curve."""
    distortions = []
    for k in k_range:
        centroids, _ = kmeans(data, k)
        distortions.append(sum((data - centroids[vq(data, centroids)[0]]) ** 2).sum())
    return distortions


def cluster_stocks(metrics: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Apply K-means clustering to return/volatility data.

    Args:
        metrics: DataFrame with columns [Ticker, Returns, Volatility]
        n_clusters: Number of clusters

    Returns:
        DataFrame with Cluster column added
    """
    data = np.asarray([
        np.asarray(metrics["Returns"]),
        np.asarray(metrics["Volatility"])
    ]).T

    # Whiten data for better clustering (normalize)
    data_whitened = data / data.std(axis=0)

    centroids, _ = kmeans(data_whitened, n_clusters)
    idx, _ = vq(data_whitened, centroids)

    result = metrics.copy()
    result["Cluster"] = idx

    return result


def label_clusters(clusters_df: pd.DataFrame) -> dict[int, str]:
    """Generate descriptive labels for each cluster based on return/volatility.

    Labels are based on relative position within the dataset:
    - High/Low Return: above/below median return
    - High/Low Vol: above/below median volatility
    """
    median_return = clusters_df["Returns"].median()
    median_vol = clusters_df["Volatility"].median()

    labels = {}
    for cluster_id in clusters_df["Cluster"].unique():
        subset = clusters_df[clusters_df["Cluster"] == cluster_id]
        avg_return = subset["Returns"].mean()
        avg_vol = subset["Volatility"].mean()

        # Determine return level
        if avg_return >= median_return * 1.5:
            ret_label = "Strong"
        elif avg_return >= median_return:
            ret_label = "Moderate"
        elif avg_return >= 0:
            ret_label = "Low"
        else:
            ret_label = "Negative"

        # Determine volatility level
        if avg_vol >= median_vol * 1.5:
            vol_label = "High Vol"
        elif avg_vol >= median_vol:
            vol_label = "Moderate Vol"
        else:
            vol_label = "Low Vol"

        labels[cluster_id] = f"{ret_label} Return, {vol_label}"

    return labels


def enrich_clusters_with_info(
    clusters_df: pd.DataFrame,
    ticker_info: dict[str, dict],
    cluster_labels: dict[int, str],
) -> pd.DataFrame:
    """Add company info and cluster labels to clusters dataframe."""
    df = clusters_df.copy()

    # Add company info from ticker skill results
    df["Name"] = df["Ticker"].apply(lambda t: ticker_info.get(t, {}).get("name", t))
    df["Sector"] = df["Ticker"].apply(lambda t: ticker_info.get(t, {}).get("sector", "") or ticker_info.get(t, {}).get("industry", ""))

    # Add cluster labels
    df["ClusterLabel"] = df["Cluster"].map(cluster_labels)

    return df


def plot_elbow(k_range: range, distortions: list[float], output: str | None = None):
    """Plot elbow curve to determine optimal cluster count."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(list(k_range), distortions, "bo-")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Distortion (Inertia)")
        ax.set_title("Elbow Curve for K-means Clustering")
        ax.grid(True, alpha=0.3)

        if output:
            plt.savefig(output, dpi=150, bbox_inches="tight")
            print(f"Elbow curve saved to {output}")
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available. Distortions:")
        for k, d in zip(k_range, distortions):
            print(f"  k={k}: {d:.2f}")


def _ensure_plotly():
    """Ensure plotly is installed, installing it if necessary."""
    try:
        import plotly.express as px
        return px
    except ImportError:
        import subprocess
        print("Installing plotly for interactive charts...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "plotly"])
        import plotly.express as px
        return px


def plot_clusters_interactive(clusters_df: pd.DataFrame, output: str | None = None):
    """Create interactive scatter plot with Plotly."""
    try:
        px = _ensure_plotly()

        # Use ClusterLabel for color if available, otherwise Cluster
        color_col = "ClusterLabel" if "ClusterLabel" in clusters_df.columns else "Cluster"

        # Build hover data based on available columns
        hover_cols = ["Ticker"]
        if "Name" in clusters_df.columns:
            hover_cols.append("Name")
        if "Sector" in clusters_df.columns:
            hover_cols.append("Sector")

        # Create custom hover template for richer info
        df = clusters_df.copy()
        df["Return %"] = (df["Returns"] * 100).round(1)
        df["Volatility %"] = (df["Volatility"] * 100).round(1)

        fig = px.scatter(
            df,
            x="Returns",
            y="Volatility",
            color=color_col,
            hover_data={
                "Ticker": True,
                "Name": True if "Name" in df.columns else False,
                "Sector": True if "Sector" in df.columns else False,
                "Return %": True,
                "Volatility %": True,
                "Returns": False,  # Hide raw values
                "Volatility": False,
            },
            title="Stock Clusters by Return and Volatility",
            labels={
                "Returns": "Annualized Return",
                "Volatility": "Annualized Volatility",
                "ClusterLabel": "Cluster",
            },
        )

        # Improve layout
        fig.update_layout(
            legend_title_text="Cluster Profile",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
            ),
        )
        fig.update_traces(
            marker=dict(size=10, symbol="diamond", line=dict(width=1, color="DarkSlateGrey"))
        )

        if output:
            fig.write_html(output)
            print(f"Interactive chart saved to {output}")
        else:
            fig.show()
        return True

    except ImportError:
        return False


def plot_clusters_static(clusters_df: pd.DataFrame, output: str | None = None):
    """Create static scatter plot with matplotlib."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))

        clusters = sorted(clusters_df["Cluster"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

        # Get cluster labels if available
        has_labels = "ClusterLabel" in clusters_df.columns

        for cluster, color in zip(clusters, colors):
            mask = clusters_df["Cluster"] == cluster
            subset = clusters_df[mask]

            # Use descriptive label if available
            if has_labels:
                label = subset["ClusterLabel"].iloc[0]
            else:
                label = f"Cluster {cluster}"

            ax.scatter(
                subset["Returns"],
                subset["Volatility"],
                c=[color],
                label=f"{label} ({len(subset)})",
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel("Annualized Return")
        ax.set_ylabel("Annualized Volatility")
        ax.set_title("Stock Clusters by Return and Volatility")
        ax.legend(title="Cluster Profile", loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output:
            plt.savefig(output, dpi=150, bbox_inches="tight")
            print(f"Static chart saved to {output}")
        else:
            plt.show()

    except ImportError:
        print("matplotlib not available for plotting", file=sys.stderr)


def print_cluster_summary(clusters_df: pd.DataFrame):
    """Print summary statistics for each cluster."""
    print("\n=== Cluster Summary ===\n")

    has_labels = "ClusterLabel" in clusters_df.columns
    has_info = "Name" in clusters_df.columns

    for cluster in sorted(clusters_df["Cluster"].unique()):
        subset = clusters_df[clusters_df["Cluster"] == cluster]
        avg_ret = subset["Returns"].mean() * 100
        avg_vol = subset["Volatility"].mean() * 100

        if has_labels:
            label = subset["ClusterLabel"].iloc[0]
            print(f"{label} ({len(subset)} stocks): "
                  f"Avg Return={avg_ret:.1f}%, Avg Volatility={avg_vol:.1f}%")
        else:
            print(f"Cluster {cluster} ({len(subset)} stocks): "
                  f"Avg Return={avg_ret:.1f}%, Avg Volatility={avg_vol:.1f}%")

    print("\n=== Top Performers by Return ===\n")
    top = clusters_df.nlargest(10, "Returns")
    for _, row in top.iterrows():
        info = ""
        if has_info and row.get("Name") and row["Name"] != row["Ticker"]:
            info = f" ({row['Name'][:30]})"
        cluster_info = row.get("ClusterLabel", f"Cluster {row['Cluster']}")
        print(f"  {row['Ticker']}{info}: Return={row['Returns']*100:.1f}%, "
              f"Volatility={row['Volatility']*100:.1f}% [{cluster_info}]")

    print("\n=== Lowest Volatility ===\n")
    low_vol = clusters_df.nsmallest(10, "Volatility")
    for _, row in low_vol.iterrows():
        info = ""
        if has_info and row.get("Name") and row["Name"] != row["Ticker"]:
            info = f" ({row['Name'][:30]})"
        cluster_info = row.get("ClusterLabel", f"Cluster {row['Cluster']}")
        print(f"  {row['Ticker']}{info}: Return={row['Returns']*100:.1f}%, "
              f"Volatility={row['Volatility']*100:.1f}% [{cluster_info}]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster stocks by return and volatility"
    )
    parser.add_argument(
        "--clusters", "-k",
        type=int,
        default=5,
        help="Number of clusters (default: 5)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for chart (HTML for interactive, PNG/PDF for static)",
    )
    parser.add_argument(
        "--csv",
        help="Export cluster data to CSV file",
    )
    parser.add_argument(
        "--elbow",
        action="store_true",
        help="Show elbow curve to determine optimal cluster count",
    )
    parser.add_argument(
        "--elbow-output",
        help="Save elbow curve to file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of tickers to process (useful for testing)",
    )
    parser.add_argument(
        "--tickers", "-t",
        help="Comma-separated list of ticker symbols (e.g., 'AAPL,MSFT,GOOGL')",
    )
    parser.add_argument(
        "--index", "-i",
        default="sp500",
        help="Market index to analyze: 'sp500', 'nasdaq100', or 'dow' (default: sp500)",
    )
    parser.add_argument(
        "--no-info",
        action="store_true",
        help="Skip fetching company names and sectors (faster but less detailed output)",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Track ticker info from any source
    ticker_info: dict[str, dict] = {}

    # Get tickers - either from user input or from index
    if args.tickers:
        # User provided custom tickers - fetch with metrics
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        if verbose:
            print(f"Using custom tickers: {', '.join(tickers)}", file=sys.stderr)
        ticker_info = fetch_ticker_info_with_metrics(tickers, verbose=verbose)
    else:
        # Fetch from index via ticker skill (with metrics)
        tickers, ticker_info = fetch_index_tickers_with_metrics(args.index, verbose=verbose)
        if not tickers:
            print("Error: Could not fetch index tickers", file=sys.stderr)
            return 1
        if args.limit:
            tickers = tickers[:args.limit]
        if verbose:
            print(f"  Processing {len(tickers)} tickers", file=sys.stderr)

    # Read pre-computed metrics from ticker cache
    if verbose:
        print("Reading metrics from ticker cache...", file=sys.stderr)
    metrics = read_metrics_from_cache(tickers)

    if metrics.empty:
        print("Error: No metrics found in cache. Run ticker skill with --refresh-metrics first.", file=sys.stderr)
        return 1

    if verbose:
        print(f"  Found metrics for {len(metrics)} tickers", file=sys.stderr)

    # Elbow curve
    if args.elbow:
        if verbose:
            print("Calculating elbow curve...", file=sys.stderr)
        data = np.asarray([
            np.asarray(metrics["Returns"]),
            np.asarray(metrics["Volatility"])
        ]).T
        data_whitened = data / data.std(axis=0)
        max_k = min(len(metrics), 20)  # Can't have more clusters than data points
        k_range = range(2, max_k)
        distortions = find_elbow(data_whitened, k_range)
        plot_elbow(k_range, distortions, args.elbow_output)
        if not args.output and not args.csv:
            return 0

    # Cluster stocks
    if verbose:
        print(f"Clustering into {args.clusters} groups...", file=sys.stderr)
    clusters_df = cluster_stocks(metrics, args.clusters)

    # Label clusters based on characteristics
    cluster_labels = label_clusters(clusters_df)

    # ticker_info was already fetched when getting tickers with metrics

    # Enrich clusters with labels and info
    clusters_df = enrich_clusters_with_info(clusters_df, ticker_info, cluster_labels)

    # Print summary
    print_cluster_summary(clusters_df)

    # Export CSV
    if args.csv:
        clusters_df.to_csv(args.csv, index=False)
        print(f"\nCluster data exported to {args.csv}")

    # Plot
    if args.output:
        if args.output.endswith(".html"):
            if not plot_clusters_interactive(clusters_df, args.output):
                print("Plotly not available, falling back to static plot", file=sys.stderr)
                static_output = args.output.replace(".html", ".png")
                plot_clusters_static(clusters_df, static_output)
        else:
            plot_clusters_static(clusters_df, args.output)
    else:
        # Try interactive first, fall back to static
        if not plot_clusters_interactive(clusters_df, None):
            plot_clusters_static(clusters_df, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
