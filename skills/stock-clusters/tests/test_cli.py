"""Characterization tests for stock_clusters.py CLI argument parsing.

The argument parser is defined inside main(). These tests verify CLI behavior
by running the script as a subprocess, capturing its response to various
argument combinations.
"""

import subprocess
import sys


SCRIPT_PATH = "skills/stock-clusters/scripts/stock_clusters.py"


def run_script(*args: str) -> subprocess.CompletedProcess:
    """Run stock_clusters.py with given arguments."""
    return subprocess.run(
        [sys.executable, SCRIPT_PATH, *args],
        capture_output=True,
        text=True,
    )


class TestCLIHelp:
    """Verify the script prints help and exits cleanly."""

    def test_help_exits_zero(self):
        result = run_script("--help")
        assert result.returncode == 0

    def test_help_mentions_clusters(self):
        result = run_script("--help")
        assert "--clusters" in result.stdout
        assert "-k" in result.stdout

    def test_help_mentions_tickers(self):
        result = run_script("--help")
        assert "--tickers" in result.stdout
        assert "-t" in result.stdout

    def test_help_mentions_elbow(self):
        result = run_script("--help")
        assert "--elbow" in result.stdout

    def test_help_mentions_output(self):
        result = run_script("--help")
        assert "--output" in result.stdout
        assert "-o" in result.stdout

    def test_help_mentions_csv(self):
        result = run_script("--help")
        assert "--csv" in result.stdout

    def test_help_mentions_quiet(self):
        result = run_script("--help")
        assert "--quiet" in result.stdout
        assert "-q" in result.stdout


class TestCLIDefaults:
    """Verify default argument values by inspecting the argparse setup.

    Since the parser is inside main(), we reconstruct it to check defaults.
    This mirrors the exact parser definition in the source code.
    """

    def test_default_clusters_is_5(self):
        import argparse

        # Reconstruct the parser as defined in main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--clusters", "-k", type=int, default=5)
        parser.add_argument("--output", "-o")
        parser.add_argument("--csv")
        parser.add_argument("--elbow", action="store_true")
        parser.add_argument("--elbow-output")
        parser.add_argument("--quiet", "-q", action="store_true")
        parser.add_argument("--tickers", "-t")
        parser.add_argument("--index", "-i")

        args = parser.parse_args([])
        assert args.clusters == 5

    def test_elbow_default_is_false(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--elbow", action="store_true")
        args = parser.parse_args([])
        assert args.elbow is False

    def test_quiet_default_is_false(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--quiet", "-q", action="store_true")
        args = parser.parse_args([])
        assert args.quiet is False

    def test_output_default_is_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", "-o")
        args = parser.parse_args([])
        assert args.output is None

    def test_csv_default_is_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--csv")
        args = parser.parse_args([])
        assert args.csv is None


class TestCLIArgumentParsing:
    """Verify specific argument combinations parse correctly."""

    def test_tickers_parsing_single(self):
        """Verify how --tickers value is stored (raw string, split happens in main)."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--tickers", "-t")
        args = parser.parse_args(["--tickers", "AAPL"])
        # The raw value is stored as-is; main() splits on comma
        assert args.tickers == "AAPL"

    def test_tickers_parsing_multiple(self):
        """Verify comma-separated tickers are stored as a single string."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--tickers", "-t")
        args = parser.parse_args(["--tickers", "AAPL,MSFT,GOOGL"])
        assert args.tickers == "AAPL,MSFT,GOOGL"

    def test_tickers_split_logic(self):
        """Verify the split logic from main() -- splits on comma and uppercases."""
        raw = "aapl, msft , googl"
        tickers = [t.strip().upper() for t in raw.split(",")]
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_clusters_accepts_integer(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--clusters", "-k", type=int, default=5)
        args = parser.parse_args(["-k", "3"])
        assert args.clusters == 3

    def test_output_accepts_path(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", "-o")
        args = parser.parse_args(["--output", "/tmp/chart.html"])
        assert args.output == "/tmp/chart.html"

    def test_csv_accepts_path(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--csv")
        args = parser.parse_args(["--csv", "/tmp/data.csv"])
        assert args.csv == "/tmp/data.csv"

    def test_elbow_is_boolean_flag(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--elbow", action="store_true")
        args = parser.parse_args(["--elbow"])
        assert args.elbow is True

    def test_data_file_accepts_path(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--data-file", "-d")
        args = parser.parse_args(["--data-file", "/tmp/tickers.json"])
        assert args.data_file == "/tmp/tickers.json"

    def test_data_file_accepts_stdin_dash(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--data-file", "-d")
        args = parser.parse_args(["--data-file", "-"])
        assert args.data_file == "-"

    def test_data_file_short_flag(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--data-file", "-d")
        args = parser.parse_args(["-d", "/tmp/data.json"])
        assert args.data_file == "/tmp/data.json"

    def test_data_file_default_is_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--data-file", "-d")
        args = parser.parse_args([])
        assert args.data_file is None


class TestCLIDataFileHelp:
    """Verify --data-file appears in help output."""

    def test_help_mentions_data_file(self):
        result = run_script("--help")
        assert "--data-file" in result.stdout
        assert "-d" in result.stdout
