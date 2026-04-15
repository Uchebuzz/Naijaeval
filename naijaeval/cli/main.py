"""NaijaEval command-line interface.

Commands
--------
naijaeval list metrics        -- list all registered metrics
naijaeval list datasets       -- list all registered datasets
naijaeval list benchmarks     -- list available benchmark YAML files
naijaeval run                 -- run a benchmark against a model
naijaeval compare             -- compare two sets of result files
naijaeval report              -- render a JSON result file as HTML
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

import naijaeval
from naijaeval.registry import list_datasets, list_metrics

app = typer.Typer(
    name="naijaeval",
    help=(
        "NaijaEval — evaluation toolkit for AI systems in African language contexts.\n\n"
        "Run 'naijaeval COMMAND --help' for details on any command."
    ),
    add_completion=False,
)
console = Console()

list_app = typer.Typer(help="List available metrics, datasets, and benchmarks.")
app.add_typer(list_app, name="list")

# ---------------------------------------------------------------------------
# list sub-commands
# ---------------------------------------------------------------------------


@list_app.command("metrics")
def list_metrics_cmd():
    """List all registered evaluation metrics."""
    # Ensure all metrics are imported/registered
    import naijaeval.metrics  # noqa: F401

    names = list_metrics()
    if not names:
        rprint("[yellow]No metrics registered.[/yellow]")
        return

    table = Table(title="Registered Metrics", show_header=True, header_style="bold blue")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    from naijaeval.registry import MetricRegistry
    for name in names:
        cls = MetricRegistry.get(name)
        instance = cls()
        desc = getattr(instance, "description", "")
        table.add_row(name, desc)

    console.print(table)


@list_app.command("datasets")
def list_datasets_cmd():
    """List all registered dataset loaders."""
    import naijaeval.datasets  # noqa: F401

    names = list_datasets()
    if not names:
        rprint("[yellow]No datasets registered.[/yellow]")
        return

    table = Table(title="Registered Datasets", show_header=True, header_style="bold blue")
    table.add_column("Name", style="cyan")
    table.add_column("Loader function")

    from naijaeval.registry import DatasetRegistry
    for name in names:
        fn = DatasetRegistry.get(name)
        table.add_row(name, fn.__name__)

    console.print(table)


@list_app.command("benchmarks")
def list_benchmarks_cmd():
    """List available benchmark YAML files."""
    benchmark_dir = Path(__file__).parent.parent.parent / "benchmarks"
    if not benchmark_dir.exists():
        rprint("[yellow]No benchmarks directory found.[/yellow]")
        return

    files = sorted(benchmark_dir.glob("*.yaml"))
    if not files:
        rprint("[yellow]No benchmark files found in benchmarks/.[/yellow]")
        return

    table = Table(title="Available Benchmarks", show_header=True, header_style="bold blue")
    table.add_column("File", style="cyan")
    table.add_column("Path")

    for f in files:
        table.add_row(f.stem, str(f))

    console.print(table)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="Benchmark name (matches benchmarks/<name>.yaml)"),
    predictions: Optional[Path] = typer.Option(None, "--predictions", "-p", help="Path to predictions file (one per line)."),
    references: Optional[Path] = typer.Option(None, "--references", "-r", help="Path to references file (one per line)."),
    sources: Optional[Path] = typer.Option(None, "--sources", "-s", help="Path to source texts file (one per line)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path."),
    model: str = typer.Option("unknown", "--model", "-m", help="Model identifier for the report."),
):
    """Run a benchmark and output results.

    Loads predictions and references from text files (one item per line),
    runs all metrics defined in the benchmark YAML, and prints results.

    Example::

        naijaeval run --benchmark naija_mt_v1 \\
            --predictions preds.txt \\
            --references refs.txt \\
            --output results.json \\
            --model Helsinki-NLP/opus-mt-en-yo
    """
    import naijaeval.metrics  # noqa: F401

    benchmark_path = _resolve_benchmark(benchmark)
    if benchmark_path is None:
        rprint(f"[red]Benchmark '{benchmark}' not found.[/red]")
        rprint("Run 'naijaeval list benchmarks' to see available benchmarks.")
        raise typer.Exit(1)

    config = _load_yaml(benchmark_path)
    rprint(f"[bold]Benchmark:[/bold] {benchmark}")
    rprint(f"[bold]Model:[/bold] {model}\n")

    # Load data
    preds = _load_lines(predictions) if predictions else None
    refs = _load_lines(references) if references else None
    srcs = _load_lines(sources) if sources else None

    if preds is None or refs is None:
        rprint("[red]--predictions and --references are required for 'run'.[/red]")
        raise typer.Exit(1)

    if len(preds) != len(refs):
        rprint(f"[red]Mismatch: {len(preds)} predictions vs {len(refs)} references.[/red]")
        raise typer.Exit(1)

    # Run metrics
    results = {}
    metric_names = config.get("metrics", [])

    from naijaeval.registry import MetricRegistry

    with console.status("[bold blue]Running metrics…[/bold blue]"):
        for metric_name in metric_names:
            try:
                cls = MetricRegistry.get(metric_name)
                metric = cls()
                result = metric.compute(preds, refs)
                results[metric_name] = result
            except Exception as exc:
                rprint(f"[yellow]Warning: metric '{metric_name}' failed: {exc}[/yellow]")

    # Print summary table
    table = Table(title="Results", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")

    for name, result in results.items():
        table.add_row(name, f"{result.score:.4f}")

    console.print(table)

    # Save JSON
    if output:
        from naijaeval.report.json_report import save_json
        save_json(results, output, model=model, benchmark=benchmark)
        rprint(f"\n[green]Results saved to {output}[/green]")

    return results


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command("compare")
def compare_cmd(
    result_files: list[Path] = typer.Argument(..., help="Two or more JSON result files to compare."),
):
    """Compare results from multiple evaluation runs side by side.

    Example::

        naijaeval compare model_a.json model_b.json
    """
    if len(result_files) < 2:
        rprint("[red]Provide at least two result files to compare.[/red]")
        raise typer.Exit(1)

    all_data = []
    for path in result_files:
        if not path.exists():
            rprint(f"[red]File not found: {path}[/red]")
            raise typer.Exit(1)
        data = json.loads(path.read_text(encoding="utf-8"))
        all_data.append(data)

    # Collect all metric names
    all_metrics: set[str] = set()
    for data in all_data:
        all_metrics.update(data.get("summary", {}).keys())

    table = Table(title="Model Comparison", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    for data in all_data:
        table.add_column(data.get("model", "unknown"), justify="right")

    for metric in sorted(all_metrics):
        row = [metric]
        for data in all_data:
            score = data.get("summary", {}).get(metric)
            row.append(f"{score:.4f}" if score is not None else "—")
        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command("report")
def report_cmd(
    input: Path = typer.Option(..., "--input", "-i", help="JSON results file."),
    output: Path = typer.Option(..., "--output", "-o", help="Output HTML path."),
    fmt: str = typer.Option("html", "--format", "-f", help="Output format: html or json."),
):
    """Render a JSON results file as an HTML report.

    Example::

        naijaeval report --input results.json --output report.html
    """
    if not input.exists():
        rprint(f"[red]Input file not found: {input}[/red]")
        raise typer.Exit(1)

    data = json.loads(input.read_text(encoding="utf-8"))

    if fmt == "html":
        from naijaeval.metrics.base import MetricResult
        from naijaeval.report.html_report import save_html

        # Reconstruct MetricResult objects from JSON
        results = {
            name: MetricResult(
                name=res["name"],
                score=res["score"],
                details=res.get("details", {}),
                metadata=res.get("metadata", {}),
            )
            for name, res in data.get("results", {}).items()
        }
        save_html(
            results,
            output,
            model=data.get("model", "unknown"),
            benchmark=data.get("benchmark", "unknown"),
        )
        rprint(f"[green]HTML report saved to {output}[/green]")
    else:
        rprint(f"[red]Unknown format '{fmt}'. Use 'html' or 'json'.[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command("version")
def version_cmd():
    """Print the NaijaEval version."""
    rprint(f"[bold]NaijaEval[/bold] v{naijaeval.__version__}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_benchmark(name: str) -> Path | None:
    """Locate a benchmark YAML by stem name or full path."""
    p = Path(name)
    if p.exists() and p.suffix in (".yaml", ".yml"):
        return p
    benchmark_dir = Path(__file__).parent.parent.parent / "benchmarks"
    for candidate in [
        benchmark_dir / f"{name}.yaml",
        benchmark_dir / f"{name}.yml",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    app()
