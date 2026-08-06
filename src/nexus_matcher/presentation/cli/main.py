"""
nexus_matcher.presentation.cli.main | Layer: PRESENTATION
Command-line interface for NexusMatcher using Typer.

## Relationships
# DEPENDS_ON → application/use_cases/match_schema :: NexusMatcher class
# DEPENDS_ON → infrastructure/config :: configuration loading
# USED_BY    → pyproject.toml :: entry point nexus-matcher

## Attributes
# Security: No secrets in CLI output, safe file handling
# Performance: Progress bars for long operations
# Reliability: Graceful error handling, clear exit codes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Initialize Typer app
app = typer.Typer(
    name="nexus-matcher",
    help="Enterprise Semantic Schema Matching System",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()


# =============================================================================
# COMMON OPTIONS
# =============================================================================


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from nexus_matcher import __version__

        rprint(f"[bold blue]NexusMatcher[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """
    NexusMatcher - Enterprise Semantic Schema Matching

    Match schema fields to data dictionary entries using semantic search.
    """
    pass


# =============================================================================
# MATCH COMMAND
# =============================================================================


@app.command()
def match(
    schema: Annotated[
        Path,
        typer.Argument(
            help="Path to schema file (Avro, JSON Schema, SQL DDL)",
            exists=True,
            readable=True,
        ),
    ],
    dictionary: Annotated[
        Path,
        typer.Option(
            "--dictionary",
            "-d",
            help="Path to data dictionary file (Excel, CSV)",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path (JSON or CSV)",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json, csv, table",
        ),
    ] = "table",
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of matches per field",
            min=1,
            max=20,
        ),
    ] = 5,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Minimum confidence threshold",
            min=0.0,
            max=1.0,
        ),
    ] = 0.0,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Show detailed output",
        ),
    ] = False,
) -> None:
    """
    Match schema fields to dictionary entries.

    Example:
        nexus-matcher match schema.avsc -d dictionary.xlsx
        nexus-matcher match schema.json -d dictionary.csv -o results.json
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize matcher
            task = progress.add_task("Initializing matcher...", total=None)
            matcher = _get_matcher()

            # Load dictionary
            progress.update(task, description="Loading dictionary...")
            stats = matcher.load_dictionary(dictionary)
            if verbose:
                rprint(f"[dim]Loaded {stats.valid_entries} dictionary entries[/dim]")

            # Match schema
            progress.update(task, description="Matching schema...")
            results = matcher.match_schema(schema)

        # Format output
        if format == "table":
            _display_table(results, top_k, threshold, verbose)
        elif format == "json":
            json_output = _format_json(results, top_k, threshold)
            if output:
                output.write_text(json_output)
                rprint(f"[green]Results written to {output}[/green]")
            else:
                print(json_output)
        elif format == "csv":
            csv_output = _format_csv(results, top_k, threshold)
            if output:
                output.write_text(csv_output)
                rprint(f"[green]Results written to {output}[/green]")
            else:
                print(csv_output)
        else:
            rprint(f"[red]Unknown format: {format}[/red]")
            raise typer.Exit(1)

        # Summary
        total_fields = len(results)
        auto_approved = sum(
            1
            for matches in results.values()
            if matches and matches[0].decision.value == "AUTO_APPROVE"
        )
        rprint(
            f"\n[bold]Summary:[/bold] {auto_approved}/{total_fields} fields auto-approved ({auto_approved / total_fields * 100:.1f}%)"
        )

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


# =============================================================================
# SYNC COMMAND
# =============================================================================


@app.command()
def sync(
    dictionary: Annotated[
        Path,
        typer.Argument(
            help="Path to data dictionary file",
            exists=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for index files",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Show detailed output",
        ),
    ] = False,
) -> None:
    """
    Sync dictionary to vector store.

    Example:
        nexus-matcher sync dictionary.xlsx
        nexus-matcher sync dictionary.csv -o ./index
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing...", total=None)
            matcher = _get_matcher()

            progress.update(task, description="Loading dictionary...")
            stats = matcher.load_dictionary(dictionary)

        # Display stats
        table = Table(title="Sync Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Rows", str(stats.total_rows))
        table.add_row("Valid Entries", str(stats.valid_entries))
        table.add_row("Skipped Rows", str(stats.skipped_rows))
        table.add_row("Error Rows", str(stats.error_rows))
        table.add_row("Success Rate", f"{stats.success_rate * 100:.1f}%")

        console.print(table)

        if stats.errors and verbose:
            rprint("\n[yellow]Errors:[/yellow]")
            for error in stats.errors[:10]:
                rprint(f"  [dim]• {error}[/dim]")
            if len(stats.errors) > 10:
                rprint(f"  [dim]... and {len(stats.errors) - 10} more[/dim]")

        rprint("\n[green]✓ Dictionary synced successfully[/green]")

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


# =============================================================================
# API COMMAND
# =============================================================================


@app.command()
def api(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Bind host",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Bind port",
        ),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload",
            "-r",
            help="Enable auto-reload for development",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Number of worker processes",
            min=1,
        ),
    ] = 1,
) -> None:
    """
    Start the REST API server.

    Example:
        nexus-matcher api
        nexus-matcher api --host 127.0.0.1 --port 8080 --reload
    """
    try:
        import uvicorn

        rprint("[bold blue]Starting NexusMatcher API[/bold blue]")
        rprint(f"  Host: [green]{host}[/green]")
        rprint(f"  Port: [green]{port}[/green]")
        rprint(f"  Reload: [green]{reload}[/green]")
        rprint(f"  Workers: [green]{workers}[/green]")
        rprint(f"\n  Docs: [link]http://{host}:{port}/docs[/link]")
        rprint("")

        uvicorn.run(
            "nexus_matcher.presentation.api.app:create_app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            factory=True,
        )

    except ImportError:
        rprint(
            "[red]Error: uvicorn not installed. Install with: pip install nexus-matcher[api][/red]"
        )
        raise typer.Exit(1) from None
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


# =============================================================================
# INFO COMMAND
# =============================================================================


@app.command()
def info() -> None:
    """
    Show system information and configuration.
    """
    from nexus_matcher import __version__

    panel_content = f"""
[bold]NexusMatcher[/bold] v{__version__}

[cyan]Deployment Modes:[/cyan]
  • Library: [green]from nexus_matcher import NexusMatcher[/green]
  • API:     [green]nexus-matcher api[/green]
  • CLI:     [green]nexus-matcher match[/green]

[cyan]Supported Formats:[/cyan]
  • Schemas:      Avro, JSON Schema, SQL DDL, CSV
  • Dictionaries: Excel (.xlsx), CSV (.csv)

[cyan]Configuration:[/cyan]
  • Environment:  NEXUS_* variables
  • Config file:  .env

[cyan]Documentation:[/cyan]
  • API Docs:     http://localhost:8000/docs
  • GitHub:       https://github.com/jpmc/nexus-matcher
"""

    console.print(Panel(panel_content, title="System Information", border_style="blue"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_matcher():
    """Get or create matcher instance."""
    from nexus_matcher.application.use_cases.match_schema import NexusMatcher

    return NexusMatcher.from_config()


def _display_table(results: dict, top_k: int, threshold: float, verbose: bool) -> None:
    """Display results as rich table."""
    table = Table(title="Match Results")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Rank", style="dim")
    table.add_column("Match", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Decision", style="bold")

    for field_path, matches in results.items():
        for i, match in enumerate(matches[:top_k]):
            if match.final_confidence < threshold:
                continue

            # Color code confidence
            conf = match.final_confidence
            if conf >= 0.75:
                conf_style = "green"
            elif conf >= 0.5:
                conf_style = "yellow"
            else:
                conf_style = "red"

            # Color code decision
            decision = match.decision.value
            if decision == "AUTO_APPROVE":
                dec_style = "green"
            elif decision == "REVIEW":
                dec_style = "yellow"
            else:
                dec_style = "red"

            table.add_row(
                field_path if i == 0 else "",
                str(match.rank),
                match.dictionary_entry.business_name[:40],
                f"[{conf_style}]{conf:.2%}[/{conf_style}]",
                f"[{dec_style}]{decision}[/{dec_style}]",
            )

        # Add separator between fields
        if verbose and matches:
            table.add_row("", "", "", "", "")

    console.print(table)


def _format_json(results: dict, top_k: int, threshold: float) -> str:
    """Format results as JSON."""
    output = {}

    for field_path, matches in results.items():
        output[field_path] = [
            {
                "rank": match.rank,
                "dictionary_entry": {
                    "id": match.dictionary_entry.id,
                    "business_name": match.dictionary_entry.business_name,
                    "logical_name": match.dictionary_entry.logical_name,
                    "data_type": match.dictionary_entry.data_type.value,
                },
                "confidence": round(match.final_confidence, 4),
                "decision": match.decision.value,
                "scores": {
                    "semantic": round(match.score_breakdown.semantic_score, 4),
                    "lexical": round(match.score_breakdown.lexical_score, 4),
                    "type": round(match.score_breakdown.type_compatibility_score, 4),
                },
            }
            for match in matches[:top_k]
            if match.final_confidence >= threshold
        ]

    return json.dumps(output, indent=2)


def _format_csv(results: dict, top_k: int, threshold: float) -> str:
    """Format results as CSV."""
    lines = ["field_path,rank,business_name,logical_name,confidence,decision"]

    for field_path, matches in results.items():
        for match in matches[:top_k]:
            if match.final_confidence < threshold:
                continue
            lines.append(
                f'"{field_path}",{match.rank},"{match.dictionary_entry.business_name}",'
                f'"{match.dictionary_entry.logical_name}",{match.final_confidence:.4f},'
                f"{match.decision.value}"
            )

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================


def run() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()
