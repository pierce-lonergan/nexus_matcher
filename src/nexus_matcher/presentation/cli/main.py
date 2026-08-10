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

import contextlib
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Annotated

import click
import typer
from rich import print as rprint
from rich.console import Console
from rich.markup import escape
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
# LEGACY CONSOLE SUPPORT
# =============================================================================


def _ascii_only(target: Console | None = None) -> bool:
    """
    Whether the console's code page can be trusted with non-ASCII decoration.

    Rich already derives this from the stream encoding and uses it to swap its
    box-drawing characters for `+---+`; it just never applies it to spinners or to text
    we hand it ourselves. Reusing its signal keeps the whole frame degrading together.

    `target` because status output does not always go to stdout -- see `_status_console`.
    The decision has to follow the stream the characters will actually be written to.
    """
    return (target or console).options.ascii_only


def _glyph(fancy: str, plain: str) -> str:
    """Pick decoration the console can actually encode."""
    return plain if _ascii_only() else fancy


def _spinner_column(target: Console | None = None) -> SpinnerColumn:
    """
    Rich's default spinner animates with Braille (U+2838, U+283C and friends), which
    cp437, cp850 and cp1252 all refuse to encode. `match` and `sync` -- the only two
    commands that do real work -- therefore died with a bare UnicodeEncodeError and exit
    1 before printing a single result, on every legacy Windows console.

    Fall back to the ASCII `-\\|/` spinner rather than forcing UTF-8 onto the user's
    terminal or removing progress output for everyone.
    """
    return SpinnerColumn(spinner_name="line" if _ascii_only(target) else "dots")


# Formats whose output is meant to be read by a program rather than by a person.
_MACHINE_FORMATS = frozenset({"json", "csv"})


def _status_console(payload_on_stdout: bool) -> Console:
    """
    Where progress, summaries and errors go.

    When there is no `--output`, the payload IS stdout, and everything else has to get
    off it. `nexus-matcher match schema.avsc -d dict.csv -f json > results.json` used to
    write a file that began with a spinner frame and ended with

        Summary: 0/1 fields auto-approved (0.0%)

    around the JSON -- so the only documented machine-readable surface produced something
    no JSON parser accepts, in the most obvious way anyone would ever script it. Rich's
    Progress and `rich.print` both default to stdout, and nothing here had ever said
    otherwise.

    Status goes to stderr only when it would otherwise corrupt a payload. With `--output`,
    or with the human `table` format, stdout is nobody's data channel and the messages
    stay exactly where users have always seen them.
    """
    return Console(stderr=True) if payload_on_stdout else console


def _soften_encoding_errors() -> None:
    """
    Last resort for the characters we do not get to choose.

    Picking safe glyphs covers our own decoration, but not Rich's: it truncates a
    `no_wrap` column with U+2026, so a field path one character too long for the window
    aborted the command on a legacy code page -- after the matching work had been paid
    for, and with a codec error rather than any hint that a name was the problem.

    Retuning the stream's error handler keeps the user's code page (nothing is forced to
    UTF-8) and turns an unencodable character into visible escape text instead of an
    exception. `backslashreplace` rather than `replace` because most of what reaches this
    path is data -- field names, dictionary values, loader errors -- and `?` would quietly
    misreport it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # test harnesses and pipes hand us streams that cannot be retuned
        with contextlib.suppress(OSError, ValueError):  # already detached: nothing to tune
            reconfigure(errors="backslashreplace")


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
    # The installed entry point is this Typer app, not run(), so the group callback is
    # the one place every command is guaranteed to pass through.
    _soften_encoding_errors()


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
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json, csv, table (default: inferred from --output, else table)",
        ),
    ] = None,
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
    resolved_format = _resolve_format(format, output)
    # No --output and a machine format means stdout carries the payload; status has to
    # move to stderr or it lands inside the document. See _status_console.
    status = _status_console(output is None and resolved_format in _MACHINE_FORMATS)

    try:
        with Progress(
            _spinner_column(status),
            TextColumn("[progress.description]{task.description}"),
            console=status,
        ) as progress:
            # Initialize matcher
            task = progress.add_task("Initializing matcher...", total=None)
            matcher = _get_matcher()

            # Load dictionary
            progress.update(task, description="Loading dictionary...")
            stats = matcher.load_dictionary(dictionary)
            if verbose:
                status.print(f"[dim]Loaded {stats.valid_entries} dictionary entries[/dim]")

            # Match schema
            progress.update(task, description="Matching schema...")
            results = matcher.match_schema(schema)

        # Format output. Every branch that has an --output path writes to it; a branch
        # that quietly skipped the write is what made the documented example a no-op.
        if resolved_format == "table":
            table = _build_results_table(results, top_k, threshold, verbose)
            if output:
                # An extension we cannot map to json or csv is STILL a request for a
                # file. Rendering to the terminal and writing nothing is precisely
                # NM-0003 -- `-o results.json` exiting 0 having produced nothing -- and
                # this branch lost the write once already while two comments above it
                # went on claiming it wrote.
                _write_output(output, _render_table_text(table))
            else:
                console.print(table)
        elif resolved_format == "json":
            json_output = _format_json(results, top_k, threshold, _scoring_weights(matcher))
            if output:
                _write_output(output, json_output)
            else:
                # `end=""` because _format_json already terminates the document with a
                # newline. Piping stdout to a file therefore produces the same bytes as
                # `-o`, which is what a script that does either one has always assumed.
                print(json_output, end="")
        elif resolved_format == "csv":
            csv_output = _format_csv(results, top_k, threshold)
            if output:
                _write_output(output, csv_output)
            else:
                print(csv_output)
        else:
            status.print(f"[red]Unknown format: {escape(resolved_format)}[/red]")
            raise typer.Exit(1)

        # Summary
        total_fields = len(results)
        auto_approved = sum(
            1
            for matches in results.values()
            if matches and matches[0].decision.value == "AUTO_APPROVE"
        )
        if not total_fields:
            # A schema that parses to zero fields is an ordinary user mistake -- an empty
            # file, or a format the parser did not recognise. Dividing by it turned that
            # into "Error: division by zero", which tells the user nothing about what to
            # do and looks like a crash in the tool rather than a problem with their input.
            status.print(
                "\n[yellow]No fields were parsed from the schema.[/yellow] "
                "Check the file is not empty and that --schema-format matches its contents."
            )
            raise typer.Exit(1)
        status.print(
            f"\n[bold]Summary:[/bold] {auto_approved}/{total_fields} fields auto-approved ({auto_approved / total_fields * 100:.1f}%)"
        )

    except click.exceptions.Exit:
        # click.exceptions.Exit, not typer.Exit: typer.Exit is a SUBCLASS of it, so
        # catching only typer.Exit let a bare click Exit fall through to `except Exception`
        # below, which appended a redundant "Error:" line to a message that had already
        # said everything useful. Catching the parent covers both.
        raise
    except Exception as e:
        status.print(f"[red]Error: {escape(str(e))}[/red]")
        if verbose:
            status.print_exception()
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
            _spinner_column(),
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
            bullet = _glyph("•", "-")
            rprint("\n[yellow]Errors:[/yellow]")
            for error in stats.errors[:10]:
                rprint(f"  [dim]{bullet} {escape(str(error))}[/dim]")
            if len(stats.errors) > 10:
                rprint(f"  [dim]... and {len(stats.errors) - 10} more[/dim]")

        if output_dir is not None:
            # `--output-dir/-o` was declared, advertised in this command's own docstring
            # example, and then referenced nowhere in the file -- so `sync dict.csv -o
            # ./index` built an index, discarded it, and exited 0 reporting success. Same
            # silent-no-op class as `match -o`, which was the other half of this fix.
            #
            # Only the sparse index has a save(); the dense vectors live in whichever
            # vector store was wired in and the in-memory one cannot persist. So this
            # writes what CAN be written and says plainly what it wrote, rather than
            # implying the whole index round-trips.
            output_dir.mkdir(parents=True, exist_ok=True)
            sparse = getattr(matcher, "_sparse_retriever", None)
            if sparse is None or not hasattr(sparse, "save"):
                rprint(
                    "[red]Error: --output-dir was given but this configuration has no "
                    "persistable index.[/red]"
                )
                raise typer.Exit(1)
            target = output_dir / "bm25_index.pkl"
            result = sparse.save(str(target))
            if not result.is_success:
                rprint(f"[red]Error: could not write index: {escape(str(result.error))}[/red]")
                raise typer.Exit(1)
            rprint(f"[green]Wrote sparse index to[/green] {escape(str(target))}")

        success_mark = _glyph("✓", "OK:")
        rprint(f"\n[green]{success_mark} Dictionary synced successfully[/green]")

    except click.exceptions.Exit:
        raise
    except Exception as e:
        rprint(f"[red]Error: {escape(str(e))}[/red]")
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

        # host is user-supplied and an IPv6 literal is written [::1] -- markup, to rich.
        safe_host = escape(host)
        rprint("[bold blue]Starting NexusMatcher API[/bold blue]")
        rprint(f"  Host: [green]{safe_host}[/green]")
        rprint(f"  Port: [green]{port}[/green]")
        rprint(f"  Reload: [green]{reload}[/green]")
        rprint(f"  Workers: [green]{workers}[/green]")
        rprint(f"\n  Docs: [link]http://{safe_host}:{port}/docs[/link]")
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
        # rich read the literal [api] as a style tag and dropped it, so the fix for a
        # missing API extra told the user to install the package without the extra --
        # i.e. to reproduce the exact state they were already in.
        hint = escape("pip install nexus-matcher[api]")
        rprint(f"[red]Error: uvicorn not installed. Install with: {hint}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        rprint(f"[red]Error: {escape(str(e))}[/red]")
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

    # U+2022 is absent from cp437 and cp850, so the bullets alone were enough to take
    # `info` down on a real DOS code page -- the panel printed halfway, then died.
    b = _glyph("•", "-")

    panel_content = f"""
[bold]NexusMatcher[/bold] v{__version__}

[cyan]Deployment Modes:[/cyan]
  {b} Library: [green]from nexus_matcher import NexusMatcher[/green]
  {b} API:     [green]nexus-matcher api[/green]
  {b} CLI:     [green]nexus-matcher match[/green]

[cyan]Supported Formats:[/cyan]
  {b} Schemas:      Avro, JSON Schema, SQL DDL, CSV
  {b} Dictionaries: Excel (.xlsx), CSV (.csv)

[cyan]Configuration:[/cyan]
  {b} Environment:  NEXUS_* variables
  {b} Config file:  .env

[cyan]Documentation:[/cyan]
  {b} API Docs:     http://localhost:8000/docs
  {b} GitHub:       https://github.com/pierce-lonergan/nexus_matcher
"""

    console.print(Panel(panel_content, title="System Information", border_style="blue"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_matcher():
    """Get or create matcher instance."""
    from nexus_matcher.application.use_cases.match_schema import NexusMatcher

    return NexusMatcher.from_config()


# Extensions we are willing to read an intent from. Anything else falls through to the
# table renderer, which still writes -- see _render_table_text.
_EXTENSION_FORMATS = {".json": "json", ".csv": "csv"}


def _resolve_format(requested: str | None, output: Path | None) -> str:
    """
    Decide what to produce from --format and --output.

    --format used to default to "table", and only the json/csv branches ever consulted
    --output, so the documented `match schema.json -d dictionary.csv -o results.json`
    wrote no file and exited 0 -- silent data loss dressed up as success. Leaving
    --format unset now means "infer from the output path", which is the promise the
    extension in that example was already making. An explicit --format still wins, so
    anything scripted against the old flag keeps its behaviour.
    """
    if requested is not None:
        return requested
    if output is not None:
        return _EXTENSION_FORMATS.get(output.suffix.lower(), "table")
    return "table"


def _write_output(path: Path, content: str) -> None:
    """
    Write results to the path the user asked for, and say so.

    Explicit UTF-8: write_text() otherwise picks up the console's code page, which on the
    legacy Windows consoles this release is about would fail on the first accented
    business name -- after the matching work was done.
    """
    path.write_text(content, encoding="utf-8")
    rprint(f"[green]Results written to {escape(str(path))}[/green]")


def _render_table_text(table: Table) -> str:
    """
    Render the results table as plain text for a file rather than a terminal.

    An --output path we cannot map to json or csv still has to produce a file; falling
    back to "table" and then writing nothing would just reinstate the bug. The dedicated
    console keeps this independent of the terminal's width and code page.
    """
    buffer = StringIO()
    Console(file=buffer, width=120, no_color=True).print(table)
    return buffer.getvalue()


def _build_results_table(results: dict, top_k: int, threshold: float, verbose: bool) -> Table:
    """Build the rich table of match results."""
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

            # Cells are markup-parsed too, so a dictionary entry called
            # "[deprecated] Customer Email" lost its prefix on the way to the screen --
            # the UI silently rewriting a data value it was asked to display.
            table.add_row(
                escape(field_path) if i == 0 else "",
                str(match.rank),
                escape(match.dictionary_entry.business_name[:40]),
                f"[{conf_style}]{conf:.2%}[/{conf_style}]",
                f"[{dec_style}]{decision}[/{dec_style}]",
            )

        # Add separator between fields
        if verbose and matches:
            table.add_row("", "", "", "", "")

    return table


# =============================================================================
# JSON OUTPUT -- the only documented machine-readable surface
# =============================================================================
#
# This is what a governance script consumes, so it has to carry the thing governance is
# about. It used to emit four identity columns and three of the five score components:
# `protection_level` -- the classification the whole stated use case exists to propagate,
# "so the object inherits that entry's classification" -- was absent, and the emitted
# numbers could not reproduce the emitted confidence, so an auditor could not check the
# arithmetic from the file. A fresh-eyes agent given a real governance task abandoned the
# CLI and rebuilt on the Python API. See docs/research/fresh-eyes.md, DX-002.

# The five signals that produce `final_confidence`, each paired with the weight that
# scales it. Rows are (json key, ScoreBreakdown attribute, MatchingConfig weight).
#
# One table rather than five inline attribute reads, for two reasons. It keeps the
# SCORE-TO-WEIGHT PAIRING in a single visible place -- pairing a component with the wrong
# weight would produce a file whose arithmetic is self-consistently wrong, the worst
# possible failure for an audit artifact. And it puts every coupling to another layer's
# field names in one place, so a rename there is one edit here rather than a hunt.
#
# The first key was "semantic" and is now "fused_retrieval", following the domain model:
# the number is the min-max-normalised FUSED RETRIEVAL score, rank-relative, and the
# rank-1 candidate sits at `fusion_alpha` whether the match is excellent or barely
# plausible. Calling it "semantic" in a file an auditor reads claims 90% similarity and
# delivers "ranked first among the candidates for this field". The other four keys keep
# the names they already had. `semantic_weight` keeps its own name because that is what
# MatchingConfig still calls it.
_SCORE_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("fused_retrieval", "fused_retrieval_score", "semantic_weight"),
    ("lexical", "lexical_score", "lexical_weight"),
    ("edit_distance", "edit_distance_score", "edit_distance_weight"),
    ("type", "type_compatibility_score", "type_weight"),
    ("domain", "domain_score", "domain_weight"),
)

# Where MatchingConfig lives on the matcher. The application layer exposes no public
# accessor, so this is a coupling to a private name -- see `_scoring_weights` for why
# that is survivable and `_verify_reproducible` for what makes it safe.
_MATCHER_CONFIG_ATTR = "_config"

# Decimals kept for every emitted number.
#
# Six, not the four this used to use, because the file now has to be self-checking: an
# auditor recomputes sum(scores[k] * weights[k]) and compares it to `confidence`. Rounding
# to N decimals moves each term by up to 5e-(N+1), so at four decimals the recomputed sum
# can disagree with the emitted confidence in the fourth decimal -- indistinguishable, to
# the person checking, from the tool getting the sum wrong. At six the disagreement is
# below 1e-5 and cannot be mistaken for an arithmetic error.
_JSON_PRECISION = 6

# How far the recomputed weighted sum may sit from the emitted confidence before the
# document is refused. Rounding to `_JSON_PRECISION` decimals moves each of the eleven
# terms by at most 5e-(P+1), so this is roughly two orders of magnitude of headroom over
# the worst case -- loose enough never to reject honest rounding, tight enough that no
# real disagreement between the components and the total can slip through.
_REPRODUCTION_TOLERANCE = 10 ** -(_JSON_PRECISION - 1)


def _drift(owner: str, attribute: str, consequence: str) -> RuntimeError:
    """
    The error for "another layer renamed something this writer reads".

    Raised rather than defaulted or skipped. A JSON document that quietly omits the
    protection level, or that carries weights which do not reproduce its own confidence,
    is worse than no document: it looks complete and is used as evidence.
    """
    return RuntimeError(
        f"{owner} has no {attribute!r}, so {consequence} The JSON output would not be "
        f"trustworthy, so it was not written. Use --format table or --format csv "
        f"meanwhile, and report this: the CLI's JSON writer and {owner} have drifted."
    )


def _scoring_weights(matcher: object) -> dict[str, float]:
    """
    The weights that actually produced the confidences in this run.

    Read off the LIVE matcher, not off `MatchingConfig()`, so a caller who tuned the
    weights gets a file that reproduces THEIR numbers rather than the shipped ones.

    A matcher that does not expose its config is not necessarily wrong; it just cannot
    say. Rather than refuse outright, fall back to the shipped defaults and let
    `_verify_reproducible` decide -- weights that reproduce every emitted confidence ARE
    the weights that produced it, whichever object they came from, and weights that do not
    are refused there. That keeps the document's guarantee resting on arithmetic anyone
    can check instead of on a private attribute name holding still.
    """
    config = getattr(matcher, _MATCHER_CONFIG_ATTR, None)
    if config is None:
        # Imported here, not at module scope: this pulls in the whole matching stack, and
        # `nexus-matcher --help` should not pay for it.
        from nexus_matcher.application.use_cases.match_schema import MatchingConfig

        config = MatchingConfig()

    weights: dict[str, float] = {}
    for key, _score_attr, weight_attr in _SCORE_COMPONENTS:
        value = getattr(config, weight_attr, None)
        if value is None:
            raise _drift(
                type(config).__name__,
                weight_attr,
                f"the {key!r} weight cannot be emitted and the confidence cannot be "
                f"reproduced from the file.",
            )
        weights[key] = round(float(value), _JSON_PRECISION)
    return weights


def _score_payload(breakdown: object) -> dict[str, float]:
    """All five components, so the weighted sum in the same record can be recomputed."""
    scores: dict[str, float] = {}
    for key, score_attr, _weight_attr in _SCORE_COMPONENTS:
        value = getattr(breakdown, score_attr, None)
        if value is None:
            raise _drift(
                type(breakdown).__name__,
                score_attr,
                f"the {key!r} component cannot be emitted and the confidence cannot be "
                f"reproduced from the file.",
            )
        scores[key] = round(float(value), _JSON_PRECISION)
    return scores


def _verify_reproducible(document: dict, weights: dict[str, float]) -> None:
    """
    Do the auditor's arithmetic before handing them the file.

    The whole promise of this output is that `sum(scores[k] * weights[k])` comes back to
    `confidence`. That promise is checked here, on the emitted numbers, so a document that
    cannot keep it is never written -- a file that looks complete and does not add up is
    worse than no file, because it is the one that gets used as evidence.

    It closes a class of drift no name-matching can: a component paired with the wrong
    weight, a sixth weighted signal this writer knows nothing about, or a matcher whose
    weights could not be read and whose confidences the shipped defaults do not explain.
    All three would otherwise produce a file that is self-consistently wrong.

    Clamped exactly as `_weighted_confidence` clamps, so weights that sum above 1.0 are
    not reported as a failure of arithmetic that never happened.
    """
    for field_path, records in document.items():
        for record in records:
            scores = record["scores"]
            total = sum(scores[key] * weight for key, weight in weights.items())
            recomputed = min(max(total, 0.0), 1.0)
            emitted = record["confidence"]
            if abs(recomputed - emitted) > _REPRODUCTION_TOLERANCE:
                raise RuntimeError(
                    f"the JSON output for {field_path!r} rank {record['rank']} would not "
                    f"be reproducible from its own numbers: the emitted components and "
                    f"weights give {recomputed!r}, the emitted confidence is {emitted!r}. "
                    f"Refusing to write a governance file whose arithmetic does not "
                    f"close. Use --format table or --format csv meanwhile, and report "
                    f"this: the CLI's JSON writer and the matcher's scoring have drifted."
                )


def _entry_payload(entry: object) -> dict[str, str]:
    """
    The dictionary entry as a governance record rather than as four identity columns.

    `protection_level` is the field the stated use case is built on. `definition` and
    `domain` are the two a reviewer needs to judge whether a match is right at all -- the
    business name alone does not distinguish "Customer Email Address" in the marketing
    domain from the same name in billing.
    """
    return {
        "id": entry.id,
        "business_name": entry.business_name,
        "logical_name": entry.logical_name,
        "definition": entry.definition,
        "data_type": entry.data_type.value,
        "protection_level": entry.protection_level.value,
        "domain": entry.domain,
    }


def _format_json(
    results: dict,
    top_k: int,
    threshold: float,
    weights: dict[str, float],
) -> str:
    """
    Format results as JSON: one self-contained, verifiable record per match.

    `weights` is repeated inside every record rather than hoisted into an envelope. An
    envelope would have to wrap the field paths in a container key, and every script
    written against the current top-level `{field_path: [...]}` shape would break; a
    sibling key alongside them would collide the first time somebody's schema contains a
    field genuinely called "weights". Repeating five numbers also means a single record
    lifted out with `jq` into a ticket still proves its own confidence.

    `sort_keys` gives every object a key order that does not depend on the order this
    function happens to build its dicts in, and orders the field paths lexicographically
    rather than by schema position -- so adding one field to a schema produces a one-hunk
    diff instead of a reordered file. The dict literals here are deliberately left in
    reading order, not alphabetical order, so that dropping `sort_keys` shows up.

    `ensure_ascii` stays at its default True. It is what makes this surface safe on the
    legacy Windows code pages NM-0001 was about: the document is pure ASCII, so an accented
    business name survives as an escape rather than being mangled by the console's
    `backslashreplace` error handler on the way to stdout. json.loads gives the original
    string back either way.
    """
    output = {}

    for field_path, matches in results.items():
        output[field_path] = [
            {
                "rank": match.rank,
                "dictionary_entry": _entry_payload(match.dictionary_entry),
                "confidence": round(match.final_confidence, _JSON_PRECISION),
                "decision": match.decision.value,
                "scores": _score_payload(match.score_breakdown),
                "weights": dict(weights),
            }
            for match in matches[:top_k]
            if match.final_confidence >= threshold
        ]

    _verify_reproducible(output, weights)

    # Trailing newline: a file without one is a permanent "\ No newline at end of file"
    # in every diff of a document whose whole point is diffing cleanly.
    return json.dumps(output, indent=2, sort_keys=True) + "\n"


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
