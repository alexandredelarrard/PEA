"""The hidden `--as-of` run date on the extraction CLI."""

from __future__ import annotations

import datetime as dt

import click
import pandas as pd

from src.data_extract import cli as extraction_cli


def test_as_of_is_a_hidden_option_resolved_to_midnight() -> None:
    for name in ("seed-universe", "extraction-status"):
        command = extraction_cli.cli.commands[name]
        options = {p.name: p for p in command.params if isinstance(p, click.Option)}
        assert "as_of" in options and options["as_of"].hidden, name
        assert "--as-of" not in (command.get_help(click.Context(command)) or ""), name

    assert extraction_cli._run_date(dt.datetime(2026, 10, 1, 13, 45)) == pd.Timestamp("2026-10-01")
    assert extraction_cli._run_date(None) == pd.Timestamp.today().normalize()

    print("\n=== SANITY CHECK: --as-of ===")
    print("  hidden on seed-universe and extraction-status; an injected date resolves to midnight, the default to today. Validated.")
