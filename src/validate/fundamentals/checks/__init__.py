"""
checks/  (src/validate/fundamentals/)
--------------------------------------------------------------------------------------------
`CHECK_REGISTRY`: every check, declared exactly once, next to its implementation.

AGENTS.md's rule -- "never hand-list what a registry drives" -- is the whole design here. The
CLI's `--tier` filter, the report renderer, the fire-rate table and the calibration pass all
enumerate this registry. A check that forgets to declare a `substrate` or an
`expected_fire_rate_ceiling` fails a contract test rather than silently running with a default.

## What a check IS

A function `(Substrates) -> list[Finding]`, registered with `@check(...)`. It gets the frames
already loaded and projected -- it must NEVER touch `store` itself, which is what makes
`tests/validate/` able to instantiate the whole validator against a synthetic frame with no
DB and no CLI, and what keeps Phase 10's "the validator re-reads the tables per check" risk
structurally impossible rather than merely avoided.

## `expected_fire_rate_ceiling` is the anti-DQC_0118 guard

XBRL-US's own documentation of DQC_0118 says it plainly: *"inconsistencies reported to filers
can be overwhelming as many don't represent real errors."* A check firing on 30% of rows has
not found 30% bad data; it has a threshold bug, and it buries every real finding under itself.

So each check DECLARES the rate it expects, as a fraction of the rows it examined, and the
calibration report flags any check over its own ceiling as a **threshold bug** rather than
leaving a human to notice a big number in a table. The ceiling is a claim about the check,
measured, and it moves only with evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:                                   # pragma: no cover - typing only
    from src.validate.fundamentals.finding import Finding
    from src.validate.fundamentals.substrate import Substrates

#: Which table a check reads (decision 41).
#:
#: TIER 1 RUNS ON `history`; TIERS 2 AND 3 RUN ON `facts`. Not a preference -- neither works
#: the other way round. On the history grain `frozen_series` and `level_outlier` fire BY
#: CONSTRUCTION over the ~20 forward-filled instant columns, and `q4_footing` / `holdout_q4`
#: cannot run at all because history stores TTM levels rather than discrete quarters. On the
#: facts grain alone, the validator would never check the table the cube actually reads.
HISTORY = "history"
FACTS = "facts"
SUBSTRATES: frozenset[str] = frozenset({HISTORY, FACTS})

#: What one finding is ABOUT, which is what `period_key` encodes. See `finding.py`.
GRAIN_CELL = "cell"        # one (ticker, as_of/period_end, field)
GRAIN_ROW = "row"          # one (ticker, as_of) publication event
GRAIN_SERIES = "series"    # one (ticker, field) span
GRAIN_TICKER = "ticker"    # one ticker, no period
GRAINS: frozenset[str] = frozenset({GRAIN_CELL, GRAIN_ROW, GRAIN_SERIES, GRAIN_TICKER})


@dataclass(frozen=True, slots=True)
class CheckSpec:
    """One check's contract. Everything the registry's consumers need, without importing it."""

    name: str
    tier: int
    substrate: str
    severity: str
    grain: str
    expected_fire_rate_ceiling: float
    fn: Callable[["Substrates"], list["Finding"]]
    doc: str = ""

    def __post_init__(self) -> None:
        if self.substrate not in SUBSTRATES:
            raise ValueError(f"{self.name}: substrate {self.substrate!r} not in {SUBSTRATES}")
        if self.grain not in GRAINS:
            raise ValueError(f"{self.name}: grain {self.grain!r} not in {GRAINS}")
        if self.tier not in (1, 2, 3, 4):
            raise ValueError(f"{self.name}: tier {self.tier} is not 1-4")
        if not 0.0 <= self.expected_fire_rate_ceiling <= 1.0:
            raise ValueError(f"{self.name}: expected_fire_rate_ceiling "
                             f"{self.expected_fire_rate_ceiling} is not a fraction")


#: name -> CheckSpec. Populated by the `@check` decorator at import time; the tier modules are
#: imported at the bottom of this file so that importing the package is enough to fill it.
CHECK_REGISTRY: dict[str, CheckSpec] = {}


def check(*, name: str, tier: int, substrate: str, severity: str, grain: str,
          expected_fire_rate_ceiling: float) -> Callable:
    """Register a check. The decorated function keeps working as a plain function.

    `severity` here is the check's DEFAULT. A check whose severity depends on what it found --
    `series_shape` maps a `late_start` matching an ASC-842 adoption date to `info` and an
    unexplained one to `high` -- overrides it per finding; the declared value is what the
    registry reports and what the report groups by when nothing fired.
    """
    def decorate(fn: Callable[["Substrates"], list["Finding"]]):
        if name in CHECK_REGISTRY:
            raise ValueError(f"check {name!r} is registered twice")
        CHECK_REGISTRY[name] = CheckSpec(
            name=name, tier=tier, substrate=substrate, severity=severity, grain=grain,
            expected_fire_rate_ceiling=expected_fire_rate_ceiling, fn=fn,
            doc=(fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return decorate


def checks_for(tiers: Iterable[int] | None = None,
               names: Iterable[str] | None = None) -> list[CheckSpec]:
    """The registry, filtered and ordered by (tier, name) -- the order the report reads in.

    Both filters are inclusive: `--tier 1,3 --field X` narrows to tiers 1 and 3, and naming a
    check that is not registered raises rather than silently running nothing, because
    "0 findings" and "the check name was a typo" must never look the same.
    """
    wanted_tiers = set(tiers) if tiers is not None else None
    if names is not None:
        names = list(names)
        unknown = sorted(set(names) - set(CHECK_REGISTRY))
        if unknown:
            raise KeyError(f"unknown check(s): {unknown}; "
                           f"registered: {sorted(CHECK_REGISTRY)}")
    selected = [
        spec for spec in CHECK_REGISTRY.values()
        if (wanted_tiers is None or spec.tier in wanted_tiers)
        and (names is None or spec.name in names)
    ]
    return sorted(selected, key=lambda s: (s.tier, s.name))


# Imported LAST and for their side effect: each module's `@check` calls populate the registry
# above. Anything importing `src.validate.fundamentals.checks` therefore gets a full registry,
# which is what lets every consumer enumerate rather than hand-list.
from src.validate.fundamentals.checks import tier1_value   # noqa: E402,F401
from src.validate.fundamentals.checks import tier2_series  # noqa: E402,F401
from src.validate.fundamentals.checks import tier3_internal  # noqa: E402,F401
