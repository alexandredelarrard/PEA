"""JCI <- TYC known-truth rows shared by the predecessor share-basis and level-factor tests."""

from __future__ import annotations

SPIN_2012 = 2.011667672500503
#: Yahoo's JCI split events (prices_splits, live 2026-10-07): Tyco's own splits, the 2012 ADT/Pentair spin and the 2016 consolidation.
JCI_YF = [
    ("JCI", "1995-11-15", 2.0),
    ("JCI", "1997-10-23", 2.0),
    ("JCI", "1999-10-22", 2.0),
    ("JCI", "2007-07-02", 0.25),
    ("JCI", "2012-10-01", SPIN_2012),
    ("JCI", "2016-09-06", 0.955),
]
#: sharadar_actions splits and spinoffs (live 2026-10-07): old JCI's own splits under JCI, Tyco's under TYC.
JCI_ACTIONS = [
    ("JCI", "2004-01-05", "split", 2.0),
    ("JCI", "2007-10-03", "split", 3.0),
    ("JCI", "2016-10-31", "spinoff", 0.1),
    ("TYC", "1999-10-22", "split", 2.0),
    ("TYC", "2007-07-02", "spinoff", 1.0),
    ("TYC", "2007-07-02", "split", 0.25),
    ("TYC", "2012-10-01", "spinoff", 0.5),
    ("TYC", "2012-10-01", "spinoff", 0.23994),
]
