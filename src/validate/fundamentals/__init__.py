"""
The fundamentals validator: `fundamentals_facts` / `fundamentals_history` -> a ranked,
explained, self-contained finding queue in `fundamentals_check`.

    validator.py       FundamentalsValidator -- the ONE implementation that judges a value
    substrate.py       Substrates -- every frame, loaded once, projected, passed down
    finding.py         the investigation packet, and the cross-run `finding_id`
    check_register.py  the settled-findings config, and the rules that keep it from
                       becoming a suppression list
    report.py          fire rates, the queue, register health
    checks/            CHECK_REGISTRY + the three tier modules

Read `../README.md` first. Domain-scoped from day one so a future prices or insider validator
has an obvious home and this does not become a fundamentals package with a generic name.
"""
