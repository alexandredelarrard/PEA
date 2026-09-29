from __future__ import annotations

import numpy as np

from src.utils import text_metrics


def test_earnings_call_quality_uses_cleaned_sections_and_rejects_boilerplate() -> None:
    assess = getattr(text_metrics, "assess_earnings_call_sections", None)
    assert assess is not None, "shared earnings-call quality gate is missing"

    useful = "Revenue growth remained strong while margins improved and customer demand supported our outlook for the coming quarter. " * 8
    valid = assess({"prepared_remarks": useful, "qa": useful})
    assert valid.valid
    assert valid.combined_word_count >= 100

    boilerplate = "Thank you. Good morning. Congratulations on the quarter. " * 35
    malformed = assess({"prepared_remarks": boilerplate, "qa": "Operator instructions."})
    assert not malformed.valid
    assert malformed.combined_word_count < 100
    assert malformed.reason

    null_section = assess({"prepared_remarks": np.nan, "qa": useful})
    assert not null_section.valid
    assert null_section.cleaned_sections["prepared_remarks"] == ""

    print("\n=== SANITY CHECK: shared earnings-call quality gate ===")
    print(
        "  useful prepared + Q&A content passes; >100 raw words of greetings/boilerplate "
        "cleans below the 100-word threshold and is classified malformed. Validated."
    )
