"""Normalize five source responses, merge useful facts and preserve paid archives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from src.data_extract.utils.common.empty_markers import marker_mask
from src.data_extract.utils.schemas.def14a_schema import (
    BeneficialOwner,
    Def14AExtract,
    DirectorCompensation,
    DirectorInfo,
    ExecutiveCompensation,
    GovernanceProfile,
)
from src.data_extract.utils.structure.def14a.flatten import (
    _CHILD_SPEC,
    _DIRECTOR_COMPONENT_COLS,
    _EXEC_COMPONENT_COLS,
    _FEE_COLS,
    _ceo_from_compensation,
    _child_frames,
    _flatten,
    _prepare_frame,
)
from src.data_extract.utils.structure.def14a.retry import SECTIONS, Section, reservation_frame
from src.data_extract.utils.structure.def14a.validate import clean_person_name
from src.data_store.schema import Table, Tables


@dataclass
class ProcessedSection:
    """Owned save frames and the normalized legacy-compatible answer."""

    frames: dict[Table, pd.DataFrame]
    legacy: Def14AExtract


def _normalize_units(section: Section, answer: BaseModel, as_of: pd.Timestamp | str) -> dict[str, Any]:
    """Convert declared source bases once; leave unsupported currency/scale unknown."""
    payload = answer.model_dump()
    scales = {"units": 1.0, "thousands": 1_000.0, "millions": 1_000_000.0}

    # Normalize monetary cells without a magnitude or neighbour-based guess.
    if section.key != "ownership":
        amount_scale: str | None = payload.get("amount_scale")
        factor = scales.get(amount_scale) if amount_scale is not None and str(payload.get("currency", "")).upper() == "USD" else None
        for row in payload.get("compensation", []):
            for field in {*_EXEC_COMPONENT_COLS, *_DIRECTOR_COMPONENT_COLS, "total"}:
                value = row.get(field)
                row[field] = float(value) * factor if value is not None and factor is not None else None
        for field in (*_FEE_COLS, "median_employee_pay"):
            if field in payload:
                value = payload[field]
                payload[field] = float(value) * factor if value is not None and factor is not None else None
        payload["amount_scale"] = "units" if factor is not None else None

    # Keep economic ownership and voting on separate fractional bases.
    if section.key == "ownership":
        percentage = 0.01 if payload["percent_unit"] == "percent" else 1.0
        shares_scale: str | None = payload.get("shares_scale")
        shares = scales.get(shares_scale) if shares_scale is not None else None
        for row in payload["holders"]:
            row["shares"] = row["shares"] * shares if row["shares"] is not None and shares is not None else None
            for field in ("percent_of_class", "percent_of_voting_power"):
                row[field] = row[field] * percentage if row[field] is not None else None
        for field in ("insider_ownership_pct", "insider_voting_pct"):
            payload[field] = payload[field] * percentage if payload[field] is not None else None
        payload["insider_shares"] = payload["insider_shares"] * shares if payload["insider_shares"] is not None and shares is not None else None
        payload.update(percent_unit="fraction", shares_scale="units" if shares is not None else None)

    # Derive tenure from this filing's own date, never a later observation.
    if section.key == "governance":
        percentage = 0.01 if payload["percent_unit"] == "percent" else 1.0
        value = payload["say_on_pay_support_pct"]
        payload["say_on_pay_support_pct"] = value * percentage if value is not None else None
        year = pd.Timestamp(as_of).year
        for row in payload["directors"]:
            since = row.get("director_since_year")
            row["tenure_years"] = year - since if since is not None and 0 < since <= year else None
        payload["percent_unit"] = "fraction"
    return payload


def _legacy_answer(section: Section, payload: dict[str, Any], parent: dict[str, Any]) -> Def14AExtract:
    """Adapt normalized values to the existing pure row builders."""
    candidate = payload.get("ceo_name") or parent.get("ceo_name_proxy")
    ceo = candidate if isinstance(candidate, str) and candidate.strip() else None

    # Compensation rows share the old column vocabulary after a small field rename.
    if section.key in ("executive_pay", "director_pay"):
        model = ExecutiveCompensation if section.key == "executive_pay" else DirectorCompensation
        aliases = {"all_other_comp_usd": "other_compensation", "total_compensation_usd": "total"}
        rows = []
        for row in payload.get("compensation", []):
            values = {field: row.get(aliases.get(field, field.removesuffix("_usd"))) for field in model.model_fields}
            if model is ExecutiveCompensation:
                values["title"] = values["title"] or ""
            rows.append(model.model_validate(values))
        key = "compensation" if section.key == "executive_pay" else "director_compensation"
        return Def14AExtract.model_validate({"ceo_name": ceo, key: rows})

    # Audit totals may be derived only from all four same-basis categories.
    if section.key == "audit":
        fields = {field + "_usd": payload.get(field) for field in _FEE_COLS}
        fields.update(auditor_name=payload.get("auditor_name"), auditor_since_year=payload.get("auditor_since_year"))
        categories = [fields[field + "_usd"] for field in _FEE_COLS[1:5]]
        if fields["auditor_fees_usd"] is None and all(value is not None for value in categories):
            fields["auditor_fees_usd"] = sum(value for value in categories if value is not None)
        return Def14AExtract.model_validate({"governance": GovernanceProfile.model_validate(fields)})

    # Group ownership remains independent from named holder rows.
    if section.key == "ownership":
        group = {field: payload.get(field) for field in ("insider_shares", "insider_ownership_pct", "insider_voting_pct")}
        return Def14AExtract.model_validate(
            {
                "ceo_name": ceo,
                "ownership_holders": [BeneficialOwner.model_validate(row) for row in payload.get("holders", [])],
                "governance": GovernanceProfile.model_validate(group),
            }
        )

    # Board summaries retain reported counts and explicit tri-state provisions.
    fields = {field: payload.get(field) for field in GovernanceProfile.model_fields if field in payload}
    fields["majority_voting_for_directors"] = payload.get("majority_voting")
    fields["median_employee_pay_usd"] = payload.get("median_employee_pay")
    return Def14AExtract.model_validate(
        {
            **{field: payload.get(field) for field in ("ceo_age", "ceo_since_year", "ceo_is_founder", "ceo_is_board_chair")},
            "ceo_name": ceo,
            "directors": [DirectorInfo.model_validate(row) for row in payload.get("directors", [])],
            "governance": GovernanceProfile.model_validate(fields),
        }
    )


def _merge_rows(table: Table, df_saved: pd.DataFrame | None, df_new: pd.DataFrame) -> pd.DataFrame:
    """Fill missing cells and retain useful people/years from partial answers."""
    df_old = df_saved.loc[~marker_mask(table, df_saved)].copy() if df_saved is not None else pd.DataFrame()

    # Existing non-null facts win; repair adds missing cells and new keys.
    if df_old.empty:
        return df_new.copy()
    if df_new.empty:
        return df_old
    keys = list(table.pk)
    df_old = df_old.drop_duplicates(keys).set_index(keys)
    df_new = df_new.drop_duplicates(keys).set_index(keys)
    return df_old.combine_first(df_new).reset_index()


def _parent_patch(section: Section, legacy: Def14AExtract, stamp: dict[str, Any], parent: dict[str, Any]) -> pd.DataFrame:
    """Write only this domain's scalar facts and Python-derived summaries."""

    # Reuse derived summaries while excluding the legacy magnitude/pay-ratio repair path.
    safe = legacy.model_copy(deep=True)
    if safe.governance is not None:
        safe.governance = safe.governance.model_copy(
            update={**{field + "_usd": None for field in _FEE_COLS}, "ceo_pay_ratio": None, "median_employee_pay_usd": None}
        )
    filing = pd.Series({**stamp, "filing_date": stamp["as_of"]})
    values = _flatten(str(stamp["ticker"]), filing, safe)
    governance = legacy.governance or GovernanceProfile.model_validate({})
    if section.key == "audit":
        values = {field: getattr(governance, field + "_usd", None) for field in _FEE_COLS}
        values.update(auditor_name=governance.auditor_name, auditor_since_year=governance.auditor_since_year)
    if section.key == "governance":
        values.update(ceo_pay_ratio=governance.ceo_pay_ratio, median_employee_pay=governance.median_employee_pay_usd)
    ceo_name = clean_person_name(legacy.ceo_name)
    if section.key == "ownership" and ceo_name is not None:
        matches = [
            row
            for row in legacy.ownership_holders
            if (holder_name := clean_person_name(row.holder_name)) is not None and holder_name.casefold() == ceo_name.casefold()
        ]
        if len(matches) == 1:
            values.update(ceo_ownership_pct=matches[0].percent_of_class, ceo_voting_pct=matches[0].percent_of_voting_power)

    # Refresh derived copies, but never clear a valid scalar on a partial response.
    derived = set(next(item.parent_columns for item in SECTIONS if item.key == "executive_pay")) | {
        "n_director_comp_rows",
        "n_ownership_rows",
        "n_five_percent_holders",
        "n_directors",
        "avg_director_age",
        "avg_board_tenure",
    }
    row = {field: stamp[field] for field in (*Tables.def14a_llm.pk, "as_of")}
    for field in section.parent_columns:
        value = values.get(field)
        if value is None or pd.isna(value):
            continue
        old = parent.get(field)
        row[field] = old if field not in derived and old is not None and pd.notna(old) else value
    return pd.DataFrame([row])


def postprocess_section(
    section: Section,
    answer: BaseModel,
    stamp: dict[str, Any],
    saved: Mapping[Table, pd.DataFrame],
    parent: dict[str, Any],
) -> ProcessedSection:
    """Normalize one domain, preserve its partial facts and emit only owned frames."""
    # Reuse the already-loaded facts for this domain.
    df_saved = saved.get(section.table)

    # Legacy archived values were already requested in USD; do not scale them again.
    if isinstance(answer, Def14AExtract):
        legacy = answer.model_copy(deep=True)
    else:
        normalized = _normalize_units(section, answer, stamp["as_of"])
        legacy = _legacy_answer(section, normalized, parent)
    previous = pd.to_numeric(df_saved["retry"], errors="coerce").max() if df_saved is not None and "retry" in df_saved else float("nan")
    retry = int(previous) if pd.notna(previous) else 1
    frames: dict[Table, pd.DataFrame] = {}

    # Merge only this filing/domain; recompute duplicated summaries from retained rows.
    if section.table is not Tables.def14a_llm:
        rows = _child_frames(str(stamp["ticker"]), pd.Series({**stamp, "filing_date": stamp["as_of"]}), legacy)[section.table.name]
        numeric, keys = _CHILD_SPEC[section.table.name]
        df_new = _prepare_frame(rows, numeric, keys) if rows else pd.DataFrame()
        df_merged = _merge_rows(section.table, df_saved, df_new)
        if not df_merged.empty:
            df_merged["retry"] = retry
            frames[section.table] = df_merged
            raw_key = {"executive_pay": "compensation", "director_pay": "compensation", "ownership": "holders", "governance": "directors"}[
                section.key
            ]
            legacy_key = {
                "executive_pay": "compensation",
                "director_pay": "director_compensation",
                "ownership": "ownership_holders",
                "governance": "directors",
            }[section.key]
            records = df_merged.astype(object).where(pd.notna(df_merged), None).to_dict("records")
            retained = _legacy_answer(section, {raw_key: records}, parent)
            legacy = legacy.model_copy(update={legacy_key: getattr(retained, legacy_key)})
        else:
            frames[section.table] = reservation_frame(section.table, stamp, df_saved, retry)

    # Merge this domain's scalar facts before resolving identity-dependent copies.
    df_parent = _parent_patch(section, legacy, stamp, parent)

    # A repaired identity must immediately select its own already-saved pay and ownership.
    ceo_name = clean_person_name(df_parent.iloc[0].get("ceo_name_proxy", parent.get("ceo_name_proxy")))
    previous_ceo = clean_person_name(parent.get("ceo_name_proxy"))
    if section.key == "governance" and ceo_name:
        for owner in (item for item in SECTIONS if item.key in ("executive_pay", "ownership")):
            df_related = saved.get(owner.table)
            if df_related is None or df_related.empty:
                continue
            df_related = df_related.loc[
                df_related["ticker"].eq(stamp["ticker"])
                & df_related["accession_number"].eq(stamp["accession_number"])
                & pd.to_datetime(df_related["as_of"], errors="coerce").eq(pd.Timestamp(stamp["as_of"]))
                & ~marker_mask(owner.table, df_related)
            ]
            if df_related.empty:
                continue
            raw_key = "compensation" if owner.key == "executive_pay" else "holders"
            records = df_related.astype(object).where(pd.notna(df_related), None).to_dict("records")
            retained = _legacy_answer(owner, {raw_key: records}, {"ceo_name_proxy": ceo_name})
            selected = _ceo_from_compensation(retained)
            selected_name = clean_person_name(selected.name) if selected is not None else None
            matched = selected_name is not None and selected_name.casefold() == ceo_name.casefold()
            if owner.key == "ownership":
                matched = df_related.holder_name.map(clean_person_name).str.casefold().eq(ceo_name.casefold()).sum() == 1
            if not matched:
                continue
            copied = _parent_patch(owner, retained, stamp, {})
            columns = [column for column in owner.parent_columns if column.startswith("ceo_")]
            copied = copied.reindex(columns=columns)

            # Keep unsupported copies only when the retained CEO identity is unchanged.
            if previous_ceo and previous_ceo.casefold() == ceo_name.casefold():
                copied = copied.combine_first(pd.DataFrame([{column: parent.get(column) for column in columns}]))
            df_parent[columns] = copied

    # The audit counter alone belongs to the parent; sibling patches omit it.
    if section.key == "audit":
        df_parent["retry"] = retry
    frames[Tables.def14a_llm] = df_parent
    return ProcessedSection(frames, legacy)


def _archive_document(raw: str | None) -> dict[str, Any]:
    """Decode an external archived answer without dropping its original bytes."""
    if not raw or raw == "_empty":
        return {}
    try:
        document = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return document if isinstance(document, dict) else {}


def append_answer(raw: str | None, section: Section, answer: BaseModel, payload: str) -> str:
    """Durably append a paid raw answer before derived processing can fail."""
    document = _archive_document(raw)
    metadata = document.get("_def14a_split")

    # The version distinguishes split reservations from legacy all-domain attempts.
    if not isinstance(metadata, dict):
        metadata = {"version": 1, "legacy_json": raw, "responses": {}}
    responses = metadata.setdefault("responses", {})
    responses.setdefault(section.key, []).append({"response": json.loads(answer.model_dump_json()), "source": payload})
    document["_def14a_split"] = metadata
    return json.dumps(document, ensure_ascii=False, allow_nan=False)


def archived_answers(raw: str | None) -> dict[str, list[BaseModel]]:
    """Reuse all already-paid legacy/split partial answers without another submission."""
    try:
        document = json.loads(raw) if raw and raw != "_empty" else {}
    except (ValueError, TypeError):
        document = {}
    if not isinstance(document, dict):
        document = {}
    metadata = document.get("_def14a_split")
    legacy_raw = metadata.get("legacy_json") if isinstance(metadata, dict) else raw
    result: dict[str, list[BaseModel]] = {section.key: [] for section in SECTIONS}

    # Replay the original legacy request before appended domain repairs.
    if legacy_raw and legacy_raw != "_empty":
        try:
            legacy = Def14AExtract.model_validate_json(legacy_raw)
        except (ValueError, ValidationError):
            legacy = None
        if legacy is not None:
            for section in SECTIONS:
                result[section.key].append(legacy)

    # An empty/failed newer answer cannot hide useful facts in an older answer.
    responses = metadata.get("responses", {}) if isinstance(metadata, dict) else {}
    for section in SECTIONS:
        for entry in responses.get(section.key, []):
            try:
                result[section.key].append(section.schema.model_validate(entry["response"]))
            except (ValueError, TypeError, KeyError, ValidationError):
                continue
    return result


def refresh_archive(raw: str | None, section: Section, legacy: Def14AExtract) -> str:
    """Refresh the legacy-readable facade while retaining original and raw responses."""
    document = _archive_document(raw)
    if not isinstance(document.get("_def14a_split"), dict):
        document["_def14a_split"] = {"version": 1, "legacy_json": raw, "responses": {}}
    payload = json.loads(legacy.model_dump_json())
    child = {"executive_pay": "compensation", "director_pay": "director_compensation", "ownership": "ownership_holders", "governance": "directors"}
    governance_fields = {
        "audit": ("auditor_name", "auditor_since_year", *(field + "_usd" for field in _FEE_COLS)),
        "ownership": (
            "insider_shares",
            "insider_ownership_pct",
            "insider_voting_pct",
            "ceo_ownership_pct",
            "ceo_voting_pct",
            "n_five_percent_holders",
        ),
        "governance": (
            "board_size",
            "n_independent_directors",
            "n_women_directors",
            "independent_chair",
            "ceo_is_board_chair",
            "lead_independent_director",
            "classified_board",
            "dual_class_shares",
            "poison_pill",
            "majority_voting_for_directors",
            "say_on_pay_support_pct",
            "ceo_pay_ratio",
            "median_employee_pay_usd",
        ),
    }

    # Keep the facade compatible with old JSON consumers, without adding a progress field.
    if section.key in child and payload[child[section.key]]:
        document[child[section.key]] = payload[child[section.key]]
    if section.key == "governance":
        for field in ("ceo_name", "ceo_age", "ceo_since_year", "ceo_is_founder", "ceo_is_board_chair"):
            if payload.get(field) is not None and document.get(field) is None:
                document[field] = payload[field]
    current = document.get("governance") or {}
    for field in governance_fields.get(section.key, ()):
        value = (payload.get("governance") or {}).get(field)
        if value is not None and current.get(field) is None:
            current[field] = value
    if current:
        document["governance"] = current
    return json.dumps(document, ensure_ascii=False, allow_nan=False)
