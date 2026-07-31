import pandas as pd
from edgar import Company, set_identity

# Set User-Agent identity required by SEC EDGAR
set_identity("Jane Doe jdoe@example.com")


def get_facts_dataframe(xb) -> pd.DataFrame | None:
    """Safely converts edgartools XBRL/FactsView objects to a pandas DataFrame."""
    if xb is None:
        return None

    # Retrieve facts container from xb or xb.instance
    facts_obj = getattr(xb, "facts", None)
    if facts_obj is None and hasattr(xb, "instance"):
        facts_obj = getattr(xb.instance, "facts", None)

    if facts_obj is None:
        return None

    # Convert FactsView / FactQuery / XBRLInstance to Pandas DataFrame
    if isinstance(facts_obj, pd.DataFrame):
        return facts_obj
    elif hasattr(facts_obj, "to_dataframe"):
        return facts_obj.to_dataframe()
    elif hasattr(facts_obj, "to_pandas"):
        return facts_obj.to_pandas()
    elif hasattr(facts_obj, "query"):
        return facts_obj.query().to_dataframe()

    return None


def extract_maa_10q_facts_robust():
    """Extracts facts directly from raw 10-Q filings with full FactsView compatibility."""

    company = Company("GOOGL")

    # Fetch all quarterly filings
    filings = company.get_filings(form=["10-Q"])

    summary_records = []
    detailed_facts = []

    print(f"Extracting facts directly from {len(filings)} 10-Q filings...")

    for filing in filings:
        try:
            xb = filing.xbrl()
            facts_df = get_facts_dataframe(xb)

            # Check DataFrame length instead of .empty on FactsView
            if facts_df is not None and len(facts_df) > 0:
                fact_count = len(facts_df)

                # Attach filing metadata to individual fact rows
                df_copy = facts_df.copy()
                df_copy["filing_date"] = filing.filing_date
                df_copy["form"] = filing.form
                df_copy["accession_number"] = filing.accession_number
                detailed_facts.append(df_copy)
            else:
                fact_count = 0

            summary_records.append(
                {
                    "filing_date": filing.filing_date,
                    "form": filing.form,
                    "period_of_report": filing.period_of_report,
                    "fact_count": fact_count,
                }
            )

        except Exception as e:
            print(
                f"Error parsing {filing.accession_number} ({filing.filing_date}): {e}"
            )
            summary_records.append(
                {
                    "filing_date": filing.filing_date,
                    "form": filing.form,
                    "period_of_report": filing.period_of_report,
                    "fact_count": 0,
                }
            )

    summary_df = (
        pd.DataFrame(summary_records)
        .sort_values("filing_date")
        .reset_index(drop=True)
    )
    all_facts_df = (
        pd.concat(detailed_facts, ignore_index=True)
        if detailed_facts
        else pd.DataFrame()
    )

    return summary_df, all_facts_df


# Run robust extraction
summary_df, all_facts_df = extract_maa_10q_facts_robust()

# Inspect the 10 facts for a single filing date
sample_date = '2023-04-26'

sample_df = all_facts_df.loc[
    (all_facts_df['form'] == '10-Q') & 
    (all_facts_df['label'] == 'Research and development') & 
    (all_facts_df['filing_date'] == sample_date)
]

# View the distinguishing metadata columns
cols_to_inspect = [c for c in ['concept', 'period_end', 'dimensions', 'numeric_value', 'units'] if c in sample_df.columns]
print(sample_df[cols_to_inspect].to_string())