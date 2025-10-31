"""
process_finance_contacts.py
---------------------------------
This script cleans and preprocesses the 'Finance Dep. Contact sheet.xlsx'
for donor outreach analysis and clustering in the Second Life e.V. project.
"""

import pandas as pd

def load_and_clean_excel(file_path: str) -> pd.DataFrame:
    """Load and clean the Finance Department contact sheet."""
    # Load raw Excel file
    raw_df = pd.read_excel(file_path, sheet_name=0)

    # Use first row as column headers
    df = raw_df.copy()
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    # Standardize column names
    df.columns.name = None
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    # Fix column name typos
    df = df.rename(columns={'first_outeach': 'first_outreach'})

    # Convert date columns
    if 'first_outreach' in df.columns:
        df['first_outreach'] = pd.to_datetime(df['first_outreach'], errors='coerce', dayfirst=True)

    # Boolean fields
    df['is_contacted'] = df['current_status'].astype(str).str.lower().fillna('').str.contains('outreach')
    df['has_contact_info'] = df['contact_information'].notna()
    df['is_english'] = df['language'].astype(str).str.lower().eq('english')

    # Define helper function
    def is_ideal(row):
        donation_type = str(row.get('type_of_donation', '')).lower()
        return (
            (row.get('type', '').lower() == 'non profit') and
            (('financial' in donation_type) or ('material' in donation_type)) and
            (not row.get('is_contacted', False)) and
            (row.get('has_contact_info', False))
        )

    df['is_ideal_candidate'] = df.apply(is_ideal, axis=1)

    # Remove unnamed/empty columns
    df = df.loc[:, ~df.columns.str.contains('^nan', na=False)]

    # Strip whitespace and normalize text fields
    for col in ['type', 'location', 'outreach_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace('nan', '')

    # Replace empty strings with NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # Report missing values summary
    print("\n=== Missing Values Summary ===")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    # Basic data summary
    print("\n✅ Data cleaned successfully.")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    file_path = "data/raw/Finance Dep. Contact sheet.xlsx"
    cleaned_df = load_and_clean_excel(file_path)

    # Save cleaned version
    output_path = "data/processed/cleaned_contacts.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\n💾 Cleaned data saved to: {output_path}")

    # Preview first few rows
    print("\n=== Preview ===")
    print(cleaned_df.head(5))
