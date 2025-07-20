import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for beautiful plots
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

# --- Constants and Global Mappings ---
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
    '%d/%m/%Y', '%-d/%-m/%Y', '%-d.%-m.%Y', # Added -%d for non-padded day
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', # Added datetime formats
    '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S', '%-d/%-m/%Y %H:%M:%S',
    '%-d.%-m.%Y %H:%M:%S'
]

FUZZY_MATCH_THRESHOLD = 60 # Threshold for fuzzy matching bank names

BANK_NAME_MAP = {
    'central bank of kenya': 'cbk', 'kenya commercial bank': 'kcb',
    'kingdom bank': 'kingdom', 'absa bank': 'absa', 'ABSA Bank': 'absa',
    'equity bank': 'equity', 'i&m bank': 'i&m', 'ncba bank kenya plc': 'ncba', 'ncba bank': 'ncba',
    'sbm bank (kenya) limited': 'sbm', 'sbm bank': 'sbm',
    'baas temporary account': 'baas', # Added for consistency
    'fx temporary account': 'fx_temp', # Added for consistency
    'other temporary account': 'other_temp', # Added for consistency
    'unclaimed funds': 'unclaimed_funds', # Added for consistency
    'yeepay': 'yeepay' # Added for consistency
}

# Define expected columns for FX Tracker and Bank Statements
FX_EXPECTED_COLUMNS = {
    'Amount': 'Amount',
    'Operation': 'Operation',
    'Completed At': 'Completed At',
    'Intermediary Account': 'Intermediary Account',
    'Currency': 'Currency',
    'Status': 'Status'
}

BANK_EXPECTED_COLUMNS = {
    'Date': 'Date',
    'Credit': 'Credit', # Or 'Deposit'
    'Debit': 'Debit',   # Or 'Withdrawal'
    'Description': 'Description'
}


# --- Helper Functions ---

def parse_date(date_str_raw):
    """Parses a date string into a datetime object using predefined formats."""
    if pd.isna(date_str_raw):
        return None
    if isinstance(date_str_raw, datetime): # Already a datetime object
        return date_str_raw
    if not isinstance(date_str_raw, str):
        return None

    # Attempt to parse as date only, stripping time if present
    date_str = str(date_str_raw).split('.')[0].strip() # Remove milliseconds if present

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def safe_float(x):
    """Safely converts a value to a float, handling commas, non-numeric inputs, and ensuring consistency."""
    if pd.isna(x) or x is None:
        return None
    try:
        cleaned_x = str(x).replace(',', '').strip()
        return float(cleaned_x)
    except (ValueError, TypeError):
        return None

def normalize_bank_key(raw_key):
    """Normalizes bank names to a consistent short code, using fuzzy matching."""
    raw_key_lower = str(raw_key).lower().strip()
    
    # First, try direct replacement from BANK_NAME_MAP
    for long, short in BANK_NAME_MAP.items():
        if raw_key_lower.startswith(long):
            return short # Return the short code if a direct prefix match is found

    # If no direct match, try fuzzy matching against known short codes/replacements
    # Create a list of all possible normalized bank names for fuzzy matching
    all_normalized_bank_names = list(set(BANK_NAME_MAP.values())) # Use only the short codes for fuzzy matching targets

    match = process.extractOne(raw_key_lower, all_normalized_bank_names, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_MATCH_THRESHOLD:
        return match[0] # Return the best fuzzy matched normalized name
    
    return raw_key_lower # Return original if no good fuzzy match or direct map

def resolve_amount_column(columns, operation):
    """Identifies the amount column based on the operation (credit/debit)."""
    columns_lower = [col.lower() for col in columns]
    if operation.lower() == 'credit':
        candidates = ['credit', 'deposit', 'amount']
    elif operation.lower() == 'debit':
        candidates = ['debit', 'withdrawal', 'amount']
    else:
        candidates = ['amount', 'value', 'credit', 'deposit', 'debit', 'withdrawal'] # Fallback for general amount

    for key in candidates:
        if key in columns_lower:
            return columns[columns_lower.index(key)]
    return None

def resolve_date_column(columns):
    """Identifies the date column from a list of column names, prioritizing common formats."""
    for candidate in ['Value Date', 'Transaction Date', 'MyUnknownColumn', 'Transaction date', 'Date', 'Activity Date']:
        if candidate in columns:
            return candidate
    return None

def get_amount_columns(columns):
    """Returns a list of potential amount columns."""
    return [col for col in columns if col.lower() in ['deposit', 'credit', 'withdrawal', 'debit', 'amount', 'value']]

def get_description_columns(columns):
    """Identifies the description column from a list of column names."""
    for desc in ['Transaction details','Transaction', 'Customer reference','Narration',
                 'Transaction Details', 'Detail',  'Transaction Remarks:',
                 'TransactionDetails', 'Description', 'Narrative', 'Remarks']:
        if desc in columns:
            return desc
    return None

def process_uploaded_file(uploaded_file, sheet_name=None):
    """Reads an uploaded file (CSV or Excel) into a DataFrame."""
    # Reset file pointer to the beginning for re-reading
    uploaded_file.seek(0)
    
    if uploaded_file.name.endswith('.csv'):
        # Try multiple encodings for CSV
        encodings = ['utf-8', 'latin1', 'ISO-8859-1']
        for enc in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=enc)
                return df
            except Exception:
                continue
        st.error(f"Failed to decode CSV file '{uploaded_file.name}' using common encodings.")
        return pd.DataFrame()
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            return df
        except Exception as e:
            st.error(f"Error reading Excel file '{uploaded_file.name}': {e}")
            return pd.DataFrame()
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return pd.DataFrame()

def get_excel_sheet_names(uploaded_file):
    """Returns sheet names for an Excel file."""
    # Reset file pointer to the beginning for re-reading
    uploaded_file.seek(0)
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as e:
        st.error(f"Error getting Excel sheet names: {e}")
        return []

# --- Reconciliation Logic ---

def reconcile_adjustment_row(
    adj_row: pd.Series,
    all_bank_dfs: dict,
    mode: str, # 'local' or 'foreign'
    date_tolerance_days: int = 3,
    amount_tolerance: float = 1.0, # Absolute tolerance for amount matching
    debug: bool = False # Control verbose printing
) -> bool:
    """
    Attempts to reconcile a single adjustment row against all uploaded bank statements.
    Returns True if a match is found, False otherwise.
    Appends to global matched_adjustments_list or unmatched_adjustments_list lists.
    """
    if debug:
        st.info(f"🔍 Processing Adjustment (Amount: {adj_row.get('Amount')}, Date: {adj_row.get('Completed At')}, Bank: {adj_row.get('Intermediary Account')}, Currency: {adj_row.get('Currency')})")

    amount = safe_float(adj_row.get('Amount'))
    if amount is None or pd.isna(amount) or abs(amount) < 0.01:
        if debug:
            st.warning("❌ Skipping row due to invalid or insignificant amount")
        return False

    parsed_date = parse_date(adj_row.get('Completed At'))
    if pd.isna(parsed_date) or parsed_date is None:
        if debug:
            st.warning("❌ Skipping row due to invalid or missing date ('Completed At')")
        return False

    ref_date = datetime(parsed_date.year, parsed_date.month, parsed_date.day)

    operation = str(adj_row.get('Operation', '')).strip().lower()
    if operation not in ['credit', 'debit']:
        if debug:
            st.warning(f"❌ Skipping row due to unrecognised operation: '{operation}'")
        st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Unrecognised operation: {operation}'})
        return False

    status = str(adj_row.get('Status', '')).strip().lower()
    if (mode == 'local' and status != 'successful') or \
       (mode == 'foreign' and status != 'completed'):
        if debug:
            st.warning(f"❌ Skipping row due for mode '{mode}' and status '{status}'")
        st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Skipped due to status "{status}" for mode "{mode}"'})
        return False

    intermediary_account = str(adj_row.get('Intermediary Account', '')).strip()
    currency = str(adj_row.get('Currency', '')).strip().upper()

    expected_bank_name_adj = None
    expected_currency_adj = None

    if mode == 'local':
        expected_bank_name_adj = normalize_bank_key(intermediary_account)
        expected_currency_adj = currency
        if not expected_bank_name_adj:
            if debug:
                st.warning(f"❌ Could not normalize bank name for local mode: '{intermediary_account}'")
            st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Could not normalize bank name for local mode'})
            return False
    elif mode == 'foreign':
        parts = intermediary_account.split('-')
        if len(parts) < 2:
            if debug:
                st.warning(f"❌ Skipping row due to malformed foreign intermediary account: '{intermediary_account}'")
            st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Malformed foreign intermediary account'})
            return False
        bank_name_raw = parts[0].strip()
        currency_raw = parts[1].strip().upper()

        expected_bank_name_adj = normalize_bank_key(bank_name_raw)
        expected_currency_adj = currency_raw # Use currency from intermediary account for foreign mode

        if not expected_bank_name_adj:
            if debug:
                st.warning(f"❌ Could not normalize bank name for foreign mode: '{bank_name_raw}'")
            st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Could not normalize bank name for foreign mode'})
            return False

        if currency != currency_raw:
             if debug:
                 st.warning(f"⚠️ Currency mismatch: Adjustment currency '{currency}' vs Intermediary Account currency '{currency_raw}'")
    else:
        if debug:
            st.warning(f"❌ Invalid reconciliation mode: {mode}")
        st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Invalid mode: {mode}'})
        return False

    if debug:
        st.info(f"   Expected (Normalized) Bank: '{expected_bank_name_adj}', Currency: '{expected_currency_adj}'")

    target_bank_df_key = None
    for bank_df_key in all_bank_dfs.keys():
        bank_df_key_lower = bank_df_key.lower() # e.g., 'kcb usd_csv'

        # Attempt to parse bank name and currency from the bank_df_key
        # This assumes file names are like "bankname currency_csv" or "bankname currency.xlsx"
        # Example: 'kcb usd_csv' -> 'kcb', 'usd'
        # Example: 'absa kes_csv' -> 'absa', 'kes'
        # Remove common file extensions and split by space or underscore
        clean_file_key = bank_df_key_lower.replace('_csv', '').replace('.csv', '').replace('_xlsx', '').replace('.xlsx', '').replace('_', ' ')
        
        parts = clean_file_key.split(' ')
        
        bank_name_from_file = parts[0] if parts else ''
        currency_from_file = parts[1] if len(parts) > 1 else ''

        # Normalize the bank name from the file key
        normalized_bank_name_from_file = normalize_bank_key(bank_name_from_file)

        if debug:
            st.info(f"   Checking bank statement file: '{bank_df_key}'")
            st.info(f"     File parsed: Normalized Bank='{normalized_bank_name_from_file}', Currency='{currency_from_file}'")
            st.info(f"     Adjustment: Normalized Bank='{expected_bank_name_adj}', Currency='{expected_currency_adj}'")

        # Primary matching: Normalized bank name and currency must match
        bank_name_match_score = fuzz.ratio(expected_bank_name_adj, normalized_bank_name_from_file)
        bank_name_match = (bank_name_match_score >= FUZZY_MATCH_THRESHOLD)
        currency_match = (expected_currency_adj.lower() == currency_from_file.lower())

        if bank_name_match and currency_match:
            target_bank_df_key = bank_df_key
            if debug:
                st.success(f"   ✅ Match found for bank DF key: {target_bank_df_key} (Bank Name & Currency Match)")
            break
        elif debug:
            st.info(f"   ❌ No match for '{bank_df_key}': Bank Name Match ({bank_name_match}, score {bank_name_match_score}), Currency Match ({currency_match})")

    if not target_bank_df_key:
        if debug:
            st.error(f"   ❌ No matching bank statement found for this adjustment in any of the uploaded files based on normalized bank name and currency.")
        st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'No matching bank statement found (normalized bank name/currency mismatch)'})
        return False

    bank_df = all_bank_dfs[target_bank_df_key]
    if bank_df.empty:
        if debug:
            st.warning(f"⚠️ Bank statement '{target_bank_df_key}' is empty.")
        st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Target bank statement ({target_bank_df_key}) is empty'})
        return False

    bank_df_columns = bank_df.columns.tolist()

    date_column = resolve_date_column(bank_df_columns)
    amount_column = resolve_amount_column(bank_df_columns, operation)

    if debug:
        st.info(f"   📅 Using date column: {date_column} | 💰 Using amount column: {amount_column}")

    if not date_column or not amount_column:
        if debug:
            st.error("❌ Missing date or amount column in bank data for reconciliation")
        st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Missing date/amount column in bank statement'})
        return False

    # Convert bank statement date column to datetime objects
    bank_df['_ParsedDate'] = bank_df[date_column].apply(parse_date)

    # Filter by date tolerance
    date_matches_df = bank_df[
        (bank_df['_ParsedDate'].notna()) &
        (bank_df['_ParsedDate'].between(
            ref_date - timedelta(days=date_tolerance_days),
            ref_date + timedelta(days=date_tolerance_days)
        ))
    ].copy()

    if debug:
        st.info(f"🔎 Found {len(date_matches_df)} date matches in bank statement '{target_bank_df_key}'")

    match_found = False
    for idx, bank_row in date_matches_df.iterrows():
        bank_amt_raw = bank_row.get(amount_column)
        bank_amt = safe_float(bank_amt_raw)

        if bank_amt is None:
            continue

        if debug:
            st.info(f"  Comparing bank amount {bank_amt} (from column '{amount_column}') with adjustment amount {amount}")

        if abs(bank_amt - amount) <= amount_tolerance:
            # Generate a unique key for the bank record to mark it as matched
            bank_record_key = (
                target_bank_df_key,
                bank_row['_ParsedDate'].strftime('%Y-%m-%d'),
                round(bank_amt, 2),
                operation
            )
            if bank_record_key not in st.session_state.matched_bank_keys:
                st.session_state.matched_adjustments_list.append({
                    'Adjustment_Date': parsed_date.strftime('%Y-%m-%d'),
                    'Adjustment_Amount': amount,
                    'Adjustment_Operation': operation,
                    'Adjustment_Intermediary_Account': intermediary_account,
                    'Adjustment_Currency': currency,
                    'Bank_Table': target_bank_df_key,
                    'Bank_Statement_Date': bank_row['_ParsedDate'].strftime('%Y-%m-%d'),
                    'Bank_Statement_Amount': bank_amt,
                    'Bank_Matched_Column': amount_column,
                    'Bank_Row_Index': idx
                })
                st.session_state.matched_bank_keys.add(bank_record_key)
                if debug:
                    st.success("✅ Match found and recorded!")
                match_found = True
                break
            else:
                if debug:
                    st.warning("⚠️ Potential duplicate match skipped (bank record already matched).")
                continue

    if not match_found:
        if debug:
            st.error("❌ No amount match found within tolerance for this adjustment.")
        st.session_state.unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'No amount match in bank statement'})
    return match_found


def perform_reconciliation():
    """Main function to perform the reconciliation process."""
    st.session_state.matched_adjustments_list = []
    st.session_state.unmatched_adjustments_list = []
    st.session_state.unmatched_bank_records_list = []
    st.session_state.matched_bank_keys = set()

    if st.session_state.fx_trade_df.empty:
        st.warning("FX Data is empty. Please upload and process FX Tracker data.")
        return
    if not st.session_state.bank_dfs:
        st.warning("No Bank Statements processed. Please upload and process bank data.")
        return

    st.subheader("--- Starting Reconciliation Process ---")
    current_mode = st.session_state.reconciliation_mode
    st.info(f"Reconciliation Mode: {current_mode.upper()}")

    # Display which bank statements are loaded for reconciliation
    if st.session_state.bank_dfs:
        st.info(f"Bank statements loaded for reconciliation: {', '.join(st.session_state.bank_dfs.keys())}")
    else:
        st.warning("No bank statements available for reconciliation.")
        return

    # Process all adjustments first
    for index, row in st.session_state.fx_trade_df.iterrows():
        reconcile_adjustment_row(
            adj_row=row,
            all_bank_dfs=st.session_state.bank_dfs,
            mode=current_mode,
            date_tolerance_days=3,
            amount_tolerance=1.0,
            debug=st.session_state.debug_mode # Use the debug mode checkbox state
        )

    st.subheader("--- Identifying Unmatched Bank Records ---")
    for bank_key, bank_df in st.session_state.bank_dfs.items():
        if bank_df.empty:
            st.warning(f"Skipping empty bank statement: {bank_key}")
            continue

        bank_df_copy = bank_df.copy()
        bank_df_copy.columns = bank_df_copy.columns.str.strip()
        date_col = resolve_date_column(bank_df_copy.columns.tolist())
        amount_cols = get_amount_columns(bank_df_copy.columns.tolist())
        description_col = get_description_columns(bank_df_copy.columns.tolist())

        if not date_col or not amount_cols or not description_col:
            st.warning(f"Skipping '{bank_key}': Missing required columns (Date, Amount, or Description).")
            continue

        bank_df_copy['_ParsedDate'] = bank_df_copy[date_col].apply(parse_date)

        for idx, row in bank_df_copy.iterrows():
            row_date = row.get('_ParsedDate')
            if not isinstance(row_date, datetime) or pd.isna(row_date):
                continue

            description = str(row.get(description_col, '')).strip()

            is_matched_in_any_way = False
            for amt_col in amount_cols:
                amt_val = safe_float(row.get(amt_col))
                if amt_val is None or abs(amt_val) < 0.01:
                    continue

                rounded_amt = round(amt_val, 2)
                operation_for_key = 'debit' if 'debit' in amt_col.lower() or amt_val < 0 else 'credit'
                if 'credit' in amt_col.lower():
                    operation_for_key = 'credit'
                elif 'debit' in amt_col.lower():
                    operation_for_key = 'debit'

                bank_record_key = (
                    bank_key,
                    row_date.strftime('%Y-%m-%d'),
                    rounded_amt,
                    operation_for_key
                )
                if bank_record_key in st.session_state.matched_bank_keys:
                    is_matched_in_any_way = True
                    break
            if is_matched_in_any_way:
                continue

            final_amt_col_for_unmatched = None
            final_amt_val_for_unmatched = None
            for amt_col in amount_cols:
                amt_val = safe_float(row.get(amt_col))
                if amt_val is not None and abs(amt_val) >= 0.01:
                    final_amt_col_for_unmatched = amt_col
                    final_amt_val_for_unmatched = round(amt_val, 2)
                    break

            if final_amt_val_for_unmatched is not None:
                st.session_state.unmatched_bank_records_list.append({
                    'Bank_Table': bank_key,
                    'Date': row_date.strftime('%Y-%m-%d'),
                    'Description': description,
                    'Transaction_Type_Column': final_amt_col_for_unmatched,
                    'Amount': final_amt_val_for_unmatched,
                    'Original_Row_Index': idx
                })

    st.session_state.df_matched_adjustments = pd.DataFrame(st.session_state.matched_adjustments_list)
    st.session_state.df_unmatched_adjustments = pd.DataFrame(st.session_state.unmatched_adjustments_list)
    st.session_state.df_unmatched_bank_records = pd.DataFrame(st.session_state.unmatched_bank_records_list)

    st.success("Reconciliation Complete!")
    st.write("---")
    st.write(f"✅ Total Adjustments Matched: {len(st.session_state.df_matched_adjustments)}")
    st.write(f"❌ Total Adjustments Unmatched: {len(st.session_state.df_unmatched_adjustments)}")
    st.write(f"📄 Total Unmatched Bank Records: {len(st.session_state.df_unmatched_bank_records)}")

    # Display results in expanders
    with st.expander("Matched Adjustments"):
        if not st.session_state.df_matched_adjustments.empty:
            st.dataframe(st.session_state.df_matched_adjustments)
            st.download_button(
                label="Download Matched Adjustments",
                data=st.session_state.df_matched_adjustments.to_csv(index=False).encode('utf-8'),
                file_name="matched_adjustments.csv",
                mime="text/csv",
            )
        else:
            st.info("No matched adjustments.")

    with st.expander("Unmatched Adjustments"):
        if not st.session_state.df_unmatched_adjustments.empty:
            st.dataframe(st.session_state.df_unmatched_adjustments)
            st.download_button(
                label="Download Unmatched Adjustments",
                data=st.session_state.df_unmatched_adjustments.to_csv(index=False).encode('utf-8'),
                file_name="unmatched_adjustments.csv",
                mime="text/csv",
            )
        else:
            st.info("No unmatched adjustments.")

    with st.expander("Unmatched Bank Records"):
        if not st.session_state.df_unmatched_bank_records.empty:
            st.dataframe(st.session_state.df_unmatched_bank_records)
            st.download_button(
                label="Download Unmatched Bank Records",
                data=st.session_state.df_unmatched_bank_records.to_csv(index=False).encode('utf-8'),
                file_name="unmatched_bank_records.csv",
                mime="text/csv",
            )
        else:
            st.info("No unmatched bank records.")

def perform_data_analysis_and_visualizations():
    """Performs data analysis and generates visualizations based on reconciliation results."""
    st.subheader("Data Analysis and Visualizations")

    if st.session_state.df_matched_adjustments.empty and \
       st.session_state.df_unmatched_adjustments.empty and \
       st.session_state.df_unmatched_bank_records.empty:
        st.warning("No data available for analysis. Please run reconciliation first.")
        return

    # 7.1 Reconciliation Overview
    st.markdown("### 7.1 Reconciliation Overview")
    reconciliation_status = pd.DataFrame({
        'Category': ['Matched Adjustments', 'Unmatched Adjustments', 'Unmatched Bank Records'],
        'Count': [len(st.session_state.df_matched_adjustments),
                  len(st.session_state.df_unmatched_adjustments),
                  len(st.session_state.df_unmatched_bank_records)]
    })
    st.write("**Counts of Matched/Unmatched Records:**")
    st.dataframe(reconciliation_status)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Category', y='Count', data=reconciliation_status, palette='viridis', ax=ax1)
    ax1.set_title('Overview of Reconciliation Status')
    ax1.set_ylabel('Number of Records')
    ax1.set_xlabel('')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)

    # 7.2 Unmatched Adjustments Analysis
    if not st.session_state.df_unmatched_adjustments.empty:
        st.markdown("### 7.2 Unmatched Adjustments Analysis")
        st.write("**Top Reasons for Unmatched Adjustments:**")
        reason_counts = st.session_state.df_unmatched_adjustments['Reason'].value_counts().reset_index()
        reason_counts.columns = ['Reason', 'Count']
        st.dataframe(reason_counts)

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        sns.barplot(x='Count', y='Reason', data=reason_counts, palette='magma', ax=ax2)
        ax2.set_title('Reasons for Unmatched Adjustments')
        ax2.set_xlabel('Number of Adjustments')
        ax2.set_ylabel('Reason')
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        st.write("**Distribution of Unmatched Adjustment Amounts:**")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(st.session_state.df_unmatched_adjustments['Amount'], bins=20, kde=True, color='red', ax=ax3)
        ax3.set_title('Distribution of Unmatched Adjustment Amounts')
        ax3.set_xlabel('Amount')
        ax3.set_ylabel('Frequency')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig3)
    else:
        st.info("No unmatched adjustments to analyze.")

    # 7.3 Unmatched Bank Records Analysis
    if not st.session_state.df_unmatched_bank_records.empty:
        st.markdown("### 7.3 Unmatched Bank Records Analysis")
        st.write("**Unmatched Bank Records by Bank/Table:**")
        bank_table_counts = st.session_state.df_unmatched_bank_records['Bank_Table'].value_counts().reset_index()
        bank_table_counts.columns = ['Bank_Table', 'Count']
        st.dataframe(bank_table_counts)

        fig4, ax4 = plt.subplots(figsize=(10, 7))
        sns.barplot(x='Count', y='Bank_Table', data=bank_table_counts, palette='cividis', ax=ax4)
        ax4.set_title('Unmatched Bank Records by Bank Statement')
        ax4.set_xlabel('Number of Records')
        ax4.set_ylabel('Bank Statement')
        ax4.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig4)

        st.write("**Distribution of Unmatched Bank Record Amounts:**")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.histplot(st.session_state.df_unmatched_bank_records['Amount'], bins=20, kde=True, color='blue', ax=ax5)
        ax5.set_title('Distribution of Unmatched Bank Record Amounts')
        ax5.set_xlabel('Amount')
        ax5.set_ylabel('Frequency')
        ax5.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig5)

        # Time series analysis for unmatched bank records (if dates are present)
        st.session_state.df_unmatched_bank_records['_ParsedDate'] = pd.to_datetime(st.session_state.df_unmatched_bank_records['Date'])
        if not st.session_state.df_unmatched_bank_records['_ParsedDate'].empty:
            st.write("**Unmatched Bank Records Over Time:**")
            daily_unmatched = st.session_state.df_unmatched_bank_records.set_index('_ParsedDate').resample('D')['Amount'].count()
            if not daily_unmatched.empty:
                fig6, ax6 = plt.subplots(figsize=(12, 6))
                daily_unmatched.plot(kind='line', marker='o', linestyle='-', color='purple', ax=ax6)
                ax6.set_title('Daily Trend of Unmatched Bank Records')
                ax6.set_xlabel('Date')
                ax6.set_ylabel('Number of Unmatched Records')
                ax6.grid(True)
                plt.tight_layout()
                st.pyplot(fig6)
            else:
                st.info("Not enough daily data to show time series trend for unmatched bank records.")
    else:
        st.info("No unmatched bank records to analyze.")

    st.success("Data Analysis and Visualizations Complete!")

# --- Streamlit App Layout ---

def fx_reconciliation_app():

    st.set_page_config(layout="wide", page_title="Adjustment Reconciliation Dashboard")

    # Define colors from the provided palette
    COLORS = {
        'white': '#FFFFFF',
        'secondary': '#798088',
        'primary': '#361371',
        'pink_alpha': '#9F6AF8CC',
        'container_alpha': '#F0EFEF4D',
        'buy_goods_color': '#F5EFFD',
        'green': '#2B9973',
        'buy_airtime_alpha': '#77CE8780',
        'utilities_alpha': '#9F6AF833',
        'red': '#E85E5D',
        'pink': '#9F6AF8',
        'color-text-input': '#FFAD6B'

    }

    # Inject custom CSS
    st.markdown(f"""
        <style>
            .reportview-container {{
                background-color: {COLORS['white']};
                color: {COLORS['primary']};
            }}
            .sidebar .sidebar-content {{
                background-color: {COLORS['buy_goods_color']};
                color: {COLORS['primary']};
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: {COLORS['primary']};
            }}
            .stButton>button {{
                background-color: {COLORS['primary']};
                color: {COLORS['white']};
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }}
            .stButton>button:hover {{
                background-color: {COLORS['pink']};
                color: {COLORS['white']};
                box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
                transform: translateY(-2px);
            }}
            .stSelectbox>div>div>div {{
                color: {COLORS['color-text-input']};
            }}
            .stTextInput>div>div>input {{
                color: {COLORS['color-text-input']};
       
            }}
            .stFileUploader>div>div>button {{
                background-color: {COLORS['green']};
                color: {COLORS['white']};
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }}
            .stFileUploader>div>div>button:hover {{
                background-color: {COLORS['buy_airtime_alpha']}; /* Lighter green */
                color: {COLORS['primary']};
                box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
                transform: translateY(-2px);
            }}
            .stDataFrame {{
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .stDataFrame table thead th {{
                background-color: {COLORS['primary']};
                color: {COLORS['white']};
            }}
            .stDataFrame table tbody tr:nth-child(even) {{
                background-color: {COLORS['container_alpha']};
            }}
            .stDataFrame table tbody tr:nth-child(odd) {{
                background-color: {COLORS['white']};
            }}
            .stInfo {{
                background-color: {COLORS['utilities_alpha']};
                color: {COLORS['primary']};
                border-left: 5px solid {COLORS['pink']};
                border-radius: 8px;
                padding: 10px;
            }}
            .stWarning {{
                background-color: rgba(232, 94, 93, 0.2); /* Red with alpha */
                color: {COLORS['red']};
                border-left: 5px solid {COLORS['red']};
                border-radius: 8px;
                padding: 10px;
            }}
            .stSuccess {{
                background-color: rgba(43, 153, 115, 0.2); /* Green with alpha */
                color: {COLORS['green']};
                border-left: 5px solid {COLORS['green']};
                border-radius: 8px;
                padding: 10px;
            }}
            .stExpander {{
                border: 1px solid {COLORS['secondary']};
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
                background-color: {COLORS['container_alpha']};
            }}
        </style>
        """, unsafe_allow_html=True)


    # Initialize session state variables
    if 'fx_trade_df' not in st.session_state:
        st.session_state.fx_trade_df = pd.DataFrame()
    if 'bank_dfs' not in st.session_state:
        st.session_state.bank_dfs = {}
    if 'matched_adjustments_list' not in st.session_state:
        st.session_state.matched_adjustments_list = []
    if 'unmatched_adjustments_list' not in st.session_state:
        st.session_state.unmatched_adjustments_list = []
    if 'unmatched_bank_records_list' not in st.session_state:
        st.session_state.unmatched_bank_records_list = []
    if 'matched_bank_keys' not in st.session_state:
        st.session_state.matched_bank_keys = set()
    if 'df_matched_adjustments' not in st.session_state:
        st.session_state.df_matched_adjustments = pd.DataFrame()
    if 'df_unmatched_adjustments' not in st.session_state:
        st.session_state.df_unmatched_adjustments = pd.DataFrame()
    if 'df_unmatched_bank_records' not in st.session_state:
        st.session_state.df_unmatched_bank_records = pd.DataFrame()
    if 'reconciliation_mode' not in st.session_state:
        st.session_state.reconciliation_mode = 'local' # Default mode
    if 'fx_uploaded_file_obj' not in st.session_state:
        st.session_state.fx_uploaded_file_obj = None
    if 'bank_uploaded_file_objs' not in st.session_state:
        st.session_state.bank_uploaded_file_objs = []
    if 'raw_bank_data_previews' not in st.session_state: # New state for raw bank data and mappings
        st.session_state.raw_bank_data_previews = {}
    if 'debug_mode' not in st.session_state: # New state for debug mode
        st.session_state.debug_mode = False


    # Sidebar for controls and uploads
    with st.sidebar:
        st.header("Upload Data")

        st.markdown("### 📥 FX Tracker Upload")
        fx_uploaded_file = st.file_uploader("Upload FX Tracker (CSV/Excel)", type=["csv", "xlsx"], key="fx_uploader")

        if fx_uploaded_file:
            # Check if a new file is uploaded or if the file object has changed
            if st.session_state.fx_uploaded_file_obj != fx_uploaded_file:
                st.session_state.fx_uploaded_file_obj = fx_uploaded_file # Store the new file object
                # Clear previous FX data when a new file is uploaded
                st.session_state.fx_trade_df = pd.DataFrame()
                st.session_state.fx_sheet_names = [] # Initialize for new file
                st.session_state.fx_selected_sheet = None # Initialize for new file

                # Process the newly uploaded file to get sheet names or initial df
                if fx_uploaded_file.name.endswith('.xlsx'):
                    st.session_state.fx_sheet_names = get_excel_sheet_names(fx_uploaded_file)
                    if st.session_state.fx_sheet_names:
                        st.session_state.fx_selected_sheet = st.session_state.fx_sheet_names[0]
                else:
                    # For CSV, directly load and store raw df
                    df_fx_raw_temp = process_uploaded_file(fx_uploaded_file)
                    if not df_fx_raw_temp.empty:
                        st.session_state.fx_raw_df = df_fx_raw_temp
                    else:
                        st.session_state.fx_raw_df = pd.DataFrame()

            file_details_fx = {"FileName": fx_uploaded_file.name, "FileType": fx_uploaded_file.type, "FileSize": fx_uploaded_file.size}
            st.write(file_details_fx)

            df_fx_raw = pd.DataFrame()
            if fx_uploaded_file.name.endswith('.xlsx'):
                selected_sheet_fx = st.selectbox("Select FX Sheet:", st.session_state.fx_sheet_names, key="fx_sheet_selector",
                                                index=st.session_state.fx_sheet_names.index(st.session_state.fx_selected_sheet) if st.session_state.fx_selected_sheet in st.session_state.fx_sheet_names else 0)
                if selected_sheet_fx != st.session_state.fx_selected_sheet: # Update selected sheet in state
                    st.session_state.fx_selected_sheet = selected_sheet_fx
                
                if selected_sheet_fx:
                    df_fx_raw = process_uploaded_file(fx_uploaded_file, sheet_name=selected_sheet_fx)
                    st.session_state.fx_raw_df = df_fx_raw # Store for mapping
            else:
                df_fx_raw = st.session_state.fx_raw_df # Use the already loaded raw df for CSV

            if not df_fx_raw.empty:
                st.write("FX Data Preview:")
                st.dataframe(df_fx_raw.head())

                st.markdown("#### Map FX Columns")
                fx_column_mappings = {}
                available_columns = df_fx_raw.columns.tolist()
                available_columns.insert(0, "") 

                for expected_col, default_val in FX_EXPECTED_COLUMNS.items():
                    # Try to pre-select if a column with a similar name exists
                    initial_selection = default_val if default_val in df_fx_raw.columns else ""
                    
                    mapped_col = st.selectbox(
                        f"Map '{expected_col}' to:",
                        options=available_columns,
                        index=available_columns.index(initial_selection) if initial_selection else 0,
                        key=f"fx_map_{expected_col}"
                    )
                    fx_column_mappings[expected_col] = mapped_col if mapped_col else None

                if st.button("Process FX Data", key="process_fx_btn"):
                    temp_df_fx = df_fx_raw.copy()
                    renamed_cols_dict = {}
                    for expected_col, mapped_col in fx_column_mappings.items():
                        if mapped_col and mapped_col in temp_df_fx.columns:
                            renamed_cols_dict[mapped_col] = expected_col
                    
                    temp_df_fx.rename(columns=renamed_cols_dict, inplace=True)
                    temp_df_fx.columns = temp_df_fx.columns.str.strip()
                    st.session_state.fx_trade_df = temp_df_fx
                    st.success("FX Data Processed!")
                    st.dataframe(st.session_state.fx_trade_df.head())
            else:
                st.error("Could not load FX data.")
        else:
            st.session_state.fx_trade_df = pd.DataFrame()
            st.session_state.fx_uploaded_file_obj = None
            st.session_state.fx_raw_df = pd.DataFrame() # Clear raw df as well


        st.markdown("### 🏦 Bank Statements Upload")
        bank_uploaded_files = st.file_uploader("Upload Bank Statement(s) (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True, key="bank_uploader")

        # Only update raw_bank_data_previews if new files are uploaded
        if bank_uploaded_files != st.session_state.bank_uploaded_file_objs:
            st.session_state.bank_uploaded_file_objs = bank_uploaded_files
            st.session_state.raw_bank_data_previews = {} # Reset for new uploads
            for i, file in enumerate(bank_uploaded_files):
                file_key = file.name.lower().replace('.', '_')
                st.session_state.raw_bank_data_previews[file_key] = {
                    'file_obj': file,
                    'df_raw': pd.DataFrame(),
                    'sheet_names': [],
                    'selected_sheet': None,
                    'column_mappings': {} # Store mappings per file
                }
                # Immediately process file to get sheet names or initial df
                if file.name.endswith('.xlsx'):
                    st.session_state.raw_bank_data_previews[file_key]['sheet_names'] = get_excel_sheet_names(file)
                    if st.session_state.raw_bank_data_previews[file_key]['sheet_names']:
                        st.session_state.raw_bank_data_previews[file_key]['selected_sheet'] = st.session_state.raw_bank_data_previews[file_key]['sheet_names'][0]
                        st.session_state.raw_bank_data_previews[file_key]['df_raw'] = process_uploaded_file(file, sheet_name=st.session_state.raw_bank_data_previews[file_key]['selected_sheet'])
                else:
                    st.session_state.raw_bank_data_previews[file_key]['df_raw'] = process_uploaded_file(file)

        # Display mapping UI for each uploaded bank file
        if st.session_state.raw_bank_data_previews:
            for file_key, data in st.session_state.raw_bank_data_previews.items():
                # Use a unique expander for each file to maintain state
                with st.expander(f"Configure {data['file_obj'].name}", expanded=True):
                    st.markdown(f"#### Configuration for {data['file_obj'].name}")
                    
                    df_bank_raw = data['df_raw']
                    
                    if data['file_obj'].name.endswith('.xlsx'):
                        current_sheet = st.selectbox(f"Select Sheet for {data['file_obj'].name}:", data['sheet_names'], 
                                                    index=data['sheet_names'].index(data['selected_sheet']) if data['selected_sheet'] in data['sheet_names'] else 0,
                                                    key=f"bank_sheet_selector_{file_key}")
                        if current_sheet != data['selected_sheet']:
                            data['selected_sheet'] = current_sheet
                            data['df_raw'] = process_uploaded_file(data['file_obj'], sheet_name=current_sheet)
                            df_bank_raw = data['df_raw'] # Update df_bank_raw for current iteration

                    if not df_bank_raw.empty:
                        st.write(f"Preview of {data['file_obj'].name}:")
                        st.dataframe(df_bank_raw.head())

                        available_columns = df_bank_raw.columns.tolist()
                        available_columns.insert(0, "") 

                        current_file_mappings = data['column_mappings'] # Use the stored mappings for this file

                        for expected_col, default_val in BANK_EXPECTED_COLUMNS.items():
                            initial_selection = current_file_mappings.get(expected_col) or (default_val if default_val in df_bank_raw.columns else "")
                            
                            mapped_col = st.selectbox(
                                f"Map '{expected_col}' (or main amount) to:",
                                options=available_columns,
                                index=available_columns.index(initial_selection) if initial_selection and initial_selection in available_columns else 0,
                                key=f"bank_map_{file_key}_{expected_col}"
                            )
                            current_file_mappings[expected_col] = mapped_col if mapped_col else None
                        data['column_mappings'] = current_file_mappings # Update mappings in state
                    else:
                        st.error(f"Could not load bank data from {data['file_obj'].name}.")
            
            # Button to process all bank statements after mapping
            if st.button("Process All Bank Statements", key="process_all_bank_btn"):
                st.session_state.bank_dfs = {}
                for file_key, data in st.session_state.raw_bank_data_previews.items():
                    df_to_process = data['df_raw'].copy()
                    renamed_cols_dict = {}
                    for expected_col, mapped_col in data['column_mappings'].items():
                        if mapped_col and mapped_col in df_to_process.columns:
                            renamed_cols_dict[mapped_col] = expected_col
                    
                    df_to_process.rename(columns=renamed_cols_dict, inplace=True)
                    df_to_process.columns = df_to_process.columns.str.strip()
                    st.session_state.bank_dfs[file_key] = df_to_process
                    st.success(f"Processed and applied mappings for {data['file_obj'].name}!")
                st.write("All Bank Statements Processed!")
        else:
            st.session_state.bank_dfs = {}
            st.session_state.bank_uploaded_file_objs = []
            st.session_state.raw_bank_data_previews = {}


    # Main content area
    st.header("Reconciliation")

    st.session_state.reconciliation_mode = st.selectbox(
        "Select Reconciliation Mode:",
        options=['local', 'foreign'],
        key="mode_selector"
    )

    # Debug mode checkbox
    st.session_state.debug_mode = st.checkbox("Enable Reconciliation Debug Logging", value=st.session_state.debug_mode)

    if st.button("Perform Reconciliation", key="reconcile_btn"):
        perform_reconciliation()

    st.header("Analysis and Visualizations")
    if st.button("Generate Analysis and Visualizations", key="analyze_btn"):
        perform_data_analysis_and_visualizations()

