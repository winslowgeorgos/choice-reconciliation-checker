import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

# Set Seaborn style for beautiful plots
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

# --- Constants and Global Mappings ---
DATE_FORMATS = [
    '%Y-%m-%d',
    '%Y/%m/%d',
    '%d.%m.%Y',
    '%Y.%m.%d',
    '%d/%m/%Y',
    '%Y-%m-%d %H:%M:%S',
    '%Y/%m/%d %H:%M:%S',
    '%d.%m.%Y %H:%M:%S',
    '%Y.%m.%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S'
]


# Simplified BANK_NAME_MAP for normalization (if still needed for other purposes, e.g., display)
# For direct matching, we'll use predefined options
BANK_NAME_MAP = {
    'central bank of kenya': 'cbk', 'kenya commercial bank': 'kcb',
    'kingdom bank': 'kingdom', 'absa bank': 'absa', 'ABSA Bank': 'absa',
    'equity bank': 'equity', 'i&m bank': 'i&m', 'ncba bank kenya plc': 'ncba', 'ncba bank': 'ncba',
    'sbm bank (kenya) limited': 'sbm', 'sbm bank': 'sbm',
    'baas temporary account': 'baas',
    'fx temporary account': 'fx_temp',
    'other temporary account': 'other_temp',
    'unclaimed funds': 'unclaimed_funds',
    'yeepay': 'yeepay'
}

# NEW: Predefined list of bank and currency combinations for user selection
# This list should be exhaustive for all possible bank statements you expect.
PREDEFINED_BANK_CURRENCY_OPTIONS = [
    "Absa KES", "Absa USD", "Absa EUR", "Absa GBP",
    "CBK KES", "CBK USD", "CBK EUR", "CBK GBP",
    "Equity KES", "Equity USD", "Equity EUR", "Equity GBP",
    "I&M KES", "I&M USD",
    "KCB KES", "KCB USD",
    "Kingdom KES", "Kingdom USD",
    "NCBA KES", "NCBA USD", "NCBA EUR",
    "SBM KES", "SBM USD",
    "BAAS Temporary KES", "BAAS Temporary USD",
    "FX Temporary KES", "FX Temporary USD",
    "Other Temporary KES", "Other Temporary USD",
    "Unclaimed Funds KES", "Unclaimed Funds USD",
    "Yeepay KES", "Yeepay USD",
    "UBA KES" , "UBA USD" , "UBA"
    # Add more as needed
]


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
    'Date': ['Date', 'Transaction Date', 'Value Date', 'Value date'],
    'Credit': ['Credit', 'Credit Amount', 'Money In', 'Deposit', 'Credit amount'], # Or 'Deposit'
    'Debit': ['Debit', 'Debit Amount', 'Money Out', 'Withdrawal', 'Debit amount'],   # Or 'Withdrawal'
    'Description': ['Description', 'Narrative', 'Transaction Details', 'Customer reference', 'Transaction Remarks:', 'Transaction Details', 'TransactionDetails', 'Transaction\nDetails']
}


# --- Helper Functions ---

def parse_date(date_str_raw):
    """Parses a date string into a datetime object using predefined formats."""
    if pd.isna(date_str_raw) or date_str_raw == pd.NaT:
        return None
    if isinstance(date_str_raw, datetime):
        return date_str_raw
    if not isinstance(date_str_raw, str):
        date_str_raw = str(date_str_raw)

    # Strip time part if present (e.g., "18.07.2025 12:34:56" => "18.07.2025")
    date_str = date_str_raw.partition(" ")[0].strip() if " " in date_str_raw.strip() else date_str_raw.strip()

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
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'windows-1252']
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
    debug: bool = False, # Control verbose printing
    matched_adjustments_list: list = None,
    unmatched_adjustments_list: list = None,
    matched_bank_keys: set = None
) -> bool:
    """
    Attempts to reconcile a single adjustment row against all uploaded bank statements.
    Returns True if a match is found, False otherwise.
    Appends to provided matched_adjustments_list or unmatched_adjustments_list lists.
    """
    if matched_adjustments_list is None or unmatched_adjustments_list is None or matched_bank_keys is None:
        raise ValueError("Matched/unmatched lists and matched_bank_keys set must be provided.")

    if debug:
        st.info(f"🔍 Processing Adjustment (Amount: {adj_row.get('Amount')}, Date: {adj_row.get('Completed At')}, Bank: {adj_row.get('Intermediary Account')}, Currency: {adj_row.get('Currency')}) for mode '{mode}'")

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
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Unrecognised operation: {operation}'})
        return False

    status = str(adj_row.get('Status', '')).strip().lower()
    if (mode == 'local' and status != 'successful') or \
       (mode == 'foreign' and status != 'completed'):
        if debug:
            st.warning(f"❌ Skipping row due for mode '{mode}' and status '{status}'")
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Skipped due to status "{status}" for mode "{mode}"'})
        return False

    intermediary_account = str(adj_row.get('Intermediary Account', '')).strip()
    currency = str(adj_row.get('Currency', '')).strip().upper()

    expected_bank_name_adj = None
    expected_currency_adj = None

    if mode == 'local':
        expected_bank_name_adj = intermediary_account.lower()
        expected_currency_adj = currency.upper()

    elif mode == 'foreign':
        parts = intermediary_account.split('-')
        if len(parts) < 2:
            if debug:
                st.warning(f"❌ Skipping row due to malformed foreign intermediary account: '{intermediary_account}'")
            unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Malformed foreign intermediary account'})
            return False
        bank_name_raw = parts[0].strip()
        currency_raw = parts[1].strip().upper()

        expected_bank_name_adj = bank_name_raw.lower()
        expected_currency_adj = currency_raw

        if currency.upper() != currency_raw.upper():
             if debug:
                 st.warning(f"⚠️ Currency mismatch: Adjustment currency '{currency}' vs Intermediary Account currency '{currency_raw}'")
    else:
        if debug:
            st.warning(f"❌ Invalid reconciliation mode: {mode}")
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Invalid mode: {mode}'})
        return False

    if debug:
        st.info(f"   Expected (from Adjustment) Bank: '{expected_bank_name_adj}', Currency: '{expected_currency_adj}'")

    target_bank_df_key = None
    for bank_df_key in all_bank_dfs.keys():
        key_parts = bank_df_key.split(' ')
        if len(key_parts) >= 2:
            bank_name_from_key = ' '.join(key_parts[:-1]).lower()
            currency_from_key = key_parts[-1].upper()
        else:
            if debug:
                st.warning(f"Skipping bank statement key '{bank_df_key}' due to unexpected format.")
            continue

        if debug:
            st.info(f"   Checking bank statement file: '{bank_df_key}'")
            st.info(f"     File parsed: Bank='{bank_name_from_key}', Currency='{currency_from_key}'")
            st.info(f"     Adjustment: Bank='{expected_bank_name_adj}', Currency='{expected_currency_adj}'")

        bank_name_from_adj_standardized = ""
        # Using the simplified direct mapping from BANK_NAME_MAP for the FX side's intermediary account
        # to match the standardized bank names chosen for bank statements.
        # This assumes 'intermediary_account' in FX Tracker is a human-readable bank name.
        matched_bank_name = False
        for long, short in BANK_NAME_MAP.items():
            if expected_bank_name_adj.startswith(long):
                bank_name_from_adj_standardized = short
                matched_bank_name = True
                break
        
        if not matched_bank_name:
            # If no direct map found, try to extract the first word or use as is
            bank_name_from_adj_standardized = expected_bank_name_adj.lower().split(' ')[0]


        bank_name_match = (bank_name_from_adj_standardized == bank_name_from_key)
        currency_match = (expected_currency_adj.lower() == currency_from_key.lower())

        if bank_name_match and currency_match:
            target_bank_df_key = bank_df_key
            if debug:
                st.success(f"   ✅ Match found for bank DF key: {target_bank_df_key} (Direct Bank Name & Currency Match)")
            break
        elif debug:
            st.info(f"   ❌ No match for '{bank_df_key}': Bank Name Match ({bank_name_match}), Currency Match ({currency_match})")

    if not target_bank_df_key:
        if debug:
            st.error(f"   ❌ No matching bank statement found for this adjustment based on selected bank statement name and currency.")
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'No matching bank statement found (direct bank name/currency mismatch)- {target_bank_df_key}'})
        return False

    bank_df = all_bank_dfs[target_bank_df_key]
    if bank_df.empty:
        if debug:
            st.warning(f"⚠️ Bank statement '{target_bank_df_key}' is empty.")
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Target bank statement ({target_bank_df_key}) is empty'})
        return False

    bank_df_columns = bank_df.columns.tolist()

    date_column = resolve_date_column(bank_df_columns)
    amount_column = resolve_amount_column(bank_df_columns, operation)

    if debug:
        st.info(f"   📅 Using date column: {date_column} | 💰 Using amount column: {amount_column}")

    if not date_column or not amount_column:
        if debug:
            st.error("❌ Missing date or amount column in bank data for reconciliation")
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Missing date/amount column in bank statement'})
        return False

    # Convert bank statement date column to datetime objects
    # This should already be done during pre-processing in main_dashboard, but a safeguard is good.
    if '_ParsedDate' not in bank_df.columns:
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
            bank_record_key_operation = 'debit' if 'debit' in amount_column.lower() or bank_amt < 0 else 'credit'
            if 'credit' in amount_column.lower(): # Specific check for credit column
                bank_record_key_operation = 'credit'
            
            # Generate a unique key for the bank record to mark it as matched
            bank_record_key = (
                target_bank_df_key,
                bank_row['_ParsedDate'].strftime('%Y-%m-%d'),
                round(amount, 2),
                bank_record_key_operation # Use the determined operation for the key
            )
            
            # Ensure we don't double-count a bank record if it matches multiple adjustments
            # if bank_record_key not in matched_bank_keys:
            matched_adjustments_list.append({
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
            matched_bank_keys.add(bank_record_key)
            if debug:
                st.success("✅ Match found and recorded!")
            match_found = True
            break
            # else:
            #     if debug:
            #         st.info(f"  Bank record {bank_record_key} already matched. Skipping duplicate.")
            #     # We count this adjustment as unmatched if its corresponding bank record is already used
            #     # or we might want to specifically track this as a "potential duplicate" if needed.
            #     # For now, let's treat it as unmatched since its target bank entry is taken.
            #     # unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Bank record already matched to another adjustment'})
            #     continue # Continue searching for another match for this adjustment
            

    if not match_found:
        if debug:
            st.error("❌ No amount match found within tolerance for this adjustment.")
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'No amount match in bank statement'})
    return match_found


def identify_unmatched_bank_records(bank_dfs: dict, matched_bank_keys: set, unmatched_bank_records_list: list, debug: bool):
    """Identifies bank records that were not matched by any adjustment."""
    for bank_key, bank_df in bank_dfs.items():
        if bank_df.empty:
            if debug:
                st.warning(f"Skipping empty bank statement: {bank_key}")
            continue

        bank_df_copy = bank_df.copy()
        bank_df_copy.columns = bank_df_copy.columns.str.strip()
        date_col = 'Date' # Standardized column name from main_dashboard preprocessing
        amount_cols = ['Credit', 'Debit'] # Standardized column names
        description_col = get_description_columns(bank_df_copy.columns.tolist()) # Still need to resolve this

        if not date_col or not amount_cols or not description_col:
            st.warning(
                f"Skipping '{bank_key}': Missing required columns for unmatched bank record identification:"
                f"{' Date,' if not date_col else ''}"
                f"{' Amount,' if not amount_cols else ''}"
                f"{' Description' if not description_col else ''}".rstrip(',')
            )
            continue

        # _ParsedDate should already exist from main_dashboard preprocessing
        if '_ParsedDate' not in bank_df_copy.columns:
            bank_df_copy['_ParsedDate'] = bank_df_copy[date_col].apply(parse_date)

        for idx, row in bank_df_copy.iterrows():
            row_date = row.get('_ParsedDate')
            if (pd.isna(row_date)): 
                # if debug:
                st.warning(f"Row {idx} in '{bank_key}' is missing parsed date: {row.get(date_col)}")
                continue

            if not isinstance(row_date, datetime):
                # if debug:
                st.warning(f"Row {idx} in '{bank_key}' has invalid or missing parsed date: {parse_date(str(row.get(date_col)))}")
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

                if bank_record_key in matched_bank_keys:
                    is_matched_in_any_way = True
                    break

            if is_matched_in_any_way:
                continue

            # Not matched: warn with reason
            final_amt_col_for_unmatched = None
            final_amt_val_for_unmatched = None
            for amt_col in amount_cols:
                amt_val = safe_float(row.get(amt_col))
                if amt_val is not None and abs(amt_val) >= 0.01:
                    final_amt_col_for_unmatched = amt_col
                    final_amt_val_for_unmatched = round(amt_val, 2)
                    break
            
            if final_amt_val_for_unmatched is not None:
                unmatched_bank_records_list.append({
                    'Bank_Table': bank_key,
                    'Date': row_date.strftime('%Y-%m-%d'),
                    'Description': description,
                    'Transaction_Type_Column': final_amt_col_for_unmatched,
                    'Amount': final_amt_val_for_unmatched,
                    'Original_Row_Index': idx
                })

def perform_reconciliation_for_mode(fx_df: pd.DataFrame, all_bank_dfs: dict, mode: str, debug: bool):
    """Performs reconciliation for a specific FX mode (local or foreign)."""
    st.subheader(f"--- Starting Reconciliation for {mode.upper()} FX Data ---")

    # Initialize mode-specific session state lists/sets
    st.session_state[f'matched_adjustments_list_{mode}'] = []
    st.session_state[f'unmatched_adjustments_list_{mode}'] = []
    st.session_state[f'matched_bank_keys_{mode}'] = set() # This set tracks bank records matched by THIS mode

    if fx_df.empty:
        st.warning(f"{mode.upper()} FX Data is empty. Skipping reconciliation for this mode.")
        return

    if not all_bank_dfs:
        st.warning("No Bank Statements processed. Please upload and process bank data.")
        return

    st.info(f"Reconciliation Mode: {mode.upper()}")
    st.info(f"Bank statements loaded for reconciliation: {', '.join(all_bank_dfs.keys())}")

    # Process all adjustments for the current mode
    for index, row in fx_df.iterrows():
        reconcile_adjustment_row(
            adj_row=row,
            all_bank_dfs=all_bank_dfs,
            mode=mode,
            date_tolerance_days=3,
            amount_tolerance=1.0,
            debug=debug,
            matched_adjustments_list=st.session_state[f'matched_adjustments_list_{mode}'],
            unmatched_adjustments_list=st.session_state[f'unmatched_adjustments_list_{mode}'],
            matched_bank_keys=st.session_state[f'matched_bank_keys_{mode}']
        )
    
    # After processing all adjustments for the current mode, identify unmatched bank records
    # Pass the matched_bank_keys specific to this mode.
    # Note: unmatched bank records are conceptually "unmatched across ALL FX data"
    # To properly implement this, we need a GLOBAL set of matched bank keys.
    # For now, let's just combine the keys from both reconciliation passes.
    # A simpler approach: Identify unmatched bank records *after* both local and foreign FX have been processed.
    # So, we'll move `identify_unmatched_bank_records` to the main `perform_reconciliation` function.

    st.session_state[f'df_matched_adjustments_{mode}'] = pd.DataFrame(st.session_state[f'matched_adjustments_list_{mode}'])
    st.session_state[f'df_unmatched_adjustments_{mode}'] = pd.DataFrame(st.session_state[f'unmatched_adjustments_list_{mode}'])

    st.success(f"Reconciliation for {mode.upper()} FX Data Complete!")
    st.write(f"--- {mode.upper()} Reconciliation Summary ---")
    st.write(f"✅ Total {mode.upper()} Adjustments Matched: {len(st.session_state[f'df_matched_adjustments_{mode}'])}")
    st.write(f"❌ Total {mode.upper()} Adjustments Unmatched: {len(st.session_state[f'df_unmatched_adjustments_{mode}'])}")


def perform_full_reconciliation(bank_dfs: dict):
    """Main function to perform the reconciliation process for both local and foreign FX data."""
    st.subheader("--- Overall Reconciliation Process ---")

    # Initialize global lists for unmatched bank records and combined matched bank keys
    st.session_state.unmatched_bank_records_list_global = []
    st.session_state.matched_bank_keys_global = set()

    if not bank_dfs:
        st.warning("No Bank Statements processed. Please upload and process bank data in 'Bank Statement Management'.")
        return

    # Perform reconciliation for LOCAL FX data
    perform_reconciliation_for_mode(
        fx_df=st.session_state.fx_trade_df_local,
        all_bank_dfs=bank_dfs,
        mode='local',
        debug=st.session_state.debug_mode
    )
    # Add keys from local reconciliation to global set
    st.session_state.matched_bank_keys_global.update(st.session_state.get('matched_bank_keys_local', set()))

    # Perform reconciliation for FOREIGN FX data
    perform_reconciliation_for_mode(
        fx_df=st.session_state.fx_trade_df_foreign,
        all_bank_dfs=bank_dfs,
        mode='foreign',
        debug=st.session_state.debug_mode
    )
    # Add keys from foreign reconciliation to global set
    st.session_state.matched_bank_keys_global.update(st.session_state.get('matched_bank_keys_foreign', set()))

    st.subheader("--- Identifying Global Unmatched Bank Records ---")
    identify_unmatched_bank_records(
        bank_dfs=bank_dfs, # Use the passed bank_dfs
        matched_bank_keys=st.session_state.matched_bank_keys_global,
        unmatched_bank_records_list=st.session_state.unmatched_bank_records_list_global,
        debug=st.session_state.debug_mode
    )
    st.session_state.df_unmatched_bank_records = pd.DataFrame(st.session_state.unmatched_bank_records_list_global)

    st.success("Overall Reconciliation Complete!")
    st.write("---")
    st.write(f"📄 Total Unmatched Bank Records (Global): {len(st.session_state.df_unmatched_bank_records)}")

    # Display results in expanders
    st.markdown("### Reconciliation Results Summary")

    with st.expander("Local FX Matched Adjustments"):
        if not st.session_state.df_matched_adjustments_local.empty:
            st.dataframe(st.session_state.df_matched_adjustments_local)
            st.download_button(
                label="Download Local Matched Adjustments",
                data=st.session_state.df_matched_adjustments_local.to_csv(index=False).encode('utf-8'),
                file_name="matched_adjustments_local.csv",
                mime="text/csv",
                key=f"download_matched_local_{uuid.uuid4()}"
            )
        else:
            st.info("No local matched adjustments.")

    with st.expander("Local FX Unmatched Adjustments"):
        if not st.session_state.df_unmatched_adjustments_local.empty:
            st.dataframe(st.session_state.df_unmatched_adjustments_local)
            st.download_button(
                label="Download Local Unmatched Adjustments",
                data=st.session_state.df_unmatched_adjustments_local.to_csv(index=False).encode('utf-8'),
                file_name="unmatched_adjustments_local.csv",
                mime="text/csv",
                key=f"download_unmatched_local_{uuid.uuid4()}"
            )
        else:
            st.info("No local unmatched adjustments.")
    
    with st.expander("Foreign FX Matched Adjustments"):
        if not st.session_state.df_matched_adjustments_foreign.empty:
            st.dataframe(st.session_state.df_matched_adjustments_foreign)
            st.download_button(
                label="Download Foreign Matched Adjustments",
                data=st.session_state.df_matched_adjustments_foreign.to_csv(index=False).encode('utf-8'),
                file_name="matched_adjustments_foreign.csv",
                mime="text/csv",
                key=f"download_matched_foreign_{uuid.uuid4()}"
            )
        else:
            st.info("No foreign matched adjustments.")

    with st.expander("Foreign FX Unmatched Adjustments"):
        if not st.session_state.df_unmatched_adjustments_foreign.empty:
            st.dataframe(st.session_state.df_unmatched_adjustments_foreign)
            st.download_button(
                label="Download Foreign Unmatched Adjustments",
                data=st.session_state.df_unmatched_adjustments_foreign.to_csv(index=False).encode('utf-8'),
                file_name="unmatched_adjustments_foreign.csv",
                mime="text/csv",
                key=f"download_unmatched_foreign_{uuid.uuid4()}"
            )
        else:
            st.info("No foreign unmatched adjustments.")

    with st.expander("Unmatched Bank Records (Global)"):
        if not st.session_state.df_unmatched_bank_records.empty:
            st.dataframe(st.session_state.df_unmatched_bank_records)
            st.download_button(
                label="Download Unmatched Bank Records (Global)",
                data=st.session_state.df_unmatched_bank_records.to_csv(index=False).encode('utf-8'),
                file_name="unmatched_bank_records_global.csv",
                mime="text/csv",
                key=f"download_unmatched_bank_global_{uuid.uuid4()}"
            )
        else:
            st.info("No unmatched bank records (global).")


def perform_data_analysis_and_visualizations():
    """Performs data analysis and generates visualizations based on reconciliation results."""
    st.subheader("Data Analysis and Visualizations")

    all_empty = (
        st.session_state.df_matched_adjustments_local.empty and
        st.session_state.df_unmatched_adjustments_local.empty and
        st.session_state.df_matched_adjustments_foreign.empty and
        st.session_state.df_unmatched_adjustments_foreign.empty and
        st.session_state.df_unmatched_bank_records.empty
    )
    if all_empty:
        st.warning("No data available for analysis. Please run reconciliation first.")
        return

    # Combine data for overall analysis where appropriate
    combined_unmatched_adjustments = pd.concat([
        st.session_state.df_unmatched_adjustments_local.assign(Mode='Local FX'),
        st.session_state.df_unmatched_adjustments_foreign.assign(Mode='Foreign FX')
    ], ignore_index=True)

    combined_matched_adjustments = pd.concat([
        st.session_state.df_matched_adjustments_local.assign(Mode='Local FX'),
        st.session_state.df_matched_adjustments_foreign.assign(Mode='Foreign FX')
    ], ignore_index=True)


    # 7.1 Reconciliation Overview
    st.markdown("### 7.1 Reconciliation Overview (Combined)")
    reconciliation_status = pd.DataFrame({
        'Category': [
            'Matched Local Adjustments', 'Unmatched Local Adjustments',
            'Matched Foreign Adjustments', 'Unmatched Foreign Adjustments',
            'Unmatched Bank Records'
        ],
        'Count': [
            len(st.session_state.df_matched_adjustments_local),
            len(st.session_state.df_unmatched_adjustments_local),
            len(st.session_state.df_matched_adjustments_foreign),
            len(st.session_state.df_unmatched_adjustments_foreign),
            len(st.session_state.df_unmatched_bank_records)
        ]
    })
    st.write("**Counts of Matched/Unmatched Records:**")
    st.dataframe(reconciliation_status)

    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Category', y='Count', data=reconciliation_status, palette='viridis', ax=ax1)
    ax1.set_title('Overview of Reconciliation Status (Combined FX)')
    ax1.set_ylabel('Number of Records')
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig1)

    # 7.2 Unmatched Adjustments Analysis (Combined)
    if not combined_unmatched_adjustments.empty:
        st.markdown("### 7.2 Unmatched Adjustments Analysis (Combined FX)")
        st.write("**Top Reasons for Unmatched Adjustments by Mode:**")
        reason_counts_by_mode = combined_unmatched_adjustments.groupby(['Mode', 'Reason']).size().reset_index(name='Count')
        st.dataframe(reason_counts_by_mode)

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Count', y='Reason', hue='Mode', data=reason_counts_by_mode, palette='magma', ax=ax2)
        ax2.set_title('Reasons for Unmatched Adjustments by FX Mode')
        ax2.set_xlabel('Number of Adjustments')
        ax2.set_ylabel('Reason')
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig2)

        st.write("**Distribution of Unmatched Adjustment Amounts (Combined):**")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(combined_unmatched_adjustments, x='Amount', hue='Mode', bins=20, kde=True, palette='coolwarm', ax=ax3)
        ax3.set_title('Distribution of Unmatched Adjustment Amounts (Combined FX)')
        ax3.set_xlabel('Amount')
        ax3.set_ylabel('Frequency')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig3)
    else:
        st.info("No unmatched adjustments to analyze for both local and foreign FX.")

    # 7.3 Unmatched Bank Records Analysis (Global)
    if not st.session_state.df_unmatched_bank_records.empty:
        st.markdown("### 7.3 Unmatched Bank Records Analysis (Global)")
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
        plt.tight_layout()
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
        df_unmatched_bank_temp = st.session_state.df_unmatched_bank_records.copy()
        df_unmatched_bank_temp['_ParsedDate'] = pd.to_datetime(df_unmatched_bank_temp['Date'])
        if not df_unmatched_bank_temp['_ParsedDate'].empty:
            st.write("**Unmatched Bank Records Over Time:**")
            daily_unmatched = df_unmatched_bank_temp.set_index('_ParsedDate').resample('D')['Amount'].count()
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
            st.info("No valid 'Date' dates found in unmatched bank records for daily trend visualization.")
    else:
        st.info("No unmatched bank records to analyze.")

    st.success("Data Analysis and Visualizations Complete!")

# --- Streamlit App Layout ---

def fx_reconciliation_app(bank_dfs: dict): # Added bank_dfs as an argument

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


    # Initialize session state variables for both local and foreign FX data
    if 'fx_trade_df_local' not in st.session_state:
        st.session_state.fx_trade_df_local = pd.DataFrame()
    if 'fx_trade_df_foreign' not in st.session_state:
        st.session_state.fx_trade_df_foreign = pd.DataFrame()
    
    # bank_dfs is now passed as an argument, no need to initialize here
    # if 'bank_dfs' not in st.session_state:
    #     st.session_state.bank_dfs = {}
    
    # Separate states for reconciliation results for local and foreign
    if 'matched_adjustments_list_local' not in st.session_state:
        st.session_state.matched_adjustments_list_local = []
    if 'unmatched_adjustments_list_local' not in st.session_state:
        st.session_state.unmatched_adjustments_list_local = []
    if 'matched_bank_keys_local' not in st.session_state: # Bank keys matched by local
        st.session_state.matched_bank_keys_local = set()

    if 'matched_adjustments_list_foreign' not in st.session_state:
        st.session_state.matched_adjustments_list_foreign = []
    if 'unmatched_adjustments_list_foreign' not in st.session_state:
        st.session_state.unmatched_adjustments_list_foreign = []
    if 'matched_bank_keys_foreign' not in st.session_state: # Bank keys matched by foreign
        st.session_state.matched_bank_keys_foreign = set()

    # Global lists for overall unmatched bank records and combined matched bank keys
    if 'unmatched_bank_records_list_global' not in st.session_state:
        st.session_state.unmatched_bank_records_list_global = []
    if 'matched_bank_keys_global' not in st.session_state:
        st.session_state.matched_bank_keys_global = set()

    if 'df_matched_adjustments_local' not in st.session_state:
        st.session_state.df_matched_adjustments_local = pd.DataFrame()
    if 'df_unmatched_adjustments_local' not in st.session_state:
        st.session_state.df_unmatched_adjustments_local = pd.DataFrame()
    if 'df_matched_adjustments_foreign' not in st.session_state:
        st.session_state.df_matched_adjustments_foreign = pd.DataFrame()
    if 'df_unmatched_adjustments_foreign' not in st.session_state:
        st.session_state.df_unmatched_adjustments_foreign = pd.DataFrame()
    if 'df_unmatched_bank_records' not in st.session_state: # This is the global one
        st.session_state.df_unmatched_bank_records = pd.DataFrame()
    
    if 'fx_uploaded_file_obj_local' not in st.session_state:
        st.session_state.fx_uploaded_file_obj_local = None
    if 'fx_uploaded_file_obj_foreign' not in st.session_state:
        st.session_state.fx_uploaded_file_obj_foreign = None

    # Removed bank_uploaded_file_objs and raw_bank_data_previews as they are handled centrally
    # if 'bank_uploaded_file_objs' not in st.session_state:
    #     st.session_state.bank_uploaded_file_objs = []
    # if 'raw_bank_data_previews' not in st.session_state:
    #     st.session_state.raw_bank_data_previews = {}
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False


    # Sidebar for controls and uploads
    with st.sidebar:
        st.header("Upload Data")

        # --- Local FX Tracker Upload ---
        st.markdown("### 📥 Local FX Tracker Upload")
        fx_uploaded_file_local = st.file_uploader("Upload Local FX Tracker (CSV/Excel)", type=["csv", "xlsx"], key="fx_uploader_local")

        if fx_uploaded_file_local:
            if st.session_state.fx_uploaded_file_obj_local != fx_uploaded_file_local:
                st.session_state.fx_uploaded_file_obj_local = fx_uploaded_file_local
                st.session_state.fx_trade_df_local = pd.DataFrame()
                st.session_state.fx_sheet_names_local = []
                st.session_state.fx_selected_sheet_local = None
                if fx_uploaded_file_local.name.endswith('.xlsx'):
                    st.session_state.fx_sheet_names_local = get_excel_sheet_names(fx_uploaded_file_local)
                    if st.session_state.fx_sheet_names_local:
                        st.session_state.fx_selected_sheet_local = st.session_state.fx_sheet_names_local[0]
                else:
                    df_fx_raw_temp = process_uploaded_file(fx_uploaded_file_local)
                    if not df_fx_raw_temp.empty:
                        st.session_state.fx_raw_df_local = df_fx_raw_temp
                    else:
                        st.session_state.fx_raw_df_local = pd.DataFrame()

            file_details_fx_local = {"FileName": fx_uploaded_file_local.name, "FileType": fx_uploaded_file_local.type, "FileSize": fx_uploaded_file_local.size}
            st.write(file_details_fx_local)

            df_fx_raw_local = pd.DataFrame()
            if fx_uploaded_file_local.name.endswith('.xlsx'):
                selected_sheet_fx_local = st.selectbox("Select Local FX Sheet:", st.session_state.fx_sheet_names_local, key="fx_sheet_selector_local",
                                                        index=st.session_state.fx_sheet_names_local.index(st.session_state.fx_selected_sheet_local) if st.session_state.fx_selected_sheet_local in st.session_state.fx_sheet_names_local else 0)
                if selected_sheet_fx_local != st.session_state.fx_selected_sheet_local:
                    st.session_state.fx_selected_sheet_local = selected_sheet_fx_local
                if selected_sheet_fx_local:
                    df_fx_raw_local = process_uploaded_file(fx_uploaded_file_local, sheet_name=selected_sheet_fx_local)
                    st.session_state.fx_raw_df_local = df_fx_raw_local
            else:
                df_fx_raw_local = st.session_state.fx_raw_df_local

            if not df_fx_raw_local.empty:
                st.write("Local FX Data Preview:")
                st.dataframe(df_fx_raw_local.head())

                st.markdown("#### Map Local FX Columns")
                fx_column_mappings_local = {}
                available_columns_local = df_fx_raw_local.columns.tolist()
                available_columns_local.insert(0, "") 

                for expected_col, default_val in FX_EXPECTED_COLUMNS.items():
                    initial_selection = default_val if default_val.strip() in [col.strip() for col in df_fx_raw_local.columns] else ""
                    mapped_col = st.selectbox(
                        f"Map '{expected_col}' to:",
                        options=available_columns_local,
                        index = [col.strip() for col in available_columns_local].index(initial_selection.strip()) if initial_selection else 0,
                        key=f"fx_map_local_{expected_col}"
                    )
                    fx_column_mappings_local[expected_col] = mapped_col if mapped_col else None

                if st.button("Process Local FX Data", key="process_fx_local_btn"):
                    temp_df_fx = df_fx_raw_local.copy()
                    renamed_cols_dict = {}
                    for expected_col, mapped_col in fx_column_mappings_local.items():
                        if mapped_col and mapped_col in temp_df_fx.columns:
                            renamed_cols_dict[mapped_col] = expected_col
                    
                    temp_df_fx.rename(columns=renamed_cols_dict, inplace=True)
                    temp_df_fx.columns = temp_df_fx.columns.str.strip()
                    st.session_state.fx_trade_df_local = temp_df_fx
                    st.success("Local FX Data Processed!")
                    st.dataframe(st.session_state.fx_trade_df_local.head())
            else:
                st.error("Could not load Local FX data.")
        else:
            st.session_state.fx_trade_df_local = pd.DataFrame()
            st.session_state.fx_uploaded_file_obj_local = None
            st.session_state.fx_raw_df_local = pd.DataFrame()

        # --- Foreign FX Tracker Upload ---
        st.markdown("### 📥 Foreign FX Tracker Upload")
        fx_uploaded_file_foreign = st.file_uploader("Upload Foreign FX Tracker (CSV/Excel)", type=["csv", "xlsx"], key="fx_uploader_foreign")

        if fx_uploaded_file_foreign:
            if st.session_state.fx_uploaded_file_obj_foreign != fx_uploaded_file_foreign:
                st.session_state.fx_uploaded_file_obj_foreign = fx_uploaded_file_foreign
                st.session_state.fx_trade_df_foreign = pd.DataFrame()
                st.session_state.fx_sheet_names_foreign = []
                st.session_state.fx_selected_sheet_foreign = None
                if fx_uploaded_file_foreign.name.endswith('.xlsx'):
                    st.session_state.fx_sheet_names_foreign = get_excel_sheet_names(fx_uploaded_file_foreign)
                    if st.session_state.fx_sheet_names_foreign:
                        st.session_state.fx_selected_sheet_foreign = st.session_state.fx_sheet_names_foreign[0]
                else:
                    df_fx_raw_temp = process_uploaded_file(fx_uploaded_file_foreign)
                    if not df_fx_raw_temp.empty:
                        st.session_state.fx_raw_df_foreign = df_fx_raw_temp
                    else:
                        st.session_state.fx_raw_df_foreign = pd.DataFrame()

            file_details_fx_foreign = {"FileName": fx_uploaded_file_foreign.name, "FileType": fx_uploaded_file_foreign.type, "FileSize": fx_uploaded_file_foreign.size}
            st.write(file_details_fx_foreign)

            df_fx_raw_foreign = pd.DataFrame()
            if fx_uploaded_file_foreign.name.endswith('.xlsx'):
                selected_sheet_fx_foreign = st.selectbox("Select Foreign FX Sheet:", st.session_state.fx_sheet_names_foreign, key="fx_sheet_selector_foreign",
                                                        index=st.session_state.fx_sheet_names_foreign.index(st.session_state.fx_selected_sheet_foreign) if st.session_state.fx_selected_sheet_foreign in st.session_state.fx_sheet_names_foreign else 0)
                if selected_sheet_fx_foreign != st.session_state.fx_selected_sheet_foreign:
                    st.session_state.fx_selected_sheet_foreign = selected_sheet_fx_foreign
                if selected_sheet_fx_foreign:
                    df_fx_raw_foreign = process_uploaded_file(fx_uploaded_file_foreign, sheet_name=selected_sheet_fx_foreign)
                    st.session_state.fx_raw_df_foreign = df_fx_raw_foreign
            else:
                df_fx_raw_foreign = st.session_state.fx_raw_df_foreign

            if not df_fx_raw_foreign.empty:
                st.write("Foreign FX Data Preview:")
                st.dataframe(df_fx_raw_foreign.head())

                st.markdown("#### Map Foreign FX Columns")
                fx_column_mappings_foreign = {}
                available_columns_foreign = df_fx_raw_foreign.columns.tolist()
                available_columns_foreign.insert(0, "") 

                for expected_col, default_val in FX_EXPECTED_COLUMNS.items():
                    initial_selection = default_val if default_val.strip() in [col.strip() for col in df_fx_raw_foreign.columns] else ""
                    mapped_col = st.selectbox(
                        f"Map '{expected_col}' to:",
                        options=available_columns_foreign,
                        index = [col.strip() for col in available_columns_foreign].index(initial_selection.strip()) if initial_selection else 0,
                        key=f"fx_map_foreign_{expected_col}"
                    )
                    fx_column_mappings_foreign[expected_col] = mapped_col if mapped_col else None

                if st.button("Process Foreign FX Data", key="process_fx_foreign_btn"):
                    temp_df_fx = df_fx_raw_foreign.copy()
                    renamed_cols_dict = {}
                    for expected_col, mapped_col in fx_column_mappings_foreign.items():
                        if mapped_col and mapped_col in temp_df_fx.columns:
                            renamed_cols_dict[mapped_col] = expected_col
                    
                    temp_df_fx.rename(columns=renamed_cols_dict, inplace=True)
                    temp_df_fx.columns = temp_df_fx.columns.str.strip()
                    st.session_state.fx_trade_df_foreign = temp_df_fx
                    st.success("Foreign FX Data Processed!")
                    st.dataframe(st.session_state.fx_trade_df_foreign.head())
            else:
                st.error("Could not load Foreign FX data.")
        else:
            st.session_state.fx_trade_df_foreign = pd.DataFrame()
            st.session_state.fx_uploaded_file_obj_foreign = None
            st.session_state.fx_raw_df_foreign = pd.DataFrame()

    # Removed the bank statement upload section from here as it's now centralized in main_dashboard.py
    # st.markdown("### 🏦 Bank Statements Upload")
    # bank_uploaded_files = st.file_uploader("Upload Bank Statement(s) (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True, key="bank_uploader")

    # bank_dfs = {} # This will now be passed as an argument
    # ... (rest of the bank upload/processing logic removed)

    # Main content area
    st.header("Reconciliation")

    # Debug mode checkbox
    st.session_state.debug_mode = st.checkbox("Enable Reconciliation Debug Logging", value=st.session_state.debug_mode)

    if st.button("Perform Full Reconciliation (Local & Foreign FX)", key="reconcile_btn"):
        # Pass the pre-processed bank_dfs from session state
        perform_full_reconciliation(bank_dfs)

    st.header("Analysis and Visualizations")
    if st.button("Generate Analysis and Visualizations", key="analyze_btn"):
        perform_data_analysis_and_visualizations()
    return (
        st.session_state.df_matched_adjustments_local,
        st.session_state.df_matched_adjustments_foreign,
        st.session_state.df_unmatched_adjustments_local,
        st.session_state.df_unmatched_adjustments_foreign,
        st.session_state.df_unmatched_bank_records
    )
