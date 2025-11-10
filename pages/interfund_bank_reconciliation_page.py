# pages/interfund_bank_reconciliation_page.py
import streamlit as st
import pandas as pd
from datetime import datetime
import io
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Helper Functions ---
def save_uploaded_file(file, filename):
    print("saving interfund bank uploaded data : ", filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def save_dataframe(df, filename):
    df.to_pickle(os.path.join(CACHE_DIR, filename))

def load_dataframe(filename):
    path = os.path.join(CACHE_DIR, filename)
    return pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame()

def save_object(obj, filename):
    with open(os.path.join(CACHE_DIR, filename), 'wb') as f:
        pickle.dump(obj, f)

def load_object(filename, default=None):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return default

# --- Restore session state for Interfund Bank Records ---
st.session_state.interfund_bank_df = load_dataframe("interfund_bank_df.pkl")
st.session_state.interfund_bank_sheet = load_object("interfund_bank_sheet.pkl")
st.session_state.interfund_bank_col_mapping = load_object("interfund_bank_col_mapping.pkl", {
    'Application ID': 'Application ID',
    'Amount (In Debit Account Currency)': 'Amount (In Debit Account Currency)',
    'Credit Account Currency': 'Credit Account Currency',
    'Intermediary Bank Account': 'Intermediary Bank Account',
    'Status': 'Status'
})

# Set Seaborn style for beautiful plots
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

# --- Configuration ---
# Output paths
out_csv_path_unmatched = 'UnmatchedInterfund.csv'
out_csv_path_bank_unmatched = 'UnmatchedBankRecords_Interfund.csv'
out_csv_path_matched = 'MatchedInterfund.csv'

# Various Date Formats
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
    '%d/%m/%Y', '%-d/%-m/%Y', '%-d.%-m/%-Y',
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
    '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S', '%-d/%-m/%Y %H:%M:%S',
    '%-d.%-m.%Y %H:%M:%S', "%d.%m.%Y"
]

# PREDEFINED LIST OF BANK NAME - CURRENCY COMBINATIONS
PREDEFINED_BANK_CURRENCY_COMBOS = sorted([
    "Absa KES", "Absa USD", "Absa EUR", "Absa GBP",
    "CBK KES", "CBK USD", "CBK EUR", "CBK GBP",
    "Equity KES", "Equity USD", "Equity EUR", "Equity GBP",
    "I&M KES", "I&M USD", "I&M EUR", "I&M GBP",
    "KCB KES", "KCB USD", "KCB EUR", "KCB GBP",
    "Kingdom KES", "Kingdom USD",
    "NCBA KES", "NCBA USD", "NCBA EUR", "NCBA GBP",
    "SBM KES", "SBM USD", "SBM EUR",
    "BAAS Temporary KES", "BAAS Temporary USD",
    "FX Temporary KES", "FX Temporary USD",
    "Other Temporary KES", "Other Temporary USD",
    "Unclaimed Funds KES", "Unclaimed Funds USD",
    "Yeepay KES", "Yeepay USD",
    "UBA KES", "UBA USD", "UBA"
])

def safe_float(x):
    """Safely converts a value to a float, handling commas, non-numeric inputs, and ensuring consistency."""
    if pd.isna(x) or x is None:
        return None
    try:
        cleaned_x = str(x).replace(',', '').strip()
        return abs(float(cleaned_x))
    except (ValueError, TypeError):
        return None

def normalize_bank_key(raw_key, debug_mode=False):
    """
    Normalizes bank names to a consistent short code, using fuzzy matching.
    """
    raw_key_lower = str(raw_key).lower().strip()
    replacements = {
        'ncba bank kenya plc': 'NCBA',
        'ncba bank': 'NCBA',
        'equity bank': 'Equity',
        'i&m bank': 'I&M',
        'central bank of kenya': 'CBK',
        'kenya commercial bank': 'KCB',
        'kcb bank': 'KCB',
        'sbm bank (kenya) limited': 'SBM',
        'sbm bank': 'SBM',
        'absa bank': 'Absa',
        'kingdom bank': 'Kingdom',
        'uba': 'UBA',
        'yeepay': 'Yeepay',
    }

    # First, try direct replacement
    for long, short in replacements.items():
        if raw_key_lower == long.lower():
            if debug_mode:
                st.info(f"DEBUG: normalize_bank_key - Direct match found: '{raw_key_lower}' -> '{short}'")
            return short
        if raw_key_lower.startswith(long.lower()):
            if debug_mode:
                st.info(f"DEBUG: normalize_bank_key - Starts with match found: '{raw_key_lower}' starts with '{long.lower()}' -> '{short}'")
            return short

    # If no direct match, try fuzzy matching
    all_target_bank_names = list(replacements.values()) + [k.capitalize() for k in replacements.keys()]
    all_target_bank_names = list(set(all_target_bank_names))

    if debug_mode:
        st.info(f"DEBUG: normalize_bank_key - Fuzzy matching '{raw_key_lower}' against set: {all_target_bank_names}")

    match = process.extractOne(raw_key_lower, all_target_bank_names, scorer=fuzz.ratio)
    if match:
        if debug_mode:
            st.info(f"DEBUG: normalize_bank_key - Fuzzy match result: '{match[0]}' with relevance value {match[1]}")
        if match[1] >= 70:  # FUZZY_MATCH_THRESHOLD
            for long, short in replacements.items():
                if match[0].lower() == long.lower():
                    return short
                if match[0].lower().startswith(long.lower()):
                    return short
            return match[0].title() if match[0].islower() else match[0]
    
    if debug_mode:
        st.info(f"DEBUG: normalize_bank_key - No good fuzzy match found for '{raw_key_lower}'. Returning original.")
    
    return str(raw_key).strip().title()

def extract_bank_info_from_intermediary_column(column_value):
    """
    Extract bank name from intermediary bank account column.
    Example: "SBM Bank - KES - 0322284570009" -> "SBM"
    """
    if pd.isna(column_value) or not column_value:
        return None
    
    parts = str(column_value).split('-')
    if len(parts) >= 1:
        bank_name = parts[0].strip()
        return normalize_bank_key(bank_name)
    return None

def parse_date(date_str_raw):
    """Parses a date string into a datetime object using predefined formats."""
    if pd.isna(date_str_raw):
        return None
    
    try:
        return pd.to_datetime(date_str_raw, infer_datetime_format=True, errors='coerce')
    except Exception:
        pass

    if not isinstance(date_str_raw, str):
        return None
        
    date_str = str(date_str_raw).strip()

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

# --- Core Matching Logic for Interfund Bank Reconciliation ---
def process_interfund_match(
    interfund_row: pd.Series,
    all_bank_dfs: dict,
    unmatched_list: list,
    matched_list: list,
    date_tolerance_days: int = 3,
    debug_mode: bool = False,
    already_matched_interfund_records: set = None,
    skipped_bank_records: dict = None,
    matched_bank_keys: set = None
) -> list or None:
    """Matches one interfund bank record against bank statement records."""

    # Initialize tracking sets if not provided
    if already_matched_interfund_records is None:
        already_matched_interfund_records = set()
    if skipped_bank_records is None:
        skipped_bank_records = {}
    if matched_bank_keys is None:
        matched_bank_keys = set()

    # Extract unique identifier for this interfund record
    application_id = interfund_row.get('Application ID', '')
    if not application_id:
        application_id = f"{interfund_row.get('Amount (In Debit Account Currency)', '')}_{interfund_row.get('Intermediary Bank Account', '')}"

    record_id = f"{application_id}_INTERFUND"
    
    # Check if this interfund record has already been matched
    if record_id in already_matched_interfund_records:
        if debug_mode:
            st.info(f"⏭️  Skipping already matched interfund record: {record_id}")
        return None

    amount = safe_float(interfund_row.get('Amount (In Debit Account Currency)'))
    if amount is None:
        if debug_mode:
            st.error(f"DEBUG: Skipping interfund row due to invalid amount ({amount}).")
        return None

    status = str(interfund_row.get('Status', '')).strip().lower()
    if status in ['declined', 'rejected', 'pending', 'not completed']:
        if debug_mode:
            st.info(f"DEBUG: Skipping interfund row due to status: {status}.")
        return None

    currency = str(interfund_row.get('Credit Account Currency', '')).strip().upper()
    if not currency:
        if debug_mode:
            st.error(f"DEBUG: Skipping interfund row due to missing currency.")
        return None

    # Extract bank info from intermediary column
    bank_info_raw = interfund_row.get('Intermediary Bank Account', '')
    normalized_bank_name = extract_bank_info_from_intermediary_column(bank_info_raw)
    
    if not normalized_bank_name:
        unmatched_record = {
            'Bank Table (Expected)': f"N/A ({bank_info_raw})",
            'Amount': amount,
            'Currency': currency,
            'Status': 'Invalid Bank Info in Interfund Record',
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        }
        unmatched_list.append(unmatched_record)
        return None

    expected_bank_key = f"{normalized_bank_name} {currency}"

    if expected_bank_key not in all_bank_dfs:
        unmatched_record = {
            'Bank Table (Expected)': expected_bank_key,
            'Amount': amount,
            'Currency': currency,
            'Status': 'No Matching Bank Statement File Found',
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        }
        unmatched_list.append(unmatched_record)
        return None

    bank_df = all_bank_dfs[expected_bank_key]
    bank_df_columns = bank_df.columns.tolist()

    # Initialize Skipped column if not exists
    if 'Skipped_By_Interfund' not in bank_df.columns:
        bank_df['Skipped_By_Interfund'] = ""

    # According to matching rules: Always match with Debit column in bank statement
    bank_amount_column = 'Debit'

    if bank_amount_column not in bank_df.columns:
        unmatched_record = {
            'Bank Table (Expected)': expected_bank_key,
            'Amount': amount,
            'Currency': currency,
            'Status': 'Missing Debit Column in Bank Statement',
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        }
        unmatched_list.append(unmatched_record)
        return None

    # Filter bank rows for matching (no date filtering for interfund)
    amount_matches = bank_df.copy()

    matched_records = []
    skipped_records = []

    for idx, bank_row in amount_matches.iterrows():
        bank_amt = safe_float(bank_row.get(bank_amount_column))
        if bank_amt is None:
            continue

        # Compare absolute values as per requirements
        amount_diff = abs(abs(bank_amt) - abs(amount))

        if amount_diff < 0.05:  # Small tolerance for floating point differences
            # Create bank record key for tracking
            bank_record_key_operation = 'debit'
            
            bank_record_key = (
                expected_bank_key,
                round(abs(bank_amt), 2),
                bank_record_key_operation
            )

            # Check if this bank record is already matched
            is_already_matched = bank_record_key in matched_bank_keys

            if is_already_matched:
                # Mark as skipped
                if debug_mode:
                    st.warning(f"⚠️ Bank record {bank_record_key} already matched, marking as skipped for interfund record {record_id}")
                
                # Mark this bank record as skipped by this interfund record
                current_skipped = bank_df.loc[idx, "Skipped_By_Interfund"]
                skipped_list = []
                if current_skipped and current_skipped != "":
                    try:
                        skipped_list = json.loads(current_skipped)
                    except:
                        skipped_list = []
                
                # Add interfund record to skipped list
                skipped_info = {
                    'interfund_id': record_id,
                    'interfund_amount': amount,
                    'interfund_currency': currency,
                    'skipped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'match_details': {
                        'amount_difference': amount_diff,
                        'bank_amount': bank_amt,
                        'amount_column': bank_amount_column
                    }
                }
                skipped_list.append(skipped_info)
                bank_df.loc[idx, "Skipped_By_Interfund"] = json.dumps(skipped_list)
                
                # Track in skipped_bank_records
                if record_id not in skipped_bank_records:
                    skipped_bank_records[record_id] = []
                skipped_records.append({
                    'bank_key': bank_record_key,
                    'bank_table': expected_bank_key,
                    'bank_amount': bank_amt,
                    'bank_row_index': idx,
                    'match_details': {
                        'amount_difference': amount_diff,
                        'bank_amount': bank_amt,
                        'amount_column': bank_amount_column
                    }
                })
                
                continue  # Skip to next potential match

            # If we get here, this is a valid unmatched bank record - proceed with matching
            matched_records.append({
                'Bank Index': idx,
                'Description': str(bank_row.get('Description', '')).strip(),
                'Debit': safe_float(bank_row.get('Debit')),
                'Credit': safe_float(bank_row.get('Credit')),
                'Matched Column': bank_amount_column,
                'Bank Amount': bank_amt,
                'Bank Record Key': bank_record_key,
                'Amount Difference': amount_diff
            })

            # Mark bank record as matched
            bank_df.at[idx, "Matched"] = True
            matched_bank_keys.add(bank_record_key)

            if debug_mode:
                st.info(f"✅ Sub-Match Found: Bank[{idx}] {bank_amt:.2f} {currency} "
                        f"≈ Interfund {amount:.2f} {currency}")

    if matched_records:
        # Convert complex objects to JSON strings for PyArrow compatibility
        all_matched_records_json = json.dumps(matched_records) if matched_records else ""
        skipped_records_json = json.dumps(skipped_records) if skipped_records else ""

        # Create base matched record
        matched_record = {
            'Bank Table': expected_bank_key,
            'Interfund Amount': amount,
            'Currency': currency,
            'Total Bank Matches': len(matched_records),
            'Skipped Bank Records': len(skipped_records),

            # Flattened first match (for CSV friendliness)
            'Matched Bank Record Index': matched_records[0]['Bank Index'],
            'Matched Bank Description': matched_records[0]['Description'],
            'Matched Bank Debit': matched_records[0]['Debit'],
            'Matched Bank Credit': matched_records[0]['Credit'],

            # JSON strings for complex objects
            'All Matched Bank Records': all_matched_records_json,
            'Skipped Bank Records Info': skipped_records_json,
            
            # Interfund record details
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw,
            'Status': status
        }

        matched_list.append(matched_record)

        # MARK THIS INTERFUND RECORD AS MATCHED
        already_matched_interfund_records.add(record_id)

        if debug_mode:
            st.success(f"✅ Interfund {amount:.2f} {currency} matched {len(matched_records)} bank entries in '{expected_bank_key}' (skipped: {len(skipped_records)}).")

        return [(expected_bank_key, m['Bank Index']) for m in matched_records]

    # If none matched but there were skipped records
    if skipped_records:
        skipped_records_json = json.dumps(skipped_records) if skipped_records else ""
        
        unmatched_record = {
            'Bank Table (Expected)': expected_bank_key,
            'Amount': amount,
            'Currency': currency,
            'Status': f'Potential matches found but already taken by other records (skipped: {len(skipped_records)})',
            'Skipped Bank Records': skipped_records_json,
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        }
        unmatched_list.append(unmatched_record)

        if debug_mode:
            st.warning(f"⚠️ Interfund {amount:.2f} {currency} had {len(skipped_records)} potential matches but all were already taken in {expected_bank_key}.")
        return None

    # If none matched and no skipped records
    unmatched_record = {
        'Bank Table (Expected)': expected_bank_key,
        'Amount': amount,
        'Currency': currency,
        'Status': 'No Bank Statement Match (Amount)',
        'Application ID': application_id,
        'Intermediary Bank Account': bank_info_raw
    }
    unmatched_list.append(unmatched_record)

    if debug_mode:
        st.warning(f"⚠️ No matches found for Interfund {amount:.2f} {currency} in {expected_bank_key}.")
    return None

def interfund_bank_reconciliation_app(all_bank_dfs: dict):
    st.title("🏦 Interfund Bank Reconciliation")
    st.markdown("""
    This dashboard helps verify interfund bank records against bank statements, identifying matched and unmatched transactions.
    Upload your Interfund Bank Records file below.
    """)

    # --- Data Loading Section ---
    st.header("1. Data Loading")

    # Interfund Bank Records Upload
    st.subheader("Upload Interfund Bank Records")
    uploaded_interfund_file = st.file_uploader("Choose Interfund Bank Records (CSV or XLSX)", type=["csv", "xlsx"], key="interfund_uploader")

    # Initialize or load interfund bank data
    interfund_df = pd.DataFrame()
    if 'interfund_bank_df' in st.session_state:
        interfund_df = st.session_state.interfund_bank_df
    else:
        try:
            loaded_df = load_dataframe('interfund_bank_df.pkl')
            if not loaded_df.empty:
                interfund_df = loaded_df
                st.session_state.interfund_bank_df = loaded_df
        except Exception as e:
            st.warning(f"Could not load cached interfund bank data: {e}")

    if uploaded_interfund_file:
        try:
            save_uploaded_file(uploaded_interfund_file, "interfund_bank_uploaded." + uploaded_interfund_file.name.split('.')[-1])
            if uploaded_interfund_file.name.endswith('.xlsx'):
                xls = pd.ExcelFile(uploaded_interfund_file)
                sheet_names = xls.sheet_names
                
                # Initialize session state if it doesn't exist
                if 'interfund_bank_sheet' not in st.session_state:
                    st.session_state.interfund_bank_sheet = sheet_names[0]
                
                selected_sheet = st.selectbox(
                    "Select sheet for Interfund Bank Records", 
                    sheet_names, 
                    key="interfund_sheet_selector",
                    index=sheet_names.index(st.session_state.interfund_bank_sheet) 
                    if st.session_state.interfund_bank_sheet in sheet_names 
                    else 0
                )
                
                # Update session state if selection changed
                if selected_sheet != st.session_state.interfund_bank_sheet:
                    st.session_state.interfund_bank_sheet = selected_sheet
                    save_object(selected_sheet, "interfund_bank_sheet.pkl")
                    
                interfund_df = pd.read_excel(uploaded_interfund_file, sheet_name=selected_sheet)
            else:
                interfund_df = pd.read_csv(uploaded_interfund_file)

            interfund_df.columns = interfund_df.columns.str.strip()
            st.success("Interfund Bank Records loaded successfully!")
            st.dataframe(interfund_df.head())

            # Initialize column mapping if it doesn't exist
            if 'interfund_bank_col_mapping' not in st.session_state:
                st.session_state.interfund_bank_col_mapping = {
                    'Application ID': 'Application ID',
                    'Amount (In Debit Account Currency)': 'Amount (In Debit Account Currency)',
                    'Credit Account Currency': 'Credit Account Currency',
                    'Intermediary Bank Account': 'Intermediary Bank Account',
                    'Status': 'Status'
                }

            # Column mapping for Interfund Bank Records
            st.subheader("Interfund Bank Records Column Mapping")
            interfund_col_options = ['-- Select Column --'] + interfund_df.columns.tolist()
            col_mapping = {}

            # Define the required columns and their default/suggested mappings
            required_cols = {
                'Application ID': 'Application ID',
                'Amount (In Debit Account Currency)': 'Amount (In Debit Account Currency)',
                'Credit Account Currency': 'Credit Account Currency',
                'Intermediary Bank Account': 'Intermediary Bank Account',
                'Status': 'Status'
            }

            for display_name, suggested_col in required_cols.items():
                initial_selection = (
                    st.session_state.interfund_bank_col_mapping.get(display_name)
                    if display_name in st.session_state.interfund_bank_col_mapping
                    else suggested_col if suggested_col in interfund_col_options
                    else '-- Select Column --'
                )
                selected_col = st.selectbox(
                    f"Map '{display_name}' to:",
                    options=interfund_col_options,
                    index=interfund_col_options.index(initial_selection) if initial_selection in interfund_col_options else 0,
                    key=f"interfund_map_select_{display_name}"
                )
                col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None

            renamed_interfund_df = pd.DataFrame()
            mapped_columns_dict = {selected: original for original, selected in col_mapping.items() if selected and selected in interfund_df.columns}

            if mapped_columns_dict:
                cols_to_keep = list(mapped_columns_dict.keys())
                renamed_interfund_df = interfund_df[cols_to_keep].rename(columns=mapped_columns_dict)
                interfund_df = renamed_interfund_df
                st.success("Interfund Bank Records columns mapped successfully!")
                st.dataframe(interfund_df.head())
            else:
                st.warning("No Interfund Bank Records columns mapped. Proceeding with original column names.")

            st.session_state.interfund_bank_df = interfund_df
            save_dataframe(interfund_df, "interfund_bank_df.pkl")
            st.session_state.interfund_bank_col_mapping = col_mapping
            save_object(col_mapping, "interfund_bank_col_mapping.pkl")

        except Exception as e:
            st.error(f"Error loading Interfund Bank Records: {e}")

    if not all_bank_dfs:
        st.warning("No bank statements loaded. Please upload and process bank statements in 'Bank Statement Management' on the main dashboard.")
        return (pd.DataFrame(), pd.DataFrame())

    # --- Reconciliation Section ---
    st.header("2. Run Reconciliation")

    debug_mode = st.checkbox("Enable Debug Mode (show detailed logs)", value=False, key="interfund_debug_toggle")

    if st.button("Run Interfund Reconciliation"):
        if interfund_df.empty:
            st.warning("Please upload Interfund Bank Records to run reconciliation.")
        else:
            # Check if essential columns are available after mapping
            required_for_recon = ['Application ID', 'Amount (In Debit Account Currency)', 'Credit Account Currency', 'Intermediary Bank Account', 'Status']
            if not all(col in interfund_df.columns for col in required_for_recon):
                missing_cols = [col for col in required_for_recon if col not in interfund_df.columns]
                st.error(f"Missing essential Interfund Bank Records columns for reconciliation: {', '.join(missing_cols)}. Please map them correctly.")
                return (pd.DataFrame(), pd.DataFrame())

            with st.spinner("Reconciling interfund transactions... This may take a moment."):
                match_count = 0
                unmatched_list = []
                matched_list = []
                
                # Create a copy of all_bank_dfs for this reconciliation run
                current_run_bank_dfs = {key: df.copy() for key, df in all_bank_dfs.items()}

                for index, row in interfund_df.iterrows():
                    status = str(row.get('Status', '')).strip().lower()

                    if status in ['declined', 'rejected', 'pending', 'not completed']:
                        if debug_mode:
                            st.info(f"DEBUG: Skipping interfund row {index} due to status: {status}.")
                        continue

                    # Process Interfund Match
                    process_interfund_match(
                        row,
                        current_run_bank_dfs,
                        unmatched_list,
                        matched_list,
                        debug_mode=debug_mode
                    )

                # Collect unmatched bank records
                unmatched_bank_records = []
                for bank_key, bank_df in current_run_bank_dfs.items():
                    bank_df.columns = bank_df.columns.str.strip()
                    
                    description_col = 'Description'
                    credit_col = 'Credit'
                    debit_col = 'Debit'

                    if description_col not in bank_df.columns or \
                       (credit_col not in bank_df.columns and debit_col not in bank_df.columns):
                        st.warning(f"Skipping bank statement '{bank_key}': Missing required mapped columns.")
                        continue

                    # Filter for rows where 'Matched' is False
                    unmatched_bank_df_for_key = bank_df[bank_df["Matched"] == False].copy()

                    for idx, row in unmatched_bank_df_for_key.iterrows():
                        amount_found = None
                        transaction_type_col_name = "N/A"
                        
                        debit_amt = safe_float(row.get(debit_col))
                        if debit_amt is not None and abs(debit_amt) > 0.01:
                            amount_found = debit_amt
                            transaction_type_col_name = debit_col
                        
                        if amount_found is not None:
                            unmatched_bank_records.append({
                                'Bank Table': bank_key,
                                'Description': str(row.get(description_col, '')).strip(),
                                'Transaction Type (Column)': transaction_type_col_name,
                                'Amount': round(amount_found, 2)
                            })

                st.session_state['unmatched_interfund_df'] = pd.DataFrame(unmatched_list)
                st.session_state['matched_interfund_df'] = pd.DataFrame(matched_list)
                st.session_state['unmatched_bank_interfund'] = pd.DataFrame(unmatched_bank_records)
                st.session_state['interfund_df'] = interfund_df

                st.success("Interfund Reconciliation complete!")

    # --- Results and Analysis Section ---
    st.header("3. Reconciliation Results and Analysis")

    if 'unmatched_interfund_df' in st.session_state:
        unmatched_interfund_df = st.session_state['unmatched_interfund_df']
        matched_interfund_df = st.session_state['matched_interfund_df']
        unmatched_bank_interfund = st.session_state['unmatched_bank_interfund']
        interfund_df = st.session_state['interfund_df']

        st.subheader("Overall Summary")
        
        # Calculate totals only for approved trades
        active_interfund_records = interfund_df[interfund_df['Status'].str.lower() == 'approved'] if 'Status' in interfund_df.columns else interfund_df
        total_interfund_records = len(active_interfund_records)

        st.write(f"✅ **Interfund Matches:** {len(matched_interfund_df)}")
        st.write(f"❌ **Interfund Unmatched:** {len(unmatched_interfund_df)}")
        st.write(f"📤 **Bank-only unmatched entries:** {len(unmatched_bank_interfund)}")
        st.markdown("---")

        # --- Reconciliation Summary Statistics ---
        st.subheader("Reconciliation Summary Statistics")
        st.write(f"Total Interfund Records (Approved only): {total_interfund_records}")
        st.write(f"Total Interfund records processed: {len(matched_interfund_df) + len(unmatched_interfund_df)}")
        
        match_rate = (len(matched_interfund_df)/(len(matched_interfund_df) + len(unmatched_interfund_df))*100) if (len(matched_interfund_df) + len(unmatched_interfund_df)) > 0 else 0

        st.write(f"Interfund Matched: {len(matched_interfund_df)} ({match_rate:.2f}%)")
        st.write(f"Interfund Unmatched: {len(unmatched_interfund_df)} ({100 - match_rate:.2f}%)")

        # --- Visualization ---
        st.subheader("Reconciliation Visualization")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Interfund Pie Chart
        labels = ['Matched', 'Unmatched']
        sizes = [len(matched_interfund_df), len(unmatched_interfund_df)]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Interfund Reconciliation')

        st.pyplot(fig)

        # --- Detailed Results ---
        st.subheader("Detailed Results")

        # Matched Interfund Records
        if not matched_interfund_df.empty:
            st.write("✅ **Matched Interfund Records**")
            st.dataframe(matched_interfund_df)
            csv_matched = matched_interfund_df.to_csv(index=False)
            st.download_button(
                label="Download Matched Interfund Records as CSV",
                data=csv_matched,
                file_name=out_csv_path_matched,
                mime="text/csv",
            )
        else:
            st.info("No matched interfund records found.")

        # Unmatched Interfund Records
        if not unmatched_interfund_df.empty:
            st.write("❌ **Unmatched Interfund Records**")
            st.dataframe(unmatched_interfund_df)
            csv_unmatched = unmatched_interfund_df.to_csv(index=False)
            st.download_button(
                label="Download Unmatched Interfund Records as CSV",
                data=csv_unmatched,
                file_name=out_csv_path_unmatched,
                mime="text/csv",
            )
        else:
            st.info("No unmatched interfund records found.")

        # Unmatched Bank Records
        if not unmatched_bank_interfund.empty:
            st.write("📤 **Unmatched Bank Records**")
            st.dataframe(unmatched_bank_interfund)
            csv_bank_unmatched = unmatched_bank_interfund.to_csv(index=False)
            st.download_button(
                label="Download Unmatched Bank Records as CSV",
                data=csv_bank_unmatched,
                file_name=out_csv_path_bank_unmatched,
                mime="text/csv",
            )
        else:
            st.info("No unmatched bank records found.")

        return (matched_interfund_df, unmatched_interfund_df)

    else:
        st.info("Reconciliation results will appear here after running the process.")
        return (pd.DataFrame(), pd.DataFrame())