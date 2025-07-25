
# pages/fx_reconciliation_page.py
import streamlit as st
import pandas as pd
from datetime import datetime
import io
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for beautiful plots
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

# --- Configuration ---
# Output paths (these will be relative to where the notebook is run or absolute paths)
# In Streamlit, we'll offer direct downloads rather than saving to disk.
out_csv_path_buy_unmatched = 'UnmatchedCounterpartyPayment.csv'
out_csv_path_sell_unmatched = 'UnmatchedChoicePayment.csv'
out_csv_path_bank_unmatched = 'UnmatchedBankRecords.csv'
out_csv_path_buy_matched = 'MatchedCounterpartyPayment.csv'
out_csv_path_sell_matched = 'MatchedChoicePayment.csv'

# Various Date Formats to handle different date representations in CSVs
date_formats = [
    '%Y-%m-%d',
    '%Y/%m/%d',
    '%d.%m.%Y',
    '%Y.%m.%d',
    '%d/%m/%Y',
    '%-d/%-m/%Y',
    '%-d.%-m.%Y',
    '%Y-%m-%d %H:%M:%S', # For datetime strings
    '%Y/%m/%d %H:%M:%S',
    '%d.%m.%Y %H:%M:%S',
    '%Y.%m.%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%-d/%-m/%Y %H:%M:%S',
    '%-d.%-m.%Y %H:%M:%S'
]

# Fuzzy matching threshold for bank names (0-100)
FUZZY_MATCH_THRESHOLD = 70

# Hardcoded FX Rates (for demonstration purposes)
FX_RATES = {
    'USDKES': 145.0,
    'EURKES': 155.0,
    'GBPUSD': 1.25,
    'USDGBP': 0.8, # Inverse rate
    'EURUSD': 1.08,
    'USDEUR': 0.92, # Inverse rate
    'KESUSD': 1/145.0, # Added for completeness
    'KESEUR': 1/155.0, # Added for completeness
    'USDGBP': 1/1.25, # Added for completeness
    # Add more as needed
}

def get_fx_rate(from_currency, to_currency, date=None):
    """
    Retrieves the FX rate for conversion.
    In a real application, this would query a database or an external API.
    For this example, it uses the hardcoded FX_RATES.
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency == to_currency:
        return 1.0

    pair = f"{from_currency}{to_currency}"
    if pair in FX_RATES:
        return FX_RATES[pair]

    # Try inverse rate
    inverse_pair = f"{to_currency}{from_currency}"
    if inverse_pair in FX_RATES:
        return 1 / FX_RATES[inverse_pair]

    st.warning(f"Warning: FX rate not found for {from_currency} to {to_currency}. Assuming 1:1 for demonstration.")
    return 1.0 # Fallback

def convert_currency(amount, from_currency, to_currency, date=None):
    """Converts an amount from one currency to another using the FX_RATES."""
    rate = get_fx_rate(from_currency, to_currency, date)
    return amount * rate

# --- Helper Functions for Data Consistency and Processing ---
def safe_float(x):
    """Safely converts a value to a float, handling commas, non-numeric inputs, and ensuring consistency."""
    if pd.isna(x) or x is None:
        return None
    try:
        # Convert to string, remove commas, and strip whitespace
        cleaned_x = str(x).replace(',', '').strip()
        return float(cleaned_x)
    except (ValueError, TypeError):
        return None

def normalize_bank_key(raw_key):
    """Normalizes bank names to a consistent short code, using fuzzy matching."""
    raw_key_lower = str(raw_key).lower().strip()
    replacements = {
        'ncba bank kenya plc': 'ncba',
        'ncba bank': 'ncba',
        'equity bank': 'equity',
        'i&m bank': 'i&m',
        'central bank of kenya': 'cbk',
        'kenya commercial bank': 'kcb',
        'kcb bank': 'kcb',
        'sbm bank (kenya) limited': 'sbm',
        'sbm bank': 'sbm',
        'absa bank': 'absa',
        'kingdom bank': 'kingdom'
    }

    # First, try direct replacement
    for long, short in replacements.items():
        if raw_key_lower == long: # Exact match for full name
            return short
        if raw_key_lower.startswith(long): # If it starts with a long name, use short
            return short # Crucial change: just return the short form

    # If no direct match, try fuzzy matching against known short codes/replacements
    all_bank_names = list(replacements.values()) + list(replacements.keys())
    all_bank_names = list(set(all_bank_names)) # Ensure uniqueness

    match = process.extractOne(raw_key_lower, all_bank_names, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_MATCH_THRESHOLD:
        for long, short in replacements.items():
            if match[0] == long: # Exact match for fuzzy result
                return short
            if match[0].startswith(long): # Fuzzy result starts with long name
                return short
        return match[0] # Return the best fuzzy match if no specific short form found
    return raw_key_lower # Return original if no good fuzzy match

def resolve_amount_column(columns, action_type, is_sell_side=False):
    """
    Identifies the correct amount column (e.g., 'Credit Amount', 'Debit Amount')
    based on action type, prioritizing the mapped standard names.
    This function is now primarily used to check for the presence of the mapped
    standard columns and resolve which of 'Credit Amount' or 'Debit Amount' to use.
    """
    columns_lower = [col.lower() for col in columns]

    # Prioritize the standardized mapped names
    if not is_sell_side: # Refers to the local currency amount in FX trade (Counterparty Payment)
        if action_type == 'Bank Buy': # We pay local currency (debit)
            if 'Debit Amount' in columns: return 'Debit Amount'
            # Fallback if mapped name not present, though it should be after pre-processing
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['withdrawal', 'debit']), None)
        elif action_type == 'Bank Sell': # We receive local currency (credit)
            if 'Credit Amount' in columns: return 'Credit Amount'
            # Fallback if mapped name not present
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['deposit', 'credit']), None)
    else: # is_sell_side refers to the foreign currency amount in FX trade (Choice Payment)
        if action_type == 'Bank Buy': # We receive foreign currency (credit)
            if 'Credit Amount' in columns: return 'Credit Amount'
            # Fallback if mapped name not present
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['deposit', 'credit']), None)
        elif action_type == 'Bank Sell': # We pay foreign currency (debit)
            if 'Debit Amount' in columns: return 'Debit Amount'
            # Fallback if mapped name not present
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['withdrawal', 'debit']), None)
    return None

def resolve_date_column(columns):
    """Identifies the date column from a list of column names, prioritizing common formats."""
    for candidate in ['Value Date', 'Transaction Date', 'MyUnknownColumn', 'Transaction date', 'Date', 'Activity Date']:
        if candidate in columns:
            return candidate
    return None

def get_description_columns(columns):
    """Identifies the description column from a list of column names."""
    for desc in ['Transaction details','Transaction', 'Customer reference','Narration',
                 'Transaction Details', 'Detail',  'Transaction Remarks:',
                 'TransactionDetails', 'Description', 'Narrative', 'Remarks']:
        if desc in columns:
            return desc
    return None

def parse_date(date_str_raw):
    """Parses a date string into a datetime object using predefined formats."""
    if pd.isna(date_str_raw):
        return None
    
    # Try direct pandas to_datetime for robustness
    try:
        # Infer format first, much faster for standard formats
        return pd.to_datetime(date_str_raw, infer_datetime_format=True, errors='coerce')
    except Exception:
        pass # Fallback to manual formats if infer fails

    # Fallback to predefined formats if pandas infer_datetime_format fails
    if not isinstance(date_str_raw, str):
        return None
        
    date_str = str(date_str_raw).strip() # Ensure it's a string

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

# --- Core Matching Logic ---
def process_fx_match(
    fx_row: pd.Series,
    all_bank_dfs: dict,
    unmatched_list: list,
    matched_list: list,
    action_type: str,
    fx_amount_field: str,
    bank_currency_info_field: str,
    is_sell_side: bool,
    date_tolerance_days: int = 3,
    debug_mode: bool = False
) -> tuple or None: # Returns (bank_key, bank_row_idx) on match, else None
    amount = safe_float(fx_row.get(fx_amount_field))
    if amount is None or action_type not in ['Bank Buy', 'Bank Sell']:
        if debug_mode:
            st.error(f"DEBUG: Skipping FX row due to invalid amount ({amount}) or action type ({action_type}).")
        return None

    parsed_date = fx_row.get('Created At')
    if parsed_date:
        # Ensure parsed_date from FX trade df is a datetime object
        if not isinstance(parsed_date, datetime):
            parsed_date = parse_date(str(parsed_date)) # Use helper to parse if it's a string

    if not isinstance(parsed_date, datetime):
        if debug_mode:
            st.error(f"DEBUG: Skipping FX row due to unparseable 'Created At' date: {fx_row.get('Created At')}.")
        return None

    counterparty_raw = str(fx_row.get(bank_currency_info_field, '')).strip()
    parts = counterparty_raw.split('-')
    if len(parts) < 2:
        if debug_mode:
            st.error(f"DEBUG: Skipping FX row due to insufficient parts in '{bank_currency_info_field}': {counterparty_raw}. Expected 'BankName-Currency'.")
        unmatched_list.append({
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': f"N/A ({counterparty_raw})",
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'Invalid Bank/Currency Info in FX Trade',
            'Source Column': bank_currency_info_field
        })
        return None

    trade_bank_name = parts[0].strip()
    trade_currency = parts[1].strip().upper()

    normalized_trade_bank_key_prefix = normalize_bank_key(trade_bank_name)
    trade_bank_key_full = f"{normalized_trade_bank_key_prefix} {trade_currency}".lower()

    if debug_mode:
        st.info(f"DEBUG: Processing FX Trade - Date: {parsed_date.strftime('%Y-%m-%d')}, Type: {action_type}, Amount: {amount}, Currency: {trade_currency}, Expected Bank Key: {trade_bank_key_full}")
        st.info(f"DEBUG: Available Bank DFS keys: {list(all_bank_dfs.keys())}")


    found_match = False
    target_bank_df_key = None

    for bank_df_key_in_dict in all_bank_dfs.keys():
        # Check for exact match first
        if trade_bank_key_full == bank_df_key_in_dict:
            target_bank_df_key = bank_df_key_in_dict
            if debug_mode:
                st.success(f"DEBUG: Direct match found for bank key: {target_bank_df_key}")
            break
        # Then fuzzy match
        elif fuzz.ratio(trade_bank_key_full, bank_df_key_in_dict) >= FUZZY_MATCH_THRESHOLD:
            target_bank_df_key = bank_df_key_in_dict
            if debug_mode:
                st.success(f"DEBUG: Fuzzy match found for bank key: {target_bank_df_key} (Ratio: {fuzz.ratio(trade_bank_key_full, bank_df_key_in_dict)})")
            break

    if not target_bank_df_key:
        unmatched_list.append({
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': trade_bank_key_full,
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'No Matching Bank Statement File Found',
            'Source Column': bank_currency_info_field
        })
        if debug_mode:
            st.warning(f"DEBUG: No matching bank statement found for expected key: {trade_bank_key_full}")
        return None

    bank_df = all_bank_dfs[target_bank_df_key]
    bank_df_columns = bank_df.columns.tolist()

    # 'Date Column' is now expected to be the standardized name after pre-processing
    date_column = 'Date Column'
    amount_column = resolve_amount_column(bank_df_columns, action_type, is_sell_side)
    
    if date_column not in bank_df.columns:
        # This should ideally not happen if pre-processing is successful
        unmatched_list.append({
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': target_bank_df_key,
            'Action Type': action_type,
            'Amount': amount,
            'Status': f"Mapped Date Column '{date_column}' Missing in Bank Statement after pre-processing",
            'Source Column': bank_currency_info_field
        })
        if debug_mode:
            st.error(f"DEBUG: Mapped date column '{date_column}' not found in bank statement '{target_bank_df_key}' during matching.")
        return None

    if not amount_column or amount_column not in bank_df.columns:
        unmatched_list.append({
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': target_bank_df_key,
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'Missing or Unresolvable Amount Column in Bank Statement',
            'Source Column': bank_currency_info_field
        })
        if debug_mode:
            st.warning(f"DEBUG: Missing or unresolvable amount column ({amount_column}) in bank statement '{target_bank_df_key}'.")
        return None
    
    # Date in bank_df['Date Column'] is already parsed to datetime during pre-processing
    # Filter based on the 'Date Column' which is already a datetime type
    date_matches = bank_df[
        bank_df['Date Column'].dt.date.between(
            parsed_date.date() - pd.Timedelta(days=date_tolerance_days),
            parsed_date.date() + pd.Timedelta(days=date_tolerance_days)
        )
    ]

    if debug_mode:
        st.info(f"DEBUG: Found {len(date_matches)} potential date matches in '{target_bank_df_key}' within ±{date_tolerance_days} days of {parsed_date.strftime('%Y-%m-%d')}.")


    bank_statement_currency_parts = target_bank_df_key.split(' ')
    bank_statement_currency = bank_statement_currency_parts[1].upper() if len(bank_statement_currency_parts) > 1 else "UNKNOWN"

    for idx, bank_row in date_matches.iterrows():
        bank_amt_raw = bank_row.get(amount_column)
        bank_amt = safe_float(bank_amt_raw)

        if debug_mode:
            st.info(f"DEBUG: Checking bank record - Date: {bank_row.get(date_column).strftime('%Y-%m-%d') if bank_row.get(date_column) else 'N/A'}, Amount (raw): {bank_amt_raw}, Amount (parsed): {bank_amt}, Column: {amount_column}")

        if bank_amt is not None:
            converted_amount = convert_currency(amount, trade_currency, bank_statement_currency, parsed_date)
            if debug_mode:
                st.info(f"DEBUG: Converting trade amount {amount} {trade_currency} to {bank_statement_currency}. Converted: {converted_amount}")

            # Match within a small tolerance for floating point comparisons
            if converted_amount is not None and abs(converted_amount) > 0.01 and abs(bank_amt - converted_amount) < 0.05: # Adjusted tolerance
                matched_list.append({
                    'Date': parsed_date.strftime('%Y-%m-%d'),
                    'Bank Table': target_bank_df_key,
                    'Action Type': action_type,
                    'Trade Amount': amount,
                    'Trade Currency': trade_currency,
                    'Bank Statement Amount': bank_amt,
                    'Bank Statement Currency': bank_statement_currency,
                    'Converted Trade Amount': converted_amount,
                    'Matched In Column': amount_column,
                    'Date Column Used': date_column,
                    'Source Column': bank_currency_info_field
                })
                found_match = True
                if debug_mode:
                    st.success(f"DEBUG: MATCH FOUND! Trade Amount: {amount:.2f} {trade_currency}, Bank Amount: {bank_amt:.2f} {bank_statement_currency}, Converted: {converted_amount:.2f}")
                return (target_bank_df_key, idx) # Return unique identifier of matched bank record
        elif debug_mode:
            st.info(f"DEBUG: Bank amount is None or invalid for this row.")

    if not found_match:
        unmatched_list.append({
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': target_bank_df_key,
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'No Bank Statement Match (Amount or Date Tolerance)',
            'Source Column': bank_currency_info_field
        })
        if debug_mode:
            st.warning(f"DEBUG: No match found for FX trade after checking all potential bank records.")

    return None # No match found

def graphed_analysis_app():
    st.title("💰 FX Trade Verification and Reconciliation")
    st.markdown("""
    This dashboard helps verify FX trade records against bank statements, identifying matched and unmatched transactions.
    Upload your FX Trade Tracker and Bank Statement files below.
    """)

    # --- Data Loading Section ---
    st.header("1. Data Loading")

    # FX Trade Tracker Upload
    st.subheader("Upload FX Trade Tracker")
    uploaded_fx_file = st.file_uploader("Choose FX Trade Tracker (CSV or XLSX)", type=["csv", "xlsx"], key="fx_uploader")

    fx_trade_df = pd.DataFrame()
    if uploaded_fx_file:
        try:
            if uploaded_fx_file.name.endswith('.xlsx'):
                # For Excel, allow sheet selection
                xls = pd.ExcelFile(uploaded_fx_file)
                sheet_names = xls.sheet_names
                selected_sheet = st.selectbox("Select sheet for FX Tracker", sheet_names, key="fx_sheet_selector")
                fx_trade_df = pd.read_excel(uploaded_fx_file, sheet_name=selected_sheet)
            else:
                fx_trade_df = pd.read_csv(uploaded_fx_file)

            fx_trade_df.columns = fx_trade_df.columns.str.strip()
            st.success("FX Trade Tracker loaded successfully!")
            st.dataframe(fx_trade_df.head())

            # Column mapping for FX Trade Tracker
            st.subheader("FX Trade Tracker Column Mapping")
            fx_col_options = ['-- Select Column --'] + fx_trade_df.columns.tolist()
            col_mapping = {}
            
            # Define the required FX columns and their default/suggested mappings
            fx_required_cols = {
                'Action Type': 'Action Type',
                'Status': 'Status',
                'Created At': 'Created At',
                'Buy Currency Amount': 'Buy Currency Amount',
                'Buy Trade Info': 'Buy Trade Info',
                'Sell Currency Amount': 'Sell Currency Amount',
                'Sell Trade Info': 'Sell Trade Info'
            }

            for display_name, suggested_col in fx_required_cols.items():
                default_index = 0
                if suggested_col and suggested_col in fx_col_options:
                    default_index = fx_col_options.index(suggested_col)
                
                selected_col = st.selectbox(
                    f"Map '{display_name}' to:",
                    options=fx_col_options,
                    index=default_index,
                    key=f"fx_map_select_{display_name}"
                )
                col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None

            # Apply mapping
            renamed_fx_df = pd.DataFrame() 
            mapped_columns_dict = {}

            for original_name, selected_map in col_mapping.items():
                if selected_map: 
                    mapped_columns_dict[selected_map] = original_name

            if mapped_columns_dict:
                cols_to_keep = [col for col in mapped_columns_dict.keys() if col in fx_trade_df.columns]
                renamed_fx_df = fx_trade_df[cols_to_keep].rename(columns=mapped_columns_dict)
                fx_trade_df = renamed_fx_df 
                st.success("FX Trade Tracker columns mapped successfully!")
                st.dataframe(fx_trade_df.head())
            else:
                st.warning("No FX Trade Tracker columns mapped. Proceeding with original column names.")
                # fx_trade_df remains unchanged if no mapping selected
                # This could lead to errors if expected columns are missing later.
                # Consider adding a check before reconciliation.


        except Exception as e:
            st.error(f"Error loading FX Trade Tracker: {e}")

    # Bank Statements Upload and Pre-processing
    st.subheader("Upload Bank Statement(s)")
    uploaded_bank_files = st.file_uploader("Choose Bank Statement(s) (CSV or XLSX)", type=["csv", "xlsx"], accept_multiple_files=True, key="bank_uploader")

    bank_dfs = {}
    if uploaded_bank_files:
        for uploaded_file in uploaded_bank_files:
            try:
                file_name = uploaded_file.name
                df = pd.DataFrame()

                if file_name.endswith('.xlsx'):
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_names = xls.sheet_names
                    selected_sheet = st.selectbox(f"Select sheet for {file_name}", sheet_names, key=f"bank_sheet_selector_{file_name}")
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    df = pd.read_csv(uploaded_file)

                df.columns = df.columns.str.strip()

                base_name = file_name.lower().replace('.csv', '').replace('.xlsx', '')
                name_parts = base_name.split('_')

                bank_name_for_key = normalize_bank_key(name_parts[0].strip())
                currency_for_key = "UNKNOWN" 

                if len(name_parts) > 1:
                    currency_for_key = name_parts[1].strip().upper()
                else:
                    st.info(f"Could not infer currency from '{file_name}'. Please enter it below.")
                    currency_input = st.text_input(f"Enter currency for '{file_name}' (e.g., KES, USD):", key=f"currency_input_{file_name}")
                    if currency_input:
                        currency_for_key = currency_input.strip().upper()
                    else:
                        st.warning(f"Currency for '{file_name}' is set to 'UNKNOWN'. This might affect matching.")


                key = f"{bank_name_for_key} {currency_for_key}".lower() 

                # Column mapping for Bank Statements (user selects original columns)
                st.subheader(f"Column Mapping for {file_name}")
                bank_col_options = ['-- Select Column --'] + df.columns.tolist()
                bank_col_mapping = {}

                bank_required_cols = {
                    'Date Column': resolve_date_column(df.columns.tolist()),
                    'Description Column': get_description_columns(df.columns.tolist()),
                    'Credit Amount': next((col for col in df.columns if col.lower() in ['deposit', 'credit', 'credit amount']), None),
                    'Debit Amount': next((col for col in df.columns if col.lower() in ['withdrawal', 'debit', 'debit amount']), None)
                }

                for display_name, suggested_col in bank_required_cols.items():
                    default_index = 0
                    if suggested_col and suggested_col in bank_col_options:
                        default_index = bank_col_options.index(suggested_col)
                    
                    selected_col = st.selectbox(
                        f"Map '{display_name}' for {file_name} to:",
                        options=bank_col_options,
                        index=default_index,
                        key=f"bank_map_select_{file_name}_{display_name}"
                    )
                    bank_col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None
                
                # Apply the selected mappings to the DataFrame and pre-process
                temp_df = df.copy()
                
                # Create a dictionary for renaming. Only rename if a mapping was selected.
                rename_dict = {
                    selected_original_col: mapped_name
                    for mapped_name, selected_original_col in bank_col_mapping.items()
                    if selected_original_col and selected_original_col in temp_df.columns
                }
                
                # Rename columns
                if rename_dict:
                    temp_df.rename(columns=rename_dict, inplace=True)
                
                # Ensure date column is datetime and filter out invalid dates
                if 'Date Column' in temp_df.columns:
                    temp_df['Date Column'] = temp_df['Date Column'].apply(parse_date)
                    temp_df = temp_df[temp_df['Date Column'].notna()].copy() # Filter and create a copy to avoid SettingWithCopyWarning
                else:
                    st.error(f"Error: 'Date Column' not found in '{file_name}' after mapping. This file cannot be processed for reconciliation.")
                    continue # Skip this bank file if date column is missing

                # Apply safe_float to amount columns
                if 'Credit Amount' in temp_df.columns:
                    temp_df['Credit Amount'] = temp_df['Credit Amount'].apply(safe_float)
                if 'Debit Amount' in temp_df.columns:
                    temp_df['Debit Amount'] = temp_df['Debit Amount'].apply(safe_float)

                bank_dfs[key] = temp_df # Store the pre-processed DataFrame
                st.success(f"Bank Statement '{file_name}' loaded and columns mapped successfully! (Internal Key: `{key}`)")
                st.dataframe(bank_dfs[key].head())

            except Exception as e:
                st.error(f"Error loading Bank Statement '{uploaded_file.name}': {e}")

    # --- Reconciliation Section ---
    st.header("2. Run Reconciliation")

    debug_mode = st.checkbox("Enable Debug Mode (show detailed logs)", value=False, key="debug_toggle")

    date_tolerance_days = st.slider(
        "Date Tolerance (± days for matching):",
        min_value=0,
        max_value=7,
        value=3,
        step=1,
        key="date_tolerance_slider"
    )

    if st.button("Run Reconciliation"):
        if fx_trade_df.empty or not bank_dfs:
            st.warning("Please upload both FX Trade Tracker and Bank Statement(s) to run reconciliation.")
        else:
            # Check if essential FX columns are available after mapping
            fx_required_for_recon = ['Action Type', 'Status', 'Created At', 'Buy Currency Amount', 'Buy Trade Info', 'Sell Currency Amount', 'Sell Trade Info']
            if not all(col in fx_trade_df.columns for col in fx_required_for_recon):
                missing_cols = [col for col in fx_required_for_recon if col not in fx_trade_df.columns]
                st.error(f"Missing essential FX Trade Tracker columns for reconciliation: {', '.join(missing_cols)}. Please map them correctly.")
                return

            with st.spinner("Reconciling transactions... This may take a moment."):
                buy_match_count = 0
                sell_match_count = 0
                unmatched_buy = []
                unmatched_sell = []
                unmatched_bank_records = []
                matched_buy = []
                matched_sell = []
                
                # To track matched bank records: (bank_key, original_bank_df_index)
                matched_bank_record_keys = set() 

                # Ensure column names are stripped of whitespace for consistent access
                fx_trade_df.columns = fx_trade_df.columns.str.strip()

                for index, row in fx_trade_df.iterrows():
                    action_type = str(row.get('Action Type', '')).strip()
                    status = str(row.get('Status', '')).strip().lower()

                    if status in ['cancelled', 'pending']: # Skip cancelled or pending trades
                        if debug_mode:
                            st.info(f"DEBUG: Skipping FX row {index} due to status: {status}.")
                        continue

                    # Process Buy Side (Counterparty Payment)
                    # process_fx_match now returns (bank_key, bank_row_idx) on match
                    matched_info = process_fx_match(
                        row,
                        bank_dfs,
                        unmatched_buy,
                        matched_buy,
                        action_type,
                        'Buy Currency Amount',
                        'Buy Trade Info',
                        False, # is_sell_side
                        date_tolerance_days=date_tolerance_days,
                        debug_mode=debug_mode
                    )
                    if matched_info:
                        buy_match_count += 1
                        matched_bank_record_keys.add(matched_info)

                    # Process Sell Side (Choice Payment)
                    matched_info = process_fx_match(
                        row,
                        bank_dfs,
                        unmatched_sell,
                        matched_sell,
                        action_type,
                        'Sell Currency Amount',
                        'Sell Trade Info',
                        True, # is_sell_side
                        date_tolerance_days=date_tolerance_days,
                        debug_mode=debug_mode
                    )
                    if matched_info:
                        sell_match_count += 1
                        matched_bank_record_keys.add(matched_info)

                # Scan for unmatched bank records
                for bank_key, bank_df in bank_dfs.items():
                    bank_df.columns = bank_df.columns.str.strip()
                    
                    # These columns are expected to exist and be pre-processed
                    date_col = 'Date Column'
                    description_col = 'Description Column'
                    credit_col = 'Credit Amount'
                    debit_col = 'Debit Amount'

                    if date_col not in bank_df.columns or description_col not in bank_df.columns or \
                       (credit_col not in bank_df.columns and debit_col not in bank_df.columns):
                        st.warning(f"Skipping bank statement '{bank_key}': Missing required mapped columns ('Date Column', 'Description Column', or neither 'Credit Amount'/'Debit Amount') after pre-processing.")
                        continue

                    for idx, row in bank_df.iterrows():
                        # Use the already parsed and validated date
                        row_date_parsed = row.get(date_col) 

                        # Check if this specific bank record (by its original index) was matched
                        if (bank_key, idx) not in matched_bank_record_keys:
                            # This bank record was not matched by any FX trade
                            amount_found = None
                            transaction_type_col_name = "N/A"
                            
                            # Check credit amount first
                            credit_amt = safe_float(row.get(credit_col))
                            if credit_amt is not None and abs(credit_amt) > 0.01:
                                amount_found = credit_amt
                                transaction_type_col_name = credit_col
                            
                            # If no credit or credit is zero, check debit
                            if amount_found is None:
                                debit_amt = safe_float(row.get(debit_col))
                                if debit_amt is not None and abs(debit_amt) > 0.01:
                                    amount_found = debit_amt
                                    transaction_type_col_name = debit_col
                            
                            if amount_found is not None:
                                unmatched_bank_records.append({
                                    'Bank Table': bank_key, 
                                    'Date': row_date_parsed.strftime('%Y-%m-%d') if row_date_parsed else None,
                                    'Description': str(row.get(description_col, '')).strip(),
                                    'Transaction Type (Column)': transaction_type_col_name,
                                    'Amount': round(amount_found, 2)
                                })
                            elif debug_mode:
                                st.info(f"DEBUG: Skipping bank record {idx} in {bank_key} - no significant amount in Credit/Debit after pre-processing.")


            st.session_state['unmatched_buy_df'] = pd.DataFrame(unmatched_buy)
            st.session_state['unmatched_sell_df'] = pd.DataFrame(unmatched_sell)
            st.session_state['matched_buy_df'] = pd.DataFrame(matched_buy)
            st.session_state['matched_sell_df'] = pd.DataFrame(matched_sell)
            st.session_state['unmatched_bank_df'] = pd.DataFrame(unmatched_bank_records)
            st.session_state['fx_trade_df'] = fx_trade_df 

            st.success("Reconciliation complete!")

    # --- Results and Analysis Section ---
    st.header("3. Reconciliation Results and Analysis")

    if 'unmatched_buy_df' in st.session_state:
        unmatched_buy_df = st.session_state['unmatched_buy_df']
        unmatched_sell_df = st.session_state['unmatched_sell_df']
        matched_buy_df = st.session_state['matched_buy_df']
        matched_sell_df = st.session_state['matched_sell_df']
        unmatched_bank_df = st.session_state['unmatched_bank_df']
        fx_trade_df = st.session_state['fx_trade_df']

        st.subheader("Overall Summary")
        # Ensure 'Action Type' and 'Status' exist before filtering
        total_fx_trades = len(fx_trade_df) if not fx_trade_df.empty else 0
        
        # Calculate totals only for non-cancelled/pending trades for more accurate reconciliation rate
        active_fx_trades = fx_trade_df[~fx_trade_df['Status'].isin(['cancelled', 'pending'])] if 'Status' in fx_trade_df.columns else fx_trade_df
        total_buy_side_trades_active = len(active_fx_trades[active_fx_trades['Action Type'] == 'Bank Buy']) if 'Action Type' in active_fx_trades.columns else 0
        total_sell_side_trades_active = len(active_fx_trades[active_fx_trades['Action Type'] == 'Bank Sell']) if 'Action Type' in active_fx_trades.columns else 0


        st.write(f"✅ **BUY Side Matches (Counterparty Payment):** {len(matched_buy_df)}")
        st.write(f"❌ **BUY Side Unmatched:** {len(unmatched_buy_df)}")
        st.write(f"✅ **SELL Side Matches (Choice Payment):** {len(matched_sell_df)}")
        st.write(f"❌ **SELL Side Unmatched:** {len(unmatched_sell_df)}")
        st.write(f"📤 **Bank-only unmatched entries:** {len(unmatched_bank_df)}")

        st.markdown("---")

        # --- 7.1. Reconciliation Summary Statistics ---
        st.subheader("Reconciliation Summary Statistics")
        st.write(f"Total FX Trade Records (excluding cancelled/pending): {len(active_fx_trades)}")
        st.write(f"Total Buy Side FX Trades processed: {total_buy_side_trades_active}")
        st.write(f"Total Sell Side FX Trades processed: {total_sell_side_trades_active}")
        
        buy_match_rate = (len(matched_buy_df)/total_buy_side_trades_active*100) if total_buy_side_trades_active > 0 else 0
        buy_unmatch_rate = (len(unmatched_buy_df)/total_buy_side_trades_active*100) if total_buy_side_trades_active > 0 else 0
        sell_match_rate = (len(matched_sell_df)/total_sell_side_trades_active*100) if total_sell_side_trades_active > 0 else 0
        sell_unmatch_rate = (len(unmatched_sell_df)/total_sell_side_trades_active*100) if total_sell_side_trades_active > 0 else 0

        st.write(f"Buy Side Matched: {len(matched_buy_df)} ({buy_match_rate:.2f}%)")
        st.write(f"Buy Side Unmatched: {len(unmatched_buy_df)} ({buy_unmatch_rate:.2f}%)")
        st.write(f"Sell Side Matched: {len(matched_sell_df)} ({sell_match_rate:.2f}%)")
        st.write(f"Sell Side Unmatched: {len(unmatched_sell_df)} ({sell_unmatch_rate:.2f}%)")
        st.write(f"Unmatched Bank Records (not found in FX trades): {len(unmatched_bank_df)}")

        st.markdown("---")

        # --- 7.2. Visualizing Reconciliation Status (Buy Side) ---
        st.subheader("Visualizing Reconciliation Status (Buy Side)")
        if not matched_buy_df.empty or not unmatched_buy_df.empty:
            buy_status_counts = pd.DataFrame({
                'Status': ['Matched Buy', 'Unmatched Buy'],
                'Count': [len(matched_buy_df), len(unmatched_buy_df)]
            })
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Status', y='Count', data=buy_status_counts, ax=ax)
            ax.set_title('FX Buy Side Reconciliation Status')
            ax.set_ylabel('Number of Trades')
            st.pyplot(fig)
        else:
            st.info("No Buy Side data for reconciliation status visualization.")

        # --- 7.3. Visualizing Reconciliation Status (Sell Side) ---
        st.subheader("Visualizing Reconciliation Status (Sell Side)")
        if not matched_sell_df.empty or not unmatched_sell_df.empty:
            sell_status_counts = pd.DataFrame({
                'Status': ['Matched Sell', 'Unmatched Sell'],
                'Count': [len(matched_sell_df), len(unmatched_sell_df)]
            })
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Status', y='Count', data=sell_status_counts, ax=ax)
            ax.set_title('FX Sell Side Reconciliation Status')
            ax.set_ylabel('Number of Trades')
            st.pyplot(fig)
        else:
            st.info("No Sell Side data for reconciliation status visualization.")

        # --- 7.4. Distribution of FX Trade Amounts ---
        st.subheader("Distribution of FX Trade Amounts")
        if not fx_trade_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            if 'Buy Currency Amount' in fx_trade_df.columns:
                sns.histplot(fx_trade_df['Buy Currency Amount'].dropna(), kde=True, bins=10, ax=axes[0])
                axes[0].set_title('Distribution of Buy Currency Amounts (FX Trades)')
                axes[0].set_xlabel('Amount')
                axes[0].set_ylabel('Frequency')
            else:
                axes[0].set_title('Buy Currency Amount Data Missing')
                axes[0].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

            if 'Sell Currency Amount' in fx_trade_df.columns:
                sns.histplot(fx_trade_df['Sell Currency Amount'].dropna(), kde=True, bins=10, color='orange', ax=axes[1])
                axes[1].set_title('Distribution of Sell Currency Amounts (FX Trades)')
                axes[1].set_xlabel('Amount')
                axes[1].set_ylabel('Frequency')
            else:
                axes[1].set_title('Sell Currency Amount Data Missing')
                axes[1].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No FX Trade data for amount distribution visualization.")

        # --- 7.5. Top Unmatched Bank Records by Amount ---
        st.subheader("Top Unmatched Bank Records by Amount")
        if not unmatched_bank_df.empty:
            # Ensure 'Amount' column is numeric for sorting
            unmatched_bank_df['Amount'] = pd.to_numeric(unmatched_bank_df['Amount'], errors='coerce')
            top_unmatched_bank = unmatched_bank_df.sort_values(by='Amount', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.barplot(x='Amount', y='Bank Table', hue='Transaction Type (Column)', data=top_unmatched_bank, dodge=True, ax=ax)
            ax.set_title('Top 10 Unmatched Bank Records by Amount')
            ax.set_xlabel('Amount')
            ax.set_ylabel('Bank Account')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No unmatched bank records for top amount visualization.")

        # --- 7.6. Transaction Volume by Bank (Unmatched Bank Records) ---
        st.subheader("Number of Unmatched Transactions per Bank Account")
        if not unmatched_bank_df.empty:
            bank_volume = unmatched_bank_df['Bank Table'].value_counts().reset_index()
            bank_volume.columns = ['Bank Table', 'Count']
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Count', y='Bank Table', data=bank_volume, palette='cubehelix', ax=ax)
            ax.set_title('Number of Unmatched Transactions per Bank Account')
            ax.set_xlabel('Number of Unmatched Transactions')
            ax.set_ylabel('Bank Account')
            st.pyplot(fig)
        else:
            st.info("No unmatched bank records for transaction volume visualization.")

        # --- 7.7. Daily Transaction Trend (FX Trades) ---
        st.subheader("Daily Transaction Trend (FX Trades)")
        if not fx_trade_df.empty and 'Created At' in fx_trade_df.columns:
            fx_trades_valid_dates = fx_trade_df.dropna(subset=['Created At']).copy()
            # Ensure 'Created At' is treated as datetime for trend analysis
            fx_trades_valid_dates['DateOnly'] = fx_trades_valid_dates['Created At'].apply(parse_date)
            fx_trades_valid_dates = fx_trades_valid_dates[fx_trades_valid_dates['DateOnly'].notna()]
            
            if not fx_trades_valid_dates.empty:
                daily_counts = fx_trades_valid_dates['DateOnly'].dt.date.value_counts().sort_index().reset_index()
                daily_counts.columns = ['Date', 'Count']

                if len(daily_counts) > 1:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x='Date', y='Count', data=daily_counts, marker='o', ax=ax)
                    ax.set_title('Daily FX Trade Transaction Count')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Number of Trades')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough date diversity in FX trades for daily trend visualization (only one unique date or less).")
            else:
                st.info("No valid 'Created At' dates found in FX trades for daily trend visualization.")
        else:
            st.info("No FX Trade data with valid 'Created At' column for daily trend visualization.")

        st.markdown("---")

        st.subheader("Download Results")
        # Helper to convert DataFrame to CSV for download
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Display previews and download buttons
        if not matched_buy_df.empty:
            st.markdown("#### Preview: Matched Buy Side FX Records")
            st.dataframe(matched_buy_df.head())
            st.download_button(
                label="Download Matched Buy Side FX Records",
                data=convert_df_to_csv(matched_buy_df),
                file_name=out_csv_path_buy_matched,
                mime="text/csv",
                key="download_matched_buy"
            )
        if not unmatched_buy_df.empty:
            st.markdown("#### Preview: Unmatched Buy Side FX Records")
            st.dataframe(unmatched_buy_df.head())
            st.download_button(
                label="Download Unmatched Buy Side FX Records",
                data=convert_df_to_csv(unmatched_buy_df),
                file_name=out_csv_path_buy_unmatched,
                mime="text/csv",
                key="download_unmatched_buy"
            )
        if not matched_sell_df.empty:
            st.markdown("#### Preview: Matched Sell Side FX Records")
            st.dataframe(matched_sell_df.head())
            st.download_button(
                label="Download Matched Sell Side FX Records",
                data=convert_df_to_csv(matched_sell_df),
                file_name=out_csv_path_sell_matched,
                mime="text/csv",
                key="download_matched_sell"
            )
        if not unmatched_sell_df.empty:
            st.markdown("#### Preview: Unmatched Sell Side FX Records")
            st.dataframe(unmatched_sell_df.head())
            st.download_button(
                label="Download Unmatched Sell Side FX Records",
                data=convert_df_to_csv(unmatched_sell_df),
                file_name=out_csv_path_sell_unmatched,
                mime="text/csv",
                key="download_unmatched_sell"
            )
        if not unmatched_bank_df.empty:
            st.markdown("#### Preview: Unmatched Bank Records (Bank-only entries)")
            st.dataframe(unmatched_bank_df.head())
            st.download_button(
                label="Download Unmatched Bank Records",
                data=convert_df_to_csv(unmatched_bank_df),
                file_name=out_csv_path_bank_unmatched,
                mime="text/csv",
                key="download_unmatched_bank"
            )

# # To run the Streamlit app directly
# if __name__ == '__main__':
#     graphed_analysis_app()

# ```