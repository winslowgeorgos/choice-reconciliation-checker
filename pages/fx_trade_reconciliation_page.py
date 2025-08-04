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
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
    '%d/%m/%Y', '%-d/%-m/%Y', '%-d.%-m/%-Y', # Added -%d for non-padded day
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', # Added datetime formats
    '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S', '%-d/%-m/%Y %H:%M:%S',
    '%-d.%-m.%Y %H:%M:%S', "%d.%m.%Y"
]

# Fuzzy matching threshold for bank names (0-100) - Less relevant with direct selection, but kept for normalize_bank_key
FUZZY_MATCH_THRESHOLD = 70

# PREDEFINED LIST OF BANK NAME - CURRENCY COMBINATIONS (Copied from main_dashboard for consistency)
PREDEFINED_BANK_CURRENCY_COMBOS = sorted([ # Sorted for better UX in dropdown
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
])


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

def normalize_bank_key(raw_key, debug_mode=False): # Added debug_mode parameter
    """
    Normalizes bank names to a consistent short code, using fuzzy matching.
    This function is primarily used for standardizing the bank *name* part of the FX trade info.
    For bank statement file naming, we will now use direct user selection.
    """
    raw_key_lower = str(raw_key).lower().strip()
    replacements = {
        'ncba bank kenya plc': 'NCBA', # Changed to Title Case
        'ncba bank': 'NCBA', # Changed to Title Case
        'equity bank': 'Equity', # Changed to Title Case
        'i&m bank': 'I&M', # Changed to Title Case
        'central bank of kenya': 'CBK', # Changed to Title Case
        'kenya commercial bank': 'KCB', # Changed to Title Case
        'kcb bank': 'KCB', # Changed to Title Case
        'sbm bank (kenya) limited': 'SBM', # Changed to Title Case
        'sbm bank': 'SBM', # Changed to Title Case
        'absa bank': 'Absa', # Changed to Title Case
        'kingdom bank': 'Kingdom', # Changed to Title Case
        'uba': 'UBA' # Added UBA, assuming it should be capitalized
    }

    # First, try direct replacement
    for long, short in replacements.items():
        if raw_key_lower == long.lower(): # Compare lowercase raw_key with lowercase long name
            if debug_mode:
                st.info(f"DEBUG: normalize_bank_key - Direct match found: '{raw_key_lower}' -> '{short}'")
            return short
        if raw_key_lower.startswith(long.lower()): # If it starts with a long name, use short
            if debug_mode:
                st.info(f"DEBUG: normalize_bank_key - Starts with match found: '{raw_key_lower}' starts with '{long.lower()}' -> '{short}'")
            return short

    # If no direct match, try fuzzy matching against known short codes/replacements
    # Create a list of all possible target bank names (both original and standardized) for fuzzy matching
    all_target_bank_names = list(replacements.values()) + [k.capitalize() for k in replacements.keys()] # Include capitalized versions for fuzzy matching
    all_target_bank_names = list(set(all_target_bank_names)) # Ensure uniqueness

    if debug_mode:
        st.info(f"DEBUG: normalize_bank_key - Fuzzy matching '{raw_key_lower}' against set: {all_target_bank_names}")

    match = process.extractOne(raw_key_lower, all_target_bank_names, scorer=fuzz.ratio)
    if match:
        if debug_mode:
            st.info(f"DEBUG: normalize_bank_key - Fuzzy match result: '{match[0]}' with relevance value {match[1]} (Threshold: {FUZZY_MATCH_THRESHOLD})")
        if match[1] >= FUZZY_MATCH_THRESHOLD:
            # If a fuzzy match is found, try to map it back to our standardized short forms
            for long, short in replacements.items():
                if match[0].lower() == long.lower():
                    return short
                if match[0].lower().startswith(long.lower()):
                    return short
            # If fuzzy match but not directly in replacements (e.g., a slightly misspelled short form), try to capitalize it
            return match[0].title() if match[0].islower() else match[0] # Capitalize if it's all lowercase
    if debug_mode:
        st.info(f"DEBUG: normalize_bank_key - No good fuzzy match found for '{raw_key_lower}'. Returning original.")
    
    # Fallback: if no match, try to title case the original raw_key for better consistency with PREDEFINED_BANK_CURRENCY_COMBOS
    return str(raw_key).strip().title() # Return Title Case of original if no match

def resolve_amount_column(columns, action_type, bank_statement_currency):
    """
    Identifies the correct amount column ('Credit' or 'Debit')
    based on the action type and bank statement currency, following the new rules.
    Assumes 'Credit' and 'Debit' are standardized column names after preprocessing.
    """
    bank_statement_currency = bank_statement_currency.upper()

    if bank_statement_currency == 'KES':
        if action_type == 'Bank Buy': # KES (Debit column) for Bank Buy
            if 'Debit' in columns: return 'Debit'
        elif action_type == 'Bank Sell': # KES (Credit column) for Bank Sell
            if 'Credit' in columns: return 'Credit'
    else: # Another currency (USD, EURO etc)
        if action_type == 'Bank Sell': # Non-KES (Debit column) for Bank Sell
            if 'Debit' in columns: return 'Debit'
        elif action_type == 'Bank Buy': # Non-KES (Credit column) for Bank Buy
            if 'Credit' in columns: return 'Credit'
            
    # Fallback if mapped name not present or rule not met.
    # This part can be made more robust if there are other column names to consider.
    columns_lower = [col.lower() for col in columns]
    if 'debit' in columns_lower: return 'Debit'
    if 'credit' in columns_lower: return 'Credit'
    
    return None


def resolve_date_column(columns):
    """Identifies the date column from a list of column names, prioritizing common formats."""
    # This function is now less critical as 'Date' is the standardized column name
    # after preprocessing in main_dashboard.py.
    for candidate in ['Value Date', 'Transaction Date', 'MyUnknownColumn', 'Transaction date', 'Date', 'Activity Date']:
        if candidate in columns:
            return candidate
    return None

def get_description_columns(columns):
    """Identifies the description column from a list of column names."""
    for desc in ['Description', 'Narrative', 'Transaction Details', 'Customer reference', 'Transaction Remarks:', 'Transaction Details', 'Transaction\nDetails']:
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

    for fmt in DATE_FORMATS:
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

    trade_bank_name_raw = parts[0].strip()
    trade_currency = parts[1].strip().upper()

    # Normalize FX trade bank name using the existing normalize_bank_key
    normalized_trade_bank_name = normalize_bank_key(trade_bank_name_raw, debug_mode) # Removed .lower()
    
    if debug_mode:
        st.info(f"DEBUG: Processing FX Trade - Date: {parsed_date.strftime('%Y-%m-%d')}, Type: {action_type}, Amount: {amount}, Trade Currency: {trade_currency}, Normalized Trade Bank Name: {normalized_trade_bank_name}")
        st.info(f"DEBUG: Available Bank Statement Keys: {list(all_bank_dfs.keys())}")


    found_match = False
    target_bank_df_key = None
    best_bank_name_match_ratio = 0
    potential_bank_df_key = None

    # Now, with user-selected bank statement keys, we prioritize exact matches first
    # The `bank_df_key_in_dict` will now be the exact 'bankname currency' string from the dropdown.
    expected_bank_key_from_fx_trade = f"{normalized_trade_bank_name} {trade_currency}" # Removed .lower()

    if expected_bank_key_from_fx_trade in all_bank_dfs:
        target_bank_df_key = expected_bank_key_from_fx_trade
        if debug_mode:
            st.success(f"DEBUG: DIRECT BANK KEY MATCH! Found bank statement: '{target_bank_df_key}' based on FX trade info and user selection.")
    else:
        # If no direct match, log and move to unmatched
        if debug_mode:
            st.warning(f"DEBUG: No exact bank statement file found matching FX trade expected key '{expected_bank_key_from_fx_trade}'.")
        unmatched_list.append({
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': expected_bank_key_from_fx_trade,
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'No Matching Bank Statement File Found (based on exact match from user selection)',
            'Source Column': bank_currency_info_field
        })
        return None


    bank_df = all_bank_dfs[target_bank_df_key]
    bank_df_columns = bank_df.columns.tolist()

    # The bank_statement_currency is now directly from the target_bank_df_key
    bank_statement_currency_parts = target_bank_df_key.split(' ')
    bank_statement_currency = bank_statement_currency_parts[1].upper() if len(bank_statement_currency_parts) > 1 else "UNKNOWN"

    # 'Date' is the standardized column name after pre-processing in main_dashboard
    date_column = 'Date'
    # Use the updated resolve_amount_column based on the new criteria
    amount_column = resolve_amount_column(bank_df_columns, action_type, bank_statement_currency)
    
    if date_column not in bank_df.columns:
        # This should ideally not happen if pre-processing is successful in main_dashboard
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
            'Status': 'Missing or Unresolvable Amount Column in Bank Statement based on new rules',
            'Source Column': bank_currency_info_field
        })
        if debug_mode:
            st.warning(f"DEBUG: Missing or unresolvable amount column ({amount_column}) in bank statement '{target_bank_df_key}'.")
        return None
    
    # Date in bank_df['Date'] is already parsed to datetime during pre-processing in main_dashboard
    # Filter based on the 'Date' column which is already a datetime type
    date_matches = bank_df[
        bank_df['Date'].dt.date.between( # Use 'Date' column
            parsed_date.date() - pd.Timedelta(days=date_tolerance_days),
            parsed_date.date() + pd.Timedelta(days=date_tolerance_days)
        )
    ]

    if debug_mode:
        st.info(f"DEBUG: Found {len(date_matches)} potential date matches in '{target_bank_df_key}' within ±{date_tolerance_days} days of {parsed_date.strftime('%Y-%m-%d')}.")


    for idx, bank_row in date_matches.iterrows():
        # Only consider bank records that have not been matched yet
        # if bank_df.at[idx, "Matched"] == True:
        #     st.warning(f"DEBUG: Skipping bank record {idx} in {target_bank_df_key} (AMOUNT {bank_row.get(amount_column)}) (Date: {bank_row.get(date_column).strftime('%Y-%m-%d') if bank_row.get(date_column) else 'N/A'}, Desc: {bank_row.get('Description Column', 'N/A')}) as it's already matched.")
        #     continue

        bank_amt_raw = bank_row.get(amount_column)
        bank_amt = safe_float(bank_amt_raw)

        if debug_mode:
            st.info(f"DEBUG: Checking bank record {idx} in '{target_bank_df_key}':")
            st.info(f"  Bank Record Details - Date: {bank_row.get(date_column).strftime('%Y-%m-%d') if bank_row.get(date_column) else 'N/A'}, Desc: {bank_row.get('Description', 'N/A')}, Amount (raw): {bank_amt_raw}, Amount (parsed): {bank_amt}, Column: {amount_column}")

        if bank_amt is not None:
            # The trade_currency is the currency of the FX amount (e.g., Buy Currency Amount or Sell Currency Amount)
            # The bank_statement_currency is the currency of the bank account (e.g., KES, USD, EUR)
            converted_amount = convert_currency(amount, trade_currency, bank_statement_currency, parsed_date)
            amount_diff = abs(bank_amt - converted_amount) if converted_amount is not None else float('inf')

            if debug_mode:
                st.info(f"DEBUG: Trade Amount: {amount} {trade_currency}, Bank Statement Currency: {bank_statement_currency}. Converted Trade Amount: {converted_amount:.2f}")
                st.info(f"DEBUG: Bank Amount: {bank_amt:.2f}, Converted Trade Amount: {converted_amount:.2f}, Difference: {amount_diff:.2f} (Tolerance: 0.05)")


            # Match within a small tolerance for floating point comparisons
            if converted_amount is not None and abs(converted_amount) > 0.01 and amount_diff < 0.05: # Adjusted tolerance
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
                bank_df.at[idx, "Matched"] = True # Mark this bank record as matched
                if debug_mode:
                    st.success(f"DEBUG: MATCH FOUND! FX Trade Date: {parsed_date.strftime('%Y-%m-%d')}, Amount: {amount:.2f} {trade_currency} (Converted: {converted_amount:.2f} {bank_statement_currency}) matched with Bank Record {idx} (Date: {bank_row.get(date_column).strftime('%Y-%m-%d')}, Amount: {bank_amt:.2f} {bank_statement_currency}).")
                return (target_bank_df_key, idx) # Return unique identifier of matched bank record
            elif debug_mode:
                st.info(f"DEBUG: No amount match for bank record {idx}. Difference: {amount_diff:.2f}. Bank Record: Date: {bank_row.get(date_column).strftime('%Y-%m-%d') if bank_row.get(date_column) else 'N/A'}, Amount: {bank_amt:.2f}, Description: {bank_row.get('Description', 'N/A')}")
        elif debug_mode:
            st.info(f"DEBUG: Bank amount is None or invalid for row {idx}.")

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
            st.warning(f"DEBUG: No match found for FX trade (Date: {parsed_date.strftime('%Y-%m-%d')}, Amount: {amount}) after checking all potential bank records in '{target_bank_df_key}'.")

    return None # No match found

def graphed_analysis_app(all_bank_dfs: dict): # Added all_bank_dfs as an argument
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

    # Removed Bank Statements Upload and Pre-processing section as it's now centralized
    # st.subheader("Upload Bank Statement(s)")
    # uploaded_bank_files = st.file_uploader("Choose Bank Statement(s) (CSV or XLSX)", type=["csv", "xlsx"], accept_multiple_files=True, key="bank_uploader")
    # bank_dfs = {} # This will now be passed as an argument

    if not all_bank_dfs:
        st.warning("No bank statements loaded. Please upload and process bank statements in 'Bank Statement Management' on the main dashboard.")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


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
        if fx_trade_df.empty:
            st.warning("Please upload FX Trade Tracker to run reconciliation.")
        else:
            # Check if essential FX columns are available after mapping
            fx_required_for_recon = ['Action Type', 'Status', 'Created At', 'Buy Currency Amount', 'Buy Trade Info', 'Sell Currency Amount', 'Sell Trade Info']
            if not all(col in fx_trade_df.columns for col in fx_required_for_recon):
                missing_cols = [col for col in fx_required_for_recon if col not in fx_trade_df.columns]
                st.error(f"Missing essential FX Trade Tracker columns for reconciliation: {', '.join(missing_cols)}. Please map them correctly.")
                return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

            with st.spinner("Reconciling transactions... This may take a moment."):
                buy_match_count = 0
                sell_match_count = 0
                unmatched_buy = []
                matched_buy = []
                unmatched_sell = []
                matched_sell = []
                
                # No longer need matched_bank_record_keys set as matching status is in DataFrame

                # Ensure column names are stripped of whitespace for consistent access
                fx_trade_df.columns = fx_trade_df.columns.str.strip()

                # Create a copy of all_bank_dfs for this reconciliation run
                # This ensures that the 'Matched' status is tracked locally for this app's run
                # without affecting the original bank_dfs in session_state from main_dashboard.
                current_run_bank_dfs = {key: df.copy() for key, df in all_bank_dfs.items()}

                for index, row in fx_trade_df.iterrows():
                    action_type = str(row.get('Action Type', '')).strip()
                    status = str(row.get('Status', '')).strip().lower()

                    if status in ['cancelled', 'pending']: # Skip cancelled or pending trades
                        if debug_mode:
                            st.info(f"DEBUG: Skipping FX row {index} due to status: {status}.")
                        continue

                    # Process Buy Side (Counterparty Payment)
                    process_fx_match(
                        row,
                        current_run_bank_dfs, # Use the copy
                        unmatched_buy,
                        matched_buy,
                        action_type,
                        'Buy Currency Amount',
                        'Buy Trade Info',
                        date_tolerance_days=date_tolerance_days,
                        debug_mode=debug_mode
                    )
                    # The process_fx_match function now directly marks the bank_df with "Matched" = True

                    # Process Sell Side (Choice Payment)
                    process_fx_match(
                        row,
                        current_run_bank_dfs, # Use the copy
                        unmatched_sell,
                        matched_sell,
                        action_type,
                        'Sell Currency Amount',
                        'Sell Trade Info',
                        date_tolerance_days=date_tolerance_days,
                        debug_mode=debug_mode
                    )
                    # The process_fx_match function now directly marks the bank_df with "Matched" = True

                # Collect unmatched bank records by filtering the 'Matched' column from current_run_bank_dfs
                unmatched_bank_records = []
                for bank_key, bank_df in current_run_bank_dfs.items():
                    bank_df.columns = bank_df.columns.str.strip()
                    
                    date_col = 'Date' # Standardized column name
                    description_col = get_description_columns(bank_df.columns.tolist()) # Need to resolve this dynamically
                    credit_col = 'Credit' # Standardized column name
                    debit_col = 'Debit' # Standardized column name

                    if date_col not in bank_df.columns or description_col not in bank_df.columns or \
                       (credit_col not in bank_df.columns and debit_col not in bank_df.columns):
                        st.warning(f"Skipping bank statement '{bank_key}': Missing required mapped columns ('Date', 'Description', or neither 'Credit'/'Debit') after pre-processing.")
                        continue

                    # Filter for rows where 'Matched' is False
                    unmatched_bank_df_for_key = bank_df[bank_df["Matched"] == False].copy() # Work on a copy

                    for idx, row in unmatched_bank_df_for_key.iterrows():
                        row_date_parsed = row.get(date_col) 
                        amount_found = None
                        transaction_type_col_name = "N/A"
                        
                        credit_amt = safe_float(row.get(credit_col))
                        if credit_amt is not None and abs(credit_amt) > 0.01:
                            amount_found = credit_amt
                            transaction_type_col_name = credit_col
                        
                        if amount_found is None:
                            debit_amt = safe_float(row.get(debit_col))
                            if debit_amt is not None and abs(debit_amt) > 0.01:
                                amount_found = debit_amt
                                transaction_type_col_name = debit_col
                        
                        if amount_found is not None:
                            unmatched_bank_records.append({
                                'Bank Table': bank_key, # Use 'Bank Table' for consistency with combined_match_results_page
                                'Date': row_date_parsed.strftime('%Y-%m-%d') if row_date_parsed else None,
                                'Description': str(row.get(description_col, '')).strip(),
                                'Transaction Type (Column)': transaction_type_col_name, # Use 'Transaction Type (Column)'
                                'Amount': round(amount_found, 2)
                            })
                        elif debug_mode:
                            st.info(f"DEBUG: Skipping bank record {idx} in {bank_key} - no significant amount in Credit/Debit after pre-processing and not matched.")


            st.session_state['unmatched_buy_df'] = pd.DataFrame(unmatched_buy)
            st.session_state['unmatched_sell_df'] = pd.DataFrame(unmatched_sell)
            st.session_state['matched_buy_df'] = pd.DataFrame(matched_buy)
            st.session_state['matched_sell_df'] = pd.DataFrame(matched_sell)
            st.session_state['unmatched_bank_trade'] = pd.DataFrame(unmatched_bank_records) # Renamed to unmatched_bank_trade
            st.session_state['fx_trade_df'] = fx_trade_df 

            st.success("Reconciliation complete!")

    # --- Results and Analysis Section ---
    st.header("3. Reconciliation Results and Analysis")

    if 'unmatched_buy_df' in st.session_state:
        unmatched_buy_df = st.session_state['unmatched_buy_df']
        unmatched_sell_df = st.session_state['unmatched_sell_df']
        matched_buy_df = st.session_state['matched_buy_df']
        matched_sell_df = st.session_state['matched_sell_df']
        unmatched_bank_trade = st.session_state['unmatched_bank_trade'] # Renamed
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
        st.write(f"📤 **Bank-only unmatched entries:** {len(unmatched_bank_trade)}") # Renamed
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
        st.write(f"Unmatched Bank Records (not found in FX trades): {len(unmatched_bank_trade)}") # Renamed

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
        if not unmatched_bank_trade.empty: # Renamed
            # Ensure 'Amount' column is numeric for sorting
            unmatched_bank_trade['Amount'] = pd.to_numeric(unmatched_bank_trade['Amount'], errors='coerce')
            top_unmatched_bank = unmatched_bank_trade.sort_values(by='Amount', ascending=False).head(10)
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
        if not unmatched_bank_trade.empty: # Renamed
            bank_volume = unmatched_bank_trade['Bank Table'].value_counts().reset_index()
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
        if not unmatched_bank_trade.empty: # Renamed
            st.markdown("#### Preview: Unmatched Bank Records (Bank-only entries)")
            st.dataframe(unmatched_bank_trade.head())
            st.download_button(
                label="Download Unmatched Bank Records",
                data=convert_df_to_csv(unmatched_bank_trade),
                file_name=out_csv_path_bank_unmatched,
                mime="text/csv",
                key="download_unmatched_bank"
            )
    # If you want to return the DataFrames, use the ones created above (e.g., matched_buy_df, matched_sell_df, unmatched_buy_df, unmatched_sell_df, unmatched_bank_trade)
    return (
        st.session_state.get('matched_buy_df', pd.DataFrame()),
        st.session_state.get('matched_sell_df', pd.DataFrame()),
        st.session_state.get('unmatched_buy_df', pd.DataFrame()),
        st.session_state.get('unmatched_sell_df', pd.DataFrame()),
        st.session_state.get('unmatched_bank_trade', pd.DataFrame()) # Renamed
    )
