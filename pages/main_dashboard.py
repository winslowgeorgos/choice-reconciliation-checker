
import streamlit as st
from io import BytesIO
import pandas as pd
from datetime import datetime, timedelta
import io

# Import functions from other pages (assuming these files exist in the same directory)
from fx_reconcilliation_app_page import fx_reconciliation_app
from fx_trade_reconciliation_page import graphed_analysis_app
from combine_match_results_page import run_cross_match_analysis, cross_match_analysis_app
from business_fx_reconciliation_page import business_reconciliation_app   # NEW

st.set_page_config(page_title="Finance(FX) Reconciliation Dashboard", layout="wide")

# --- Constants and Global Mappings ---
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d', '%d/%m/%Y',
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d.%m.%Y %H:%M:%S',
    '%Y.%m.%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'
]

PREDEFINED_BANK_CURRENCY_OPTIONS = [
    "Absa KES", "Absa USD", "Absa EUR", "Absa GBP", "ABSA KES-SPECIAL", "ABSA USD-SPECIAL", "ABSA EUR-SPECIAL", "ABSA GBP-SPECIAL", "ABSA Bank USD-DCD",
    "CBK KES", "CBK USD", "CBK EUR", "CBK GBP", "CBK UGX", "CBK TZS", "CBK RWF", "CBK ZAR", "CBK CNY",
    "Equity KES", "Equity USD", "Equity EUR", "Equity GBP",
    "I&M KES", "I&M USD", "I&M EUR", "I&M GBP",
    "KCB KES", "KCB USD", "KCB EUR", "KCB GBP", "KCB GBP - 1343013054",
    "Kingdom KES", "Kingdom USD", "Kingdom EUR", "Kingdom GBP",
    "NCBA KES", "NCBA USD", "NCBA EUR", "NCBA GBP",
    "SBM KES", "SBM USD", "SBM EUR", "SBM GBP",
    "UBA KES", "UBA USD", "UBA EUR", "UBA GBP",
    "BAAS Temporary KES", "BAAS Temporary USD", "BAAS Temporary EUR", "BAAS Temporary GBP",
    "FX Temporary KES", "FX Temporary USD", "FX Temporary EUR", "FX Temporary GBP",
    "Other Temporary KES", "Other Temporary USD", "Other Temporary EUR", "Other Temporary GBP",
    "Unclaimed Funds KES", "Unclaimed Funds USD", "Unclaimed Funds EUR", "Unclaimed Funds GBP",
    "Yeepay KES", "Yeepay USD", "Yeepay EUR", "Yeepay GBP", "Yeepay CNY"
]

FX_EXPECTED_COLUMNS = {
    'Amount': 'Amount', 'Operation': 'Operation', 'Completed At': 'Completed At',
    'Intermediary Account': 'Intermediary Account', 'Currency': 'Currency', 'Status': 'Status'
}

BANK_EXPECTED_COLUMNS = {
    'Date': ['Date', 'Transaction Date', 'Value Date', 'Value date'],
    'Credit': ['Credit', 'Credit Amount', 'Money In', 'Deposit', 'Credit amount'],
    'Debit': ['Debit', 'Debit Amount', 'Money Out', 'Withdrawal', 'Debit amount'],
    'Description': ['Description', 'Narrative', 'Transaction Details', 'Customer reference', 'Transaction Remarks:', 'Transaction Details', 'TransactionDetails', 'Transaction\nDetails'],
    'Running Balances': ['Running Balances', 'Running Balance', 'Running Balance (KES)', 'Running Balance (USD)', 'Running Balance (EUR)', 'Running Balance (GBP)', 'RUNNING BALANCES', 'RUNNING BALANCE', 'RUNNING BALANCE (KES)', 'RUNNING BALANCE (USD)', 'RUNNING BALANCE (EUR)', 'RUNNING BALANCE (GBP)']
}

# --- Helper Functions ---
def parse_date(date_str_raw):
    """Parses a date string into a datetime object using predefined formats."""
    if pd.isna(date_str_raw) or date_str_raw == pd.NaT: return None
    if isinstance(date_str_raw, datetime): return date_str_raw
    if not isinstance(date_str_raw, str): date_str_raw = str(date_str_raw)
    date_str = date_str_raw.partition(" ")[0].strip() if " " in date_str_raw.strip() else date_str_raw.strip()
    for fmt in DATE_FORMATS:
        try: return datetime.strptime(date_str, fmt)
        except ValueError: continue
    return None

def safe_float(x):
    """Safely converts a value to a float, handling commas, non-numeric inputs, and ensuring consistency."""
    if pd.isna(x) or x is None: return None
    try:
        cleaned_x = str(x).replace(',', '').strip()
        return float(cleaned_x)
    except (ValueError, TypeError): return None

def process_uploaded_file(uploaded_file, sheet_name=None):
    """Reads an uploaded file (CSV or Excel) into a DataFrame."""
    uploaded_file.seek(0)
    if uploaded_file.name.endswith('.csv'):
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=enc)
                return df
            except Exception: continue
        st.error(f"Failed to decode CSV file '{uploaded_file.name}' using common encodings.")
        return pd.DataFrame()
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        try:
            # Handle multiple sheet selection
            if isinstance(sheet_name, list):
                # Read multiple sheets and return as dictionary
                dfs = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                return dfs
            else:
                # Single sheet
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                return df
        except Exception as e:
            st.error(f"Error reading Excel file '{uploaded_file.name}': {e}")
            return pd.DataFrame() if not isinstance(sheet_name, list) else {}
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return pd.DataFrame()

def get_excel_sheet_names(uploaded_file):
    """Returns sheet names for an Excel file."""
    uploaded_file.seek(0)
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as e:
        st.error(f"Error getting Excel sheet names: {e}")
        return []

# --- Session State Initialization ---
if 'df_matched_adjustments_local' not in st.session_state: st.session_state.df_matched_adjustments_local = pd.DataFrame()
if 'df_matched_adjustments_foreign' not in st.session_state: st.session_state.df_matched_adjustments_foreign = pd.DataFrame()
if 'df_unmatched_adjustments_local' not in st.session_state: st.session_state.df_unmatched_adjustments_local = pd.DataFrame()
if 'df_unmatched_adjustments_foreign' not in st.session_state: st.session_state.df_unmatched_adjustments_foreign = pd.DataFrame()
if 'df_unmatched_bank_recon' not in st.session_state: st.session_state.df_unmatched_bank_recon = pd.DataFrame()
if 'df_matched_counterparty' not in st.session_state: st.session_state.df_matched_counterparty = pd.DataFrame()
if 'df_matched_choice' not in st.session_state: st.session_state.df_matched_choice = pd.DataFrame()
if 'df_unmatched_counterparty' not in st.session_state: st.session_state.df_unmatched_counterparty = pd.DataFrame()
if 'df_unmatched_choice' not in st.session_state: st.session_state.df_unmatched_choice = pd.DataFrame()
if 'df_unmatched_bank_trade' not in st.session_state: st.session_state.df_unmatched_bank_trade = pd.DataFrame()
if 'df_unmatched_bank_records' not in st.session_state: st.session_state.df_unmatched_bank_records = pd.DataFrame()
if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
if 'bank_dfs' not in st.session_state: st.session_state.bank_dfs = {}
if 'bank_uploaded_file_objs' not in st.session_state: st.session_state.bank_uploaded_file_objs = []
if 'raw_bank_data_previews' not in st.session_state: st.session_state.raw_bank_data_previews = {}
if 'merged_bank_statement' not in st.session_state: st.session_state.merged_bank_statement = pd.DataFrame()
if "cached_bank_files" not in st.session_state: st.session_state.cached_bank_files = {}
page_selection = st.sidebar.radio("Go to", ["Bank Statement Management", "Adjacements Reconciliation", "FX Trade Reconciliation", "Business FX Reconciliation", "Cross-Match Analysis"])

# --- Main App Logic ---
if page_selection == "Bank Statement Management":
    st.title("Bank Statement Management")
    st.markdown("Upload and configure your bank statements here. These statements will then be available for all reconciliation modules.")

    uploaded_files = st.file_uploader("Upload Bank Statement(s) (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True, key="bank_uploader_main")
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.cached_bank_files:
                file_bytes = file.read()
                file_type = file.type
                st.session_state.cached_bank_files[file.name] = {"content": file_bytes, "type": file_type}

    files_to_delete = []

    if st.session_state.cached_bank_files:
        st.markdown("### Uploaded Bank Statements:")
        for file_name, file_data in st.session_state.cached_bank_files.items():
            file_key = file_name.lower().replace('.', '_')

            with st.expander(f"🗂️ {file_name}", expanded=True):
                col1, col2 = st.columns([8, 2])
                with col1: st.markdown(f"**File Name:** `{file_name}`")
                with col2:
                    if st.button("❌ Remove", key=f"remove_{file_name}"):
                        files_to_delete.append(file_name)
                        continue

                if file_key not in st.session_state.raw_bank_data_previews:
                    fake_file = BytesIO(file_data["content"])
                    fake_file.name = file_name

                    if file_name.endswith('.xlsx'):
                        sheet_names = get_excel_sheet_names(fake_file)
                        selected_sheets = [sheet_names[0]] if sheet_names else []
                        # Read all selected sheets
                        if selected_sheets:
                            dfs = process_uploaded_file(fake_file, sheet_name=selected_sheets)
                            # Store as dictionary of DataFrames
                            df_dict = dfs if isinstance(dfs, dict) else {selected_sheets[0]: dfs}
                        else:
                            df_dict = {}
                    else:
                        sheet_names = []
                        selected_sheets = []
                        df_dict = {"CSV": process_uploaded_file(fake_file)}

                    # Initialize standardized names for each sheet
                    standardized_names = {}
                    for sheet_name in df_dict.keys():
                        standardized_names[sheet_name] = ""

                    st.session_state.raw_bank_data_previews[file_key] = {
                        'file_obj': fake_file, 
                        'df_raw_dict': df_dict, 
                        'sheet_names': sheet_names,
                        'selected_sheets': selected_sheets,  # Correct key name
                        'column_mappings': {}, 
                        'standardized_names': standardized_names
                    }

                data = st.session_state.raw_bank_data_previews[file_key]
                df_bank_raw_dict = data['df_raw_dict']

                if file_name.endswith('.xlsx') and data['sheet_names']:
                    # Multi-select for sheets - FIXED VERSION
                    current_sheets = st.multiselect(
                        f"Select Sheets for {file_name}:", 
                        data['sheet_names'],
                        default=data['selected_sheets'],  # Use the correct key
                        key=f"bank_sheet_selector_{file_key}"
                    )
                    
                    # Check if sheet selection changed
                    if set(current_sheets) != set(data['selected_sheets']):
                        data['selected_sheets'] = current_sheets
                        if current_sheets:
                            fake_file = BytesIO(file_data["content"])
                            fake_file.name = file_name
                            dfs = process_uploaded_file(fake_file, sheet_name=current_sheets)
                            # Ensure we have a dictionary of DataFrames
                            df_bank_raw_dict = dfs if isinstance(dfs, dict) else {current_sheets[0]: dfs}
                            # Clean column names for each sheet
                            for sheet_name, df in df_bank_raw_dict.items():
                                if df is not None:
                                    df.columns = df.columns.str.strip()
                            
                            # Initialize standardized names for newly selected sheets
                            for sheet_name in df_bank_raw_dict.keys():
                                if sheet_name not in data['standardized_names']:
                                    data['standardized_names'][sheet_name] = ""
                            
                            st.info(f"Selected {len(current_sheets)} sheet(s) for {file_name}.")
                        else:
                            df_bank_raw_dict = {}
                        data['df_raw_dict'] = df_bank_raw_dict

                # Display standardized name selector for each sheet
                if df_bank_raw_dict:
                    for sheet_name, df_bank_raw in df_bank_raw_dict.items():
                        if df_bank_raw is not None and not df_bank_raw.empty:
                            st.markdown(f"---")
                            st.subheader(f"Sheet: {sheet_name}")
                            
                            # Standardized name selector for this specific sheet
                            selected_standardized_name = st.selectbox(
                                f"Select Standardized Name for '{sheet_name}':", 
                                options=[""] + PREDEFINED_BANK_CURRENCY_OPTIONS,
                                index=PREDEFINED_BANK_CURRENCY_OPTIONS.index(data['standardized_names'].get(sheet_name, "")) + 1 
                                if data['standardized_names'].get(sheet_name, "") in PREDEFINED_BANK_CURRENCY_OPTIONS else 0,
                                key=f"standardized_name_selector_{file_key}_{sheet_name}"
                            )
                            data['standardized_names'][sheet_name] = selected_standardized_name

                            st.write(f"**Preview - {sheet_name}:**")
                            st.dataframe(df_bank_raw.head())

                            available_columns = df_bank_raw.columns.tolist()
                            available_columns.insert(0, "")
                            # Initialize column mappings for this sheet if it doesn't exist
                            if sheet_name not in data['column_mappings']:
                                data['column_mappings'][sheet_name] = {}
                            current_mappings = data['column_mappings'][sheet_name]

                            st.write(f"**Column Mapping - {sheet_name}:**")
                            col_map_cols = st.columns(2)
                            for expected_col, default_val_list in BANK_EXPECTED_COLUMNS.items():
                                initial_selection = current_mappings.get(expected_col)
                                if not initial_selection:
                                    for default_val in default_val_list:
                                        if default_val.strip() in [col.strip() for col in df_bank_raw.columns]:
                                            initial_selection = default_val
                                            break
                                
                                with col_map_cols[0]: st.markdown(f"**{expected_col}**")
                                with col_map_cols[1]:
                                    mapped_col = st.selectbox(
                                        f"Map '{expected_col}' to ({sheet_name}):", options=available_columns,
                                        index=available_columns.index(initial_selection) if initial_selection and initial_selection in available_columns else 0,
                                        key=f"bank_map_{file_key}_{sheet_name}_{expected_col}",
                                        label_visibility="collapsed"
                                    )
                                    data['column_mappings'][sheet_name][expected_col] = mapped_col if mapped_col else None
                        else:
                            st.warning(f"No data loaded for sheet '{sheet_name}' in {file_name}.")
                else:
                    st.error(f"Could not load data from {file_name}.")
    
    for file_name in files_to_delete:
        st.session_state.cached_bank_files.pop(file_name, None)
        file_key = file_name.lower().replace('.', '_')
        st.session_state.raw_bank_data_previews.pop(file_key, None)
        st.success(f"File '{file_name}' and its data have been removed.")

    st.session_state.bank_dfs = {}
    st.session_state.merged_bank_statement = pd.DataFrame()

    # for file_name in files_to_delete:
    #     st.session_state.cached_bank_files.pop(file_name, None)
    #     file_key = file_name.lower().replace('.', '_')
    #     st.session_state.raw_bank_data_previews.pop(file_key, None)
    #     st.success(f"File '{file_name}' and its data have been removed.")

    # st.session_state.bank_dfs = {}
    # st.session_state.merged_bank_statement = pd.DataFrame()

    if st.button("Process All Bank Statements", key="process_all_bank_btn_main"):
        st.session_state.bank_dfs = {}
        all_success = True
        dfs_to_concat = []
        st.session_state.running_balances_col = None

        for file_key, data in st.session_state.raw_bank_data_previews.items():
            st.info(f"Processing '{data['file_obj'].name}'...")

            # Process each sheet separately
            sheet_dfs = []
            for sheet_name, df_raw in data['df_raw_dict'].items():
                if df_raw is None or df_raw.empty:
                    st.warning(f"Skipping empty sheet '{sheet_name}' in '{data['file_obj'].name}'")
                    continue

                # Check standardized name for this specific sheet - FIXED
                sheet_standardized_name = data['standardized_names'].get(sheet_name, "")
                if not sheet_standardized_name:
                    st.error(f"Missing standardized name for sheet '{sheet_name}' in '{data['file_obj'].name}'")
                    all_success = False
                    continue

                # Check for duplicate standardized names across all sheets
                if sheet_standardized_name in st.session_state.bank_dfs:
                    st.error(f"Duplicate standardized name '{sheet_standardized_name}' detected for sheet '{sheet_name}'. Please choose a unique name for each sheet.")
                    all_success = False
                    continue

                df_to_process = df_raw.copy()
                
                # Get column mappings for this specific sheet
                sheet_mappings = data['column_mappings'].get(sheet_name, {})
                
                renamed_cols = {}
                for expected_col, mapped_col in sheet_mappings.items():
                    if mapped_col and mapped_col in df_to_process.columns:
                        renamed_cols[mapped_col] = expected_col
                
                if renamed_cols:
                    df_to_process.rename(columns=renamed_cols, inplace=True)
                df_to_process.columns = df_to_process.columns.str.strip()
                
                # --- Advanced Data Validation ---
                required_cols = ['Date', 'Credit', 'Debit', 'Running Balances']
                missing_cols = [col for col in required_cols if col not in df_to_process.columns]
                if missing_cols:
                    st.error(f"Validation failed for sheet '{sheet_name}' in '{data['file_obj'].name}'. Missing columns: {', '.join(missing_cols)}.")
                    all_success = False
                    continue

                # Process data
                df_to_process['Date'] = df_to_process['Date'].apply(parse_date)
                invalid_dates_mask = df_to_process['Date'].isna()
                if invalid_dates_mask.any():
                    num_errors = invalid_dates_mask.sum()
                    st.warning(f"Warning in sheet '{sheet_name}' of '{data['file_obj'].name}': {num_errors} invalid dates found. These rows will be dropped.")
                    df_to_process = df_to_process[~invalid_dates_mask].copy()

                df_to_process['Credit'] = df_to_process['Credit'].apply(safe_float)
                df_to_process['Debit'] = df_to_process['Debit'].apply(safe_float)
                df_to_process['Running Balances'] = df_to_process['Running Balances'].apply(safe_float)

                df_to_process["Matched"] = False
                df_to_process['Bank'] = sheet_standardized_name  # Use sheet-specific standardized name
                df_to_process['Source_Sheet'] = sheet_name  # Track which sheet this data came from
                df_to_process['Source_File'] = data['file_obj'].name  # Track which file this data came from
                
                sheet_dfs.append(df_to_process)
                st.success(f"Processed: {data['file_obj'].name} - Sheet '{sheet_name}' as '{sheet_standardized_name}'")

            # Combine all sheets from this file
            if sheet_dfs:
                # Store each sheet's data in bank_dfs
                for df in sheet_dfs:
                    bank_name = df['Bank'].iloc[0] if not df.empty else None
                    if bank_name:
                        st.session_state.bank_dfs[bank_name] = df
                dfs_to_concat.extend(sheet_dfs)
            else:
                st.error(f"No valid sheets found in '{data['file_obj'].name}'")
                all_success = False

            # Combine all sheets from this file
            if sheet_dfs:
                file_df = pd.concat(sheet_dfs, ignore_index=True)
                st.session_state.bank_dfs[f"{data['file_obj'].name}_combined"] = file_df
                dfs_to_concat.append(file_df)
            else:
                st.error(f"No valid sheets found in '{data['file_obj'].name}'")
                all_success = False

        if all_success and dfs_to_concat:
            st.session_state.merged_bank_statement = pd.concat(dfs_to_concat, ignore_index=True)
            st.write("✅ All bank statements processed and merged.")
            
            if not st.session_state.merged_bank_statement.empty:
                df_bal = st.session_state.merged_bank_statement.copy()
                rb_col = 'Running Balances' # This column is now standardized in all dataframes
                
                df_bal.rename(columns={'Date': 'date', 'Debit': 'debit', 'Credit': 'credit', 'Bank': 'bank'}, inplace=True)
                df_bal["currency"] = df_bal["bank"].apply(lambda x: str(x).split()[-1].upper())
                df_bal = df_bal.sort_values(by=['bank', 'date'])
                
                per_bank_rows = []
                for bank_name, df_bank in df_bal.groupby("bank"):
                    df_bank = df_bank.sort_values("date").reset_index(drop=True)
                    first_row = df_bank.iloc[0]
                    last_row = df_bank.iloc[-1]
                    currency = str(bank_name).split()[-1].upper()
                    running_balance_first = first_row[rb_col] if pd.notna(first_row[rb_col]) else 0
                    debit_first = first_row["debit"] if pd.notna(first_row["debit"]) else 0
                    credit_first = first_row["credit"] if pd.notna(first_row["credit"]) else 0

                    opening_balance = running_balance_first - credit_first + debit_first
                    closing_balance = last_row[rb_col] if pd.notna(last_row[rb_col]) else 0

                    per_bank_rows.append({"Bank": bank_name, "Currency": currency, "Opening Balance": round(opening_balance, 2), "Closing Balance": round(closing_balance, 2)})

                per_bank_df = pd.DataFrame(per_bank_rows).sort_values(by=["Currency", "Bank"]).reset_index(drop=True)
                st.subheader("Per-Bank Opening & Closing Balances")
                st.dataframe(per_bank_df)
                csv_per_bank = per_bank_df.to_csv(index=False).encode("utf-8")
                st.download_button(label="⬇️ Download Per-Bank Balances CSV", data=csv_per_bank, file_name="per_bank_balances.csv", mime="text/csv")
                
                currency_summary = (per_bank_df.groupby("Currency").agg({"Opening Balance": "sum", "Closing Balance": "sum"}).round(2).reset_index().sort_values(by="Currency").reset_index(drop=True))
                st.subheader("Opening & Closing Balance Summary by Currency")
                st.dataframe(currency_summary)
                csv_summary = currency_summary.to_csv(index=False).encode("utf-8")
                st.download_button(label="⬇️ Download Currency Summary CSV", data=csv_summary, file_name="currency_balance_summary.csv", mime="text/csv")

                st.markdown("---")
                st.subheader("Monthly Transaction Volume")
                df_chart = st.session_state.merged_bank_statement.copy()
                df_chart['YearMonth'] = pd.to_datetime(df_chart['Date']).dt.to_period('M').astype(str)
                df_chart['Credit'] = pd.to_numeric(df_chart['Credit'], errors='coerce').fillna(0)
                df_chart['Debit'] = pd.to_numeric(df_chart['Debit'], errors='coerce').fillna(0)
                monthly_volume = df_chart.groupby(['Bank', 'YearMonth']).agg(
                    Total_Credit=('Credit', 'sum'),
                    Total_Debit=('Debit', 'sum')
                ).reset_index()
                st.bar_chart(monthly_volume, x='YearMonth', y=['Total_Credit', 'Total_Debit'], color=['#008000', '#FF0000'])
            
        elif all_success and not dfs_to_concat: 
            st.info("⚠️ No valid files processed.")
        else: 
            st.warning("⚠️ Some files could not be processed. See messages above.")

        st.markdown("---")
        st.header("Merged Bank Statement for Display and Download")
        if not st.session_state.get("merged_bank_statement", pd.DataFrame()).empty:
            st.write("### Combined Merged Statement:")
            st.dataframe(st.session_state.merged_bank_statement)
            csv = st.session_state.merged_bank_statement.to_csv(index=False).encode("utf-8")
            st.download_button(label="⬇️ Download Merged Bank Statement as CSV", data=csv, file_name="merged_bank_statement.csv", mime="text/csv")
        else: 
            st.info("No merged bank statement available yet.")
#---

elif page_selection == "Adjacements Reconciliation":
    st.title("Local & Foreign Adjacements Reconciliation App")
    if not st.session_state.bank_dfs: st.warning("Please go to 'Bank Statement Management' to upload and process bank statements first.")
    else: (st.session_state.df_matched_adjustments_local, st.session_state.df_matched_adjustments_foreign, st.session_state.df_unmatched_adjustments_local, st.session_state.df_unmatched_adjustments_foreign, st.session_state.df_unmatched_bank_records) = fx_reconciliation_app(st.session_state.bank_dfs)
elif page_selection == "FX Trade Reconciliation":
    st.title("FX Trade Reconciliation App")
    if not st.session_state.bank_dfs: st.warning("Please go to 'Bank Statement Management' to upload and process bank statements first.")
    else: (st.session_state.df_matched_counterparty, st.session_state.df_matched_choice, st.session_state.df_unmatched_counterparty, st.session_state.df_unmatched_choice, st.session_state.df_unmatched_bank_trade) = graphed_analysis_app(st.session_state.bank_dfs)
elif page_selection == "Business FX Reconciliation":
    st.title("Business FX Reconciliation App")
    if not st.session_state.bank_dfs: st.warning("Please go to 'Bank Statement Management' to upload and process bank statements first.")
    else:  business_reconciliation_app(st.session_state.df_matched_counterparty, st.session_state.df_matched_choice, debug_mode=st.session_state.debug_mode )
    
elif page_selection == "Cross-Match Analysis":
    st.title("Cross-Match Analysis")
    st.write("This section combines and compares the results from the two reconciliation applications to find potential missed matches.")
    if (st.session_state.df_matched_adjustments_local.empty and st.session_state.df_matched_adjustments_foreign.empty and st.session_state.df_matched_counterparty.empty and st.session_state.df_matched_choice.empty):
        st.warning("Please first run the 'Adjacements Reconciliation' and 'FX Trade Reconciliation' apps to populate the dataframes needed for cross-matching.")
    else:
        if st.button("Perform Cross-Match Analysis"):
            with st.spinner("Performing cross-match analysis..."):
                run_cross_match_analysis(
                    st.session_state.df_matched_adjustments_local,
                      st.session_state.df_matched_adjustments_foreign,
                    st.session_state.df_matched_counterparty,
                      st.session_state.df_matched_choice,
                    st.session_state.bank_dfs,
                    debug_mode=st.session_state.debug_mode
                )
        else: st.info("Click the button above to run the cross-match analysis.")
        cross_match_analysis_app()
