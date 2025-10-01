# combine_match_results_page.py
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt

st.session_state.newly_matched_unmatched_bank_records_df = pd.DataFrame()
st.session_state.still_unmatched_bank_records_df  = pd.DataFrame()
st.session_state.newly_matched_unmatched_adjustments_df  = pd.DataFrame()
st.session_state.still_unmatched_adjustments_df  = pd.DataFrame()
st.session_state.combined_unmatched_bank_records_df  = pd.DataFrame()
st.session_state.unique_still_unmatched_bank_records_df = pd.DataFrame()

def run_cross_match_analysis(
    df_matched_adjustments_local: pd.DataFrame,
    df_matched_adjustments_foreign: pd.DataFrame,
    df_matched_counterparty: pd.DataFrame,
    df_matched_choice: pd.DataFrame,
    df_bank_dfs: dict,
    debug_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a cross-reconciliation check by matching bank records against all matched data
    from the two different reconciliation apps to find potential missed matches.

    Args:
        df_matched_adjustments_local (pd.DataFrame): Matched local adjustments from fx_reconciliation_app.
        df_matched_adjustments_foreign (pd.DataFrame): Matched foreign adjustments from fx_reconciliation_app.
        df_matched_counterparty (pd.DataFrame): Buy side matches from fx_trade_reconciliation.
        df_matched_choice (pd.DataFrame): Sell side matches from fx_trade_reconciliation.
        df_bank_dfs (dict): Dictionary of bank DataFrames to check for matches.
        debug_mode (bool): Enables or disables debug logging.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing
        - newly_matched_unmatched_bank_records_df: Bank records that were unmatched but found matches in other apps.
        - still_unmatched_bank_records_df: Bank records that remain unmatched after cross-analysis.
        - newly_matched_unmatched_adjustments_df: Placeholder (not actively cross-matched in this version).
        - still_unmatched_adjustments_df: Combined unmatched adjustments.
        - combined_unmatched_bank_records_df: Combined bank records for display.
    """
    st.header("Cross-Match Analysis")

    # --- Step 1: Handle Bank Records Dictionary ---
    if debug_mode:
        st.write(f"DEBUG: Type of df_bank_dfs: {type(df_bank_dfs)}")
        st.write(f"DEBUG: Keys in df_bank_dfs: {list(df_bank_dfs.keys()) if df_bank_dfs else 'Empty'}")

    # Check if df_bank_dfs is a dictionary
    if not isinstance(df_bank_dfs, dict):
        st.error(f"Expected df_bank_dfs to be a dictionary, but got {type(df_bank_dfs)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not df_bank_dfs:
        st.error("No bank data provided. The df_bank_dfs dictionary is empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Combine all bank DataFrames from the dictionary into one
    all_bank_records = []
    for bank_name, bank_df in df_bank_dfs.items():
        if debug_mode:
            st.write(f"DEBUG: Processing bank '{bank_name}' with shape {bank_df.shape}")
            st.write(f"DEBUG: Columns in {bank_name}: {list(bank_df.columns)}")
        
        # Add bank table name as a column for tracking
        bank_df_copy = bank_df.copy()
        bank_df_copy['Bank_Table_Name'] = bank_name
        all_bank_records.append(bank_df_copy)

    # Combine all bank records
    if all_bank_records:
        combined_bank_df = pd.concat(all_bank_records, ignore_index=True)
    else:
        st.error("No valid bank records found in the dictionary.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if debug_mode:
        st.write(f"DEBUG: Combined bank records shape: {combined_bank_df.shape}")
        st.write(f"DEBUG: Combined bank records columns: {list(combined_bank_df.columns)}")

    # --- Step 2: Prepare Bank Records Data ---
    df_bank_records = combined_bank_df.copy()
    
    # Map expected columns to actual columns in bank data
    column_mapping = {
        'Date': None,
        'Bank': None,
        'Credit': None,
        'Debit': None,
        'Amount': None,
        'Description': None
    }
    
    # Try to find matching columns (case insensitive)
    available_columns = [col.lower() for col in df_bank_records.columns]
    
    for expected_col in column_mapping.keys():
        if expected_col.lower() in available_columns:
            # Find the actual column name
            actual_col = [col for col in df_bank_records.columns if col.lower() == expected_col.lower()][0]
            column_mapping[expected_col] = actual_col
        else:
            # Try partial matches
            matches = [col for col in df_bank_records.columns if expected_col.lower() in col.lower()]
            if matches:
                column_mapping[expected_col] = matches[0]
    
    if debug_mode:
        st.write(f"DEBUG: Column mapping results: {column_mapping}")

    # Check if we found the essential columns
    missing_essential = []
    for col in ['Date']:  # Only Date is absolutely essential
        if column_mapping[col] is None:
            missing_essential.append(col)
    
    if missing_essential:
        st.error(f"Could not find essential columns in bank records: {missing_essential}")
        st.info(f"Available columns: {list(df_bank_records.columns)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Rename columns for consistent processing
    rename_dict = {}
    for expected_col, actual_col in column_mapping.items():
        if actual_col:
            rename_dict[actual_col] = expected_col
    
    df_bank_records_processed = df_bank_records.rename(columns=rename_dict)
    
    # Handle Bank column - if not found, use Bank_Table_Name
    if 'Bank' not in df_bank_records_processed.columns and 'Bank_Table_Name' in df_bank_records_processed.columns:
        df_bank_records_processed['Bank'] = df_bank_records_processed['Bank_Table_Name']
    
    # Handle Credit/Debit/Amount columns
    if 'Credit' not in df_bank_records_processed.columns and 'Amount' in df_bank_records_processed.columns:
        # If we have Amount but no Credit/Debit, we need to determine based on sign or other logic
        df_bank_records_processed['Credit'] = df_bank_records_processed['Amount'].apply(
            lambda x: x if x > 0 else 0
        )
        df_bank_records_processed['Debit'] = df_bank_records_processed['Amount'].apply(
            lambda x: abs(x) if x < 0 else 0
        )
    else:
        # Handle individual Credit/Debit columns
        if 'Credit' in df_bank_records_processed.columns:
            df_bank_records_processed['Credit'] = pd.to_numeric(df_bank_records_processed['Credit'], errors='coerce').fillna(0)
        else:
            df_bank_records_processed['Credit'] = 0
        
        if 'Debit' in df_bank_records_processed.columns:
            df_bank_records_processed['Debit'] = pd.to_numeric(df_bank_records_processed['Debit'], errors='coerce').fillna(0)
        else:
            df_bank_records_processed['Debit'] = 0
    
    # Clean and standardize bank records
    df_bank_records_processed['Date'] = pd.to_datetime(df_bank_records_processed['Date'], errors='coerce')
    if 'Bank' in df_bank_records_processed.columns:
        df_bank_records_processed['Bank'] = df_bank_records_processed['Bank'].astype(str)
    else:
        df_bank_records_processed['Bank'] = 'Unknown'

    if debug_mode:
        st.write(f"DEBUG: Processed bank records columns: {list(df_bank_records_processed.columns)}")
        st.write(f"DEBUG: Sample processed records:")
        st.dataframe(df_bank_records_processed.head(3))

    # --- Step 3: Prepare Matched Data Sources ---
    # Debug information about matched data sources
    if debug_mode:
        st.write("DEBUG: Matched data sources information:")
        st.write(f"  - Local Adjustments: {len(df_matched_adjustments_local)} records")
        st.write(f"  - Foreign Adjustments: {len(df_matched_adjustments_foreign)} records")
        st.write(f"  - Counterparty Trades: {len(df_matched_counterparty)} records")
        st.write(f"  - Choice Trades: {len(df_matched_choice)} records")
        
        if not df_matched_counterparty.empty:
            st.write(f"DEBUG: Counterparty columns: {list(df_matched_counterparty.columns)}")
            st.write(f"DEBUG: Counterparty sample: {df_matched_counterparty.head(1).to_dict('records')}")
        
        if not df_matched_choice.empty:
            st.write(f"DEBUG: Choice columns: {list(df_matched_choice.columns)}")
            st.write(f"DEBUG: Choice sample: {df_matched_choice.head(1).to_dict('records')}")

    # --- Step 4: Define Matching Functions for ALL Sources ---
    def match_adjustments_local(bank_row: pd.Series, matched_df: pd.DataFrame) -> Dict[str, Any]:
        """Match bank records with local adjustments"""
        matches_found = []
        for adj_index, adj_row in matched_df.iterrows():
            try:
                # Date matching
                adj_date = pd.to_datetime(adj_row.get('Adjustment_Date'), errors='coerce')
                if pd.isna(adj_date) or pd.isna(bank_row['Date']) or adj_date != bank_row['Date']:
                    continue
                    
                # Bank table matching
                adj_bank_table = str(adj_row.get('Bank_Table', ''))
                if adj_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                # Amount and operation matching
                adj_amount = float(adj_row.get('Adjustment_Amount', 0))
                adj_operation = str(adj_row.get('Adjustment_Operation', '')).lower()
                
                if adj_operation == 'credit' and abs(bank_row['Credit'] - adj_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Local Adjustments',
                        'matched_index': adj_index,
                        'match_reason': f"Credit amount {adj_amount} matches with operation {adj_operation}",
                        'confidence': 'high'
                    })
                elif adj_operation == 'debit' and abs(bank_row['Debit'] - adj_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Local Adjustments',
                        'matched_index': adj_index,
                        'match_reason': f"Debit amount {adj_amount} matches with operation {adj_operation}",
                        'confidence': 'high'
                    })
            except (ValueError, TypeError) as e:
                if debug_mode:
                    st.warning(f"DEBUG: Error in local adjustments matching: {e}")
                continue
        
        if matches_found:
            # Return the best match (first one for now)
            return matches_found[0]
        return {'matched': False, 'reason': 'No match found in local adjustments'}

    def match_adjustments_foreign(bank_row: pd.Series, matched_df: pd.DataFrame) -> Dict[str, Any]:
        """Match bank records with foreign adjustments"""
        matches_found = []
        for adj_index, adj_row in matched_df.iterrows():
            try:
                # Date matching
                adj_date = pd.to_datetime(adj_row.get('Adjustment_Date'), errors='coerce')
                if pd.isna(adj_date) or pd.isna(bank_row['Date']) or adj_date != bank_row['Date']:
                    continue
                    
                # Bank table matching
                adj_bank_table = str(adj_row.get('Bank_Table', ''))
                if adj_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                # Amount and operation matching
                adj_amount = float(adj_row.get('Adjustment_Amount', 0))
                adj_operation = str(adj_row.get('Adjustment_Operation', '')).lower()
                
                if adj_operation == 'credit' and abs(bank_row['Credit'] - adj_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Foreign Adjustments',
                        'matched_index': adj_index,
                        'match_reason': f"Credit amount {adj_amount} matches with operation {adj_operation}",
                        'confidence': 'high'
                    })
                elif adj_operation == 'debit' and abs(bank_row['Debit'] - adj_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Foreign Adjustments',
                        'matched_index': adj_index,
                        'match_reason': f"Debit amount {adj_amount} matches with operation {adj_operation}",
                        'confidence': 'high'
                    })
            except (ValueError, TypeError) as e:
                if debug_mode:
                    st.warning(f"DEBUG: Error in foreign adjustments matching: {e}")
                continue
        
        if matches_found:
            return matches_found[0]
        return {'matched': False, 'reason': 'No match found in foreign adjustments'}

    def match_counterparty(bank_row: pd.Series, matched_df: pd.DataFrame) -> Dict[str, Any]:
        """Match bank records with counterparty trades"""
        matches_found = []
        for trade_index, trade_row in matched_df.iterrows():
            try:
                # Date matching
                trade_date = pd.to_datetime(trade_row.get('Date'), errors='coerce')
                if pd.isna(trade_date) or pd.isna(bank_row['Date']) or trade_date != bank_row['Date']:
                    continue
                    
                # Bank table matching - check different possible column names
                trade_bank_table = str(trade_row.get('Bank_Table', trade_row.get('Bank Table', '')))
                if trade_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                # Amount and column matching
                trade_amount = float(trade_row.get('Trade Amount', trade_row.get('Amount', 0)))
                matched_column = str(trade_row.get('Matched In Column', trade_row.get('Matched Column', ''))).lower()
                
                # Check both credit and debit scenarios
                if matched_column == 'credit' and abs(bank_row['Credit'] - trade_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Counterparty Trades',
                        'matched_index': trade_index,
                        'match_reason': f"Credit amount {trade_amount} matches in {matched_column} column",
                        'confidence': 'high'
                    })
                elif matched_column == 'debit' and abs(bank_row['Debit'] - trade_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Counterparty Trades',
                        'matched_index': trade_index,
                        'match_reason': f"Debit amount {trade_amount} matches in {matched_column} column",
                        'confidence': 'high'
                    })
                # If no matched column specified, try both credit and debit
                elif not matched_column or matched_column == '':
                    if abs(bank_row['Credit'] - trade_amount) < 0.01:
                        matches_found.append({
                            'matched': True,
                            'source': 'Counterparty Trades',
                            'matched_index': trade_index,
                            'match_reason': f"Credit amount {trade_amount} matches (auto-detected)",
                            'confidence': 'medium'
                        })
                    elif abs(bank_row['Debit'] - trade_amount) < 0.01:
                        matches_found.append({
                            'matched': True,
                            'source': 'Counterparty Trades',
                            'matched_index': trade_index,
                            'match_reason': f"Debit amount {trade_amount} matches (auto-detected)",
                            'confidence': 'medium'
                        })
            except (ValueError, TypeError) as e:
                if debug_mode:
                    st.warning(f"DEBUG: Error in counterparty matching: {e}")
                continue
        
        if matches_found:
            # Return the highest confidence match
            return sorted(matches_found, key=lambda x: x.get('confidence', 'low'))[0]
        return {'matched': False, 'reason': 'No match found in counterparty trades'}

    def match_choice(bank_row: pd.Series, matched_df: pd.DataFrame) -> Dict[str, Any]:
        """Match bank records with choice trades"""
        matches_found = []
        for trade_index, trade_row in matched_df.iterrows():
            try:
                # Date matching
                trade_date = pd.to_datetime(trade_row.get('Date'), errors='coerce')
                if pd.isna(trade_date) or pd.isna(bank_row['Date']) or trade_date != bank_row['Date']:
                    continue
                    
                # Bank table matching - check different possible column names
                trade_bank_table = str(trade_row.get('Bank_Table', trade_row.get('Bank Table', '')))
                if trade_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                # Amount and column matching
                trade_amount = float(trade_row.get('Trade Amount', trade_row.get('Amount', 0)))
                matched_column = str(trade_row.get('Matched In Column', trade_row.get('Matched Column', ''))).lower()
                
                # Check both credit and debit scenarios
                if matched_column == 'credit' and abs(bank_row['Credit'] - trade_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Choice Trades',
                        'matched_index': trade_index,
                        'match_reason': f"Credit amount {trade_amount} matches in {matched_column} column",
                        'confidence': 'high'
                    })
                elif matched_column == 'debit' and abs(bank_row['Debit'] - trade_amount) < 0.01:
                    matches_found.append({
                        'matched': True,
                        'source': 'Choice Trades',
                        'matched_index': trade_index,
                        'match_reason': f"Debit amount {trade_amount} matches in {matched_column} column",
                        'confidence': 'high'
                    })
                # If no matched column specified, try both credit and debit
                elif not matched_column or matched_column == '':
                    if abs(bank_row['Credit'] - trade_amount) < 0.01:
                        matches_found.append({
                            'matched': True,
                            'source': 'Choice Trades',
                            'matched_index': trade_index,
                            'match_reason': f"Credit amount {trade_amount} matches (auto-detected)",
                            'confidence': 'medium'
                        })
                    elif abs(bank_row['Debit'] - trade_amount) < 0.01:
                        matches_found.append({
                            'matched': True,
                            'source': 'Choice Trades',
                            'matched_index': trade_index,
                            'match_reason': f"Debit amount {trade_amount} matches (auto-detected)",
                            'confidence': 'medium'
                        })
            except (ValueError, TypeError) as e:
                if debug_mode:
                    st.warning(f"DEBUG: Error in choice matching: {e}")
                continue
        
        if matches_found:
            # Return the highest confidence match
            return sorted(matches_found, key=lambda x: x.get('confidence', 'low'))[0]
        return {'matched': False, 'reason': 'No match found in choice trades'}

    # --- Step 5: Perform Cross-Matching Against ALL Sources ---
    st.subheader("Cross-Matching Bank Records Against All Matched Data")
    
    newly_matched_unmatched_bank_records = []
    still_unmatched_bank_records = []

    total_records = len(df_bank_records_processed)
    if debug_mode:
        st.write(f"DEBUG: Starting cross-matching for {total_records} bank records against ALL sources")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for bank_index, bank_row in df_bank_records_processed.iterrows():
        # Update progress
        if total_records > 0:
            progress = (bank_index + 1) / total_records
            progress_bar.progress(progress)
            status_text.text(f"Processing record {bank_index + 1} of {total_records}")

        if pd.isna(bank_row['Date']):
            # Skip records with invalid dates
            unmatched_record = bank_row.to_dict()
            unmatched_record.update({
                'Bank_Record_Index': bank_index,
                'Mismatch_Reason': 'Invalid date'
            })
            still_unmatched_bank_records.append(unmatched_record)
            continue

        if debug_mode and bank_index < 3:  # Only show first 3 for debug
            st.write(f"DEBUG: Processing bank record {bank_index}: Date={bank_row['Date']}, Bank={bank_row['Bank']}, Credit={bank_row['Credit']}, Debit={bank_row['Debit']}")

        # Try matching against ALL four sources
        match_results = []
        
        # Match against local adjustments
        if not df_matched_adjustments_local.empty:
            local_match = match_adjustments_local(bank_row, df_matched_adjustments_local)
            match_results.append(local_match)
            if debug_mode and bank_index < 3 and local_match.get('matched'):
                st.success(f"DEBUG: Bank {bank_index} matched with LOCAL ADJUSTMENTS")
        
        # Match against foreign adjustments
        if not df_matched_adjustments_foreign.empty:
            foreign_match = match_adjustments_foreign(bank_row, df_matched_adjustments_foreign)
            match_results.append(foreign_match)
            if debug_mode and bank_index < 3 and foreign_match.get('matched'):
                st.success(f"DEBUG: Bank {bank_index} matched with FOREIGN ADJUSTMENTS")
        
        # Match against counterparty trades
        if not df_matched_counterparty.empty:
            counterparty_match = match_counterparty(bank_row, df_matched_counterparty)
            match_results.append(counterparty_match)
            if debug_mode and bank_index < 3 and counterparty_match.get('matched'):
                st.success(f"DEBUG: Bank {bank_index} matched with COUNTERPARTY TRADES")
        
        # Match against choice trades
        if not df_matched_choice.empty:
            choice_match = match_choice(bank_row, df_matched_choice)
            match_results.append(choice_match)
            if debug_mode and bank_index < 3 and choice_match.get('matched'):
                st.success(f"DEBUG: Bank {bank_index} matched with CHOICE TRADES")

        # Check if any match was found
        successful_matches = [result for result in match_results if result.get('matched', False)]
        
        if successful_matches:
            # Take the best match (highest confidence)
            best_match = sorted(successful_matches, key=lambda x: x.get('confidence', 'low'))[0]
            matched_record = bank_row.to_dict()
            matched_record.update({
                'Match_Source': best_match['source'],
                'Matched_Index': best_match['matched_index'],
                'Match_Reason': best_match['match_reason'],
                'Match_Confidence': best_match.get('confidence', 'medium'),
                'Bank_Record_Index': bank_index
            })
            newly_matched_unmatched_bank_records.append(matched_record)
            
            if debug_mode and bank_index < 3:
                st.success(f"DEBUG: Bank record {bank_index} matched with {best_match['source']} at index {best_match['matched_index']}")
        else:
            # No match found in ANY source
            unmatched_record = bank_row.to_dict()
            reasons = [result.get('reason', 'Unknown') for result in match_results if not result.get('matched', False)]
            mismatch_reason = ' | '.join(reasons) if reasons else 'No matches in any source'
            
            unmatched_record.update({
                'Bank_Record_Index': bank_index,
                'Mismatch_Reason': mismatch_reason
            })
            still_unmatched_bank_records.append(unmatched_record)
            
            if debug_mode and bank_index < 3:
                st.info(f"DEBUG: Bank record {bank_index} remains unmatched: {mismatch_reason}")

    # Clear progress bar
    progress_bar.empty()
    status_text.empty()

    # Convert to DataFrames
    newly_matched_unmatched_bank_records_df = pd.DataFrame(newly_matched_unmatched_bank_records)
    still_unmatched_bank_records_df = pd.DataFrame(still_unmatched_bank_records)

    # --- Step 6: Display Match Statistics by Source ---
    st.markdown("---")
    st.subheader("Match Statistics by Source")
    
    if not newly_matched_unmatched_bank_records_df.empty:
        match_stats = newly_matched_unmatched_bank_records_df['Match_Source'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        sources = ['Local Adjustments', 'Foreign Adjustments', 'Counterparty Trades', 'Choice Trades']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, source in enumerate(sources):
            count = match_stats.get(source, 0)
            with [col1, col2, col3, col4][i]:
                st.metric(f"Matches in {source}", count)
        
        # Pie chart of match sources
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(match_stats.values, labels=match_stats.index, autopct='%1.1f%%', 
               colors=colors[:len(match_stats)], startangle=90)
        ax.set_title("Distribution of Matches by Source")
        ax.axis('equal')
        st.pyplot(fig)
        plt.close()

    # --- Step 7: Create unique unmatched records ---
    if not still_unmatched_bank_records_df.empty:
        unique_cols = ['Bank', 'Date', 'Credit', 'Debit']
        available_unique_cols = [col for col in unique_cols if col in still_unmatched_bank_records_df.columns]
        if available_unique_cols:
            unique_still_unmatched_bank_records_df = still_unmatched_bank_records_df.drop_duplicates(
                subset=available_unique_cols, keep='first'
            ).copy()
        else:
            unique_still_unmatched_bank_records_df = still_unmatched_bank_records_df.copy()
    else:
        unique_still_unmatched_bank_records_df = pd.DataFrame()

    # --- Step 8: Prepare Other Output DataFrames ---
    combined_unmatched_adjustments_df = pd.DataFrame()
    newly_matched_unmatched_adjustments_df = pd.DataFrame()
    still_unmatched_adjustments_df = pd.DataFrame()
    combined_unmatched_bank_records_df = df_bank_records_processed.copy()

    # --- Step 9: Display Overall Results ---
    st.markdown("---")
    st.subheader("Cross-Match Results Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bank Records", total_records)
    with col2:
        st.metric("Newly Matched", len(newly_matched_unmatched_bank_records_df))
    with col3:
        st.metric("Still Unmatched", len(still_unmatched_bank_records_df))

    # Store results in session state
    st.session_state.newly_matched_unmatched_bank_records_df = newly_matched_unmatched_bank_records_df
    st.session_state.still_unmatched_bank_records_df = still_unmatched_bank_records_df
    st.session_state.newly_matched_unmatched_adjustments_df = newly_matched_unmatched_adjustments_df
    st.session_state.still_unmatched_adjustments_df = still_unmatched_adjustments_df
    st.session_state.combined_unmatched_bank_records_df = combined_unmatched_bank_records_df
    st.session_state.unique_still_unmatched_bank_records_df = unique_still_unmatched_bank_records_df

    return (
        newly_matched_unmatched_bank_records_df,
        still_unmatched_bank_records_df,
        newly_matched_unmatched_adjustments_df,
        still_unmatched_adjustments_df,
        combined_unmatched_bank_records_df
    )

# The cross_match_analysis_app function remains the same...
def cross_match_analysis_app():
    newly_matched_unmatched_bank_records_df = st.session_state.newly_matched_unmatched_bank_records_df 
    still_unmatched_bank_records_df = st.session_state.still_unmatched_bank_records_df
    newly_matched_unmatched_adjustments_df = st.session_state.newly_matched_unmatched_adjustments_df
    still_unmatched_adjustments_df = st.session_state.still_unmatched_adjustments_df  
    combined_unmatched_bank_records_df = st.session_state.combined_unmatched_bank_records_df
    unique_still_unmatched_bank_records_df = st.session_state.unique_still_unmatched_bank_records_df

    st.markdown("---")
    st.subheader("Original Bank Records")
    if not combined_unmatched_bank_records_df.empty:
        st.dataframe(combined_unmatched_bank_records_df)
        st.download_button(
            label="Download Original Bank Records",
            data=combined_unmatched_bank_records_df.to_csv(index=False).encode('utf-8'),
            file_name="Original_Bank_Records.csv",
            mime="text/csv"
        )
    else:
        st.info("No bank records to display.")

    st.markdown("---")
    st.subheader("Newly Found Matches (Previously Unmatched Bank Records)")
    if not newly_matched_unmatched_bank_records_df.empty:
        st.write("These bank records were found to have matches in other reconciliation apps during cross-analysis.")
        st.dataframe(newly_matched_unmatched_bank_records_df)
        st.download_button(
            label="Download Newly Matched Bank Records",
            data=newly_matched_unmatched_bank_records_df.to_csv(index=False).encode('utf-8'),
            file_name="Newly_Matched_Bank_Records.csv",
            mime="text/csv"
        )
    else:
        st.info("No new matches were found during cross-analysis.")

    st.markdown("---")
    st.subheader("Still Unmatched Bank Records (After Cross-Match)")
    if not still_unmatched_bank_records_df.empty:
        st.write("These records remain unmatched even after cross-match analysis against all matched data.")
        st.dataframe(still_unmatched_bank_records_df)
        st.download_button(
            label="Download All Still Unmatched Bank Records",
            data=still_unmatched_bank_records_df.to_csv(index=False).encode('utf-8'),
            file_name="Still_Unmatched_Bank_Records.csv",
            mime="text/csv"
        )
    else:
        st.success("All bank records were matched during cross-analysis.")

    st.markdown("---")
    st.subheader("Unique Still Unmatched Bank Records")
    if not unique_still_unmatched_bank_records_df.empty:
        st.write("This table shows a de-duplicated view of the 'Still Unmatched Bank Records'.")
        st.dataframe(unique_still_unmatched_bank_records_df)
        st.download_button(
            label="Download Unique Still Unmatched Bank Records",
            data=unique_still_unmatched_bank_records_df.to_csv(index=False).encode('utf-8'),
            file_name="Unique_Still_Unmatched_Bank_Records.csv",
            mime="text/csv"
        )
    else:
        st.info("No unique unmatched bank records to display.")

if __name__ == '__main__':
    st.title("Cross-Match Analysis App")
    st.warning("This file is intended to be imported by main_dashboard.py. This is a placeholder.")