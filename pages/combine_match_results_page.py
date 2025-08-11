# combine_match_results_page.py
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple
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
    df_unmatched_adjustments_local: pd.DataFrame,
    df_unmatched_adjustments_foreign: pd.DataFrame,
    df_matched_counterparty: pd.DataFrame,
    df_matched_choice: pd.DataFrame,
    df_unmatched_bank_recon: pd.DataFrame,
    df_unmatched_bank_trade: pd.DataFrame,
    debug_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a cross-reconciliation check by combining matched and unmatched data
    from the two different reconciliation apps to find potential missed matches.

    Args:
        df_matched_adjustments_local (pd.DataFrame): Matched local adjustments from fx_reconciliation_app.
        df_matched_adjustments_foreign (pd.DataFrame): Matched foreign adjustments from fx_reconciliation_app.
        df_unmatched_adjustments_local (pd.DataFrame): Unmatched local adjustments.
        df_unmatched_adjustments_foreign (pd.DataFrame): Unmatched foreign adjustments.
        df_matched_counterparty (pd.DataFrame): Buy side matches from fx_trade_reconciliation.
        df_matched_choice (pd.DataFrame): Sell side matches from fx_trade_reconciliation.
        df_unmatched_bank_recon (pd.DataFrame): Unmatched bank records from fx_reconciliation_app.
        df_unmatched_bank_trade (pd.DataFrame): Unmatched bank records from fx_trade_reconciliation.
        debug_mode (bool): Enables or disables debug logging.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing
        - newly_matched_unmatched_bank_records_df: Bank records that were unmatched in one app but matched in another.
        - still_unmatched_bank_records_df: Bank records that remain unmatched after cross-analysis.
        - newly_matched_unmatched_adjustments_df: Placeholder (not actively cross-matched in this version).
        - still_unmatched_adjustments_df: Combined unmatched adjustments.
        - combined_unmatched_bank_records_df (for display).
    """
    st.header("Cross-Match Analysis")

    # --- Step 1: Combine All Matched Bank Records ---
    # The goal here is to create a comprehensive list of all bank records
    # that were successfully matched by *either* reconciliation application.
    # We standardize column names to allow concatenation.
    
    # Standardize matched adjustments from fx_reconciliation_app
    df_matched_adjustments = pd.concat([
        df_matched_adjustments_local.rename(columns={
            'Bank_Table': 'BankTable',
            'Bank_Statement_Date': 'BankDate',
            'Bank_Statement_Amount': 'BankAmount'
        }),
        df_matched_adjustments_foreign.rename(columns={
            'Bank_Table': 'BankTable',
            'Bank_Statement_Date': 'BankDate',
            'Bank_Statement_Amount': 'BankAmount'
        })
    ], ignore_index=True)
    df_matched_adjustments = df_matched_adjustments[['BankTable', 'BankDate', 'BankAmount']].copy()
    df_matched_adjustments['Source'] = 'FX Reconciliation App'
    
    # Standardize matched trade payments from fx_trade_reconciliation_page
    df_matched_trades = pd.concat([
        df_matched_counterparty.rename(columns={
            'Date': 'BankDate',
            'Bank Table': 'BankTable',
            'Bank Statement Amount': 'BankAmount'
        }),
        df_matched_choice.rename(columns={
            'Date': 'BankDate',
            'Bank Table': 'BankTable',
            'Bank Statement Amount': 'BankAmount'
        })
    ], ignore_index=True)
    df_matched_trades = df_matched_trades[['BankTable', 'BankDate', 'BankAmount']].copy()
    df_matched_trades['Source'] = 'FX Trade Reconciliation App'

    combined_matched_df = pd.concat([df_matched_adjustments, df_matched_trades], ignore_index=True)
    
    if 'BankDate' in combined_matched_df.columns:
        combined_matched_df['BankDate'] = pd.to_datetime(combined_matched_df['BankDate'], errors='coerce')
    if 'BankAmount' in combined_matched_df.columns:
        combined_matched_df['BankAmount'] = pd.to_numeric(combined_matched_df['BankAmount'], errors='coerce')
        combined_matched_df.dropna(subset=['BankAmount'], inplace=True)
        combined_matched_df['BankAmount'] = combined_matched_df['BankAmount'].round(2)

    combined_matched_df['MatchKey'] = combined_matched_df.apply(
        lambda row: f"{str(row['BankTable']).lower().strip()}_"
                    f"{row['BankDate'].strftime('%Y-%m-%d')}_"
                    f"{row['BankAmount']}"
        if pd.notna(row['BankDate']) and pd.notna(row['BankAmount']) else None,
        axis=1
    )
    combined_matched_df.dropna(subset=['MatchKey'], inplace=True)
    matched_keys_set = set(combined_matched_df['MatchKey'].unique())
    if debug_mode:
        st.info(f"DEBUG: Total unique matched bank record keys: {len(matched_keys_set)}")


    # --- Step 2: Combine All Unmatched Bank Records ---
    df_unmatched_bank_recon['Source'] = 'FX Reconciliation App'
    df_unmatched_bank_trade['Source'] = 'FX Trade Reconciliation App'

    combined_unmatched_bank_records_df = pd.concat([
        df_unmatched_bank_recon.rename(columns={
            'Bank_Table': 'BankTable',
            'Transaction_Type_Column': 'TransactionType'
        }),
        df_unmatched_bank_trade.rename(columns={
            'Bank Table': 'BankTable',
            'Transaction Type (Column)': 'TransactionType'
        })
    ], ignore_index=True)

    if 'Date' in combined_unmatched_bank_records_df.columns:
        combined_unmatched_bank_records_df['Date'] = pd.to_datetime(combined_unmatched_bank_records_df['Date'], errors='coerce')
    if 'Amount' in combined_unmatched_bank_records_df.columns:
        combined_unmatched_bank_records_df['Amount'] = pd.to_numeric(combined_unmatched_bank_records_df['Amount'], errors='coerce')
        combined_unmatched_bank_records_df.dropna(subset=['Amount'], inplace=True)
        combined_unmatched_bank_records_df['Amount'] = combined_unmatched_bank_records_df['Amount'].round(2)


    # --- Step 3: Cross-Match Unmatched Bank Records against All Matched Records ---
    st.subheader("Cross-Matching Unmatched Bank Records")
    
    newly_matched_unmatched_bank_records = []
    still_unmatched_bank_records = []

    if combined_unmatched_bank_records_df.empty:
        st.info("No combined unmatched bank records to cross-match.")
    elif combined_matched_df.empty:
        st.info("No combined matched records available for cross-matching. All unmatched bank records will remain 'still unmatched'.")
        still_unmatched_bank_records = combined_unmatched_bank_records_df.to_dict('records')
    else:
        for index, row in combined_unmatched_bank_records_df.iterrows():
            if pd.isna(row.get('Date')) or pd.isna(row.get('Amount')) or pd.isna(row.get('BankTable')):
                still_unmatched_bank_records.append(row.to_dict())
                if debug_mode:
                    st.warning(f"DEBUG: Skipping unmatched bank record {index} due to missing key components.")
                continue

            unmatched_key = f"{str(row['BankTable']).lower().strip()}_" \
                            f"{row['Date'].strftime('%Y-%m-%d')}_" \
                            f"{row['Amount']}"
            
            if unmatched_key in matched_keys_set:
                newly_matched_unmatched_bank_records.append(row.to_dict())
                if debug_mode:
                    st.success(f"DEBUG: Found a NEW match for previously unmatched bank record: {unmatched_key}")
            else:
                still_unmatched_bank_records.append(row.to_dict())
                if debug_mode:
                    st.info(f"DEBUG: Unmatched bank record {unmatched_key} remains unmatched after cross-check.")
        
    newly_matched_unmatched_bank_records_df = pd.DataFrame(newly_matched_unmatched_bank_records)
    still_unmatched_bank_records_df = pd.DataFrame(still_unmatched_bank_records)

    if not still_unmatched_bank_records_df.empty:
        unique_cols = ['BankTable', 'Description', 'Date', 'TransactionType', 'Amount']
        unique_still_unmatched_bank_records_df = still_unmatched_bank_records_df.drop_duplicates(subset=unique_cols, keep='first').copy()

    else:
        unique_still_unmatched_bank_records_df = pd.DataFrame()


    

    combined_unmatched_adjustments_df = pd.concat([
        df_unmatched_adjustments_local.assign(Source='FX Reconciliation App - Local'),
        df_unmatched_adjustments_foreign.assign(Source='FX Reconciliation App - Foreign')
    ], ignore_index=True)

    newly_matched_unmatched_adjustments_df = pd.DataFrame()
    still_unmatched_adjustments_df = combined_unmatched_adjustments_df
    


    
    # --- NEW VISUALIZATION AND ANALYSIS SECTION START ---
    st.markdown("---")
    st.header("Final Data Analysis & Visualization")
    
    if not still_unmatched_bank_records_df.empty:
        st.markdown("""
        ### Summary of Unmatched Records
        These visualizations provide a high-level overview of the `Still Unmatched Bank Records` after the cross-match analysis. This helps in understanding the source and characteristics of the remaining discrepancies.
        """)
        
        # Create a subplot for the visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pie Chart: Distribution of unmatched records by Source
        source_counts = still_unmatched_bank_records_df['Source'].value_counts()
        ax1.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax1.set_title("Distribution of Still Unmatched Records by Source App")
        ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Bar Chart: Count of unmatched records by Bank Table
        bank_table_counts = still_unmatched_bank_records_df['BankTable'].value_counts()
        sns.barplot(x=bank_table_counts.index, y=bank_table_counts.values, ax=ax2, palette="viridis")
        ax2.set_title("Count of Still Unmatched Records by Bank Table")
        ax2.set_xlabel("Bank Table")
        ax2.set_ylabel("Number of Records")
        plt.xticks(rotation=45, ha='right')
        
        # Display the visualizations
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("""
        ### Analysis of Unmatched Transaction Amounts
        This box plot shows the distribution of the transaction amounts for the still unmatched records, categorized by the source application. This can help identify if a particular application has a higher frequency of large-value unmatched transactions, which may require further investigation.
        """)
        
        # Box Plot: Distribution of 'Amount' by 'Source'
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Source', y='Amount', data=still_unmatched_bank_records_df, palette="coolwarm", ax=ax)
        ax.set_title("Distribution of Amounts for Still Unmatched Records")
        ax.set_xlabel("Source Application")
        ax.set_ylabel("Transaction Amount")
        st.pyplot(fig)

    else:
        st.info("No data available for visualization as all bank records were matched.")

    # --- NEW VISUALIZATION AND ANALYSIS SECTION END ---

    st.session_state.newly_matched_unmatched_bank_records_df = newly_matched_unmatched_bank_records_df
    st.session_state.still_unmatched_bank_records_df  = still_unmatched_bank_records_df
    st.session_state.newly_matched_unmatched_adjustments_df  = newly_matched_unmatched_adjustments_df
    st.session_state.still_unmatched_adjustments_df  = still_unmatched_adjustments_df
    st.session_state.combined_unmatched_bank_records_df  = combined_unmatched_bank_records_df
    st.session_state.unique_still_unmatched_bank_records_df = unique_still_unmatched_bank_records_df



    return (
        newly_matched_unmatched_bank_records_df,
        still_unmatched_bank_records_df,
        newly_matched_unmatched_adjustments_df,
        still_unmatched_adjustments_df,
        combined_unmatched_bank_records_df
    )


# A small helper function to simulate the app flow for testing.
def cross_match_analysis_app():
    
    
    newly_matched_unmatched_bank_records_df = st.session_state.newly_matched_unmatched_bank_records_df 
    still_unmatched_bank_records_df = st.session_state.still_unmatched_bank_records_df
    newly_matched_unmatched_adjustments_df = st.session_state.newly_matched_unmatched_adjustments_df
    still_unmatched_adjustments_df = st.session_state.still_unmatched_adjustments_df  
    combined_unmatched_bank_records_df = st.session_state.combined_unmatched_bank_records_df
    unique_still_unmatched_bank_records_df = st.session_state.unique_still_unmatched_bank_records_df

    # st.info("The Cross-Match Analysis will be triggered by the main dashboard.")
    st.markdown("---")
    st.subheader("Combined Unmatched Adjustments (from FX Reconciliation App)")
    if not still_unmatched_adjustments_df.empty:
        st.write("These are the adjustments that remain unmatched from the FX Reconciliation App, even after the initial reconciliation.")
        st.dataframe(still_unmatched_adjustments_df)
        st.download_button(
            label="Download Combined Unmatched Adjustments",
            data=still_unmatched_adjustments_df.to_csv(index=False).encode('utf-8'),
            file_name="Combined_Unmatched_Adjustments.csv",
            mime="text/csv"
        )
    else:
        st.info("No combined unmatched adjustments to display.")
    
    # --- Step 5: Display Results ---
    st.markdown("---")
    st.subheader("Combined Unmatched Bank Records (Before Cross-Match)")

    if not combined_unmatched_bank_records_df.empty:
        st.dataframe(combined_unmatched_bank_records_df)
        st.download_button(
            label="Download Combined Unmatched Bank Records (Original)",
            data=combined_unmatched_bank_records_df.to_csv(index=False).encode('utf-8'),
            file_name="Combined_Unmatched_Bank_Records_Original.csv",
            mime="text/csv"
        )
    else:
        st.info("No combined unmatched bank records to display.")

    st.markdown("---")
    st.subheader("Newly Found Matches (from Unmatched Bank Records)")
    if not newly_matched_unmatched_bank_records_df.empty:
        st.write("These bank records were previously flagged as unmatched by one reconciliation app but were found to have a corresponding match in the other reconciliation app's matched data.")
        st.dataframe(newly_matched_unmatched_bank_records_df)
        st.download_button(
            label="Download Newly Matched Bank Records",
            data=newly_matched_unmatched_bank_records_df.to_csv(index=False).encode('utf-8'),
            file_name="Newly_Matched_Bank_Records.csv",
            mime="text/csv"
        )
    else:
        st.success("No new matches were found in the unmatched bank records from cross-analysis.")

    st.markdown("---")
    st.subheader("Still Unmatched Bank Records (After Cross-Match)")
    if not still_unmatched_bank_records_df.empty:
        st.write("These records remain unmatched even after the cross-match analysis against all matched data.")
        st.dataframe(still_unmatched_bank_records_df)
        st.download_button(
            label="Download All Still Unmatched Bank Records",
            data=still_unmatched_bank_records_df.to_csv(index=False).encode('utf-8'),
            file_name="Still_Unmatched_Bank_Records.csv",
            mime="text/csv"
        )
    else:
        st.success("All bank records were matched or found to have a corresponding entry after cross-match.")

    st.markdown("---")
    st.subheader("Unique Still Unmatched Bank Records")
    if not unique_still_unmatched_bank_records_df.empty:
        st.write("This table shows a de-duplicated view of the 'Still Unmatched Bank Records', returning only unique records based on **BankTable, Description, Date, TransactionType, and Amount**.")
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