# fx_trade_reconciliation_page.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns
import json
import uuid
import os
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Pickle tracking keys for change detection
PICKLE_TRACKING_KEYS = [
    'matched_buy_df', 'matched_sell_df', 'unmatched_buy_df', 'unmatched_sell_df', 'unmatched_bank_trade',
    'fx_trade_df', 'moved_buy_matched', 'moved_buy_unmatched', 'moved_sell_matched', 
    'moved_sell_unmatched', 'moved_bank_records_trade', 'audit_moves_log_trade',
    'deleted_buy_matched', 'deleted_buy_unmatched', 'deleted_sell_matched',
    'deleted_sell_unmatched', 'deleted_bank_trade', 'audit_deletes_log_trade'
]

# --- Helper Functions for Record Management (copied from fx_reconciliation_app) ---
def generate_record_id():
    """Generate a unique ID for a record"""
    return str(uuid.uuid4())

def add_unique_ids(df):
    """Add a unique ID column to the dataframe if it doesn't exist"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    if '_record_id' not in df_copy.columns:
        df_copy['_record_id'] = [generate_record_id() for _ in range(len(df_copy))]
    return df_copy

def ensure_record_ids(df):
    """Ensure dataframe has record IDs, return dataframe with record IDs"""
    if df is None or df.empty:
        return df
    if '_record_id' not in df.columns:
        return add_unique_ids(df)
    return df

def add_audit_columns(df):
    """Add audit trail columns to dataframe if they don't exist"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    if 'deleted_by' not in df_copy.columns:
        df_copy['deleted_by'] = ''
    if 'deleted_at' not in df_copy.columns:
        df_copy['deleted_at'] = ''
    if 'delete_reason' not in df_copy.columns:
        df_copy['delete_reason'] = ''
    if 'source_dataframe' not in df_copy.columns:
        df_copy['source_dataframe'] = ''
    if 'deleted_from' not in df_copy.columns:
        df_copy['deleted_from'] = ''
    if 'moved_by' not in df_copy.columns:
        df_copy['moved_by'] = ''
    if 'moved_from' not in df_copy.columns:
        df_copy['moved_from'] = ''
    if 'moved_at' not in df_copy.columns:
        df_copy['moved_at'] = ''
    if 'move_reason' not in df_copy.columns:
        df_copy['move_reason'] = ''
    if 'move_type' not in df_copy.columns:
        df_copy['move_type'] = ''
    if 'moved_to' not in df_copy.columns:
        df_copy['moved_to'] = ''
    
    if 'moved_at' in df_copy.columns:
        df_copy['moved_at'] = df_copy['moved_at'].astype(str)
    if 'deleted_at' in df_copy.columns:
        df_copy['deleted_at'] = df_copy['deleted_at'].astype(str)
    
    return df_copy

def add_row_numbers(df):
    """Add row numbers to dataframe for easy reference"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    if '#' in df_copy.columns:
        df_copy = df_copy.drop(columns=['#'])
    
    df_copy.insert(0, '#', range(1, len(df_copy) + 1))
    return df_copy

def remove_row_numbers(df):
    """Remove row number column from dataframe"""
    if df is None or df.empty:
        return df
    if '#' in df.columns:
        return df.drop(columns=['#'])
    return df

def get_current_user():
    """Get the current username for audit trail"""
    if 'user' in st.session_state:
        return st.session_state['user'].get('username', 'unknown')
    return 'unknown_user'

def get_deleted_df_name(source_name):
    """Generate a consistent name for the deleted records dataframe"""
    source_clean = source_name.lower().replace(' ', '_')
    
    if 'buy' in source_clean and 'matched' in source_clean:
        return 'deleted_buy_matched'
    elif 'buy' in source_clean and 'unmatched' in source_clean:
        return 'deleted_buy_unmatched'
    elif 'sell' in source_clean and 'matched' in source_clean:
        return 'deleted_sell_matched'
    elif 'sell' in source_clean and 'unmatched' in source_clean:
        return 'deleted_sell_unmatched'
    elif 'bank' in source_clean:
        return 'deleted_bank_trade'
    else:
        return f"deleted_{source_clean}"

def get_moved_df_name(source_name, target_name):
    """Generate a consistent name for the moved records dataframe"""
    source_clean = source_name.lower().replace(' ', '_')
    target_clean = target_name.lower().replace(' ', '_')
    
    if 'buy_matched' in target_clean:
        return 'moved_buy_matched'
    elif 'buy_unmatched' in target_clean:
        return 'moved_buy_unmatched'
    elif 'sell_matched' in target_clean:
        return 'moved_sell_matched'
    elif 'sell_unmatched' in target_clean:
        return 'moved_sell_unmatched'
    elif 'bank' in target_clean:
        return 'moved_bank_records_trade'
    else:
        return f"moved_{target_clean}"

def move_records_to_new_df(source_df, selected_record_ids, source_name, target_name, move_reason=""):
    """Move selected records from source to a NEW moved dataframe"""
    if not selected_record_ids:
        st.warning(f"No records selected to move from {source_name}")
        return pd.DataFrame(), source_df
    
    source_df_copy = source_df.copy() if source_df is not None else pd.DataFrame()
    source_df_copy = ensure_record_ids(source_df_copy)
    
    if '#' in source_df_copy.columns:
        source_df_copy = source_df_copy.drop(columns=['#'])
    
    selected_records = source_df_copy[source_df_copy['_record_id'].isin(selected_record_ids)].copy()
    remaining_source = source_df_copy[~source_df_copy['_record_id'].isin(selected_record_ids)].reset_index(drop=True)
    
    if '#' in remaining_source.columns:
        remaining_source = remaining_source.drop(columns=['#'])
    
    if selected_records.empty:
        logger.warning(f"No records found with IDs: {selected_record_ids}")
        return pd.DataFrame(), source_df
    
    current_user = get_current_user()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    selected_records = add_audit_columns(selected_records)
    selected_records['moved_by'] = current_user
    selected_records['moved_from'] = source_name
    selected_records['moved_to'] = target_name
    selected_records['moved_at'] = current_time
    selected_records['move_reason'] = move_reason
    selected_records['move_type'] = f"{source_name} → {target_name}"
    
    return selected_records, remaining_source

def delete_records_to_new_df(source_df, selected_record_ids, source_name, delete_reason=""):
    """Delete selected records and store in deleted dataframe"""
    if not selected_record_ids:
        st.warning(f"No records selected to delete from {source_name}")
        return pd.DataFrame(), source_df
    
    source_df_copy = source_df.copy() if source_df is not None else pd.DataFrame()
    source_df_copy = ensure_record_ids(source_df_copy)
    
    if '#' in source_df_copy.columns:
        source_df_copy = source_df_copy.drop(columns=['#'])
    
    selected_records = source_df_copy[source_df_copy['_record_id'].isin(selected_record_ids)].copy()
    mask = source_df_copy['_record_id'].isin(selected_record_ids)
    remaining_source = source_df_copy[~mask].reset_index(drop=True)
    
    if selected_records.empty:
        logger.warning(f"No records found with IDs: {selected_record_ids}")
        return pd.DataFrame(), source_df
    
    current_user = get_current_user()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    selected_records = add_audit_columns(selected_records)
    selected_records['deleted_by'] = current_user
    selected_records['deleted_at'] = current_time
    selected_records['delete_reason'] = delete_reason
    selected_records['deleted_from'] = source_name
    selected_records['source_dataframe'] = source_name
    
    return selected_records, remaining_source

def delete_selected_rows_with_audit(df, selected_record_ids, source_name, delete_reason="", df_name=None, on_data_change=None):
    """Delete selected rows and store them in a deleted audit dataframe"""
    if not selected_record_ids:
        return df, 0
    
    if isinstance(selected_record_ids, str):
        selected_record_ids = [selected_record_ids]
    
    source_df = df.copy() if df is not None else pd.DataFrame()
    
    if source_df.empty:
        return df, 0
    
    source_df = ensure_record_ids(source_df)
    
    if '#' in source_df.columns:
        source_df = source_df.drop(columns=['#'])
    
    deleted_records, remaining_source = delete_records_to_new_df(
        source_df, selected_record_ids, source_name, delete_reason
    )
    
    if deleted_records.empty:
        return df, 0
    
    deleted_df_name = get_deleted_df_name(source_name)
    
    if deleted_df_name not in st.session_state:
        st.session_state[deleted_df_name] = deleted_records
    else:
        existing = st.session_state[deleted_df_name]
        existing_ids = set(existing['_record_id'].tolist()) if not existing.empty else set()
        new_records = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_records.empty:
            st.session_state[deleted_df_name] = pd.concat([existing, new_records], ignore_index=True)
    
    if 'audit_deletes_log_trade' not in st.session_state:
        st.session_state.audit_deletes_log_trade = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.audit_deletes_log_trade
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log_entries = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log_entries.empty:
            st.session_state.audit_deletes_log_trade = pd.concat([existing_log, new_log_entries[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)
    
    remaining_source_with_numbers = add_row_numbers(remaining_source)
    
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_row_numbers(remaining_source.copy())
    
    if on_data_change:
        on_data_change(remaining_source.copy())
    
    save_dataframe(st.session_state[deleted_df_name], f"{deleted_df_name}.pkl")
    if 'audit_deletes_log_trade' in st.session_state and not st.session_state.audit_deletes_log_trade.empty:
        save_dataframe(st.session_state.audit_deletes_log_trade, "audit_deletes_log_trade.pkl")
    
    update_deleted_stats_cards_trade()
    
    return remaining_source_with_numbers, len(selected_record_ids)

def clear_selection_state(key_prefix):
    """Clear selection state for a given dataframe"""
    selection_key = f"{key_prefix}_selection_state"
    if selection_key in st.session_state:
        st.session_state[selection_key] = {}
    logger.debug(f"Cleared selection state for {key_prefix}")

def update_moved_stats_cards_trade():
    """Update the statistics for moved records cards"""
    moved_counts = {
        'moved_buy_matched': 0,
        'moved_buy_unmatched': 0,
        'moved_sell_matched': 0,
        'moved_sell_unmatched': 0,
        'moved_bank_records_trade': 0,
        'total_moved': 0
    }
    
    for key in moved_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            moved_counts[key] = len(st.session_state[key])
    
    moved_counts['total_moved'] = sum([
        moved_counts['moved_buy_matched'],
        moved_counts['moved_buy_unmatched'],
        moved_counts['moved_sell_matched'],
        moved_counts['moved_sell_unmatched'],
        moved_counts['moved_bank_records_trade']
    ])
    
    st.session_state.moved_stats_trade = moved_counts
    return moved_counts

def update_deleted_stats_cards_trade():
    """Update the statistics for deleted records cards"""
    deleted_counts = {
        'deleted_buy_matched': 0,
        'deleted_buy_unmatched': 0,
        'deleted_sell_matched': 0,
        'deleted_sell_unmatched': 0,
        'deleted_bank_trade': 0,
        'total_deleted': 0
    }
    
    for key in deleted_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            deleted_counts[key] = len(st.session_state[key])
    
    deleted_counts['total_deleted'] = sum([
        deleted_counts['deleted_buy_matched'],
        deleted_counts['deleted_buy_unmatched'],
        deleted_counts['deleted_sell_matched'],
        deleted_counts['deleted_sell_unmatched'],
        deleted_counts['deleted_bank_trade']
    ])
    
    st.session_state.deleted_stats_trade = deleted_counts
    return deleted_counts

def sync_all_display_dataframes_trade():
    """Synchronize all display dataframes with their original versions"""
    for key in list(st.session_state.keys()):
        if key.endswith('_display_df'):
            base_key = key.replace('_display_df', '')
            if base_key in st.session_state and not st.session_state[base_key].empty:
                st.session_state[key] = add_row_numbers(st.session_state[base_key].copy())
    logger.debug("Synchronized all display dataframes")

def refresh_analytics_dataframes_trade():
    """Refresh analytics dataframes from current session state"""
    analytics_dataframes = [
        ('matched_buy_df', 'matched_buy_analytics'),
        ('matched_sell_df', 'matched_sell_analytics'),
        ('unmatched_buy_df', 'unmatched_buy_analytics'),
        ('unmatched_sell_df', 'unmatched_sell_analytics'),
        ('unmatched_bank_trade', 'unmatched_bank_analytics')
    ]
    
    for session_key, df_key in analytics_dataframes:
        if session_key in st.session_state and not st.session_state[session_key].empty:
            st.session_state[df_key] = st.session_state[session_key].copy()
    
    logger.debug("Refreshed analytics dataframes")

# --- File Operations ---
def save_uploaded_file(file, filename):
    print("saving fx uploaded data : ", filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def save_dataframe(df, filename):
    if df is not None and not df.empty:
        df.to_pickle(os.path.join(CACHE_DIR, filename))
        logger.debug(f"Saved dataframe to {filename}")

def load_dataframe(filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            df = pd.read_pickle(path)
            logger.debug(f"Loaded dataframe from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_object(obj, filename):
    with open(os.path.join(CACHE_DIR, filename), 'wb') as f:
        pickle.dump(obj, f)
    logger.debug(f"Saved object to {filename}")

def load_object(filename, default=None):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
                logger.debug(f"Loaded object from {filename}")
                return obj
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return default
    return default

# --- Render Moved Records Tab ---
def render_moved_records_tab_trade():
    """Render a tab that shows all moved records with audit trail"""
    st.markdown("### 📋 Moved Records - Audit Trail")
    st.markdown("This section shows all records that have been moved between dataframes with their audit trail.")
    
    moved_stats = update_moved_stats_cards_trade()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📋 Moved Buy Matched", moved_stats['moved_buy_matched'])
    with col2:
        st.metric("⚠️ Moved Buy Unmatched", moved_stats['moved_buy_unmatched'])
    with col3:
        st.metric("📋 Moved Sell Matched", moved_stats['moved_sell_matched'])
    with col4:
        st.metric("⚠️ Moved Sell Unmatched", moved_stats['moved_sell_unmatched'])
    with col5:
        st.metric("🏦 Moved Bank Records", moved_stats['moved_bank_records_trade'])
    with col6:
        st.metric("📊 Total Moved", moved_stats['total_moved'])
    
    st.markdown("---")
    
    moved_df_names = [
        'moved_buy_matched', 'moved_buy_unmatched', 'moved_sell_matched',
        'moved_sell_unmatched', 'moved_bank_records_trade'
    ]
    
    moved_dfs = {}
    for df_name in moved_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            df_copy = st.session_state[df_name].copy()
            if 'moved_at' in df_copy.columns:
                df_copy['moved_at'] = pd.to_datetime(df_copy['moved_at'], errors='coerce')
            moved_dfs[df_name] = df_copy
    
    if not moved_dfs:
        st.info("No moved records found.")
        return
    
    tabs = st.tabs([name.replace('_', ' ').title() for name in moved_dfs.keys()])
    
    for tab, (df_name, df) in zip(tabs, moved_dfs.items()):
        with tab:
            st.markdown(f"#### {df_name.replace('_', ' ').title()} - {len(df)} records")
            
            if 'moved_by' in df.columns:
                user_counts = df['moved_by'].value_counts().head(10)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Users who moved records:**")
                    st.dataframe(user_counts.reset_index().rename(columns={'index': 'User', 'moved_by': 'Count'}))
                with col2:
                    st.markdown("**Recent moves:**")
                    if 'moved_at' in df.columns:
                        df_sorted = df.copy()
                        df_sorted['moved_at'] = pd.to_datetime(df_sorted['moved_at'], errors='coerce')
                        df_sorted = df_sorted.dropna(subset=['moved_at'])
                        if not df_sorted.empty:
                            recent = df_sorted.sort_values('moved_at', ascending=False).head(10)
                            display_cols = ['moved_at', 'moved_by', 'moved_from', 'move_reason']
                            display_cols = [col for col in display_cols if col in recent.columns]
                            st.dataframe(recent[display_cols])
            
            st.markdown("---")
            st.markdown("**Detailed Moved Records:**")
            
            display_df = df.copy()
            if 'moved_at' in display_df.columns:
                display_df['moved_at'] = pd.to_datetime(display_df['moved_at'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            cols_to_drop = ['_record_id']
            for col in cols_to_drop:
                if col in display_df.columns:
                    display_df = display_df.drop(columns=[col])
            
            audit_cols = ['moved_at', 'moved_by', 'moved_from', 'moved_to', 'move_reason', 'move_type']
            existing_audit_cols = [col for col in audit_cols if col in display_df.columns]
            other_cols = [col for col in display_df.columns if col not in existing_audit_cols + ['#']]
            display_df = display_df[existing_audit_cols + other_cols]
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {df_name} as CSV",
                data=csv,
                file_name=f"{df_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{df_name}"
            )

# --- Render Deleted Records Tab ---
def render_deleted_records_tab_trade():
    """Render a tab that shows all deleted records with audit trail"""
    st.markdown("### 🗑️ Deleted Records - Audit Trail")
    st.markdown("This section shows all records that have been deleted from dataframes with their audit trail.")
    
    deleted_stats = update_deleted_stats_cards_trade()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🗑️ Deleted Buy Matched", deleted_stats['deleted_buy_matched'])
    with col2:
        st.metric("🗑️ Deleted Buy Unmatched", deleted_stats['deleted_buy_unmatched'])
    with col3:
        st.metric("🗑️ Deleted Sell Matched", deleted_stats['deleted_sell_matched'])
    with col4:
        st.metric("🗑️ Deleted Sell Unmatched", deleted_stats['deleted_sell_unmatched'])
    with col5:
        st.metric("🗑️ Deleted Bank Records", deleted_stats['deleted_bank_trade'])
    with col6:
        st.metric("📊 Total Deleted", deleted_stats['total_deleted'])
    
    st.markdown("---")
    
    deleted_df_names = [
        'deleted_buy_matched', 'deleted_buy_unmatched', 'deleted_sell_matched',
        'deleted_sell_unmatched', 'deleted_bank_trade'
    ]
    
    deleted_dfs = {}
    for df_name in deleted_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            df_copy = st.session_state[df_name].copy()
            if 'deleted_at' in df_copy.columns:
                df_copy['deleted_at'] = pd.to_datetime(df_copy['deleted_at'], errors='coerce')
            deleted_dfs[df_name] = df_copy
    
    if not deleted_dfs:
        st.info("No deleted records found.")
        return
    
    tabs = st.tabs([name.replace('_', ' ').title() for name in deleted_dfs.keys()])
    
    for tab, (df_name, df) in zip(tabs, deleted_dfs.items()):
        with tab:
            st.markdown(f"#### {df_name.replace('_', ' ').title()} - {len(df)} records")
            
            if 'deleted_by' in df.columns:
                user_counts = df['deleted_by'].value_counts().head(10)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Users who deleted records:**")
                    st.dataframe(user_counts.reset_index().rename(columns={'index': 'User', 'deleted_by': 'Count'}))
                with col2:
                    st.markdown("**Recent deletes:**")
                    if 'deleted_at' in df.columns:
                        df_sorted = df.copy()
                        df_sorted['deleted_at'] = pd.to_datetime(df_sorted['deleted_at'], errors='coerce')
                        df_sorted = df_sorted.dropna(subset=['deleted_at'])
                        if not df_sorted.empty:
                            recent = df_sorted.sort_values('deleted_at', ascending=False).head(10)
                            display_cols = ['deleted_at', 'deleted_by', 'deleted_from', 'delete_reason']
                            display_cols = [col for col in display_cols if col in recent.columns]
                            st.dataframe(recent[display_cols])
            
            st.markdown("---")
            st.markdown("**Detailed Deleted Records:**")
            
            display_df = df.copy()
            if 'deleted_at' in display_df.columns:
                display_df['deleted_at'] = pd.to_datetime(display_df['deleted_at'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            cols_to_drop = ['_record_id']
            for col in cols_to_drop:
                if col in display_df.columns:
                    display_df = display_df.drop(columns=[col])
            
            audit_cols = ['deleted_at', 'deleted_by', 'deleted_from', 'delete_reason']
            existing_audit_cols = [col for col in audit_cols if col in display_df.columns]
            other_cols = [col for col in display_df.columns if col not in existing_audit_cols + ['#']]
            display_df = display_df[existing_audit_cols + other_cols]
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {df_name} as CSV",
                data=csv,
                file_name=f"{df_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{df_name}"
            )

# --- Render Editable Dataframe (similar to fx_reconciliation_app) ---
def render_editable_dataframe_trade(df, title, key_prefix, on_data_change=None, show_delete=True, show_move=True, move_targets=None):
    """Render a single editable dataframe with full functionality"""
    print(f"\n{'='*60}")
    print(f"🖥️ RENDER_EDITABLE_DATAFRAME called")
    print(f"   Title: {title}")
    print(f"   Key Prefix: {key_prefix}")
    print(f"   Input df shape: {df.shape if df is not None else 'None'}")
    print(f"{'='*60}")
    
    if df is None or df.empty:
        st.info(f"No {title} to display.")
        return df if df is not None else pd.DataFrame()
    
    st.markdown(f"### {title}")
    st.markdown(f"**Total Records: {len(df)}**")
    
    df = ensure_record_ids(df)
    df = add_audit_columns(df)
    
    display_df_key = f"{key_prefix}_display_df"
    original_df_key = key_prefix
    
    if display_df_key not in st.session_state:
        if '#' not in df.columns:
            st.session_state[display_df_key] = add_row_numbers(df.copy())
        else:
            st.session_state[display_df_key] = df.copy()
        if original_df_key not in st.session_state:
            st.session_state[original_df_key] = remove_row_numbers(df.copy())
    
    action_reason = st.text_input(
        "Action Reason (optional):",
        key=f"{key_prefix}_action_reason",
        placeholder="Enter reason for moving or deleting these records..."
    )
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.markdown("**Select rows to delete/move:**")
    
    with col2:
        if show_delete and st.button(f"🗑️ Delete Selected", key=f"{key_prefix}_delete_btn"):
            selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
            selected_record_ids = [
                record_id for record_id, is_selected in selection_state.items() 
                if is_selected and record_id.startswith(f"{key_prefix}_select_")
            ]
            selected_ids = [rid.replace(f"{key_prefix}_select_", "") for rid in selected_record_ids]
            
            if selected_ids:
                source_df = st.session_state[display_df_key].copy()
                
                updated_df, deleted_count = delete_selected_rows_with_audit(
                    source_df, selected_ids, title, action_reason,
                    df_name=display_df_key, on_data_change=on_data_change
                )
                
                if original_df_key in st.session_state:
                    original_updated = remove_row_numbers(updated_df.copy())
                    st.session_state[original_df_key] = original_updated
                
                sync_all_display_dataframes_trade()
                clear_selection_state(key_prefix)
                refresh_analytics_dataframes_trade()
                update_deleted_stats_cards_trade()
                
                st.success(f"✅ Deleted {deleted_count} record(s) - Audit trail created")
                st.rerun()
            else:
                st.warning("No rows selected for deletion")
    
    with col3:
        if show_move and move_targets:
            if st.button(f"➡️ Move Selected", key=f"{key_prefix}_move_btn"):
                selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
                selected_record_ids = [
                    record_id for record_id, is_selected in selection_state.items() 
                    if is_selected and record_id.startswith(f"{key_prefix}_select_")
                ]
                selected_ids = [rid.replace(f"{key_prefix}_select_", "") for rid in selected_record_ids]
                
                if selected_ids:
                    target_selection_key = f"{key_prefix}_selected_target"
                    selected_target = st.session_state.get(target_selection_key, list(move_targets.keys())[0] if move_targets else None)
                    
                    if selected_target and selected_target in move_targets:
                        source_key = key_prefix
                        source_df = st.session_state.get(source_key, pd.DataFrame()).copy()
                        source_df = ensure_record_ids(source_df)
                        
                        moved_records, new_source = move_records_to_new_df(
                            source_df, selected_ids, title, selected_target, action_reason
                        )
                        
                        if not moved_records.empty:
                            moved_df_name = get_moved_df_name(title, selected_target)
                            
                            if moved_df_name not in st.session_state:
                                st.session_state[moved_df_name] = moved_records
                            else:
                                existing = st.session_state[moved_df_name]
                                existing_ids = set(existing['_record_id'].tolist()) if not existing.empty else set()
                                new_records = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_records.empty:
                                    st.session_state[moved_df_name] = pd.concat([existing, new_records], ignore_index=True)
                            
                            if 'audit_moves_log_trade' not in st.session_state:
                                st.session_state.audit_moves_log_trade = moved_records[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type']].copy() if 'move_type' in moved_records.columns else moved_records[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason']].copy()
                            else:
                                existing_log = st.session_state.audit_moves_log_trade
                                existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
                                new_log_entries = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_log_entries.empty:
                                    st.session_state.audit_moves_log_trade = pd.concat([existing_log, new_log_entries[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type'] if 'move_type' in new_log_entries.columns else ['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason']]], ignore_index=True)
                            
                            st.session_state[source_key] = new_source
                            st.session_state[display_df_key] = add_row_numbers(new_source)
                            
                            if on_data_change:
                                on_data_change(new_source)
                            
                            clear_selection_state(key_prefix)
                            refresh_analytics_dataframes_trade()
                            update_moved_stats_cards_trade()
                            
                            save_dataframe(st.session_state[moved_df_name], f"{moved_df_name}.pkl")
                            if 'audit_moves_log_trade' in st.session_state and not st.session_state.audit_moves_log_trade.empty:
                                save_dataframe(st.session_state.audit_moves_log_trade, "audit_moves_log_trade.pkl")
                            
                            st.success(f"✅ Moved {len(selected_ids)} record(s) to {moved_df_name}")
                            st.rerun()
                    else:
                        st.warning("Please select a target from the dropdown above")
                else:
                    st.warning("No rows selected for moving")
    
    with col4:
        df_download = st.session_state[display_df_key].copy()
        if '#' in df_download.columns:
            df_download = df_download.drop(columns=['#'])
        if '_record_id' in df_download.columns:
            df_download = df_download.drop(columns=['_record_id'])
        
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"{key_prefix}_download"
        )
    
    with col5:
        if st.button(f"🔄 Refresh", key=f"{key_prefix}_refresh"):
            sync_all_display_dataframes_trade()
            clear_selection_state(key_prefix)
            st.rerun()
    
    with st.container():
        st.markdown("---")
        st.markdown("### Edit Data Directly")
        st.info("💡 Tip: Double-click any cell to edit its content. Use checkboxes below for batch operations.")
        
        df_for_edit = st.session_state[display_df_key].copy()
        columns_to_drop = []
        if '#' in df_for_edit.columns:
            columns_to_drop.append('#')
        if '_record_id' in df_for_edit.columns:
            columns_to_drop.append('_record_id')
        
        if columns_to_drop:
            df_for_edit_for_display = df_for_edit.drop(columns=columns_to_drop)
        else:
            df_for_edit_for_display = df_for_edit
        
        edited_df = st.data_editor(
            df_for_edit_for_display,
            use_container_width=True,
            height=min(400, len(df_for_edit_for_display) * 35 + 38),
            key=f"{key_prefix}_data_editor_{datetime.now().timestamp()}",
            num_rows="dynamic"
        )
        
        if not edited_df.equals(df_for_edit_for_display):
            edited_with_ids = ensure_record_ids(edited_df.copy())
            edited_with_audit = add_audit_columns(edited_with_ids)
            updated_with_numbers = add_row_numbers(edited_with_audit)
            st.session_state[display_df_key] = updated_with_numbers
            
            if original_df_key in st.session_state:
                st.session_state[original_df_key] = remove_row_numbers(edited_with_audit.copy())
            
            if on_data_change:
                on_data_change(remove_row_numbers(edited_with_audit.copy()))
            
            refresh_analytics_dataframes_trade()
            st.success("✅ Data updated!")
            st.rerun()
        
        st.markdown("### Select Rows for Batch Operations")
        
        if show_move and move_targets:
            st.markdown("#### Move Target Selection")
            target_options = list(move_targets.keys())
            selected_target = st.selectbox(
                "Select target dataframe for moving records:",
                options=target_options,
                key=f"{key_prefix}_selected_target"
            )
            
            if selected_target and selected_target in move_targets:
                target_key = move_targets[selected_target]
                target_df = st.session_state.get(target_key, pd.DataFrame())
                st.info(f"📌 Moving to: {selected_target} (currently {len(target_df)} records)")
                st.caption("Note: Moved records will be stored in separate audit dataframes.")
            
            st.markdown("---")
        
        selection_key = f"{key_prefix}_selection_state"
        if selection_key not in st.session_state:
            st.session_state[selection_key] = {}
        
        df_for_selection = st.session_state[display_df_key].copy()
        
        if '_record_id' in df_for_selection.columns:
            record_ids = df_for_selection['_record_id'].tolist()
        else:
            df_for_selection = ensure_record_ids(df_for_selection)
            record_ids = df_for_selection['_record_id'].tolist()
            st.session_state[display_df_key] = add_row_numbers(df_for_selection)
            st.session_state[original_df_key] = remove_row_numbers(df_for_selection.copy())
        
        for idx in range(len(df_for_selection)):
            col1_check, col2_content = st.columns([0.1, 0.9])
            
            row_num = df_for_selection.iloc[idx]['#'] if '#' in df_for_selection.columns else idx + 1
            record_id = record_ids[idx]
            checkbox_key = f"{key_prefix}_select_{record_id}"
            
            is_selected = st.session_state[selection_key].get(checkbox_key, False)
            
            if col1_check.checkbox("", value=is_selected, key=checkbox_key):
                st.session_state[selection_key][checkbox_key] = True
            else:
                st.session_state[selection_key][checkbox_key] = False
            
            with col2_content:
                row_summary = []
                for col in df_for_selection.columns:
                    if col not in ['#', '_record_id']:
                        val = df_for_selection.iloc[idx][col]
                        if pd.notna(val):
                            str_val = str(val)
                            if len(str_val) > 50:
                                str_val = str_val[:47] + "..."
                            row_summary.append(f"**{col}:** {str_val}")
                
                if row_summary:
                    st.markdown(f"**Row {row_num}:** " + " | ".join(row_summary[:3]))
                    if len(row_summary) > 3:
                        with st.expander(f"Show all columns for row {row_num}"):
                            for item in row_summary:
                                st.markdown(item)
        
        selected_count = sum(1 for v in st.session_state[selection_key].values() if v)
        if selected_count > 0:
            st.success(f"✅ {selected_count} row(s) selected for batch operations")
            if show_move and move_targets:
                current_target = st.session_state.get(f"{key_prefix}_selected_target", "Not selected")
                st.info(f"📌 These rows will be moved to audit dataframe for: **{current_target}**")
    
    result_df = st.session_state[display_df_key].copy()
    if '_record_id' in result_df.columns and '#' in result_df.columns:
        result_df = result_df.drop(columns=['_record_id', '#'])
    elif '_record_id' in result_df.columns:
        result_df = result_df.drop(columns=['_record_id'])
    elif '#' in result_df.columns:
        result_df = result_df.drop(columns=['#'])
    
    return result_df



# [Keep all the existing reconciliation logic - parse_date, safe_float, convert_currency, 
# get_fx_rate, normalize_bank_key, resolve_amount_column, resolve_date_column, 
# get_description_columns, process_fx_match, etc.]


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
        return abs(float(cleaned_x))
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
        'uba': 'UBA', # Added UBA, assuming it should be capitalized
        'yeepay' : 'Yeepay', # Added Yeepay, assuming it should be capitalized
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


#--- Main Application Function ---

# --- Core Matching Logic ---

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
    debug_mode: bool = False,
    already_matched_fx_trades: set = None,
    skipped_bank_records: dict = None,
    matched_bank_keys: set = None
) -> list or None:
    """Matches one FX trade against all potential bank statement records (can be multiple)."""

    # Initialize tracking sets if not provided
    if already_matched_fx_trades is None:
        already_matched_fx_trades = set()
    if skipped_bank_records is None:
        skipped_bank_records = {}
    if matched_bank_keys is None:
        matched_bank_keys = set()

    # Extract unique identifier for this FX trade
    fx_trade_id = fx_row.get('Trade ID', '')
    if not fx_trade_id:
        fx_trade_id = f"{fx_row.get('Created At', '')}_{fx_row.get(fx_amount_field, '')}_{fx_row.get(bank_currency_info_field, '')}"

    # Check if this FX trade has already been matched
    if fx_trade_id in already_matched_fx_trades:
        if debug_mode:
            st.info(f"⏭️  Skipping already matched FX trade: {fx_trade_id}")
        return None

    amount = safe_float(fx_row.get(fx_amount_field))
    if amount is None or action_type not in ['Bank Buy', 'Bank Sell']:
        if debug_mode:
            st.error(f"DEBUG: Skipping FX row due to invalid amount ({amount}) or action type ({action_type}).")
        return None

    parsed_date = fx_row.get('Created At')
    if parsed_date and not isinstance(parsed_date, datetime):
        parsed_date = parse_date(str(parsed_date))
    if not isinstance(parsed_date, datetime):
        if debug_mode:
            st.error(f"DEBUG: Skipping FX row due to unparseable 'Created At' date: {fx_row.get('Created At')}.")
        return None

    # Extract FX row details for tracking
    fx_details = {
        'Vendor ID': fx_row.get('Vendor ID'),
        'Vendor Name': fx_row.get('Vendor Name'),
        'Counterparty Dealer': fx_row.get('Counterparty Dealer'),
        'FX Trade ID': fx_trade_id,
        'FX Reference': fx_row.get('Reference'),
        'FX Created At': parsed_date.strftime('%Y-%m-%d') if parsed_date else None,
        'FX Amount': amount,
        'Source Column': bank_currency_info_field,
        'Action Type': action_type
    }

    counterparty_raw = str(fx_row.get(bank_currency_info_field, '')).strip()
    parts = counterparty_raw.split('-')
    if len(parts) < 2:
        unmatched_record = {
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': f"N/A ({counterparty_raw})",
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'Invalid Bank/Currency Info in FX Trade',
            **fx_details  # Include all FX details
        }
        unmatched_list.append(unmatched_record)
        return None

    trade_bank_name_raw = parts[0].strip()
    trade_currency = parts[1].strip().upper()
    normalized_trade_bank_name = normalize_bank_key(trade_bank_name_raw, debug_mode)
    expected_bank_key = f"{normalized_trade_bank_name} {trade_currency}"

    if expected_bank_key not in all_bank_dfs:
        unmatched_record = {
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': expected_bank_key,
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'No Matching Bank Statement File Found',
            **fx_details  # Include all FX details
        }
        unmatched_list.append(unmatched_record)
        return None

    bank_df = all_bank_dfs[expected_bank_key]
    bank_df_columns = bank_df.columns.tolist()
    bank_currency = expected_bank_key.split(' ')[1].upper() if ' ' in expected_bank_key else "UNKNOWN"

    # NEW: Initialize Skipped column if not exists
    if 'Skipped_By_FX_Trades' not in bank_df.columns:
        bank_df['Skipped_By_FX_Trades'] = ""

    date_column = 'Date'
    amount_column = resolve_amount_column(bank_df_columns, action_type, bank_currency)
    if date_column not in bank_df.columns or not amount_column or amount_column not in bank_df.columns:
        unmatched_record = {
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': expected_bank_key,
            'Action Type': action_type,
            'Amount': amount,
            'Status': 'Missing Required Columns in Bank Statement',
            **fx_details  # Include all FX details
        }
        unmatched_list.append(unmatched_record)
        return None

    # Filter bank rows within date tolerance window
    date_matches = bank_df[
        bank_df['Date'].dt.date.between(
            parsed_date.date() - pd.Timedelta(days=date_tolerance_days),
            parsed_date.date() + pd.Timedelta(days=date_tolerance_days)
        )
    ]

    matched_records = []
    skipped_records = []  # NEW: Track skipped bank records

    for idx, bank_row in date_matches.iterrows():
        bank_amt = safe_float(bank_row.get(amount_column))
        if bank_amt is None:
            continue

        converted_amount = convert_currency(amount, trade_currency, bank_currency, parsed_date)
        amount_diff = abs(abs(bank_amt) - abs(converted_amount)) if converted_amount is not None else float('inf')

        if converted_amount and abs(converted_amount) > 0.01 and amount_diff < 0.05:
            # Create bank record key for tracking
            bank_record_key_operation = 'debit' if 'debit' in amount_column.lower() or bank_amt < 0 else 'credit'
            if 'credit' in amount_column.lower():
                bank_record_key_operation = 'credit'
            
            bank_record_key = (
                expected_bank_key,
                bank_row[date_column].strftime('%Y-%m-%d') if hasattr(bank_row[date_column], 'strftime') else str(bank_row[date_column]),
                round(bank_amt, 2),
                bank_record_key_operation
            )

            # Check if this bank record is already matched
            is_already_matched = bank_record_key in matched_bank_keys

            if is_already_matched:
                # Mark as skipped
                if debug_mode:
                    st.warning(f"⚠️ Bank record {bank_record_key} already matched, marking as skipped for FX trade {fx_trade_id}")
                
                # Mark this bank record as skipped by this FX trade
                current_skipped = bank_df.loc[idx, "Skipped_By_FX_Trades"]
                skipped_list = []
                if current_skipped and current_skipped != "":
                    try:
                        skipped_list = json.loads(current_skipped)
                    except:
                        skipped_list = []
                
                # Add FX trade to skipped list
                skipped_info = {
                    'fx_trade_id': fx_trade_id,
                    'fx_date': parsed_date.strftime('%Y-%m-%d'),
                    'fx_amount': amount,
                    'fx_action_type': action_type,
                    'skipped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'match_details': {
                        'amount_difference': amount_diff,
                        'converted_amount': converted_amount,
                        'bank_amount': bank_amt,
                        'amount_column': amount_column
                    }
                }
                skipped_list.append(skipped_info)
                bank_df.loc[idx, "Skipped_By_FX_Trades"] = json.dumps(skipped_list)
                
                # Track in skipped_bank_records
                if fx_trade_id not in skipped_bank_records:
                    skipped_bank_records[fx_trade_id] = []
                skipped_records.append({
                    'bank_key': bank_record_key,
                    'bank_table': expected_bank_key,
                    'bank_date': bank_row[date_column].strftime('%Y-%m-%d') if hasattr(bank_row[date_column], 'strftime') else str(bank_row[date_column]),
                    'bank_amount': bank_amt,
                    'bank_row_index': idx,
                    'match_details': {
                        'amount_difference': amount_diff,
                        'converted_amount': converted_amount,
                        'bank_amount': bank_amt,
                        'amount_column': amount_column
                    }
                })
                
                continue  # Skip to next potential match

            # If we get here, this is a valid unmatched bank record - proceed with matching
            matched_records.append({
                'Bank Index': idx,
                'Bank Date': bank_row.get(date_column).strftime('%Y-%m-%d') if bank_row.get(date_column) else None,
                'Description': str(bank_row.get('Description', '')).strip(),
                'Debit': safe_float(bank_row.get('Debit')),
                'Credit': safe_float(bank_row.get('Credit')),
                'Matched Column': amount_column,
                'Bank Amount': bank_amt,
                'Bank Record Key': bank_record_key,  # NEW: Store for tracking
                'Amount Difference': amount_diff,
                'Converted Amount': converted_amount
            })

            # Mark bank record as matched
            bank_df.at[idx, "Matched"] = True
            matched_bank_keys.add(bank_record_key)

            if debug_mode:
                st.info(f"✅ Sub-Match Found: Bank[{idx}] {bank_amt:.2f} {bank_currency} "
                        f"≈ FX {amount:.2f} {trade_currency} (Converted {converted_amount:.2f})")

    if matched_records:
        # Convert complex objects to JSON strings for PyArrow compatibility
        all_matched_records_json = json.dumps(matched_records) if matched_records else ""
        skipped_records_json = json.dumps(skipped_records) if skipped_records else ""

        # Create base matched record with all FX details
        matched_record = {
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table': expected_bank_key,
            'Action Type': action_type,
            'Trade Amount': amount,
            'Trade Currency': trade_currency,
            'Bank Statement Currency': bank_currency,
            'Converted Trade Amount': converted_amount,
            'Total Bank Matches': len(matched_records),
            'Skipped Bank Records': len(skipped_records),  # NEW: Track skipped count

            # Flattened first match (for CSV friendliness)
            'Matched Bank Record Index': matched_records[0]['Bank Index'],
            'Matched Bank Record Date': matched_records[0]['Bank Date'],
            'Matched Bank Description': matched_records[0]['Description'],
            'Matched Bank Debit': matched_records[0]['Debit'],
            'Matched Bank Credit': matched_records[0]['Credit'],

            # JSON strings for complex objects (PyArrow compatible)
            'All Matched Bank Records': all_matched_records_json,
            
            # NEW: Include skipped records info as JSON string
            'Skipped Bank Records Info': skipped_records_json,
            
            # Add all FX row details
            **fx_details
        }

        matched_list.append(matched_record)

        # MARK THIS FX TRADE AS MATCHED
        already_matched_fx_trades.add(fx_trade_id)

        if debug_mode:
            st.success(f"✅ FX {amount:.2f} {trade_currency} matched {len(matched_records)} bank entries in '{expected_bank_key}' (skipped: {len(skipped_records)}).")

        return [(expected_bank_key, m['Bank Index']) for m in matched_records]

    # If none matched but there were skipped records
    if skipped_records:
        # Convert skipped records to JSON string
        skipped_records_json = json.dumps(skipped_records) if skipped_records else ""
        
        unmatched_record = {
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank Table (Expected)': expected_bank_key,
            'Action Type': action_type,
            'Amount': amount,
            'Status': f'Potential matches found but already taken by other trades (skipped: {len(skipped_records)})',
            'Skipped Bank Records': skipped_records_json,  # NEW: Include skipped records details as JSON
            **fx_details  # Include all FX details
        }
        unmatched_list.append(unmatched_record)

        if debug_mode:
            st.warning(f"⚠️ FX {amount:.2f} {trade_currency} had {len(skipped_records)} potential matches but all were already taken in {expected_bank_key}.")
        return None

    # If none matched and no skipped records
    unmatched_record = {
        'Date': parsed_date.strftime('%Y-%m-%d'),
        'Bank Table (Expected)': expected_bank_key,
        'Action Type': action_type,
        'Amount': amount,
        'Status': 'No Bank Statement Match (Amount or Date Tolerance)',
        **fx_details  # Include all FX details
    }
    unmatched_list.append(unmatched_record)

    if debug_mode:
        st.warning(f"⚠️ No matches found for FX {amount:.2f} {trade_currency} in {expected_bank_key}.")
    return None

# Note: The reconciliation functions remain unchanged from your original file.
# They are omitted here for brevity but should be kept exactly as they were.

# --- Main App Function ---
def graphed_analysis_app(all_bank_dfs: dict):
    st.markdown("# 💱 FX Trade Reconciliation Dashboard")
    
    # ========== INITIALIZE SESSION STATE ==========
    if 'matched_buy_df' not in st.session_state:
        st.session_state.matched_buy_df = pd.DataFrame()
    if 'matched_sell_df' not in st.session_state:
        st.session_state.matched_sell_df = pd.DataFrame()
    if 'unmatched_buy_df' not in st.session_state:
        st.session_state.unmatched_buy_df = pd.DataFrame()
    if 'unmatched_sell_df' not in st.session_state:
        st.session_state.unmatched_sell_df = pd.DataFrame()
    if 'unmatched_bank_trade' not in st.session_state:
        st.session_state.unmatched_bank_trade = pd.DataFrame()
    if 'fx_trade_df' not in st.session_state:
        st.session_state.fx_trade_df = pd.DataFrame()
    
    # Moved records
    if 'moved_buy_matched' not in st.session_state:
        st.session_state.moved_buy_matched = pd.DataFrame()
    if 'moved_buy_unmatched' not in st.session_state:
        st.session_state.moved_buy_unmatched = pd.DataFrame()
    if 'moved_sell_matched' not in st.session_state:
        st.session_state.moved_sell_matched = pd.DataFrame()
    if 'moved_sell_unmatched' not in st.session_state:
        st.session_state.moved_sell_unmatched = pd.DataFrame()
    if 'moved_bank_records_trade' not in st.session_state:
        st.session_state.moved_bank_records_trade = pd.DataFrame()
    
    # Deleted records
    if 'deleted_buy_matched' not in st.session_state:
        st.session_state.deleted_buy_matched = pd.DataFrame()
    if 'deleted_buy_unmatched' not in st.session_state:
        st.session_state.deleted_buy_unmatched = pd.DataFrame()
    if 'deleted_sell_matched' not in st.session_state:
        st.session_state.deleted_sell_matched = pd.DataFrame()
    if 'deleted_sell_unmatched' not in st.session_state:
        st.session_state.deleted_sell_unmatched = pd.DataFrame()
    if 'deleted_bank_trade' not in st.session_state:
        st.session_state.deleted_bank_trade = pd.DataFrame()
    
    # Audit logs
    if 'audit_moves_log_trade' not in st.session_state:
        st.session_state.audit_moves_log_trade = pd.DataFrame()
    if 'audit_deletes_log_trade' not in st.session_state:
        st.session_state.audit_deletes_log_trade = pd.DataFrame()
    
    # Stats
    if 'moved_stats_trade' not in st.session_state:
        st.session_state.moved_stats_trade = {
            'moved_buy_matched': 0, 'moved_buy_unmatched': 0,
            'moved_sell_matched': 0, 'moved_sell_unmatched': 0,
            'moved_bank_records_trade': 0, 'total_moved': 0
        }
    if 'deleted_stats_trade' not in st.session_state:
        st.session_state.deleted_stats_trade = {
            'deleted_buy_matched': 0, 'deleted_buy_unmatched': 0,
            'deleted_sell_matched': 0, 'deleted_sell_unmatched': 0,
            'deleted_bank_trade': 0, 'total_deleted': 0
        }
    
    # Load saved data
    try:
        for key in ['matched_buy_df', 'matched_sell_df', 'unmatched_buy_df', 'unmatched_sell_df', 'unmatched_bank_trade', 'fx_trade_df']:
            loaded_df = load_dataframe(f"{key}.pkl")
            if not loaded_df.empty:
                st.session_state[key] = loaded_df
                st.session_state[key] = add_unique_ids(st.session_state[key])
                st.session_state[key] = add_audit_columns(st.session_state[key])
    except Exception as e:
        logger.error(f"Error loading saved data: {e}")
    
    update_moved_stats_cards_trade()
    update_deleted_stats_cards_trade()
    
    # Load FX trade data from pickle
    st.session_state.fx_trade_tracker_df = load_dataframe("fx_trade_tracker_df.pkl")
    st.session_state.fx_trade_tracker_sheet = load_object("fx_trade_tracker_sheet.pkl")
    st.session_state.fx_trade_tracker_col_mapping = load_object("fx_trade_tracker_col_mapping.pkl", {
        'Action Type': 'Action Type', 'Status': 'Status', 'Created At': 'Created At',
        'Buy Currency Amount': 'Buy Currency Amount', 'Buy Trade Info': 'Buy Trade Info',
        'Sell Currency Amount': 'Sell Currency Amount', 'Sell Trade Info': 'Sell Trade Info',
        'Vendor ID': 'Vendor ID', 'Vendor Name': 'Vendor Name', 'Counterparty Dealer': 'Counterparty Dealer',
    })
    
    # Colors for styling
    COLORS = {
        'white': '#FFFFFF', 'secondary': '#798088', 'primary': '#361371',
        'pink_alpha': '#9F6AF8CC', 'container_alpha': '#F0EFEF4D',
        'buy_goods_color': '#F5EFFD', 'green': '#2B9973',
        'red': '#E85E5D', 'pink': '#9F6AF8'
    }
    
    st.markdown(f"""
        <style>
            .metric-card {{
                background: white; border-radius: 15px; padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-left: 5px solid {COLORS['pink']}; margin-bottom: 20px;
            }}
            .metric-title {{ font-size: 14px; color: {COLORS['secondary']}; text-transform: uppercase; }}
            .metric-value {{ font-size: 32px; font-weight: bold; color: {COLORS['primary']}; margin: 10px 0; }}
            .stButton>button {{
                background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['pink']});
                color: white; border-radius: 25px; border: none;
                padding: 12px 24px; font-weight: 600; width: 100%;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 💱 FX Trade Reconciliation")
        
        if 'user' in st.session_state:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Logged In User</div>
                <div class="metric-value">{st.session_state['user']['username']}</div>
                <div class="metric-change">Role: {st.session_state['user']['role']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Upload Section
        st.markdown("### 📤 Data Upload")
        
        with st.expander("📊 FX Trade Tracker", expanded=True):
            fx_uploaded_file = st.file_uploader("Upload FX Trade Tracker", type=["csv", "xlsx"], key="fx_uploader")
            
            if fx_uploaded_file:
                try:
                    save_uploaded_file(fx_uploaded_file, "fx_trade_uploaded." + fx_uploaded_file.name.split('.')[-1])
                    
                    if fx_uploaded_file.name.endswith('.xlsx'):
                        xls = pd.ExcelFile(fx_uploaded_file)
                        sheet_names = xls.sheet_names
                        selected_sheet = st.selectbox("Select sheet", sheet_names, key="fx_sheet_selector")
                        fx_trade_df = pd.read_excel(fx_uploaded_file, sheet_name=selected_sheet)
                    else:
                        fx_trade_df = pd.read_csv(fx_uploaded_file)
                    
                    fx_trade_df.columns = fx_trade_df.columns.str.strip()
                    st.dataframe(fx_trade_df.head(3))
                    
                    # Column mapping
                    st.markdown("#### Map Columns")
                    fx_col_options = ['-- Select Column --'] + fx_trade_df.columns.tolist()
                    col_mapping = {}
                    
                    fx_required_cols = {
                        'Action Type': 'Action Type', 'Status': 'Status', 'Created At': 'Created At',
                        'Buy Currency Amount': 'Buy Currency Amount', 'Buy Trade Info': 'Buy Trade Info',
                        'Sell Currency Amount': 'Sell Currency Amount', 'Sell Trade Info': 'Sell Trade Info',
                        'Vendor ID': 'Vendor ID', 'Vendor Name': 'Vendor Name', 'Counterparty Dealer': 'Counterparty Dealer',
                    }
                    
                    for display_name, suggested_col in fx_required_cols.items():
                        initial_selection = st.session_state.fx_trade_tracker_col_mapping.get(display_name, suggested_col if suggested_col in fx_col_options else '-- Select Column --')
                        selected_col = st.selectbox(
                            f"Map '{display_name}'",
                            options=fx_col_options,
                            index=fx_col_options.index(initial_selection) if initial_selection in fx_col_options else 0,
                            key=f"fx_map_{display_name}"
                        )
                        col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None
                    
                    if st.button("✅ Process Data", key="process_fx_btn"):
                        renamed_cols_dict = {selected: original for original, selected in col_mapping.items() if selected and selected in fx_trade_df.columns}
                        if renamed_cols_dict:
                            cols_to_keep = list(renamed_cols_dict.keys())
                            fx_trade_df = fx_trade_df[cols_to_keep].rename(columns=renamed_cols_dict)
                        
                        st.session_state.fx_trade_df = fx_trade_df
                        save_dataframe(fx_trade_df, "fx_trade_df.pkl")
                        st.session_state.fx_trade_tracker_col_mapping = col_mapping
                        save_object(col_mapping, "fx_trade_tracker_col_mapping.pkl")
                        st.success("✅ Data processed successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Reconciliation Controls
        st.markdown("### ⚙️ Reconciliation Settings")
        debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.get('debug_mode', False))
        st.session_state.debug_mode = debug_mode
        
        date_tolerance_days = st.slider("Date Tolerance (± days)", min_value=0, max_value=7, value=3, step=1)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Run Reconciliation", use_container_width=True):
                if not all_bank_dfs:
                    st.error("No bank statements loaded!")
                elif st.session_state.fx_trade_df.empty:
                    st.error("Please upload FX Trade Tracker first!")
                else:
                    with st.spinner("Running reconciliation..."):
                        # Run reconciliation logic here (keep your existing logic)
                        # [Your existing reconciliation code here]
                        pass
        
        with col_btn2:
            if st.button("🗑️ Reset All Data", use_container_width=True):
                for key in PICKLE_TRACKING_KEYS:
                    if key in st.session_state:
                        st.session_state[key] = pd.DataFrame()
                st.success("All data reset!")
                st.rerun()
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Session Stats")
        st.metric("Buy Matched", len(st.session_state.matched_buy_df))
        st.metric("Buy Unmatched", len(st.session_state.unmatched_buy_df))
        st.metric("Sell Matched", len(st.session_state.matched_sell_df))
        st.metric("Sell Unmatched", len(st.session_state.unmatched_sell_df))
        st.metric("Bank Unmatched", len(st.session_state.unmatched_bank_trade))
        
        st.markdown("---")
        st.markdown("### 📋 Audit Stats")
        st.metric("Total Moved", st.session_state.moved_stats_trade['total_moved'])
        st.metric("Total Deleted", st.session_state.deleted_stats_trade['total_deleted'])
    
    # Main content - KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">✅ Buy Matched</div>
            <div class="metric-value">{len(st.session_state.matched_buy_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚠️ Buy Unmatched</div>
            <div class="metric-value">{len(st.session_state.unmatched_buy_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">✅ Sell Matched</div>
            <div class="metric-value">{len(st.session_state.matched_sell_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚠️ Sell Unmatched</div>
            <div class="metric-value">{len(st.session_state.unmatched_sell_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🏦 Bank Unmatched</div>
            <div class="metric-value">{len(st.session_state.unmatched_bank_trade)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Buy Matched", "⚠️ Buy Unmatched", "📋 Sell Matched", "⚠️ Sell Unmatched",
        "🏦 Bank Records", "📊 Audit Trail"
    ])
    
    # Define move targets
    move_targets_buy_matched = {
        "Buy Unmatched": "unmatched_buy_df",
        "Sell Matched": "matched_sell_df",
        "Sell Unmatched": "unmatched_sell_df"
    }
    
    move_targets_buy_unmatched = {
        "Buy Matched": "matched_buy_df",
        "Sell Matched": "matched_sell_df",
        "Sell Unmatched": "unmatched_sell_df"
    }
    
    move_targets_sell_matched = {
        "Buy Matched": "matched_buy_df",
        "Buy Unmatched": "unmatched_buy_df",
        "Sell Unmatched": "unmatched_sell_df"
    }
    
    move_targets_sell_unmatched = {
        "Buy Matched": "matched_buy_df",
        "Buy Unmatched": "unmatched_buy_df",
        "Sell Matched": "matched_sell_df"
    }
    
    with tab1:
        def update_buy_matched(df):
            st.session_state.matched_buy_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_buy_df.empty:
                st.session_state.matched_buy_df = add_audit_columns(st.session_state.matched_buy_df)
            if not df.empty:
                save_dataframe(df, "matched_buy_df.pkl")
        
        render_editable_dataframe_trade(
            st.session_state.matched_buy_df, "Buy Matched Records",
            "matched_buy", on_data_change=update_buy_matched,
            show_delete=True, show_move=True, move_targets=move_targets_buy_matched
        )
    
    with tab2:
        def update_buy_unmatched(df):
            st.session_state.unmatched_buy_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_buy_df.empty:
                st.session_state.unmatched_buy_df = add_audit_columns(st.session_state.unmatched_buy_df)
            if not df.empty:
                save_dataframe(df, "unmatched_buy_df.pkl")
        
        render_editable_dataframe_trade(
            st.session_state.unmatched_buy_df, "Buy Unmatched Records",
            "unmatched_buy", on_data_change=update_buy_unmatched,
            show_delete=True, show_move=True, move_targets=move_targets_buy_unmatched
        )
    
    with tab3:
        def update_sell_matched(df):
            st.session_state.matched_sell_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_sell_df.empty:
                st.session_state.matched_sell_df = add_audit_columns(st.session_state.matched_sell_df)
            if not df.empty:
                save_dataframe(df, "matched_sell_df.pkl")
        
        render_editable_dataframe_trade(
            st.session_state.matched_sell_df, "Sell Matched Records",
            "matched_sell", on_data_change=update_sell_matched,
            show_delete=True, show_move=True, move_targets=move_targets_sell_matched
        )
    
    with tab4:
        def update_sell_unmatched(df):
            st.session_state.unmatched_sell_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_sell_df.empty:
                st.session_state.unmatched_sell_df = add_audit_columns(st.session_state.unmatched_sell_df)
            if not df.empty:
                save_dataframe(df, "unmatched_sell_df.pkl")
        
        render_editable_dataframe_trade(
            st.session_state.unmatched_sell_df, "Sell Unmatched Records",
            "unmatched_sell", on_data_change=update_sell_unmatched,
            show_delete=True, show_move=True, move_targets=move_targets_sell_unmatched
        )
    
    with tab5:
        def update_bank_trade(df):
            st.session_state.unmatched_bank_trade = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_bank_trade.empty:
                st.session_state.unmatched_bank_trade = add_audit_columns(st.session_state.unmatched_bank_trade)
            if not df.empty:
                save_dataframe(df, "unmatched_bank_trade.pkl")
        
        render_editable_dataframe_trade(
            st.session_state.unmatched_bank_trade, "Unmatched Bank Records",
            "bank_trade", on_data_change=update_bank_trade,
            show_delete=True, show_move=False
        )
    
    with tab6:
        st.markdown("### 📋 Audit Trail")
        st.markdown("Track all moved and deleted records")
        
        audit_tab1, audit_tab2 = st.tabs(["📋 Moved Records", "🗑️ Deleted Records"])
        
        with audit_tab1:
            render_moved_records_tab_trade()
        
        with audit_tab2:
            render_deleted_records_tab_trade()
    
    # Return dataframes
    return (
        st.session_state.matched_buy_df if not st.session_state.matched_buy_df.empty else pd.DataFrame(),
        st.session_state.matched_sell_df if not st.session_state.matched_sell_df.empty else pd.DataFrame(),
        st.session_state.unmatched_buy_df if not st.session_state.unmatched_buy_df.empty else pd.DataFrame(),
        st.session_state.unmatched_sell_df if not st.session_state.unmatched_sell_df.empty else pd.DataFrame(),
        st.session_state.unmatched_bank_trade if not st.session_state.unmatched_bank_trade.empty else pd.DataFrame()
    )