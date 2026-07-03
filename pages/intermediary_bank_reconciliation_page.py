# pages/intermediary_bank_reconciliation_page.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
import uuid
import sqlite3
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from auth_system import  log_audit, require_auth

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
DB_PATH = "data/intermediary_reconciliation.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Custom CSS for better UI
CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .custom-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
"""

# Database table names
INTERMEDIARY_TABLES = {
    'matched_credit': 'intermediary_matched_credit',
    'matched_debit': 'intermediary_matched_debit',
    'unmatched_credit': 'intermediary_unmatched_credit',
    'unmatched_debit': 'intermediary_unmatched_debit',
    'unmatched_bank': 'intermediary_unmatched_bank',
    'moved_credit': 'intermediary_moved_credit',
    'moved_debit': 'intermediary_moved_debit',
    'deleted_credit': 'intermediary_deleted_credit',
    'deleted_debit': 'intermediary_deleted_debit',
    'audit_moves': 'intermediary_audit_moves',
    'audit_deletes': 'intermediary_audit_deletes'
}

# Date formats for parsing
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
    '%d/%m/%Y', '%-d/%-m/%Y', '%-d.%-m/%-Y',
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
    '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S', '%-d/%-m/%Y %H:%M:%S',
    '%-d.%-m.%Y %H:%M:%S', "%d.%m.%Y"
]

# Predefined bank-currency combinations
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

# Output paths
out_csv_path_credit_unmatched = 'UnmatchedIntermediaryCredit.csv'
out_csv_path_debit_unmatched = 'UnmatchedIntermediaryDebit.csv'
out_csv_path_bank_unmatched = 'UnmatchedBankRecords_Intermediary.csv'
out_csv_path_credit_matched = 'MatchedIntermediaryCredit.csv'
out_csv_path_debit_matched = 'MatchedIntermediaryDebit.csv'

# Column mapping configuration
EXPECTED_COLUMNS = {
    'Application ID': 'Application ID',
    'Amount': 'Amount',
    'Currency': 'Currency',
    'Intermediary Bank Account - Credit': 'Intermediary Bank Account - Credit',
    'Intermediary Bank Account - Debit': 'Intermediary Bank Account - Debit',
    'Created At': 'Created At',
    'Status': 'Status'
}

# --- Helper Functions ---
def save_uploaded_file(file, filename):
    print("saving intermediary bank uploaded data : ", filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def save_dataframe(df, filename):
    if df is not None and not df.empty:
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

# --- Database Manager Class ---
class IntermediaryDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Matched Credit table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_matched_credit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table TEXT,
                Account_Type TEXT,
                Intermediary_Amount REAL,
                Currency TEXT,
                Total_Bank_Matches INTEGER,
                Skipped_Bank_Records INTEGER,
                Matched_Bank_Record_Index INTEGER,
                Matched_Bank_Record_Date TEXT,
                Matched_Bank_Description TEXT,
                Matched_Bank_Debit REAL,
                Matched_Bank_Credit REAL,
                All_Matched_Bank_Records TEXT,
                Skipped_Bank_Records_Info TEXT,
                Application_ID TEXT,
                Intermediary_Column TEXT,
                Bank_Info_Raw TEXT,
                Status TEXT,
                moved_by TEXT,
                moved_at TEXT,
                move_reason TEXT,
                moved_from TEXT,
                moved_to TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Matched Debit table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_matched_debit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table TEXT,
                Account_Type TEXT,
                Intermediary_Amount REAL,
                Currency TEXT,
                Total_Bank_Matches INTEGER,
                Skipped_Bank_Records INTEGER,
                Matched_Bank_Record_Index INTEGER,
                Matched_Bank_Record_Date TEXT,
                Matched_Bank_Description TEXT,
                Matched_Bank_Debit REAL,
                Matched_Bank_Credit REAL,
                All_Matched_Bank_Records TEXT,
                Skipped_Bank_Records_Info TEXT,
                Application_ID TEXT,
                Intermediary_Column TEXT,
                Bank_Info_Raw TEXT,
                Status TEXT,
                moved_by TEXT,
                moved_at TEXT,
                move_reason TEXT,
                moved_from TEXT,
                moved_to TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Unmatched Credit table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_unmatched_credit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table_Expected TEXT,
                Account_Type TEXT,
                Amount REAL,
                Currency TEXT,
                Status TEXT,
                Skipped_Bank_Records TEXT,
                Application_ID TEXT,
                Intermediary_Column TEXT,
                Bank_Info_Raw TEXT,
                moved_by TEXT,
                moved_at TEXT,
                move_reason TEXT,
                moved_from TEXT,
                moved_to TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Unmatched Debit table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_unmatched_debit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table_Expected TEXT,
                Account_Type TEXT,
                Amount REAL,
                Currency TEXT,
                Status TEXT,
                Skipped_Bank_Records TEXT,
                Application_ID TEXT,
                Intermediary_Column TEXT,
                Bank_Info_Raw TEXT,
                moved_by TEXT,
                moved_at TEXT,
                move_reason TEXT,
                moved_from TEXT,
                moved_to TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Unmatched Bank table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_unmatched_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Bank_Table TEXT,
                Date TEXT,
                Description TEXT,
                Transaction_Type_Column TEXT,
                Amount REAL,
                moved_by TEXT,
                moved_at TEXT,
                move_reason TEXT,
                moved_from TEXT,
                moved_to TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Moved records tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_moved_credit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                source_table TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_to TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                original_record_json TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_moved_debit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                source_table TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_to TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                original_record_json TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Deleted records tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_deleted_credit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                source_table TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                original_record_json TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_deleted_debit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                source_table TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                original_record_json TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Audit logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_audit_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT,
                timestamp TEXT,
                user TEXT,
                record_type TEXT,
                record_id TEXT,
                from_location TEXT,
                to_location TEXT,
                details TEXT,
                import_date TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_audit_deletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT,
                timestamp TEXT,
                user TEXT,
                record_type TEXT,
                record_id TEXT,
                details TEXT,
                deleted_record TEXT,
                import_date TEXT
            )
        ''')
        
        # Metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intermediary_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_intermediary_matched_credit_date ON intermediary_matched_credit(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_intermediary_matched_debit_date ON intermediary_matched_debit(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_intermediary_unmatched_credit_date ON intermediary_unmatched_credit(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_intermediary_unmatched_debit_date ON intermediary_unmatched_debit(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_intermediary_bank_date ON intermediary_unmatched_bank(record_date)',
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except:
                pass
        
        conn.commit()
        conn.close()
        logger.info("Intermediary Bank Reconciliation database initialized")
    
    def _serialize_value(self, value):
        if value is None:
            return None
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        return str(value) if not isinstance(value, (float, int)) else value
    
    def save_dataframe(self, table_name, df, record_date=None):
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"DELETE FROM {table_name} WHERE record_date = ?", (record_date,))
        except Exception as e:
            logger.debug(f"Could not clear {table_name}: {e}")
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared {table_name} for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        records = df.to_dict('records')
        
        for record in records:
            _record_id = str(record.get('_record_id', generate_record_id()))
            
            columns = list(record.keys())
            placeholders = ','.join(['?' for _ in columns])
            columns_str = ','.join([f'"{col}"' for col in columns])
            
            values = []
            for col in columns:
                values.append(self._serialize_value(record.get(col)))
            
            if 'record_date' not in columns:
                columns_str += ',"record_date"'
                placeholders += ',?'
                values.append(record_date)
            if 'import_date' not in columns:
                columns_str += ',"import_date"'
                placeholders += ',?'
                values.append(import_date)
            
            try:
                cursor.execute(f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})", values)
            except Exception as e:
                logger.error(f"Error inserting into {table_name}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to {table_name}")
    
    def load_dataframe(self, table_name, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE record_date = ?", conn, params=(target_date,))
        except Exception as e:
            logger.error(f"Error loading from {table_name}: {e}")
            df = pd.DataFrame()
        conn.close()
        
        if not df.empty:
            cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def save_metadata(self, key, value):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO intermediary_metadata (key, value, updated_at) VALUES (?, ?, ?)',
                    (key, json.dumps(value), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
    
    def load_metadata(self, key, default=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT value FROM intermediary_metadata WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else default
    
    def get_available_dates(self):
        conn = sqlite3.connect(self.db_path)
        dates = set()
        for table in INTERMEDIARY_TABLES.values():
            try:
                cursor = conn.execute(f"SELECT DISTINCT record_date FROM {table} WHERE record_date IS NOT NULL")
                for row in cursor.fetchall():
                    if row[0]:
                        dates.add(row[0])
            except:
                pass
        conn.close()
        return sorted(list(dates), reverse=True)



db = IntermediaryDB()

# --- Helper Functions for Record Management ---
def generate_record_id():
    return str(uuid.uuid4())

def add_unique_ids(df):
    """Add unique _record_id column to dataframe"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    if '_record_id' not in df_copy.columns:
        df_copy['_record_id'] = [generate_record_id() for _ in range(len(df_copy))]
    return df_copy

def ensure_record_ids(df):
    if df is None or df.empty:
        return df
    if '_record_id' not in df.columns:
        return add_unique_ids(df)
    return df

def add_audit_columns(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    audit_cols = ['deleted_by', 'deleted_at', 'delete_reason', 'source_dataframe', 'deleted_from',
                  'moved_by', 'moved_from', 'moved_at', 'move_reason', 'move_type', 'moved_to']
    for col in audit_cols:
        if col not in df_copy.columns:
            df_copy[col] = ''
    if 'moved_at' in df_copy.columns:
        df_copy['moved_at'] = df_copy['moved_at'].astype(str)
    if 'deleted_at' in df_copy.columns:
        df_copy['deleted_at'] = df_copy['deleted_at'].astype(str)
    return df_copy

def add_row_numbers(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '#' in df_copy.columns:
        df_copy = df_copy.drop(columns=['#'])
    df_copy.insert(0, '#', range(1, len(df_copy) + 1))
    return df_copy

def remove_row_numbers(df):
    if df is None or df.empty:
        return df
    if '#' in df.columns:
        return df.drop(columns=['#'])
    return df

def get_current_user():
    if 'user' in st.session_state:
        return st.session_state['user'].get('username', 'unknown')
    return 'unknown_user'

def get_deleted_df_name(source_name):
    source_lower = source_name.lower()
    if 'matched credit' in source_lower or 'unmatched credit' in source_lower:
        return 'deleted_credit_df'
    elif 'matched debit' in source_lower or 'unmatched debit' in source_lower:
        return 'deleted_debit_df'
    elif 'bank' in source_lower:
        return 'deleted_bank_df'
    return f"deleted_{source_lower.replace(' ', '_')}"

def get_moved_df_name(source_name, target_name):
    target_lower = target_name.lower()
    if 'credit' in target_lower:
        return 'moved_credit_df'
    elif 'debit' in target_lower:
        return 'moved_debit_df'
    return f"moved_{target_lower.replace(' ', '_')}"

def move_records_to_new_df(source_df, selected_record_ids, source_name, target_name, move_reason=""):
    if not selected_record_ids:
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
    if not selected_record_ids:
        return pd.DataFrame(), source_df
    
    source_df_copy = source_df.copy() if source_df is not None else pd.DataFrame()
    source_df_copy = ensure_record_ids(source_df_copy)
    if '#' in source_df_copy.columns:
        source_df_copy = source_df_copy.drop(columns=['#'])
    
    selected_records = source_df_copy[source_df_copy['_record_id'].isin(selected_record_ids)].copy()
    remaining_source = source_df_copy[~source_df_copy['_record_id'].isin(selected_record_ids)].reset_index(drop=True)
    
    if selected_records.empty:
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
    
    deleted_records, remaining_source = delete_records_to_new_df(source_df, selected_record_ids, source_name, delete_reason)
    
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
    
    # Save to database
    if deleted_df_name == 'deleted_credit_df':
        db.save_dataframe('intermediary_deleted_credit', st.session_state[deleted_df_name])
    elif deleted_df_name == 'deleted_debit_df':
        db.save_dataframe('intermediary_deleted_debit', st.session_state[deleted_df_name])
    
    if 'audit_deletes_log' not in st.session_state:
        st.session_state.audit_deletes_log = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.audit_deletes_log
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log_entries = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log_entries.empty:
            st.session_state.audit_deletes_log = pd.concat([existing_log, new_log_entries[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)
    
    if not st.session_state.audit_deletes_log.empty:
        db.save_dataframe('intermediary_audit_deletes', st.session_state.audit_deletes_log)
    
    remaining_source_with_numbers = add_row_numbers(remaining_source)
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_row_numbers(remaining_source.copy())
    
    # Update main dataframes
    main_df_mapping = {
        'Matched Credit Records': 'matched_credit_df',
        'Matched Debit Records': 'matched_debit_df',
        'Unmatched Credit Records': 'unmatched_credit_df',
        'Unmatched Debit Records': 'unmatched_debit_df',
        'Unmatched Bank Records': 'unmatched_bank_intermediary'
    }
    if source_name in main_df_mapping:
        main_key = main_df_mapping[source_name]
        if main_key in st.session_state:
            st.session_state[main_key] = remove_row_numbers(remaining_source.copy())
    
    if on_data_change:
        on_data_change(remaining_source.copy())
    
    # Update stats
    update_moved_stats_cards()
    update_deleted_stats_cards()
    
    return remaining_source_with_numbers, len(selected_record_ids)

def clear_selection_state(key_prefix):
    selection_key = f"{key_prefix}_selection_state"
    if selection_key in st.session_state:
        # Clear all checkbox states
        for checkbox_key in list(st.session_state[selection_key].keys()):
            st.session_state[selection_key][checkbox_key] = False

def sync_all_display_dataframes():
    """Sync display dataframes with original data"""
    display_mappings = [
        ('matched_credit_df', 'matched_credit_display_df'),
        ('matched_debit_df', 'matched_debit_display_df'),
        ('unmatched_credit_df', 'unmatched_credit_display_df'),
        ('unmatched_debit_df', 'unmatched_debit_display_df'),
        ('unmatched_bank_intermediary', 'unmatched_bank_display_df')
    ]
    
    for source_key, display_key in display_mappings:
        if source_key in st.session_state and not st.session_state[source_key].empty:
            df_copy = add_row_numbers(st.session_state[source_key].copy())
            st.session_state[display_key] = df_copy
        elif display_key not in st.session_state:
            st.session_state[display_key] = pd.DataFrame()

def refresh_analytics_dataframes():
    """Refresh analytics dataframes from current session state"""
    analytics_dataframes = [
        ('matched_credit_df', 'matched_credit_analytics'),
        ('matched_debit_df', 'matched_debit_analytics'),
        ('unmatched_credit_df', 'unmatched_credit_analytics'),
        ('unmatched_debit_df', 'unmatched_debit_analytics'),
        ('unmatched_bank_intermediary', 'unmatched_bank_analytics')
    ]
    for session_key, df_key in analytics_dataframes:
        if session_key in st.session_state and not st.session_state[session_key].empty:
            st.session_state[df_key] = st.session_state[session_key].copy()

def update_moved_stats_cards():
    moved_counts = {
        'moved_credit': 0,
        'moved_debit': 0,
        'total_moved': 0
    }
    
    if 'moved_credit_df' in st.session_state and not st.session_state.moved_credit_df.empty:
        moved_counts['moved_credit'] = len(st.session_state.moved_credit_df)
    if 'moved_debit_df' in st.session_state and not st.session_state.moved_debit_df.empty:
        moved_counts['moved_debit'] = len(st.session_state.moved_debit_df)
    
    moved_counts['total_moved'] = moved_counts['moved_credit'] + moved_counts['moved_debit']
    st.session_state.moved_stats = moved_counts
    return moved_counts

def update_deleted_stats_cards():
    deleted_counts = {
        'deleted_credit': 0,
        'deleted_debit': 0,
        'total_deleted': 0
    }
    
    if 'deleted_credit_df' in st.session_state and not st.session_state.deleted_credit_df.empty:
        deleted_counts['deleted_credit'] = len(st.session_state.deleted_credit_df)
    if 'deleted_debit_df' in st.session_state and not st.session_state.deleted_debit_df.empty:
        deleted_counts['deleted_debit'] = len(st.session_state.deleted_debit_df)
    
    deleted_counts['total_deleted'] = deleted_counts['deleted_credit'] + deleted_counts['deleted_debit']
    st.session_state.deleted_stats = deleted_counts
    return deleted_counts

def initialize_session_state_intermediary():
    """Initialize all Intermediary Bank related session state variables"""
    
    # Main dataframes
    if 'matched_credit_df' not in st.session_state:
        st.session_state.matched_credit_df = pd.DataFrame()
    if 'matched_debit_df' not in st.session_state:
        st.session_state.matched_debit_df = pd.DataFrame()
    if 'unmatched_credit_df' not in st.session_state:
        st.session_state.unmatched_credit_df = pd.DataFrame()
    if 'unmatched_debit_df' not in st.session_state:
        st.session_state.unmatched_debit_df = pd.DataFrame()
    if 'unmatched_bank_intermediary' not in st.session_state:
        st.session_state.unmatched_bank_intermediary = pd.DataFrame()
    if 'intermediary_raw_df' not in st.session_state:
        st.session_state.intermediary_raw_df = pd.DataFrame()
    
    # Moved records dataframes
    if 'moved_credit_df' not in st.session_state:
        st.session_state.moved_credit_df = pd.DataFrame()
    if 'moved_debit_df' not in st.session_state:
        st.session_state.moved_debit_df = pd.DataFrame()
    
    # Deleted records dataframes
    if 'deleted_credit_df' not in st.session_state:
        st.session_state.deleted_credit_df = pd.DataFrame()
    if 'deleted_debit_df' not in st.session_state:
        st.session_state.deleted_debit_df = pd.DataFrame()
    
    # Audit logs
    if 'audit_moves_log' not in st.session_state:
        st.session_state.audit_moves_log = pd.DataFrame()
    if 'audit_deletes_log' not in st.session_state:
        st.session_state.audit_deletes_log = pd.DataFrame()
    
    # Column mapping
    if 'intermediary_col_mapping' not in st.session_state:
        st.session_state.intermediary_col_mapping = {}
    
    # Statistics
    if 'moved_stats' not in st.session_state:
        st.session_state.moved_stats = {'moved_credit': 0, 'moved_debit': 0, 'total_moved': 0}
    if 'deleted_stats' not in st.session_state:
        st.session_state.deleted_stats = {'deleted_credit': 0, 'deleted_debit': 0, 'total_deleted': 0}
    
    # Current date tracking
    if 'intermediary_current_date' not in st.session_state:
        st.session_state.intermediary_current_date = datetime.now().strftime('%Y-%m-%d')
    if 'intermediary_last_save_date' not in st.session_state:
        st.session_state.intermediary_last_save_date = None
    
    # Debug mode
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

def safe_get_dataframe(df_name, default=pd.DataFrame()):
    """Safely get a dataframe from session state with default if not exists or empty"""
    if df_name in st.session_state and st.session_state[df_name] is not None:
        return st.session_state[df_name]
    return default

def save_current_state_to_db(target_date=None):
    """Save ALL Intermediary Reconciliation data to database"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get current session state data
    matched_credit_df = st.session_state.get('matched_credit_df', pd.DataFrame())
    matched_debit_df = st.session_state.get('matched_debit_df', pd.DataFrame())
    unmatched_credit_df = st.session_state.get('unmatched_credit_df', pd.DataFrame())
    unmatched_debit_df = st.session_state.get('unmatched_debit_df', pd.DataFrame())
    unmatched_bank_df = st.session_state.get('unmatched_bank_intermediary', pd.DataFrame())
    
    # Save main dataframes
    if not matched_credit_df.empty:
        db.save_dataframe('intermediary_matched_credit', add_unique_ids(matched_credit_df), target_date)
    if not matched_debit_df.empty:
        db.save_dataframe('intermediary_matched_debit', add_unique_ids(matched_debit_df), target_date)
    if not unmatched_credit_df.empty:
        db.save_dataframe('intermediary_unmatched_credit', add_unique_ids(unmatched_credit_df), target_date)
    if not unmatched_debit_df.empty:
        db.save_dataframe('intermediary_unmatched_debit', add_unique_ids(unmatched_debit_df), target_date)
    if not unmatched_bank_df.empty:
        db.save_dataframe('intermediary_unmatched_bank', add_unique_ids(unmatched_bank_df), target_date)
    
    # Save moved records
    moved_credit = st.session_state.get('moved_credit_df', pd.DataFrame())
    moved_debit = st.session_state.get('moved_debit_df', pd.DataFrame())
    if not moved_credit.empty:
        db.save_dataframe('intermediary_moved_credit', moved_credit, target_date)
    if not moved_debit.empty:
        db.save_dataframe('intermediary_moved_debit', moved_debit, target_date)
    
    # Save deleted records
    deleted_credit = st.session_state.get('deleted_credit_df', pd.DataFrame())
    deleted_debit = st.session_state.get('deleted_debit_df', pd.DataFrame())
    if not deleted_credit.empty:
        db.save_dataframe('intermediary_deleted_credit', deleted_credit, target_date)
    if not deleted_debit.empty:
        db.save_dataframe('intermediary_deleted_debit', deleted_debit, target_date)
    
    # Save audit logs
    audit_moves = st.session_state.get('audit_moves_log', pd.DataFrame())
    audit_deletes = st.session_state.get('audit_deletes_log', pd.DataFrame())
    if not audit_moves.empty:
        db.save_dataframe('intermediary_audit_moves', audit_moves, target_date)
    if not audit_deletes.empty:
        db.save_dataframe('intermediary_audit_deletes', audit_deletes, target_date)
    
    # Save stats as DataFrames (to avoid the .empty error on dicts)
    moved_stats = st.session_state.get('moved_stats', {})
    deleted_stats = st.session_state.get('deleted_stats', {})
    if moved_stats:
        moved_stats_df = pd.DataFrame([moved_stats])
        db.save_dataframe('intermediary_moved_stats', moved_stats_df, target_date)
    if deleted_stats:
        deleted_stats_df = pd.DataFrame([deleted_stats])
        db.save_dataframe('intermediary_deleted_stats', deleted_stats_df, target_date)
    
    # Save metadata
    db.save_metadata('intermediary_last_save_date', target_date)
    db.save_metadata('intermediary_col_mapping', st.session_state.get('intermediary_col_mapping', {}))
    
    # Save summary
    save_summary = {
        'matched_credit_count': len(matched_credit_df),
        'matched_debit_count': len(matched_debit_df),
        'unmatched_credit_count': len(unmatched_credit_df),
        'unmatched_debit_count': len(unmatched_debit_df),
        'unmatched_bank_count': len(unmatched_bank_df),
        'moved_credit_count': len(moved_credit),
        'moved_debit_count': len(moved_debit),
        'deleted_credit_count': len(deleted_credit),
        'deleted_debit_count': len(deleted_debit)
    }
    db.save_metadata('intermediary_save_summary', save_summary)
    
    st.session_state.intermediary_last_save_date = target_date
    
    with st.container():
        st.markdown('<div class="custom-success">', unsafe_allow_html=True)
        st.success(f"✅ Intermediary Reconciliation data saved for date: {target_date}")
        
        summary = []
        if not matched_credit_df.empty:
            summary.append(f"• matched_credit_df: {len(matched_credit_df)} records")
        if not matched_debit_df.empty:
            summary.append(f"• matched_debit_df: {len(matched_debit_df)} records")
        if not unmatched_credit_df.empty:
            summary.append(f"• unmatched_credit_df: {len(unmatched_credit_df)} records")
        if not unmatched_debit_df.empty:
            summary.append(f"• unmatched_debit_df: {len(unmatched_debit_df)} records")
        if not unmatched_bank_df.empty:
            summary.append(f"• unmatched_bank_df: {len(unmatched_bank_df)} records")
        
        if summary:
            st.info("Saved data:\n" + "\n".join(summary))
        st.markdown('</div>', unsafe_allow_html=True)
    
    return True

def load_state_from_db(target_date=None):
    """Load ALL Intermediary Reconciliation data from database"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Load main dataframes
    st.session_state.matched_credit_df = db.load_dataframe('intermediary_matched_credit', target_date)
    st.session_state.matched_debit_df = db.load_dataframe('intermediary_matched_debit', target_date)
    st.session_state.unmatched_credit_df = db.load_dataframe('intermediary_unmatched_credit', target_date)
    st.session_state.unmatched_debit_df = db.load_dataframe('intermediary_unmatched_debit', target_date)
    st.session_state.unmatched_bank_intermediary = db.load_dataframe('intermediary_unmatched_bank', target_date)
    
    # Load moved records
    st.session_state.moved_credit_df = db.load_dataframe('intermediary_moved_credit', target_date)
    st.session_state.moved_debit_df = db.load_dataframe('intermediary_moved_debit', target_date)
    
    # Load deleted records
    st.session_state.deleted_credit_df = db.load_dataframe('intermediary_deleted_credit', target_date)
    st.session_state.deleted_debit_df = db.load_dataframe('intermediary_deleted_debit', target_date)
    
    # Load audit logs
    st.session_state.audit_moves_log = db.load_dataframe('intermediary_audit_moves', target_date)
    st.session_state.audit_deletes_log = db.load_dataframe('intermediary_audit_deletes', target_date)
    
    # Load stats from DataFrames and convert back to dict
    moved_stats_df = db.load_dataframe('intermediary_moved_stats', target_date)
    if not moved_stats_df.empty:
        st.session_state.moved_stats = moved_stats_df.iloc[0].to_dict()
    else:
        update_moved_stats_cards()
    
    deleted_stats_df = db.load_dataframe('intermediary_deleted_stats', target_date)
    if not deleted_stats_df.empty:
        st.session_state.deleted_stats = deleted_stats_df.iloc[0].to_dict()
    else:
        update_deleted_stats_cards()
    
    # Add unique IDs and audit columns if missing
    for df_name in ['matched_credit_df', 'matched_debit_df', 'unmatched_credit_df', 
                    'unmatched_debit_df', 'unmatched_bank_intermediary']:
        if not st.session_state[df_name].empty:
            if '_record_id' not in st.session_state[df_name].columns:
                st.session_state[df_name] = add_unique_ids(st.session_state[df_name])
            st.session_state[df_name] = add_audit_columns(st.session_state[df_name])
    
    # Load column mapping
    col_mapping = db.load_metadata('intermediary_col_mapping', {})
    st.session_state.intermediary_col_mapping = col_mapping
    
    # Reinitialize display dataframes
    sync_all_display_dataframes()
    refresh_analytics_dataframes()
    
    st.session_state.intermediary_current_date = target_date
    
    with st.container():
        st.markdown('<div class="custom-success">', unsafe_allow_html=True)
        st.success(f"✅ Intermediary Reconciliation data loaded for date: {target_date}")
        
        summary = []
        if not st.session_state.matched_credit_df.empty:
            summary.append(f"• matched_credit_df: {len(st.session_state.matched_credit_df)} records")
        if not st.session_state.matched_debit_df.empty:
            summary.append(f"• matched_debit_df: {len(st.session_state.matched_debit_df)} records")
        if not st.session_state.unmatched_credit_df.empty:
            summary.append(f"• unmatched_credit_df: {len(st.session_state.unmatched_credit_df)} records")
        if not st.session_state.unmatched_debit_df.empty:
            summary.append(f"• unmatched_debit_df: {len(st.session_state.unmatched_debit_df)} records")
        if not st.session_state.unmatched_bank_intermediary.empty:
            summary.append(f"• unmatched_bank_df: {len(st.session_state.unmatched_bank_intermediary)} records")
        
        if summary:
            st.info("Loaded data:\n" + "\n".join(summary))
        st.markdown('</div>', unsafe_allow_html=True)
    
    return len(summary)

def reset_all_module_dataframes():
    """Reset all Intermediary Bank module dataframes to empty state"""
    with st.spinner("Resetting all dataframes..."):
        # Main dataframes
        st.session_state.matched_credit_df = pd.DataFrame()
        st.session_state.matched_debit_df = pd.DataFrame()
        st.session_state.unmatched_credit_df = pd.DataFrame()
        st.session_state.unmatched_debit_df = pd.DataFrame()
        st.session_state.unmatched_bank_intermediary = pd.DataFrame()
        st.session_state.intermediary_raw_df = pd.DataFrame()
        
        # Moved records dataframes
        st.session_state.moved_credit_df = pd.DataFrame()
        st.session_state.moved_debit_df = pd.DataFrame()
        
        # Deleted records dataframes
        st.session_state.deleted_credit_df = pd.DataFrame()
        st.session_state.deleted_debit_df = pd.DataFrame()
        
        # Audit logs
        st.session_state.audit_moves_log = pd.DataFrame()
        st.session_state.audit_deletes_log = pd.DataFrame()
        
        # Clear display dataframes
        display_keys = [key for key in st.session_state.keys() if key.endswith('_display_df')]
        for key in display_keys:
            st.session_state[key] = pd.DataFrame()
        
        # Clear selection states
        selection_keys = [key for key in st.session_state.keys() if key.endswith('_selection_state')]
        for key in selection_keys:
            st.session_state[key] = {}
        
        # Reset statistics
        st.session_state.moved_stats = {'moved_credit': 0, 'moved_debit': 0, 'total_moved': 0}
        st.session_state.deleted_stats = {'deleted_credit': 0, 'deleted_debit': 0, 'total_deleted': 0}
        
        logger.info("All Intermediary Bank module dataframes have been reset")
    
    return True

# --- Render Functions ---
def render_editable_dataframe_intermediary(df, title, key_prefix, on_data_change=None, show_delete=True, show_move=True, move_targets=None):
    """Render editable dataframe with proper display sync to main dataframe"""
    
    if df is None or df.empty:
        st.info(f"No {title} to display.")
        display_df_key = f"{key_prefix}_display_df"
        if display_df_key in st.session_state:
            st.session_state[display_df_key] = pd.DataFrame()
        return df if df is not None else pd.DataFrame()
    
    df = ensure_record_ids(df)
    df = add_audit_columns(df)
    
    st.markdown(f"### {title}")
    st.markdown(f"**Total Records: {len(df)}**")
    
    display_df_key = f"{key_prefix}_display_df"
    original_df_key = key_prefix
    
    # Always sync display dataframe with main dataframe
    if '#' not in df.columns:
        display_df = add_row_numbers(df.copy())
    else:
        display_df = df.copy()
    
    st.session_state[display_df_key] = display_df
    st.session_state[original_df_key] = remove_row_numbers(df.copy())
    
    # Use stable key without timestamp
    stable_key = key_prefix
    
    with st.expander("📝 Batch Operations", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            action_reason = st.text_input("Action Reason (optional):", key=f"{stable_key}_action_reason", placeholder="Enter reason for moving or deleting these records...")
        
        with col2:
            if show_delete and st.button(f"🗑️ Delete Selected", key=f"{stable_key}_delete_btn", use_container_width=True):
                selection_state = st.session_state.get(f"{stable_key}_selection_state", {})
                # Fix: Extract record IDs correctly from checkbox keys
                selected_record_ids = []
                for checkbox_key, is_selected in selection_state.items():
                    if is_selected and checkbox_key.startswith(f"{stable_key}_select_"):
                        # Extract record ID by removing the prefix
                        record_id = checkbox_key.replace(f"{stable_key}_select_", "")
                        selected_record_ids.append(record_id)
                
                if selected_record_ids:
                    source_df = st.session_state[display_df_key].copy()
                    updated_df, deleted_count = delete_selected_rows_with_audit(source_df, selected_record_ids, title, action_reason, df_name=display_df_key, on_data_change=on_data_change)
                    if original_df_key in st.session_state:
                        st.session_state[original_df_key] = remove_row_numbers(updated_df.copy())
                    
                    main_df_mapping = {
                        'Matched Credit Records': 'matched_credit_df',
                        'Matched Debit Records': 'matched_debit_df',
                        'Unmatched Credit Records': 'unmatched_credit_df',
                        'Unmatched Debit Records': 'unmatched_debit_df',
                        'Unmatched Bank Records': 'unmatched_bank_intermediary'
                    }
                    if title in main_df_mapping:
                        main_key = main_df_mapping[title]
                        st.session_state[main_key] = remove_row_numbers(updated_df.copy())
                    
                    sync_all_display_dataframes()
                    clear_selection_state(stable_key)
                    refresh_analytics_dataframes()
                    update_deleted_stats_cards()
                    update_moved_stats_cards()
                    st.success(f"✅ Deleted {deleted_count} record(s)")
                    st.rerun()
                else:
                    st.warning("No rows selected for deletion")
        
        with col3:
            if show_move and move_targets:
                selected_target = st.selectbox("Move to:", options=list(move_targets.keys()), key=f"{stable_key}_selected_target")
                if st.button(f"➡️ Move Selected", key=f"{stable_key}_move_btn", use_container_width=True):
                    selection_state = st.session_state.get(f"{stable_key}_selection_state", {})
                    selected_record_ids = []
                    for checkbox_key, is_selected in selection_state.items():
                        if is_selected and checkbox_key.startswith(f"{stable_key}_select_"):
                            record_id = checkbox_key.replace(f"{stable_key}_select_", "")
                            selected_record_ids.append(record_id)
                    
                    if selected_record_ids and selected_target:
                        source_key = key_prefix
                        source_df = st.session_state.get(source_key, pd.DataFrame()).copy()
                        source_df = ensure_record_ids(source_df)
                        moved_records, new_source = move_records_to_new_df(source_df, selected_record_ids, title, selected_target, action_reason)
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
                            
                            st.session_state[source_key] = new_source
                            st.session_state[display_df_key] = add_row_numbers(new_source)
                            
                            main_df_mapping = {
                                'Matched Credit Records': 'matched_credit_df',
                                'Matched Debit Records': 'matched_debit_df',
                                'Unmatched Credit Records': 'unmatched_credit_df',
                                'Unmatched Debit Records': 'unmatched_debit_df',
                                'Unmatched Bank Records': 'unmatched_bank_intermediary'
                            }
                            if title in main_df_mapping:
                                main_key = main_df_mapping[title]
                                st.session_state[main_key] = remove_row_numbers(new_source.copy())
                            
                            # Update target main dataframe
                            target_main_mapping = {
                                'Move to Matched Credit': 'matched_credit_df',
                                'Move to Matched Debit': 'matched_debit_df',
                                'Move to Unmatched Credit': 'unmatched_credit_df',
                                'Move to Unmatched Debit': 'unmatched_debit_df'
                            }
                            if selected_target in target_main_mapping:
                                target_key = target_main_mapping[selected_target]
                                target_current = st.session_state.get(target_key, pd.DataFrame()).copy()
                                moved_records_clean = remove_row_numbers(moved_records.copy())
                                st.session_state[target_key] = pd.concat([target_current, moved_records_clean], ignore_index=True)
                            
                            if on_data_change:
                                on_data_change(new_source)
                            
                            clear_selection_state(stable_key)
                            refresh_analytics_dataframes()
                            update_moved_stats_cards()
                            
                            st.success(f"✅ Moved {len(selected_record_ids)} record(s)")
                            st.rerun()
                    else:
                        st.warning("No rows selected or target not specified")
    
    # Main data editor
    st.markdown("---")
    st.markdown("### 📝 Data Editor")
    st.info("💡 Tip: Double-click any cell to edit its content directly.")
    
    df_for_edit = st.session_state[display_df_key].copy()
    if df_for_edit.empty:
        st.warning("No data available to edit.")
        return df
    
    columns_to_drop = []
    if '#' in df_for_edit.columns:
        columns_to_drop.append('#')
    if '_record_id' in df_for_edit.columns:
        columns_to_drop.append('_record_id')
    df_for_edit_for_display = df_for_edit.drop(columns=columns_to_drop) if columns_to_drop else df_for_edit
    
    edited_df = st.data_editor(
        df_for_edit_for_display,
        use_container_width=True,
        height=min(500, len(df_for_edit_for_display) * 35 + 38),
        key=f"{stable_key}_data_editor",
        num_rows="dynamic"
    )
    
    if not edited_df.equals(df_for_edit_for_display):
        edited_with_ids = ensure_record_ids(edited_df.copy())
        edited_with_audit = add_audit_columns(edited_with_ids)
        updated_with_numbers = add_row_numbers(edited_with_audit)
        
        st.session_state[display_df_key] = updated_with_numbers
        main_df = remove_row_numbers(edited_with_audit.copy())
        st.session_state[original_df_key] = main_df
        
        main_df_mapping = {
            'Matched Credit Records': 'matched_credit_df',
            'Matched Debit Records': 'matched_debit_df',
            'Unmatched Credit Records': 'unmatched_credit_df',
            'Unmatched Debit Records': 'unmatched_debit_df',
            'Unmatched Bank Records': 'unmatched_bank_intermediary'
        }
        if title in main_df_mapping:
            main_key = main_df_mapping[title]
            st.session_state[main_key] = main_df.copy()
        
        if on_data_change:
            on_data_change(main_df)
        
        refresh_analytics_dataframes()
        update_deleted_stats_cards()
        update_moved_stats_cards()
        st.success("✅ Data updated!")
        st.rerun()
    
    # Row selection for batch operations
    st.markdown("---")
    st.markdown("### ☑️ Select Rows for Batch Operations")
    
    if show_move and move_targets:
        st.markdown("#### Move Target Selection")
        selected_target = st.selectbox("Select target for moving records:", options=list(move_targets.keys()), key=f"{stable_key}_selected_target_main")
        if selected_target and selected_target in move_targets:
            target_key = move_targets[selected_target]
            target_df = st.session_state.get(target_key, pd.DataFrame())
            st.info(f"📌 Moving to: {selected_target} (currently {len(target_df)} records)")
        st.markdown("---")
    
    selection_key = f"{stable_key}_selection_state"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}
    
    df_for_selection = st.session_state[display_df_key].copy()
    if df_for_selection.empty:
        st.info("No rows available for selection.")
        return df
    
    if '_record_id' not in df_for_selection.columns:
        df_for_selection = ensure_record_ids(df_for_selection)
        st.session_state[display_df_key] = add_row_numbers(df_for_selection)
        st.session_state[original_df_key] = remove_row_numbers(df_for_selection.copy())
    
    record_ids = df_for_selection['_record_id'].tolist() if '_record_id' in df_for_selection.columns else []
    
    # Create a container for all checkboxes to avoid individual st.rerun on each click
    st.markdown("**Select rows by checking the boxes below:**")
    
    # Display rows with checkboxes - USING STABLE KEYS
    for idx in range(len(df_for_selection)):
        row_num = df_for_selection.iloc[idx]['#'] if '#' in df_for_selection.columns else idx + 1
        record_id = record_ids[idx] if idx < len(record_ids) else str(idx)
        # CRITICAL: Use stable key without timestamp
        checkbox_key = f"{stable_key}_select_{record_id}"
        is_selected = st.session_state[selection_key].get(checkbox_key, False)
        
        # Use columns for layout
        cols = st.columns([0.1, 0.9])
        
        with cols[0]:
            # Use a non-empty label (row number) but hide it
            checkbox_label = f"Select row {row_num}"
            if st.checkbox(checkbox_label, value=is_selected, key=checkbox_key, label_visibility="collapsed"):
                st.session_state[selection_key][checkbox_key] = True
            else:
                st.session_state[selection_key][checkbox_key] = False
        
        with cols[1]:
            row_summary_parts = []
            for col in ['Date', 'Intermediary_Amount', 'Amount', 'Currency', 'Application_ID']:
                if col in df_for_selection.columns:
                    val = df_for_selection.iloc[idx][col]
                    if pd.notna(val):
                        val_str = str(val)
                        if len(val_str) > 30:
                            val_str = val_str[:27] + "..."
                        row_summary_parts.append(f"**{col}:** {val_str}")
            st.markdown(f"**Row {row_num}:** " + " | ".join(row_summary_parts[:3]))
    
    selected_count = sum(1 for v in st.session_state[selection_key].values() if v)
    if selected_count > 0:
        st.success(f"✅ {selected_count} row(s) selected")
    
    # Download button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        df_download = st.session_state[display_df_key].copy()
        if '#' in df_download.columns:
            df_download = df_download.drop(columns=['#'])
        if '_record_id' in df_download.columns:
            df_download = df_download.drop(columns=['_record_id'])
        if not df_download.empty:
            csv = df_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{stable_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"{stable_key}_download",
                use_container_width=True
            )
    
    result_df = st.session_state[display_df_key].copy()
    for col in ['_record_id', '#']:
        if col in result_df.columns:
            result_df = result_df.drop(columns=[col])
    return result_df

def render_full_statistics_dashboard_intermediary():
    """Render comprehensive statistics dashboard"""
    st.markdown("### 📊 Comprehensive Statistics Dashboard")
    
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            update_moved_stats_cards()
            update_deleted_stats_cards()
            st.rerun()
    
    matched_credit = safe_get_dataframe('matched_credit_df')
    matched_debit = safe_get_dataframe('matched_debit_df')
    unmatched_credit = safe_get_dataframe('unmatched_credit_df')
    unmatched_debit = safe_get_dataframe('unmatched_debit_df')
    unmatched_bank = safe_get_dataframe('unmatched_bank_intermediary')
    
    matched_credit_count = len(matched_credit) if not matched_credit.empty else 0
    matched_debit_count = len(matched_debit) if not matched_debit.empty else 0
    unmatched_credit_count = len(unmatched_credit) if not unmatched_credit.empty else 0
    unmatched_debit_count = len(unmatched_debit) if not unmatched_debit.empty else 0
    bank_count = len(unmatched_bank) if not unmatched_bank.empty else 0
    
    total_records = matched_credit_count + matched_debit_count + unmatched_credit_count + unmatched_debit_count
    total_matched = matched_credit_count + matched_debit_count
    match_rate = (total_matched / total_records * 100) if total_records > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("✅ Matched Credit", matched_credit_count)
    with col2:
        st.metric("✅ Matched Debit", matched_debit_count)
    with col3:
        st.metric("⚠️ Unmatched Credit", unmatched_credit_count)
    with col4:
        st.metric("⚠️ Unmatched Debit", unmatched_debit_count)
    with col5:
        st.metric("🏦 Unmatched Bank", bank_count)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Total Records", total_records)
    with col2:
        st.metric("✅ Total Matched", total_matched)
    with col3:
        st.metric("📈 Match Rate", f"{match_rate:.1f}%")
    
    if total_records > 0:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            match_data = pd.DataFrame({
                'Status': ['Matched', 'Unmatched'],
                'Count': [total_matched, total_records - total_matched]
            })
            fig = px.pie(match_data, values='Count', names='Status', title='Match Status Distribution',
                        color_discrete_sequence=['#28a745', '#dc3545'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            trade_data = pd.DataFrame({
                'Type': ['Credit Matched', 'Credit Unmatched', 'Debit Matched', 'Debit Unmatched'],
                'Count': [matched_credit_count, unmatched_credit_count, matched_debit_count, unmatched_debit_count]
            })
            fig = px.bar(trade_data, x='Type', y='Count', title='Distribution by Type',
                        color='Type', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📦 Audit Summary")
        moved_stats = update_moved_stats_cards()
        deleted_stats = update_deleted_stats_cards()
        
        col1, col2 = st.columns(2)
        with col1:
            if moved_stats and moved_stats.get('total_moved', 0) > 0:
                st.markdown("#### Moved Records")
                moved_df = pd.DataFrame([
                    {'Category': 'Credit', 'Count': moved_stats.get('moved_credit', 0)},
                    {'Category': 'Debit', 'Count': moved_stats.get('moved_debit', 0)}
                ])
                fig = px.bar(moved_df, x='Category', y='Count', title='Moved Records by Type',
                            color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if deleted_stats and deleted_stats.get('total_deleted', 0) > 0:
                st.markdown("#### Deleted Records")
                deleted_df = pd.DataFrame([
                    {'Category': 'Credit', 'Count': deleted_stats.get('deleted_credit', 0)},
                                        {'Category': 'Debit', 'Count': deleted_stats.get('deleted_debit', 0)}
                ])
                fig = px.bar(deleted_df, x='Category', y='Count', title='Deleted Records by Type',
                            color='Count', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)

# --- Core Matching Functions (Keep original logic) ---
def safe_float(x):
    if pd.isna(x) or x is None:
        return None
    try:
        cleaned_x = str(x).replace(',', '').strip()
        return abs(float(cleaned_x))
    except (ValueError, TypeError):
        return None

def parse_date(date_str_raw):
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

def normalize_bank_key(raw_key, debug_mode=False):
    raw_key_lower = str(raw_key).lower().strip()
    replacements = {
        'ncba bank kenya plc': 'NCBA', 'ncba bank': 'NCBA',
        'equity bank': 'Equity', 'i&m bank': 'I&M',
        'central bank of kenya': 'CBK', 'kenya commercial bank': 'KCB',
        'kcb bank': 'KCB', 'sbm bank (kenya) limited': 'SBM',
        'sbm bank': 'SBM', 'absa bank': 'Absa',
        'kingdom bank': 'Kingdom', 'uba': 'UBA', 'yeepay': 'Yeepay',
    }
    for long, short in replacements.items():
        if raw_key_lower == long.lower():
            return short
        if raw_key_lower.startswith(long.lower()):
            return short
    all_target_bank_names = list(replacements.values()) + [k.capitalize() for k in replacements.keys()]
    all_target_bank_names = list(set(all_target_bank_names))
    match = process.extractOne(raw_key_lower, all_target_bank_names, scorer=fuzz.ratio)
    if match and match[1] >= 70:
        for long, short in replacements.items():
            if match[0].lower() == long.lower():
                return short
            if match[0].lower().startswith(long.lower()):
                return short
        return match[0].title() if match[0].islower() else match[0]
    return str(raw_key).strip().title()

def extract_bank_info_from_intermediary_column(column_value):
    if pd.isna(column_value) or not column_value:
        return None
    parts = str(column_value).split('-')
    if len(parts) >= 1:
        bank_name = parts[0].strip()
        return normalize_bank_key(bank_name)
    return None

def process_intermediary_match(
    intermediary_row: pd.Series,
    all_bank_dfs: dict,
    unmatched_list: list,
    matched_list: list,
    account_type: str,
    intermediary_column: str,
    date_tolerance_days: int = 3,
    debug_mode: bool = False,
    already_matched_intermediary_records: set = None,
    skipped_bank_records: dict = None,
    matched_bank_keys: set = None
):
    if already_matched_intermediary_records is None:
        already_matched_intermediary_records = set()
    if skipped_bank_records is None:
        skipped_bank_records = {}
    if matched_bank_keys is None:
        matched_bank_keys = set()

    application_id = intermediary_row.get('Application ID', '')
    if not application_id:
        application_id = f"{intermediary_row.get('Created At', '')}_{intermediary_row.get('Amount', '')}_{intermediary_row.get(intermediary_column, '')}"

    record_id = f"{application_id}_{account_type}"
    
    if record_id in already_matched_intermediary_records:
        return None

    amount = safe_float(intermediary_row.get('Amount'))
    if amount is None:
        return None

    status = str(intermediary_row.get('Status', '')).strip().lower()
    if status in ['declined', 'rejected', 'pending', 'not completed']:
        return None

    parsed_date = intermediary_row.get('Created At')
    if parsed_date and not isinstance(parsed_date, datetime):
        parsed_date = parse_date(str(parsed_date))
    if not isinstance(parsed_date, datetime):
        return None

    currency = str(intermediary_row.get('Currency', '')).strip().upper()
    if not currency:
        return None

    bank_info_raw = intermediary_row.get(intermediary_column, '')
    normalized_bank_name = extract_bank_info_from_intermediary_column(bank_info_raw)
    
    if not normalized_bank_name:
        unmatched_record = {
            '_record_id': generate_record_id(),
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank_Table_Expected': f"N/A ({bank_info_raw})",
            'Account_Type': account_type,
            'Amount': amount,
            'Currency': currency,
            'Status': 'Invalid Bank Info in Intermediary Record',
            'Application_ID': application_id,
            'Intermediary_Column': intermediary_column,
            'Bank_Info_Raw': bank_info_raw
        }
        unmatched_list.append(unmatched_record)
        return None

    expected_bank_key = f"{normalized_bank_name} {currency}"

    if expected_bank_key not in all_bank_dfs:
        unmatched_record = {
            '_record_id': generate_record_id(),
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank_Table_Expected': expected_bank_key,
            'Account_Type': account_type,
            'Amount': amount,
            'Currency': currency,
            'Status': 'No Matching Bank Statement File Found',
            'Application_ID': application_id,
            'Intermediary_Column': intermediary_column,
            'Bank_Info_Raw': bank_info_raw
        }
        unmatched_list.append(unmatched_record)
        return None

    bank_df = all_bank_dfs[expected_bank_key]
    bank_df_columns = bank_df.columns.tolist()

    if 'Skipped_By_Intermediary' not in bank_df.columns:
        bank_df['Skipped_By_Intermediary'] = ""

    date_column = 'Date'
    
    if account_type == 'Credit':
        bank_amount_column = 'Debit'
    else:
        bank_amount_column = 'Credit'

    if date_column not in bank_df.columns or bank_amount_column not in bank_df.columns:
        unmatched_record = {
            '_record_id': generate_record_id(),
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank_Table_Expected': expected_bank_key,
            'Account_Type': account_type,
            'Amount': amount,
            'Currency': currency,
            'Status': 'Missing Required Columns in Bank Statement',
            'Application_ID': application_id,
            'Intermediary_Column': intermediary_column,
            'Bank_Info_Raw': bank_info_raw
        }
        unmatched_list.append(unmatched_record)
        return None

    if bank_df[date_column].dtype == 'object':
        bank_df[date_column] = pd.to_datetime(bank_df[date_column], errors='coerce')
    
    date_matches = bank_df[
        bank_df[date_column].dt.date.between(
            parsed_date.date() - timedelta(days=date_tolerance_days),
            parsed_date.date() + timedelta(days=date_tolerance_days)
        )
    ]

    matched_records = []
    skipped_records = []

    for idx, bank_row in date_matches.iterrows():
        bank_amt = safe_float(bank_row.get(bank_amount_column))
        if bank_amt is None:
            continue

        amount_diff = abs(abs(bank_amt) - abs(amount))

        if amount_diff < 0.05:
            bank_record_key_operation = 'debit' if 'debit' in bank_amount_column.lower() or bank_amt < 0 else 'credit'
            
            bank_record_key = (
                expected_bank_key,
                bank_row[date_column].strftime('%Y-%m-%d') if hasattr(bank_row[date_column], 'strftime') else str(bank_row[date_column]),
                round(abs(bank_amt), 2),
                bank_record_key_operation
            )

            is_already_matched = bank_record_key in matched_bank_keys

            if is_already_matched:
                current_skipped = bank_df.loc[idx, "Skipped_By_Intermediary"]
                skipped_list = []
                if current_skipped and current_skipped != "":
                    try:
                        skipped_list = json.loads(current_skipped)
                    except:
                        skipped_list = []
                
                skipped_info = {
                    'intermediary_id': record_id,
                    'intermediary_date': parsed_date.strftime('%Y-%m-%d'),
                    'intermediary_amount': amount,
                    'intermediary_account_type': account_type,
                    'intermediary_currency': currency,
                    'skipped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                skipped_list.append(skipped_info)
                bank_df.loc[idx, "Skipped_By_Intermediary"] = json.dumps(skipped_list)
                
                skipped_records.append({
                    'bank_key': str(bank_record_key),
                    'bank_table': expected_bank_key,
                    'bank_date': bank_row[date_column].strftime('%Y-%m-%d') if hasattr(bank_row[date_column], 'strftime') else str(bank_row[date_column]),
                    'bank_amount': bank_amt,
                    'bank_row_index': idx,
                })
                continue

            matched_records.append({
                'Bank Index': idx,
                'Bank Date': bank_row.get(date_column).strftime('%Y-%m-%d') if bank_row.get(date_column) else None,
                'Description': str(bank_row.get('Description', '')).strip(),
                'Debit': safe_float(bank_row.get('Debit')),
                'Credit': safe_float(bank_row.get('Credit')),
                'Matched Column': bank_amount_column,
                'Bank Amount': bank_amt,
                'Bank Record Key': str(bank_record_key),
                'Amount Difference': amount_diff
            })

            bank_df.at[idx, "Matched"] = True
            matched_bank_keys.add(bank_record_key)

    if matched_records:
        all_matched_records_json = json.dumps(matched_records) if matched_records else ""
        skipped_records_json = json.dumps(skipped_records) if skipped_records else ""

        matched_record = {
            '_record_id': generate_record_id(),
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank_Table': expected_bank_key,
            'Account_Type': account_type,
            'Intermediary_Amount': amount,
            'Currency': currency,
            'Total_Bank_Matches': len(matched_records),
            'Skipped_Bank_Records': len(skipped_records),
            'Matched_Bank_Record_Index': matched_records[0]['Bank Index'],
            'Matched_Bank_Record_Date': matched_records[0]['Bank Date'],
            'Matched_Bank_Description': matched_records[0]['Description'],
            'Matched_Bank_Debit': matched_records[0]['Debit'],
            'Matched_Bank_Credit': matched_records[0]['Credit'],
            'All_Matched_Bank_Records': all_matched_records_json,
            'Skipped_Bank_Records_Info': skipped_records_json,
            'Application_ID': application_id,
            'Intermediary_Column': intermediary_column,
            'Bank_Info_Raw': bank_info_raw,
            'Status': status
        }

        matched_list.append(matched_record)
        already_matched_intermediary_records.add(record_id)

        return [(expected_bank_key, m['Bank Index']) for m in matched_records]

    if skipped_records:
        skipped_records_json = json.dumps(skipped_records) if skipped_records else ""
        
        unmatched_record = {
            '_record_id': generate_record_id(),
            'Date': parsed_date.strftime('%Y-%m-%d'),
            'Bank_Table_Expected': expected_bank_key,
            'Account_Type': account_type,
            'Amount': amount,
            'Currency': currency,
            'Status': f'Potential matches found but already taken (skipped: {len(skipped_records)})',
            'Skipped_Bank_Records': skipped_records_json,
            'Application_ID': application_id,
            'Intermediary_Column': intermediary_column,
            'Bank_Info_Raw': bank_info_raw
        }
        unmatched_list.append(unmatched_record)
        return None

    unmatched_record = {
        '_record_id': generate_record_id(),
        'Date': parsed_date.strftime('%Y-%m-%d'),
        'Bank_Table_Expected': expected_bank_key,
        'Account_Type': account_type,
        'Amount': amount,
        'Currency': currency,
        'Status': 'No Bank Statement Match (Amount or Date Tolerance)',
        'Application_ID': application_id,
        'Intermediary_Column': intermediary_column,
        'Bank_Info_Raw': bank_info_raw
    }
    unmatched_list.append(unmatched_record)
    return None

def run_intermediary_reconciliation(intermediary_df, all_bank_dfs, date_tolerance_days, debug_mode):
    """Run the full intermediary reconciliation process"""
    if intermediary_df.empty:
        st.warning("No intermediary data to reconcile.")
        return
    
    unmatched_credit = []
    matched_credit = []
    unmatched_debit = []
    matched_debit = []

    current_run_bank_dfs = {key: df.copy() for key, df in all_bank_dfs.items()}
    
    for bank_df in current_run_bank_dfs.values():
        if "Matched" not in bank_df.columns:
            bank_df["Matched"] = False

    progress_bar = st.progress(0)
    total_rows = len(intermediary_df)
    
    for idx, (index, row) in enumerate(intermediary_df.iterrows()):
        status = str(row.get('Status', '')).strip().lower()
        
        if status in ['declined', 'rejected', 'pending', 'not completed']:
            progress_bar.progress((idx + 1) / total_rows)
            continue
        
        process_intermediary_match(
            row, current_run_bank_dfs, unmatched_credit, matched_credit,
            'Credit', 'Intermediary Bank Account - Credit',
            date_tolerance_days, debug_mode
        )
        
        process_intermediary_match(
            row, current_run_bank_dfs, unmatched_debit, matched_debit,
            'Debit', 'Intermediary Bank Account - Debit',
            date_tolerance_days, debug_mode
        )
        
        progress_bar.progress((idx + 1) / total_rows)
    
    progress_bar.empty()
    
    # Collect unmatched bank records
    unmatched_bank_records = []
    for bank_key, bank_df in current_run_bank_dfs.items():
        bank_df.columns = bank_df.columns.str.strip()
        
        date_col = 'Date'
        description_col = 'Description'
        credit_col = 'Credit'
        debit_col = 'Debit'
        
        if date_col not in bank_df.columns or description_col not in bank_df.columns:
            continue
        
        unmatched_bank_df_for_key = bank_df[bank_df["Matched"] == False].copy()
        
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
                    '_record_id': generate_record_id(),
                    'Bank_Table': bank_key,
                    'Date': row_date_parsed.strftime('%Y-%m-%d') if row_date_parsed else None,
                    'Description': str(row.get(description_col, '')).strip(),
                    'Transaction_Type_Column': transaction_type_col_name,
                    'Amount': round(amount_found, 2)
                })
    
    # Convert to DataFrames and add unique IDs
    st.session_state.matched_credit_df = add_unique_ids(pd.DataFrame(matched_credit)) if matched_credit else pd.DataFrame()
    st.session_state.matched_debit_df = add_unique_ids(pd.DataFrame(matched_debit)) if matched_debit else pd.DataFrame()
    st.session_state.unmatched_credit_df = add_unique_ids(pd.DataFrame(unmatched_credit)) if unmatched_credit else pd.DataFrame()
    st.session_state.unmatched_debit_df = add_unique_ids(pd.DataFrame(unmatched_debit)) if unmatched_debit else pd.DataFrame()
    st.session_state.unmatched_bank_intermediary = add_unique_ids(pd.DataFrame(unmatched_bank_records)) if unmatched_bank_records else pd.DataFrame()
    
    for df_name in ['matched_credit_df', 'matched_debit_df', 'unmatched_credit_df', 
                    'unmatched_debit_df', 'unmatched_bank_intermediary']:
        if not st.session_state[df_name].empty:
            st.session_state[df_name] = add_audit_columns(st.session_state[df_name])
    
    sync_all_display_dataframes()
    
    st.success("Intermediary Reconciliation complete!")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Matched Credit", len(matched_credit))
    with col2:
        st.metric("Unmatched Credit", len(unmatched_credit))
    with col3:
        st.metric("Matched Debit", len(matched_debit))
    with col4:
        st.metric("Unmatched Debit", len(unmatched_debit))
    with col5:
        st.metric("Unmatched Bank", len(unmatched_bank_records))

# --- Main App Function ---
@require_auth
def intermediary_bank_reconciliation_app(all_bank_dfs: dict):
    # Initialize session state
    initialize_session_state_intermediary()
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    # st.markdown("""
    # <div class="main-header">
    #     <h1>🏦 Intermediary Bank Reconciliation Dashboard</h1>
    #     <p>Verify intermediary bank records against bank statements, manage exceptions, and track audit history</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # ========== DATA MANAGEMENT SECTION ==========
    st.markdown("### 📅 Data Management")
    
    available_dates = db.get_available_dates()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        if available_dates:
            selected_load_date = st.selectbox(
                "📅 Select date to load:",
                options=available_dates,
                index=0,
                key="intermediary_load_date_select"
            )
        else:
            st.selectbox("📅 Select date to load:", options=["No data available"], disabled=True, key="intermediary_load_date_select")
            selected_load_date = None
    
    with col2:
        if selected_load_date and available_dates:
            if st.button("📂 Load Data", use_container_width=True, key="load_intermediary_btn"):
                load_state_from_db(selected_load_date)
                st.rerun()
    
    with col3:
        current_date = datetime.now().strftime('%Y-%m-%d')
        st.metric("Current Date", current_date)
    
    with col4:
        if st.button("💾 Save Data", type="primary", use_container_width=True, key="save_intermediary_btn"):
            save_current_state_to_db()
            st.rerun()
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Reset Current Module Data", type="secondary", use_container_width=True, key="reset_module_btn"):
            reset_all_module_dataframes()
            st.success("✅ All current module dataframes have been reset!")
            st.balloons()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Reset All Data (Including Saved)", type="secondary", use_container_width=True, key="reset_all_btn"):
            target_date = datetime.now().strftime('%Y-%m-%d')
            reset_all_module_dataframes()
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            for table_name in INTERMEDIARY_TABLES.values():
                try:
                    cursor.execute(f"DELETE FROM {table_name} WHERE record_date = ? OR import_date LIKE ?", 
                                  (target_date, f"{target_date}%"))
                except:
                    pass
            
            conn.commit()
            conn.close()
            
            st.success("✅ All data (session and database) has been reset!")
            st.balloons()
            st.rerun()
    
    with col3:
        if st.button("📊 Refresh Dashboard", type="primary", use_container_width=True, key="refresh_dashboard_btn"):
            update_moved_stats_cards()
            update_deleted_stats_cards()
            refresh_analytics_dataframes()
            sync_all_display_dataframes()
            st.success("✅ Dashboard refreshed!")
            st.rerun()
    
    st.markdown("---")
    
    # ========== DATA UPLOAD SECTION ==========
    with st.expander("📤 Upload Intermediary Bank Records", expanded=False):
        uploaded_intermediary_file = st.file_uploader("Choose Intermediary Bank Records (CSV or XLSX)", type=["csv", "xlsx"], key="intermediary_uploader")

        if uploaded_intermediary_file:
            try:
                save_uploaded_file(uploaded_intermediary_file, "intermediary_bank_uploaded." + uploaded_intermediary_file.name.split('.')[-1])
                
                if uploaded_intermediary_file.name.endswith('.xlsx'):
                    xls = pd.ExcelFile(uploaded_intermediary_file)
                    sheet_names = xls.sheet_names
                    selected_sheet = st.selectbox("Select sheet", sheet_names, key="intermediary_sheet_selector")
                    intermediary_df = pd.read_excel(uploaded_intermediary_file, sheet_name=selected_sheet)
                else:
                    intermediary_df = pd.read_csv(uploaded_intermediary_file)

                intermediary_df.columns = intermediary_df.columns.str.strip()
                st.success(f"✅ File loaded successfully! Found {len(intermediary_df)} rows and {len(intermediary_df.columns)} columns")
                st.dataframe(intermediary_df.head(5))

                st.markdown("#### Map Columns")
                intermediary_col_options = ['-- Select Column --'] + intermediary_df.columns.tolist()
                col_mapping = {}

                saved_mapping = db.load_metadata('intermediary_col_mapping', {})

                for display_name, suggested_col in EXPECTED_COLUMNS.items():
                    initial_selection = saved_mapping.get(display_name, suggested_col if suggested_col in intermediary_col_options else '-- Select Column --')
                    selected_col = st.selectbox(
                        f"Map '{display_name}'",
                        options=intermediary_col_options,
                        index=intermediary_col_options.index(initial_selection) if initial_selection in intermediary_col_options else 0,
                        key=f"intermediary_map_{display_name}"
                    )
                    col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None

                if st.button("✅ Process Data", type="primary", key="process_intermediary_btn", use_container_width=True):
                    renamed_cols_dict = {selected: original for original, selected in col_mapping.items() if selected and selected in intermediary_df.columns}
                    if renamed_cols_dict:
                        cols_to_keep = list(renamed_cols_dict.keys())
                        intermediary_df = intermediary_df[cols_to_keep].rename(columns=renamed_cols_dict)
                    st.session_state.intermediary_raw_df = intermediary_df
                    db.save_metadata('intermediary_col_mapping', col_mapping)
                    st.success("✅ Data processed successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")

    st.markdown("---")
    
    # ========== RECONCILIATION CONTROLS ==========
    if not st.session_state.intermediary_raw_df.empty:
        st.markdown("### ⚙️ Reconciliation Settings")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.get('debug_mode', False))
            st.session_state.debug_mode = debug_mode
        
        with col2:
            date_tolerance_days = st.slider("Date Tolerance (± days)", min_value=0, max_value=7, value=3, step=1, key="intermediary_date_tolerance")
        
        with col3:
            if st.button("🔄 Run Reconciliation", type="primary", use_container_width=True):
                if not all_bank_dfs:
                    st.error("❌ No bank statements loaded! Please upload bank statements first.")
                else:
                    required_cols = ['Application ID', 'Amount', 'Currency', 
                                    'Intermediary Bank Account - Credit', 
                                    'Intermediary Bank Account - Debit', 
                                    'Created At', 'Status']
                    
                    missing_required = [col for col in required_cols if col not in st.session_state.intermediary_raw_df.columns]
                    if missing_required:
                        st.error(f"Missing required columns: {', '.join(missing_required)}. Please check your column mapping.")
                    else:
                        with st.spinner("Running reconciliation..."):
                            run_intermediary_reconciliation(
                                st.session_state.intermediary_raw_df,
                                all_bank_dfs,
                                date_tolerance_days,
                                debug_mode
                            )
                            st.success("✅ Reconciliation complete!")
                            st.balloons()
                            st.rerun()
        
        st.markdown("---")
        
        # Quick Stats Section
        st.markdown("### 📊 Quick Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("✅ Matched Credit", len(st.session_state.get('matched_credit_df', pd.DataFrame())))
        with col2:
            st.metric("✅ Matched Debit", len(st.session_state.get('matched_debit_df', pd.DataFrame())))
        with col3:
            st.metric("⚠️ Unmatched Credit", len(st.session_state.get('unmatched_credit_df', pd.DataFrame())))
        with col4:
            st.metric("⚠️ Unmatched Debit", len(st.session_state.get('unmatched_debit_df', pd.DataFrame())))
        with col5:
            st.metric("🏦 Unmatched Bank", len(st.session_state.get('unmatched_bank_intermediary', pd.DataFrame())))
        
        st.markdown("---")
        st.markdown("### 📋 Audit Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📦 Total Moved Records", st.session_state.get('moved_stats', {}).get('total_moved', 0))
        with col2:
            st.metric("🗑️ Total Deleted Records", st.session_state.get('deleted_stats', {}).get('total_deleted', 0))
    
    st.markdown("---")
    
    # Main Dashboard
    render_full_statistics_dashboard_intermediary()
    
    # Move targets configuration
    move_targets_credit = {
        "Move to Matched Debit": "matched_debit_df",
        "Move to Unmatched Credit": "unmatched_credit_df",
        "Move to Unmatched Debit": "unmatched_debit_df"
    }
    
    move_targets_debit = {
        "Move to Matched Credit": "matched_credit_df",
        "Move to Unmatched Credit": "unmatched_credit_df",
        "Move to Unmatched Debit": "unmatched_debit_df"
    }
    
    move_targets_unmatched = {
        "Move to Matched Credit": "matched_credit_df",
        "Move to Matched Debit": "matched_debit_df"
    }
    
    # Results Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "✅ Matched Credit", "✅ Matched Debit",
        "❌ Unmatched Credit", "❌ Unmatched Debit",
        "🏦 Unmatched Bank", "📋 Moved Records", "🔍 Audit Trail"
    ])
    
    with tab1:
        def update_matched_credit(df):
            st.session_state.matched_credit_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_credit_df.empty:
                st.session_state.matched_credit_df = add_audit_columns(st.session_state.matched_credit_df)
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe_intermediary(
            st.session_state.matched_credit_df,
            "Matched Credit Records",
            "matched_credit",
            on_data_change=update_matched_credit,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_credit
        )
    
    with tab2:
        def update_matched_debit(df):
            st.session_state.matched_debit_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_debit_df.empty:
                st.session_state.matched_debit_df = add_audit_columns(st.session_state.matched_debit_df)
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe_intermediary(
            st.session_state.matched_debit_df,
            "Matched Debit Records",
            "matched_debit",
            on_data_change=update_matched_debit,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_debit
        )
    
    with tab3:
        def update_unmatched_credit(df):
            st.session_state.unmatched_credit_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_credit_df.empty:
                st.session_state.unmatched_credit_df = add_audit_columns(st.session_state.unmatched_credit_df)
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe_intermediary(
            st.session_state.unmatched_credit_df,
            "Unmatched Credit Records",
            "unmatched_credit",
            on_data_change=update_unmatched_credit,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_unmatched
        )
    
    with tab4:
        def update_unmatched_debit(df):
            st.session_state.unmatched_debit_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_debit_df.empty:
                st.session_state.unmatched_debit_df = add_audit_columns(st.session_state.unmatched_debit_df)
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe_intermediary(
            st.session_state.unmatched_debit_df,
            "Unmatched Debit Records",
            "unmatched_debit",
            on_data_change=update_unmatched_debit,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_unmatched
        )
    
    with tab5:
        def update_unmatched_bank(df):
            st.session_state.unmatched_bank_intermediary = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_bank_intermediary.empty:
                st.session_state.unmatched_bank_intermediary = add_audit_columns(st.session_state.unmatched_bank_intermediary)
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe_intermediary(
            st.session_state.unmatched_bank_intermediary,
            "Unmatched Bank Records",
            "unmatched_bank",
            on_data_change=update_unmatched_bank,
            show_delete=True,
            show_move=False
        )
    
    with tab6:
        st.markdown("### 📋 Moved Records")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Moved Credit Records")
            if not st.session_state.moved_credit_df.empty:
                st.dataframe(st.session_state.moved_credit_df, use_container_width=True)
                csv_moved_credit = st.session_state.moved_credit_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Moved Credit CSV",
                    data=csv_moved_credit,
                    file_name="moved_credit_records.csv",
                    mime="text/csv",
                    key="download_moved_credit"
                )
            else:
                st.info("No moved credit records")
        
        with col2:
            st.subheader("Moved Debit Records")
            if not st.session_state.moved_debit_df.empty:
                st.dataframe(st.session_state.moved_debit_df, use_container_width=True)
                csv_moved_debit = st.session_state.moved_debit_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Moved Debit CSV",
                    data=csv_moved_debit,
                    file_name="moved_debit_records.csv",
                    mime="text/csv",
                    key="download_moved_debit"
                )
            else:
                st.info("No moved debit records")
    
    with tab7:
        st.markdown("### 🔍 Audit Trail")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Move History")
            if not st.session_state.audit_moves_log.empty:
                st.dataframe(st.session_state.audit_moves_log, use_container_width=True)
                csv_moves = st.session_state.audit_moves_log.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Move Log",
                    data=csv_moves,
                    file_name="audit_moves_log.csv",
                    mime="text/csv",
                    key="download_moves_log"
                )
            else:
                st.info("No move records found")
        
        with col2:
            st.subheader("Delete History")
            if not st.session_state.audit_deletes_log.empty:
                st.dataframe(st.session_state.audit_deletes_log, use_container_width=True)
                csv_deletes = st.session_state.audit_deletes_log.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Delete Log",
                    data=csv_deletes,
                    file_name="audit_deletes_log.csv",
                    mime="text/csv",
                    key="download_deletes_log"
                )
            else:
                st.info("No delete records found")
    
    # Return dataframes for compatibility with main_dashboard
    return (
        safe_get_dataframe('matched_credit_df'),
        safe_get_dataframe('matched_debit_df'),
        safe_get_dataframe('unmatched_credit_df'),
        safe_get_dataframe('unmatched_debit_df'),
        safe_get_dataframe('unmatched_bank_intermediary')
    )