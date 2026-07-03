# fx_reconcilliation_app_page.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
import pickle
import json
import logging
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Import authentication functions ---
try:
    from auth_system import (
        
        log_audit, require_auth, 
         get_available_data_types, get_available_modules,
        load_results_by_date, restore_session_from_loaded_results
    )
except ImportError:
    # Mock functions if auth_system is not available
    def require_auth(func):
        return func
    def get_active_version_id():
        return None
    def save_reconciliation_data(*args, **kwargs):
        pass
    def load_reconciliation_data(*args, **kwargs):
        return pd.DataFrame()
    def get_all_versions():
        return pd.DataFrame()
    def log_audit(*args, **kwargs):
        pass
    def save_all_reconciliation_results(*args, **kwargs):
        return 0
    def load_all_saved_results(*args, **kwargs):
        return {}
    def get_available_data_types():
        return ['All']
    def get_available_modules():
        return ['All']
    def load_results_by_date(*args, **kwargs):
        return {}
    def restore_session_from_loaded_results(*args, **kwargs):
        return 0

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
DB_PATH = "data/fx_reconciliation.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Custom CSS for better UI
CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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

# Database table names for FX Reconciliation module
FX_RECON_TABLES = {
    'matched_local': 'fx_recon_matched_local',
    'unmatched_local': 'fx_recon_unmatched_local',
    'matched_foreign': 'fx_recon_matched_foreign',
    'unmatched_foreign': 'fx_recon_unmatched_foreign',
    'bank_records': 'fx_recon_bank_records',
    'moved_local_matched': 'fx_recon_moved_local_matched',
    'moved_local_unmatched': 'fx_recon_moved_local_unmatched',
    'moved_foreign_matched': 'fx_recon_moved_foreign_matched',
    'moved_foreign_unmatched': 'fx_recon_moved_foreign_unmatched',
    'moved_bank_records': 'fx_recon_moved_bank_records',
    'deleted_local_matched': 'fx_recon_deleted_local_matched',
    'deleted_local_unmatched': 'fx_recon_deleted_local_unmatched',
    'deleted_foreign_matched': 'fx_recon_deleted_foreign_matched',
    'deleted_foreign_unmatched': 'fx_recon_deleted_foreign_unmatched',
    'deleted_bank_records': 'fx_recon_deleted_bank_records',
    'audit_moves_log': 'fx_recon_audit_moves',
    'audit_deletes_log': 'fx_recon_audit_deletes',
    'df_matched_adjustments_local': 'fx_recon_df_matched_local',
    'df_unmatched_adjustments_local': 'fx_recon_df_unmatched_local',
    'df_matched_adjustments_foreign': 'fx_recon_df_matched_foreign',
    'df_unmatched_adjustments_foreign': 'fx_recon_df_unmatched_foreign',
    'df_unmatched_bank_records': 'fx_recon_df_unmatched_bank'
}

# All dataframe keys for FX Reconciliation module
FX_RECON_KEYS = [
    'matched_local', 'unmatched_local', 'matched_foreign', 'unmatched_foreign', 'bank_records',
    'moved_local_matched', 'moved_local_unmatched', 'moved_foreign_matched', 
    'moved_foreign_unmatched', 'moved_bank_records', 'audit_moves_log',
    'deleted_local_matched', 'deleted_local_unmatched', 'deleted_foreign_matched',
    'deleted_foreign_unmatched', 'deleted_bank_records', 'audit_deletes_log',
    'moved_stats', 'deleted_stats', 'df_matched_adjustments_local', 'df_unmatched_adjustments_local',
    'df_matched_adjustments_foreign', 'df_unmatched_adjustments_foreign', 'df_unmatched_bank_records'
]

# Change tracking keys
CHANGE_TRACKING_KEYS = [
    'matched_local', 'unmatched_local', 'matched_foreign', 'unmatched_foreign', 'bank_records',
    'moved_local_matched', 'moved_local_unmatched', 'moved_foreign_matched', 
    'moved_foreign_unmatched', 'moved_bank_records', 'audit_moves_log',
    'deleted_local_matched', 'deleted_local_unmatched', 'deleted_foreign_matched',
    'deleted_foreign_unmatched', 'deleted_bank_records', 'audit_deletes_log'
]

# --- Database Manager Class for FX Reconciliation ---
class FXReconDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create matched tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_matched_local (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Adjustment_Date TEXT,
                Adjustment_Amount REAL,
                Adjustment_Operation TEXT,
                Adjustment_Intermediary_Account TEXT,
                Adjustment_Currency TEXT,
                Bank_Table TEXT,
                Bank_Statement_Date TEXT,
                Bank_Statement_Amount REAL,
                Bank_Matched_Column TEXT,
                Bank_Row_Index INTEGER,
                Matched_Bank_Record TEXT,
                Match_Details TEXT,
                Request_ID TEXT,
                Payment_Channel TEXT,
                Counterparty_Bank TEXT,
                Counterparty_Account_ID TEXT,
                Counterparty_Name TEXT,
                Transfer_Reference_No TEXT,
                Transaction_Narrative TEXT,
                TX_ID TEXT,
                Customer_Account_Number TEXT,
                Account_Name TEXT,
                Account_Channel TEXT,
                Product TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                moved_to TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_matched_foreign (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Adjustment_Date TEXT,
                Adjustment_Amount REAL,
                Adjustment_Operation TEXT,
                Adjustment_Intermediary_Account TEXT,
                Adjustment_Currency TEXT,
                Bank_Table TEXT,
                Bank_Statement_Date TEXT,
                Bank_Statement_Amount REAL,
                Bank_Matched_Column TEXT,
                Bank_Row_Index INTEGER,
                Matched_Bank_Record TEXT,
                Match_Details TEXT,
                Request_ID TEXT,
                Payment_Channel TEXT,
                Counterparty_Bank TEXT,
                Counterparty_Account_ID TEXT,
                Counterparty_Name TEXT,
                Transfer_Reference_No TEXT,
                Transaction_Narrative TEXT,
                TX_ID TEXT,
                Customer_Account_Number TEXT,
                Account_Name TEXT,
                Account_Channel TEXT,
                Product TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                moved_to TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Create unmatched tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_unmatched_local (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Adjustment_Date TEXT,
                Adjustment_Amount REAL,
                Adjustment_Operation TEXT,
                Adjustment_Intermediary_Account TEXT,
                Adjustment_Currency TEXT,
                Status TEXT,
                Reason TEXT,
                Request_ID TEXT,
                Payment_Channel TEXT,
                Counterparty_Bank TEXT,
                Counterparty_Account_ID TEXT,
                Counterparty_Name TEXT,
                Transfer_Reference_No TEXT,
                Transaction_Narrative TEXT,
                TX_ID TEXT,
                Customer_Account_Number TEXT,
                Account_Name TEXT,
                Account_Channel TEXT,
                Product TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                moved_to TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_unmatched_foreign (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Adjustment_Date TEXT,
                Adjustment_Amount REAL,
                Adjustment_Operation TEXT,
                Adjustment_Intermediary_Account TEXT,
                Adjustment_Currency TEXT,
                Status TEXT,
                Reason TEXT,
                Request_ID TEXT,
                Payment_Channel TEXT,
                Counterparty_Bank TEXT,
                Counterparty_Account_ID TEXT,
                Counterparty_Name TEXT,
                Transfer_Reference_No TEXT,
                Transaction_Narrative TEXT,
                TX_ID TEXT,
                Customer_Account_Number TEXT,
                Account_Name TEXT,
                Account_Channel TEXT,
                Product TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                moved_to TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Bank records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_bank_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Bank_Table TEXT,
                Date TEXT,
                Description TEXT,
                Transaction_Type_Column TEXT,
                Amount REAL,
                Original_Row_Index INTEGER,
                Skipped_By_Adjustments TEXT,
                Skipped_Count INTEGER,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                moved_to TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Moved records tables
        moved_tables = [
            'fx_recon_moved_local_matched', 'fx_recon_moved_local_unmatched',
            'fx_recon_moved_foreign_matched', 'fx_recon_moved_foreign_unmatched',
            'fx_recon_moved_bank_records'
        ]
        for table in moved_tables:
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table} (
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
        deleted_tables = [
            'fx_recon_deleted_local_matched', 'fx_recon_deleted_local_unmatched',
            'fx_recon_deleted_foreign_matched', 'fx_recon_deleted_foreign_unmatched',
            'fx_recon_deleted_bank_records'
        ]
        for table in deleted_tables:
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    _record_id TEXT UNIQUE,
                    record_date TEXT,
                    created_at TEXT,
                    source_table TEXT,
                    deleted_by TEXT,
                    deleted_at TEXT,
                    delete_reason TEXT,
                    deleted_from TEXT,
                    original_record_json TEXT,
                    import_date TEXT,
                    last_modified TEXT
                )
            ''')
        
        # Audit logs tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_audit_moves (
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
            CREATE TABLE IF NOT EXISTS fx_recon_audit_deletes (
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
            CREATE TABLE IF NOT EXISTS fx_recon_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("FX Reconciliation database initialized")
    
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
        except:
            pass
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        records = df.to_dict('records')
        
        for record in records:
            _record_id = str(record.get('_record_id', str(uuid.uuid4())))
            
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
                cursor.execute(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})", values)
            except Exception as e:
                logger.error(f"Error inserting: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to {table_name}")
    
    def load_dataframe(self, table_name, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE record_date = ?", conn, params=(target_date,))
        except:
            df = pd.DataFrame()
        conn.close()
        
        if not df.empty:
            cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def save_metadata(self, key, value):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO fx_recon_metadata (key, value, updated_at) VALUES (?, ?, ?)',
                    (key, json.dumps(value), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
    
    def load_metadata(self, key, default=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT value FROM fx_recon_metadata WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else default
    
    def get_available_dates(self):
        conn = sqlite3.connect(self.db_path)
        dates = set()
        for table in FX_RECON_TABLES.values():
            try:
                cursor = conn.execute(f"SELECT DISTINCT record_date FROM {table} WHERE record_date IS NOT NULL")
                for row in cursor.fetchall():
                    if row[0]:
                        dates.add(row[0])
            except:
                pass
        conn.close()
        return sorted(list(dates), reverse=True)



# Add this function to alter existing tables and add missing columns
def update_database_schema_dynamic():
    """Dynamically add ANY missing columns that appear in dataframes"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'fx_recon_%'")
    existing_tables = [row[0] for row in cursor.fetchall()]
    
    # Define all possible columns that might appear in ANY dataframe
    all_possible_columns = [
        # Basic columns
        '_record_id', 'record_date', 'created_at', 'import_date', 'last_modified',
        
        # Adjustment columns
        'Adjustment_Date', 'Adjustment_Amount', 'Adjustment_Operation', 
        'Adjustment_Intermediary_Account', 'Adjustment_Currency',
        
        # Bank statement columns
        'Bank_Table', 'Bank_Statement_Date', 'Bank_Statement_Amount', 
        'Bank_Matched_Column', 'Bank_Row_Index', 'Matched_Bank_Record', 'Match_Details',
        
        # Transaction columns (with underscores - dataframe format)
        'Request_ID', 'Payment_Channel', 'Counterparty_Bank', 'Counterparty_Account_ID',
        'Counterparty_Name', 'Transfer_Reference_No', 'Transaction_Narrative', 'TX_ID',
        'Customer_Account_Number', 'Account_Name', 'Account_Channel', 'Product',
        'Counterparty_Sub_account', 'Counterparty_Bank_SWIFT_Code_BIC',
        
        # Transaction columns (with spaces - database format)
        'Request ID', 'Payment Channel', 'Counterparty Bank', 'Counterparty Account ID',
        'Counterparty Name', 'Transfer Reference No.', 'Transaction Narrative', 'TX ID',
        'Customer Account Number', 'Account Name', 'Account Channel', 'Product',
        'Counterparty Sub-account', 'Counterparty Bank SWIFT Code / BIC',
        'Intermediary Account', 'Completed At',
        
        # Status and reason columns
        'Status', 'Reason',
        
        # Audit columns
        'deleted_by', 'deleted_at', 'delete_reason', 'deleted_from', 'source_dataframe',
        'moved_by', 'moved_from', 'moved_at', 'move_reason', 'move_type', 'moved_to',
        'source_table', 'moved_from_table',
        
        # Bank record specific
        'Date', 'Description', 'Transaction_Type_Column', 'Amount', 'Original_Row_Index',
        'Skipped_By_Adjustments', 'Skipped_Count'
    ]
    
    added_count = 0
    
    # For each table, add all possible columns that don't exist
    for table_name in existing_tables:
        # Get existing columns in this table
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        # Determine which columns to add
        columns_to_add = []
        for col in all_possible_columns:
            if col not in existing_columns:
                # Determine appropriate data type
                if 'Amount' in col or 'amount' in col or 'Count' in col:
                    col_type = 'REAL'
                elif 'Date' in col or 'date' in col or 'At' in col:
                    col_type = 'TEXT'
                else:
                    col_type = 'TEXT'
                
                # Handle column names with spaces or special characters
                if ' ' in col or '/' in col or '-' in col:
                    columns_to_add.append(f'"{col}" {col_type}')
                else:
                    columns_to_add.append(f'{col} {col_type}')
        
        # Add missing columns
        for column_def in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")
                logger.info(f"✓ Added column {column_def} to {table_name}")
                added_count += 1
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                if "duplicate column name" not in error_msg:
                    logger.warning(f"Could not add column to {table_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Dynamic schema update completed: Added {added_count} columns across all tables")
    return added_count


def map_dataframe_columns_for_save(df, table_name):
    """Map dataframe column names to database column names dynamically"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    # Comprehensive column mappings (underscore to space versions)
    column_mappings = {
        # Request related
        'Request_ID': 'Request ID',
        
        # Party information
        'Counterparty_Sub_account': 'Counterparty Sub-account',
        'Counterparty_Bank_SWIFT_Code_BIC': 'Counterparty Bank SWIFT Code / BIC',
        'Counterparty_Bank': 'Counterparty Bank',
        'Counterparty_Account_ID': 'Counterparty Account ID',
        'Counterparty_Name': 'Counterparty Name',
        
        # Transaction details
        'Intermediary_Account': 'Intermediary Account',
        'Completed_At': 'Completed At',
        'Payment_Channel': 'Payment Channel',
        'Transfer_Reference_No': 'Transfer Reference No.',
        'Transaction_Narrative': 'Transaction Narrative',
        'TX_ID': 'TX ID',
        
        # Customer information
        'Customer_Account_Number': 'Customer Account Number',
        'Account_Name': 'Account Name',
        'Account_Channel': 'Account Channel',
        
        # Other
        'Product': 'Product'
    }
    
    # Apply mappings for unmatched tables
    if any(keyword in table_name for keyword in ['unmatched', 'df_unmatched']):
        for old_name, new_name in column_mappings.items():
            if old_name in df_copy.columns and new_name not in df_copy.columns:
                df_copy[new_name] = df_copy[old_name]
                df_copy = df_copy.drop(columns=[old_name])
                logger.debug(f"Mapped '{old_name}' -> '{new_name}' for {table_name}")
    
    # Also handle reverse mapping (if database has underscore but dataframe has space)
    reverse_mappings = {v: k for k, v in column_mappings.items()}
    for old_name, new_name in reverse_mappings.items():
        if old_name in df_copy.columns and new_name not in df_copy.columns:
            df_copy[new_name] = df_copy[old_name]
            df_copy = df_copy.drop(columns=[old_name])
            logger.debug(f"Mapped '{old_name}' -> '{new_name}' for {table_name}")
    
    return df_copy

# Call this function right after initializing the database
# Add this line in your FXReconDB class's _init_database method or right after db = FXReconDB()

# Call this function after db initialization
# alter_tables_add_missing_columns()

db = FXReconDB()

# # Run dynamic schema update to add all possible columns
# try:
#     added = update_database_schema_dynamic()
#     st.success(f"✅ Database schema updated: {added} columns added")
# except Exception as e:
#     st.warning(f"Schema update warning: {e}")

# --- Helper Functions for Record Management ---
def generate_record_id():
    return str(uuid.uuid4())

def add_unique_ids(df):
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
    if 'local matched' in source_lower:
        return 'deleted_local_matched'
    elif 'local unmatched' in source_lower:
        return 'deleted_local_unmatched'
    elif 'foreign matched' in source_lower:
        return 'deleted_foreign_matched'
    elif 'foreign unmatched' in source_lower:
        return 'deleted_foreign_unmatched'
    elif 'bank' in source_lower:
        return 'deleted_bank_records'
    return f"deleted_{source_lower.replace(' ', '_')}"

def get_moved_df_name(source_name, target_name):
    target_lower = target_name.lower()
    if 'local matched' in target_lower:
        return 'moved_local_matched'
    elif 'local unmatched' in target_lower:
        return 'moved_local_unmatched'
    elif 'foreign matched' in target_lower:
        return 'moved_foreign_matched'
    elif 'foreign unmatched' in target_lower:
        return 'moved_foreign_unmatched'
    elif 'bank' in target_lower:
        return 'moved_bank_records'
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
    if deleted_df_name in FX_RECON_TABLES:
        db.save_dataframe(FX_RECON_TABLES[deleted_df_name], st.session_state[deleted_df_name])
    
    if 'audit_deletes_log' not in st.session_state:
        st.session_state.audit_deletes_log = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.audit_deletes_log
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log_entries = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log_entries.empty:
            st.session_state.audit_deletes_log = pd.concat([existing_log, new_log_entries[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)
    
    if 'audit_deletes_log' in FX_RECON_TABLES and not st.session_state.audit_deletes_log.empty:
        db.save_dataframe(FX_RECON_TABLES['audit_deletes_log'], st.session_state.audit_deletes_log)
    
    remaining_source_with_numbers = add_row_numbers(remaining_source)
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_row_numbers(remaining_source.copy())
    
    main_df_mapping = {
        'Local Matched Adjustments': 'matched_local',
        'Local Unmatched Adjustments': 'unmatched_local',
        'Foreign Matched Adjustments': 'matched_foreign',
        'Foreign Unmatched Adjustments': 'unmatched_foreign',
        'Unmatched Bank Records': 'bank_records'
    }
    if source_name in main_df_mapping:
        main_key = main_df_mapping[source_name]
        if main_key in st.session_state:
            st.session_state[main_key] = remove_row_numbers(remaining_source.copy())
    
    if on_data_change:
        on_data_change(remaining_source.copy())
    
    update_deleted_stats_cards()
    update_moved_stats_cards()
    
    return remaining_source_with_numbers, len(selected_record_ids)

def sync_all_display_dataframes():
    for key in list(st.session_state.keys()):
        if key.endswith('_display_df'):
            base_key = key.replace('_display_df', '')
            if base_key in st.session_state and not st.session_state[base_key].empty:
                st.session_state[key] = add_row_numbers(st.session_state[base_key].copy())

def clear_selection_state(key_prefix):
    selection_key = f"{key_prefix}_selection_state"
    if selection_key in st.session_state:
        st.session_state[selection_key] = {}

def refresh_analytics_dataframes():
    analytics_dataframes = [
        ('matched_local', 'df_matched_adjustments_local'),
        ('matched_foreign', 'df_matched_adjustments_foreign'),
        ('unmatched_local', 'df_unmatched_adjustments_local'),
        ('unmatched_foreign', 'df_unmatched_adjustments_foreign'),
        ('bank_records', 'df_unmatched_bank_records')
    ]
    for session_key, df_key in analytics_dataframes:
        if session_key in st.session_state and not st.session_state[session_key].empty:
            st.session_state[df_key] = st.session_state[session_key].copy()

def update_moved_stats_cards():
    moved_counts = {
        'moved_local_matched': 0, 'moved_local_unmatched': 0,
        'moved_foreign_matched': 0, 'moved_foreign_unmatched': 0,
        'moved_bank_records': 0, 'total_moved': 0
    }
    for key in moved_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            moved_counts[key] = len(st.session_state[key])
    moved_counts['total_moved'] = sum([moved_counts['moved_local_matched'], moved_counts['moved_local_unmatched'],
                                       moved_counts['moved_foreign_matched'], moved_counts['moved_foreign_unmatched'],
                                       moved_counts['moved_bank_records']])
    st.session_state.moved_stats = moved_counts
    return moved_counts

def update_deleted_stats_cards():
    deleted_counts = {
        'deleted_local_matched': 0, 'deleted_local_unmatched': 0,
        'deleted_foreign_matched': 0, 'deleted_foreign_unmatched': 0,
        'deleted_bank_records': 0, 'total_deleted': 0
    }
    for key in deleted_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            deleted_counts[key] = len(st.session_state[key])
    deleted_counts['total_deleted'] = sum([deleted_counts['deleted_local_matched'], deleted_counts['deleted_local_unmatched'],
                                           deleted_counts['deleted_foreign_matched'], deleted_counts['deleted_foreign_unmatched'],
                                           deleted_counts['deleted_bank_records']])
    st.session_state.deleted_stats = deleted_counts
    return deleted_counts

# --- Save/Load Functions ---
def save_current_fx_recon_state_to_db(target_date=None):
    """Save ALL FX Reconciliation data to database - dynamically handles any column"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    saved_count = 0
    saved_items = []
    errors = []
    
    # Dataframes to save with their table mappings
    dataframes_to_save = {
        'matched_local': 'fx_recon_matched_local',
        'unmatched_local': 'fx_recon_unmatched_local',
        'matched_foreign': 'fx_recon_matched_foreign',
        'unmatched_foreign': 'fx_recon_unmatched_foreign',
        'bank_records': 'fx_recon_bank_records',
        'df_matched_adjustments_local': 'fx_recon_df_matched_local',
        'df_unmatched_adjustments_local': 'fx_recon_df_unmatched_local',
        'df_matched_adjustments_foreign': 'fx_recon_df_matched_foreign',
        'df_unmatched_adjustments_foreign': 'fx_recon_df_unmatched_foreign',
        'df_unmatched_bank_records': 'fx_recon_df_unmatched_bank',
        'moved_local_matched': 'fx_recon_moved_local_matched',
        'moved_local_unmatched': 'fx_recon_moved_local_unmatched',
        'moved_foreign_matched': 'fx_recon_moved_foreign_matched',
        'moved_foreign_unmatched': 'fx_recon_moved_foreign_unmatched',
        'moved_bank_records': 'fx_recon_moved_bank_records',
        'deleted_local_matched': 'fx_recon_deleted_local_matched',
        'deleted_local_unmatched': 'fx_recon_deleted_local_unmatched',
        'deleted_foreign_matched': 'fx_recon_deleted_foreign_matched',
        'deleted_foreign_unmatched': 'fx_recon_deleted_foreign_unmatched',
        'deleted_bank_records': 'fx_recon_deleted_bank_records'
    }
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Clear existing data for this date
        for table_name in dataframes_to_save.values():
            try:
                cursor.execute(f"DELETE FROM {table_name} WHERE record_date = ?", (target_date,))
                logger.info(f"Cleared {table_name} for date {target_date}")
            except Exception as e:
                logger.warning(f"Could not clear {table_name}: {e}")
        
        conn.commit()
        
        # Save each dataframe
        for session_key, table_name in dataframes_to_save.items():
            if session_key in st.session_state:
                df = st.session_state[session_key]
                
                if df is None or df.empty:
                    logger.info(f"Skipping {session_key} - empty dataframe")
                    continue
                
                # Prepare the dataframe
                df = df.copy()
                df = ensure_record_ids(df)
                df = add_audit_columns(df)
                
                # Map column names
                df = map_dataframe_columns_for_save(df, table_name)
                
                # Add metadata columns
                import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df['record_date'] = target_date
                df['import_date'] = import_date
                df['last_modified'] = import_date
                
                # Get database columns for this table
                cursor.execute(f"PRAGMA table_info({table_name})")
                db_columns = [row[1] for row in cursor.fetchall()]
                
                # Filter dataframe to only columns that exist in the database
                valid_columns = [col for col in df.columns if col in db_columns]
                
                if not valid_columns:
                    logger.warning(f"No valid columns for {table_name}")
                    continue
                
                # Filter dataframe
                df_filtered = df[valid_columns].copy()
                
                # Insert records
                records = df_filtered.to_dict('records')
                columns = list(df_filtered.columns)
                
                # Build INSERT statement with proper quoting
                quoted_columns = [f'"{col}"' if ' ' in col or '/' in col or '-' in col else f'"{col}"' for col in columns]
                columns_str = ','.join(quoted_columns)
                placeholders = ','.join(['?' for _ in columns])
                
                inserted = 0
                for idx, record in enumerate(records):
                    try:
                        values = []
                        for col in columns:
                            val = record.get(col)
                            if pd.isna(val) or val == '':
                                values.append(None)
                            elif isinstance(val, (datetime, pd.Timestamp)):
                                values.append(val.strftime('%Y-%m-%d %H:%M:%S'))
                            elif isinstance(val, (list, dict)):
                                values.append(json.dumps(val, default=str))
                            else:
                                values.append(val)
                        
                        cursor.execute(f'INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})', values)
                        inserted += 1
                    except Exception as e:
                        logger.error(f"Error inserting record {idx} in {table_name}: {e}")
                        errors.append(f"{session_key} row {idx}: {str(e)[:100]}")
                
                if inserted > 0:
                    saved_count += 1
                    saved_items.append(f"{session_key} ({inserted} records)")
                    logger.info(f"Saved {inserted} records to {table_name}")
        
        conn.commit()
        
        # Show results
        if saved_count > 0:
            with st.container():
                st.markdown('<div class="custom-success">', unsafe_allow_html=True)
                st.success(f"✅ FX Reconciliation data saved for date: {target_date}")
                
                if saved_items:
                    st.info("Saved data:\n" + "\n".join(f"• {item}" for item in saved_items[:10]))
                
                if errors:
                    st.warning(f"⚠️ {len(errors)} errors occurred during save (first 5):")
                    for err in errors[:5]:
                        st.code(err)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No data was saved")
            
    except Exception as e:
        logger.error(f"Error in save: {e}", exc_info=True)
        st.error(f"Error saving data: {str(e)}")
        conn.rollback()
        saved_count = 0
    finally:
        conn.close()
    
    return saved_count

def load_single_dataframe_from_db(table_name, target_date):
    """Helper function to load a single dataframe from database"""
    conn = sqlite3.connect(DB_PATH)
    try:
        query = f"SELECT * FROM {table_name} WHERE record_date = ? ORDER BY id"
        df = pd.read_sql_query(query, conn, params=(target_date,))
    except Exception as e:
        logger.error(f"Error loading from {table_name}: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    # Remove internal columns
    if not df.empty:
        cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df

def load_fx_recon_state_from_db(target_date=None):
    """Load ALL FX Reconciliation data from database including audit trails"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load main dataframes directly from their specific tables
    st.session_state.matched_local = load_single_dataframe_from_db('fx_recon_matched_local', target_date)
    st.session_state.unmatched_local = load_single_dataframe_from_db('fx_recon_unmatched_local', target_date)
    st.session_state.matched_foreign = load_single_dataframe_from_db('fx_recon_matched_foreign', target_date)
    st.session_state.unmatched_foreign = load_single_dataframe_from_db('fx_recon_unmatched_foreign', target_date)
    st.session_state.bank_records = load_single_dataframe_from_db('fx_recon_bank_records', target_date)
    
    # Load analytics dataframes
    st.session_state.df_matched_adjustments_local = load_single_dataframe_from_db('fx_recon_df_matched_local', target_date)
    st.session_state.df_unmatched_adjustments_local = load_single_dataframe_from_db('fx_recon_df_unmatched_local', target_date)
    st.session_state.df_matched_adjustments_foreign = load_single_dataframe_from_db('fx_recon_df_matched_foreign', target_date)
    st.session_state.df_unmatched_adjustments_foreign = load_single_dataframe_from_db('fx_recon_df_unmatched_foreign', target_date)
    st.session_state.df_unmatched_bank_records = load_single_dataframe_from_db('fx_recon_df_unmatched_bank', target_date)
    
    # Load moved records from individual tables
    st.session_state.moved_local_matched = load_single_dataframe_from_db('fx_recon_moved_local_matched', target_date)
    st.session_state.moved_local_unmatched = load_single_dataframe_from_db('fx_recon_moved_local_unmatched', target_date)
    st.session_state.moved_foreign_matched = load_single_dataframe_from_db('fx_recon_moved_foreign_matched', target_date)
    st.session_state.moved_foreign_unmatched = load_single_dataframe_from_db('fx_recon_moved_foreign_unmatched', target_date)
    st.session_state.moved_bank_records = load_single_dataframe_from_db('fx_recon_moved_bank_records', target_date)
    
    # Load deleted records from individual tables
    st.session_state.deleted_local_matched = load_single_dataframe_from_db('fx_recon_deleted_local_matched', target_date)
    st.session_state.deleted_local_unmatched = load_single_dataframe_from_db('fx_recon_deleted_local_unmatched', target_date)
    st.session_state.deleted_foreign_matched = load_single_dataframe_from_db('fx_recon_deleted_foreign_matched', target_date)
    st.session_state.deleted_foreign_unmatched = load_single_dataframe_from_db('fx_recon_deleted_foreign_unmatched', target_date)
    st.session_state.deleted_bank_records = load_single_dataframe_from_db('fx_recon_deleted_bank_records', target_date)
    
    # Load audit logs
    try:
        query = "SELECT * FROM fx_recon_audit_moves WHERE import_date LIKE ?"
        audit_moves = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.audit_moves_log = audit_moves if not audit_moves.empty else pd.DataFrame()
    except:
        st.session_state.audit_moves_log = pd.DataFrame()
    
    try:
        query = "SELECT * FROM fx_recon_audit_deletes WHERE import_date LIKE ?"
        audit_deletes = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.audit_deletes_log = audit_deletes if not audit_deletes.empty else pd.DataFrame()
    except:
        st.session_state.audit_deletes_log = pd.DataFrame()
    
    conn.close()
    
    # Add unique IDs and audit columns to main dataframes if missing
    for df_name in ['matched_local', 'unmatched_local', 'matched_foreign', 'unmatched_foreign', 'bank_records',
                    'df_matched_adjustments_local', 'df_unmatched_adjustments_local', 
                    'df_matched_adjustments_foreign', 'df_unmatched_adjustments_foreign', 
                    'df_unmatched_bank_records']:
        if not st.session_state[df_name].empty:
            if '_record_id' not in st.session_state[df_name].columns:
                st.session_state[df_name] = add_unique_ids(st.session_state[df_name])
            st.session_state[df_name] = add_audit_columns(st.session_state[df_name])
    
    # Recalculate stats from loaded data
    update_moved_stats_cards()
    update_deleted_stats_cards()
    
    st.session_state.fx_recon_current_date = target_date
    
    # Load column mappings
    local_col_mapping = db.load_metadata('fx_recon_local_col_mapping', {})
    foreign_col_mapping = db.load_metadata('fx_recon_foreign_col_mapping', {})
    st.session_state.fx_recon_local_col_mapping = local_col_mapping
    st.session_state.fx_recon_foreign_col_mapping = foreign_col_mapping
    
    # Get saved summary for verification
    save_summary = db.load_metadata('fx_recon_save_summary', {})
    
    # CRITICAL: Reinitialize display dataframes for ALL loaded original data
    df_mappings = [
        ('matched_local', 'matched_local_display_df'),
        ('unmatched_local', 'unmatched_local_display_df'),
        ('matched_foreign', 'matched_foreign_display_df'),
        ('unmatched_foreign', 'unmatched_foreign_display_df'),
        ('bank_records', 'bank_records_display_df')
    ]
    
    for source_key, display_key in df_mappings:
        if source_key in st.session_state and not st.session_state[source_key].empty:
            # Create display dataframe with row numbers
            df_with_numbers = add_row_numbers(st.session_state[source_key].copy())
            st.session_state[display_key] = df_with_numbers
        elif display_key not in st.session_state:
            st.session_state[display_key] = pd.DataFrame()
    
    # Ensure moved and deleted dataframes exist
    moved_dfs = ['moved_local_matched', 'moved_local_unmatched', 'moved_foreign_matched', 
                 'moved_foreign_unmatched', 'moved_bank_records', 'audit_moves_log']
    for df_name in moved_dfs:
        if df_name not in st.session_state:
            st.session_state[df_name] = pd.DataFrame()
    
    deleted_dfs = ['deleted_local_matched', 'deleted_local_unmatched', 'deleted_foreign_matched',
                   'deleted_foreign_unmatched', 'deleted_bank_records', 'audit_deletes_log']
    for df_name in deleted_dfs:
        if df_name not in st.session_state:
            st.session_state[df_name] = pd.DataFrame()
    
    # CRITICAL: Reinitialize move targets with KEYS
    st.session_state.move_targets_local_matched = {
        "Local Unmatched": "unmatched_local",
        "Foreign Matched": "matched_foreign",
        "Foreign Unmatched": "unmatched_foreign"
    }
    
    st.session_state.move_targets_local_unmatched = {
        "Local Matched": "matched_local",
        "Foreign Matched": "matched_foreign",
        "Foreign Unmatched": "unmatched_foreign"
    }
    
    st.session_state.move_targets_foreign_matched = {
        "Local Matched": "matched_local",
        "Local Unmatched": "unmatched_local",
        "Foreign Unmatched": "unmatched_foreign"
    }
    
    st.session_state.move_targets_foreign_unmatched = {
        "Local Matched": "matched_local",
        "Local Unmatched": "unmatched_local",
        "Foreign Matched": "matched_foreign"
    }
    
    st.session_state.move_targets_bank = {
        "Local Matched": "matched_local",
        "Local Unmatched": "unmatched_local",
        "Foreign Matched": "matched_foreign",
        "Foreign Unmatched": "unmatched_foreign"
    }
    
    # Sync everything
    sync_all_display_dataframes()
    refresh_analytics_dataframes()
    
    with st.container():
        st.markdown('<div class="custom-success">', unsafe_allow_html=True)
        st.success(f"✅ FX Reconciliation data loaded for date: {target_date}")
        
        # Show summary of loaded data
        summary = []
        if not st.session_state.matched_local.empty:
            summary.append(f"• matched_local: {len(st.session_state.matched_local)} records")
        if not st.session_state.unmatched_local.empty:
            summary.append(f"• unmatched_local: {len(st.session_state.unmatched_local)} records")
        if not st.session_state.matched_foreign.empty:
            summary.append(f"• matched_foreign: {len(st.session_state.matched_foreign)} records")
        if not st.session_state.unmatched_foreign.empty:
            summary.append(f"• unmatched_foreign: {len(st.session_state.unmatched_foreign)} records")
        if not st.session_state.bank_records.empty:
            summary.append(f"• bank_records: {len(st.session_state.bank_records)} records")
        if not st.session_state.moved_local_matched.empty:
            summary.append(f"• moved_local_matched: {len(st.session_state.moved_local_matched)} records")
        if not st.session_state.moved_local_unmatched.empty:
            summary.append(f"• moved_local_unmatched: {len(st.session_state.moved_local_unmatched)} records")
        if not st.session_state.moved_foreign_matched.empty:
            summary.append(f"• moved_foreign_matched: {len(st.session_state.moved_foreign_matched)} records")
        if not st.session_state.moved_foreign_unmatched.empty:
            summary.append(f"• moved_foreign_unmatched: {len(st.session_state.moved_foreign_unmatched)} records")
        if not st.session_state.moved_bank_records.empty:
            summary.append(f"• moved_bank_records: {len(st.session_state.moved_bank_records)} records")
        if not st.session_state.deleted_local_matched.empty:
            summary.append(f"• deleted_local_matched: {len(st.session_state.deleted_local_matched)} records")
        if not st.session_state.deleted_local_unmatched.empty:
            summary.append(f"• deleted_local_unmatched: {len(st.session_state.deleted_local_unmatched)} records")
        if not st.session_state.deleted_foreign_matched.empty:
            summary.append(f"• deleted_foreign_matched: {len(st.session_state.deleted_foreign_matched)} records")
        if not st.session_state.deleted_foreign_unmatched.empty:
            summary.append(f"• deleted_foreign_unmatched: {len(st.session_state.deleted_foreign_unmatched)} records")
        if not st.session_state.deleted_bank_records.empty:
            summary.append(f"• deleted_bank_records: {len(st.session_state.deleted_bank_records)} records")
        
        if summary:
            st.info("Loaded data:\n" + "\n".join(summary))
        st.markdown('</div>', unsafe_allow_html=True)
    
    return len([item for item in summary if item.startswith("•")])

def reset_all_module_dataframes():
    with st.spinner("Resetting all dataframes..."):
        for session_key in FX_RECON_KEYS:
            if session_key in st.session_state:
                if session_key in ['moved_stats', 'deleted_stats']:
                    st.session_state[session_key] = {'total_moved': 0, 'total_deleted': 0}
                else:
                    st.session_state[session_key] = pd.DataFrame()
        
        display_keys = [key for key in st.session_state.keys() if key.endswith('_display_df')]
        for key in display_keys:
            st.session_state[key] = pd.DataFrame()
        
        selection_keys = [key for key in st.session_state.keys() if key.endswith('_selection_state')]
        for key in selection_keys:
            st.session_state[key] = {}
        
        update_moved_stats_cards()
        update_deleted_stats_cards()
    
    return True

# --- Render Functions ---
def render_editable_dataframe(df, title, key_prefix, on_data_change=None, show_delete=True, show_move=True, move_targets=None):
    """Render a single editable dataframe with full functionality"""
    if df is None or df.empty:
        st.info(f"No {title} to display.")
        return df if df is not None else pd.DataFrame()
    
    st.markdown(f"### {title}")
    st.markdown(f"**Total Records: {len(df)}**")
    
    df = ensure_record_ids(df)
    df = add_audit_columns(df)
    
    display_df_key = f"{key_prefix}_display_df"
    original_df_key = key_prefix
    
    if display_df_key not in st.session_state or st.session_state[display_df_key].empty:
        if '#' not in df.columns:
            st.session_state[display_df_key] = add_row_numbers(df.copy())
        else:
            st.session_state[display_df_key] = df.copy()
        if original_df_key not in st.session_state:
            st.session_state[original_df_key] = remove_row_numbers(df.copy())
    
    action_reason = st.text_input("Action Reason (optional):", key=f"{key_prefix}_action_reason", placeholder="Enter reason for moving or deleting these records...")
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.markdown("**Select rows to delete/move:**")
    
    with col2:
        if show_delete and st.button(f"🗑️ Delete Selected", key=f"{key_prefix}_delete_btn"):
            selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
            selected_record_ids = [rid.replace(f"{key_prefix}_select_", "") for rid in selection_state.keys() if selection_state[rid] and rid.startswith(f"{key_prefix}_select_")]
            if selected_record_ids:
                source_df = st.session_state[display_df_key].copy()
                updated_df, deleted_count = delete_selected_rows_with_audit(source_df, selected_record_ids, title, action_reason, df_name=display_df_key, on_data_change=on_data_change)
                if original_df_key in st.session_state:
                    st.session_state[original_df_key] = remove_row_numbers(updated_df.copy())
                sync_all_display_dataframes()
                clear_selection_state(key_prefix)
                refresh_analytics_dataframes()
                update_deleted_stats_cards()
                st.success(f"✅ Deleted {deleted_count} record(s)")
                st.rerun()
            else:
                st.warning("No rows selected for deletion")
    
    with col3:
        if show_move and move_targets:
            if st.button(f"➡️ Move Selected", key=f"{key_prefix}_move_btn"):
                selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
                selected_record_ids = [rid.replace(f"{key_prefix}_select_", "") for rid in selection_state.keys() if selection_state[rid] and rid.startswith(f"{key_prefix}_select_")]
                if selected_record_ids:
                    selected_target = st.session_state.get(f"{key_prefix}_selected_target", list(move_targets.keys())[0] if move_targets else None)
                    if selected_target and selected_target in move_targets:
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
                            
                            # Update main dataframe
                            main_df_mapping = {
                                'Local Matched Adjustments': 'matched_local',
                                'Local Unmatched Adjustments': 'unmatched_local',
                                'Foreign Matched Adjustments': 'matched_foreign',
                                'Foreign Unmatched Adjustments': 'unmatched_foreign',
                                'Unmatched Bank Records': 'bank_records'
                            }
                            if title in main_df_mapping:
                                main_key = main_df_mapping[title]
                                st.session_state[main_key] = remove_row_numbers(new_source.copy())
                            
                            if on_data_change:
                                on_data_change(new_source)
                            
                            clear_selection_state(key_prefix)
                            refresh_analytics_dataframes()
                            update_moved_stats_cards()
                            
                            st.success(f"✅ Moved {len(selected_record_ids)} record(s)")
                            st.rerun()
                    else:
                        st.warning("Please select a target from the dropdown")
                else:
                    st.warning("No rows selected for moving")
    
    with col4:
        df_download = st.session_state[display_df_key].copy()
        if '#' in df_download.columns:
            df_download = df_download.drop(columns=['#'])
        if '_record_id' in df_download.columns:
            df_download = df_download.drop(columns=['_record_id'])
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download CSV", data=csv, file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", key=f"{key_prefix}_download")
    
    with col5:
        if st.button(f"🔄 Refresh", key=f"{key_prefix}_refresh"):
            sync_all_display_dataframes()
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
        df_for_edit_for_display = df_for_edit.drop(columns=columns_to_drop) if columns_to_drop else df_for_edit
        
        edited_df = st.data_editor(df_for_edit_for_display, use_container_width=True, height=min(400, len(df_for_edit_for_display) * 35 + 38), key=f"{key_prefix}_data_editor", num_rows="dynamic")
        
        if not edited_df.equals(df_for_edit_for_display):
            edited_with_ids = ensure_record_ids(edited_df.copy())
            edited_with_audit = add_audit_columns(edited_with_ids)
            updated_with_numbers = add_row_numbers(edited_with_audit)
            st.session_state[display_df_key] = updated_with_numbers
            if original_df_key in st.session_state:
                st.session_state[original_df_key] = remove_row_numbers(edited_with_audit.copy())
            if on_data_change:
                on_data_change(remove_row_numbers(edited_with_audit.copy()))
            refresh_analytics_dataframes()
            update_deleted_stats_cards()
            update_moved_stats_cards()
            st.success("✅ Data updated!")
            st.rerun()
        
        st.markdown("### Select Rows for Batch Operations")
        
        if show_move and move_targets:
            st.markdown("#### Move Target Selection")
            selected_target = st.selectbox("Select target for moving records:", options=list(move_targets.keys()), key=f"{key_prefix}_selected_target")
            if selected_target and selected_target in move_targets:
                target_key = move_targets[selected_target]
                target_df = st.session_state.get(target_key, pd.DataFrame())
                st.info(f"📌 Moving to: {selected_target} (currently {len(target_df)} records)")
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
            
            # Fixed: Added a space as label to avoid empty label warning
            if col1_check.checkbox(" ", value=is_selected, key=checkbox_key, label_visibility="collapsed"):
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
            st.success(f"✅ {selected_count} row(s) selected")
    
    result_df = st.session_state[display_df_key].copy()
    for col in ['_record_id', '#']:
        if col in result_df.columns:
            result_df = result_df.drop(columns=[col])
    return result_df

def render_moved_records_tab():
    st.markdown("### 📋 Moved Records - Audit Trail")
    moved_stats = update_moved_stats_cards()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: st.metric("📋 Moved to Local Matched", moved_stats['moved_local_matched'])
    with col2: st.metric("⚠️ Moved to Local Unmatched", moved_stats['moved_local_unmatched'])
    with col3: st.metric("📋 Moved to Foreign Matched", moved_stats['moved_foreign_matched'])
    with col4: st.metric("⚠️ Moved to Foreign Unmatched", moved_stats['moved_foreign_unmatched'])
    with col5: st.metric("🏦 Moved to Bank Records", moved_stats['moved_bank_records'])
    with col6: st.metric("📊 Total Moved", moved_stats['total_moved'])
    
    st.markdown("---")
    
    moved_df_names = ['moved_local_matched', 'moved_local_unmatched', 'moved_foreign_matched', 'moved_foreign_unmatched', 'moved_bank_records']
    moved_dfs = {}
    for df_name in moved_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            moved_dfs[df_name] = st.session_state[df_name].copy()
    
    if not moved_dfs:
        st.info("No moved records found.")
        return
    
    tabs = st.tabs([name.replace('_', ' ').title() for name in moved_dfs.keys()])
    for tab, (df_name, df) in zip(tabs, moved_dfs.items()):
        with tab:
            st.markdown(f"#### {df_name.replace('_', ' ').title()} - {len(df)} records")
            col1, col2 = st.columns(2)
            with col1:
                if 'moved_by' in df.columns:
                    user_counts = df['moved_by'].value_counts().head(10)
                    st.dataframe(user_counts.reset_index().rename(columns={'index': 'User', 'moved_by': 'Count'}), use_container_width=True)
            with col2:
                if 'moved_at' in df.columns:
                    df_sorted = df.dropna(subset=['moved_at']).copy()
                    if not df_sorted.empty:
                        df_sorted['moved_at'] = pd.to_datetime(df_sorted['moved_at'], errors='coerce')
                        recent = df_sorted.sort_values('moved_at', ascending=False).head(10)
                        display_cols = [col for col in ['moved_at', 'moved_by', 'moved_from', 'move_reason'] if col in recent.columns]
                        st.dataframe(recent[display_cols], use_container_width=True)
            
            st.markdown("---")
            display_df = df.copy()
            cols_to_drop = ['_record_id', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)

def render_deleted_records_tab():
    st.markdown("### 🗑️ Deleted Records - Audit Trail")
    deleted_stats = update_deleted_stats_cards()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: st.metric("🗑️ Deleted from Local Matched", deleted_stats['deleted_local_matched'])
    with col2: st.metric("🗑️ Deleted from Local Unmatched", deleted_stats['deleted_local_unmatched'])
    with col3: st.metric("🗑️ Deleted from Foreign Matched", deleted_stats['deleted_foreign_matched'])
    with col4: st.metric("🗑️ Deleted from Foreign Unmatched", deleted_stats['deleted_foreign_unmatched'])
    with col5: st.metric("🗑️ Deleted from Bank Records", deleted_stats['deleted_bank_records'])
    with col6: st.metric("📊 Total Deleted", deleted_stats['total_deleted'])
    
    st.markdown("---")
    
    deleted_df_names = ['deleted_local_matched', 'deleted_local_unmatched', 'deleted_foreign_matched', 'deleted_foreign_unmatched', 'deleted_bank_records']
    deleted_dfs = {}
    for df_name in deleted_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            deleted_dfs[df_name] = st.session_state[df_name].copy()
    
    if not deleted_dfs:
        st.info("No deleted records found.")
        return
    
    tabs = st.tabs([name.replace('_', ' ').title() for name in deleted_dfs.keys()])
    for tab, (df_name, df) in zip(tabs, deleted_dfs.items()):
        with tab:
            st.markdown(f"#### {df_name.replace('_', ' ').title()} - {len(df)} records")
            col1, col2 = st.columns(2)
            with col1:
                if 'deleted_by' in df.columns:
                    user_counts = df['deleted_by'].value_counts().head(10)
                    st.dataframe(user_counts.reset_index().rename(columns={'index': 'User', 'deleted_by': 'Count'}), use_container_width=True)
            with col2:
                if 'deleted_at' in df.columns:
                    df_sorted = df.dropna(subset=['deleted_at']).copy()
                    if not df_sorted.empty:
                        df_sorted['deleted_at'] = pd.to_datetime(df_sorted['deleted_at'], errors='coerce')
                        recent = df_sorted.sort_values('deleted_at', ascending=False).head(10)
                        display_cols = [col for col in ['deleted_at', 'deleted_by', 'deleted_from', 'delete_reason'] if col in recent.columns]
                        st.dataframe(recent[display_cols], use_container_width=True)
            
            st.markdown("---")
            display_df = df.copy()
            cols_to_drop = ['_record_id', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)

# --- Reconciliation Functions ---
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
    '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
    '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'
]

BANK_NAME_MAP = {
    'central bank of kenya': 'cbk', 'kenya commercial bank': 'kcb',
    'kingdom bank': 'kingdom', 'absa bank': 'absa', 'ABSA Bank': 'absa',
    'equity bank': 'equity', 'i&m bank': 'i&m', 'ncba bank kenya plc': 'ncba', 'ncba bank': 'ncba',
    'sbm bank (kenya) limited': 'sbm', 'sbm bank': 'sbm',
    'baas temporary account': 'baas', 'fx temporary account': 'fx_temp',
    'other temporary account': 'other_temp', 'unclaimed funds': 'unclaimed_funds',
    'yeepay': 'yeepay', 'uba kenya bank': 'uba',
}

FX_EXPECTED_COLUMNS_RECON = {
    'Amount': 'Amount',
    'Operation': 'Operation ',
    'Completed At': 'Completed At',
    'Intermediary Account': 'Intermediary Account',
    'Currency': 'Currency',
    'Status': 'Status'
}

def parse_date(date_str_raw):
    if pd.isna(date_str_raw) or date_str_raw == pd.NaT:
        return None
    if isinstance(date_str_raw, datetime):
        return date_str_raw
    if not isinstance(date_str_raw, str):
        date_str_raw = str(date_str_raw)
    date_str = date_str_raw.partition(" ")[0].strip() if " " in date_str_raw.strip() else date_str_raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def safe_float(x):
    if pd.isna(x) or x is None:
        return None
    try:
        cleaned_x = str(x).replace(',', '').strip()
        return abs(float(cleaned_x))
    except (ValueError, TypeError):
        return None

def resolve_date_column(columns):
    for candidate in ['Value Date', 'Transaction Date', 'MyUnknownColumn', 'Transaction date', 'Date', 'Activity Date']:
        if candidate in columns:
            return candidate
    return None

def resolve_amount_column(columns, operation):
    columns_lower = [col.lower() for col in columns]
    if operation.lower() == 'credit':
        candidates = ['credit', 'deposit', 'amount']
    elif operation.lower() == 'debit':
        candidates = ['debit', 'withdrawal', 'amount']
    else:
        candidates = ['amount', 'value', 'credit', 'deposit', 'debit', 'withdrawal']
    for key in candidates:
        if key in columns_lower:
            return columns[columns_lower.index(key)]
    return None

def get_description_columns(columns):
    for desc in ['Transaction details', 'Transaction', 'Customer reference', 'Narration',
                 'Transaction Details', 'Detail', 'Transaction Remarks:',
                 'TransactionDetails', 'Description', 'Narrative', 'Remarks']:
        if desc in columns:
            return desc
    return None

def get_excel_sheet_names(uploaded_file):
    uploaded_file.seek(0)
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as e:
        return []

def process_uploaded_file(uploaded_file, sheet_name=None):
    uploaded_file.seek(0)
    if uploaded_file.name.endswith('.csv'):
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=enc)
                return df
            except Exception:
                continue
        return pd.DataFrame()
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            return df
        except Exception as e:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def save_uploaded_file(file, filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def save_dataframe(df, filename):
    if df is not None and not df.empty:
        df.to_pickle(os.path.join(CACHE_DIR, filename))

def load_dataframe(filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def reconcile_adjustment_row(adj_row, all_bank_dfs, mode, date_tolerance_days=3, amount_tolerance=1.0,
                             debug=False, matched_adjustments_list=None, unmatched_adjustments_list=None,
                             matched_bank_keys=None, already_matched_adjustments=None, skipped_bank_records=None):
    if matched_adjustments_list is None or unmatched_adjustments_list is None or matched_bank_keys is None:
        raise ValueError("Matched/unmatched lists and matched_bank_keys set must be provided.")
    if skipped_bank_records is None:
        skipped_bank_records = {}
    adjustment_id = adj_row.get('Request ID', '')
    if not adjustment_id:
        adjustment_id = f"{adj_row.get('Completed At', '')}_{adj_row.get('Amount', '')}_{adj_row.get('Intermediary Account', '')}"
    if already_matched_adjustments and adjustment_id in already_matched_adjustments:
        return True
    amount = safe_float(adj_row.get('Amount'))
    if amount is None or pd.isna(amount) or abs(amount) < 0.01:
        return False
    parsed_date = parse_date(adj_row.get('Completed At'))
    if pd.isna(parsed_date) or parsed_date is None:
        return False
    ref_date = datetime(parsed_date.year, parsed_date.month, parsed_date.day)
    operation = str(adj_row.get('Operation', '')).strip().lower()
    if operation not in ['credit', 'debit']:
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Unrecognised operation: {operation}'})
        return False
    status = str(adj_row.get('Status', '')).strip().lower()
    if (mode == 'local' and status != 'successful') or (mode == 'foreign' and status != 'completed'):
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Skipped due to status "{status}" for mode "{mode}"'})
        return False
    tracking_id = adj_row.get('Request ID', '')
    intermediary_account = str(adj_row.get('Intermediary Account', '')).strip()
    payment_channel = adj_row.get('Payment Channel', '')
    counterparty_bank = adj_row.get('Counterparty Bank', '')
    counterparty_account_id = adj_row.get('Counterparty Account ID', '')
    counterparty_name = adj_row.get('Counterparty Name', '')
    transfer_reference_no = adj_row.get('Transfer Reference No.', '')
    transaction_narrative = adj_row.get('Transaction Narrative', '')
    tx_id = adj_row.get('TX ID', '')
    customer_account_number = adj_row.get('Customer Account Number', '')
    account_name = adj_row.get('Account Name', '')
    account_channel = adj_row.get('Account Channel', '')
    product = adj_row.get('Product', '')
    currency = str(adj_row.get('Currency', '')).strip().upper()
    expected_bank_name_adj = None
    expected_currency_adj = None
    if mode == 'local':
        expected_bank_name_adj = intermediary_account.lower()
        expected_currency_adj = currency.upper()
    elif mode == 'foreign':
        parts = intermediary_account.split('-')
        if len(parts) < 2:
            unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Malformed foreign intermediary account'})
            return False
        bank_name_raw = parts[0].strip()
        currency_raw = parts[1].strip().upper()
        expected_bank_name_adj = bank_name_raw.lower()
        expected_currency_adj = currency_raw
    else:
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Invalid mode: {mode}'})
        return False
    target_bank_df_key = None
    for bank_df_key in all_bank_dfs.keys():
        key_parts = bank_df_key.split(' ')
        if len(key_parts) >= 2:
            bank_name_from_key = ' '.join(key_parts[:-1]).lower()
            currency_from_key = key_parts[-1].upper()
        else:
            continue
        bank_name_from_adj_standardized = ""
        for long, short in BANK_NAME_MAP.items():
            if expected_bank_name_adj.startswith(long):
                bank_name_from_adj_standardized = short
                break
        if not bank_name_from_adj_standardized:
            bank_name_from_adj_standardized = expected_bank_name_adj.lower().split(' ')[0]
        bank_name_match = (bank_name_from_adj_standardized == bank_name_from_key)
        currency_match = (expected_currency_adj.lower() == currency_from_key.lower())
        if bank_name_match and currency_match:
            target_bank_df_key = bank_df_key
            break
    if not target_bank_df_key:
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'No matching bank statement found'})
        return False
    bank_df = all_bank_dfs[target_bank_df_key]
    if bank_df.empty:
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': f'Target bank statement ({target_bank_df_key}) is empty'})
        return False
    bank_df_columns = bank_df.columns.tolist()
    date_column = resolve_date_column(bank_df_columns)
    amount_column = resolve_amount_column(bank_df_columns, operation)
    if not date_column or not amount_column:
        unmatched_adjustments_list.append({**adj_row.to_dict(), 'Reason': 'Missing date/amount column in bank statement'})
        return False
    if '_ParsedDate' not in bank_df.columns:
        bank_df['_ParsedDate'] = bank_df[date_column].apply(parse_date)
    if 'Skipped_By_Adjustments' not in bank_df.columns:
        bank_df['Skipped_By_Adjustments'] = ""
    date_matches_df = bank_df[(bank_df['_ParsedDate'].notna()) & (bank_df['_ParsedDate'].between(ref_date - timedelta(days=date_tolerance_days), ref_date + timedelta(days=date_tolerance_days)))].copy()
    match_found = False
    for idx, bank_row in date_matches_df.iterrows():
        bank_amt = safe_float(bank_row.get(amount_column))
        if bank_amt is None or bank_row['_ParsedDate'] is None or abs(bank_amt) < 0.01:
            continue
        amount_diff = abs(abs(bank_amt) - abs(amount))
        if amount_diff <= amount_tolerance:
            bank_record_key_operation = 'debit' if 'debit' in amount_column.lower() or bank_amt < 0 else 'credit'
            if 'credit' in amount_column.lower():
                bank_record_key_operation = 'credit'
            bank_record_key = (target_bank_df_key, bank_row['_ParsedDate'].strftime('%Y-%m-%d'), round(amount, 2), bank_record_key_operation)
            is_already_matched = bank_record_key in matched_bank_keys
            if is_already_matched:
                continue
            matched_record = {
                'Adjustment_Date': parsed_date.strftime('%Y-%m-%d'),
                'Adjustment_Amount': amount,
                'Adjustment_Operation': operation,
                'Adjustment_Intermediary_Account': intermediary_account,
                'Adjustment_Currency': currency,
                'Bank_Table': target_bank_df_key,
                'Bank_Statement_Date': bank_row['_ParsedDate'].strftime('%Y-%m-%d'),
                'Bank_Statement_Amount': bank_amt,
                'Bank_Matched_Column': amount_column,
                'Bank_Row_Index': int(idx),
                'Match_Details': json.dumps({'amount_diff': amount_diff}),
                'Request_ID': tracking_id,
                'Payment_Channel': payment_channel,
                'Counterparty_Bank': counterparty_bank,
                'Counterparty_Account_ID': counterparty_account_id,
                'Counterparty_Name': counterparty_name,
                'Transfer_Reference_No': transfer_reference_no,
                'Transaction_Narrative': transaction_narrative,
                'TX_ID': tx_id,
                'Customer_Account_Number': customer_account_number,
                'Account_Name': account_name,
                'Account_Channel': account_channel,
                'Product': product
            }
            matched_adjustments_list.append(matched_record)
            if "Matched" not in bank_df.columns:
                bank_df["Matched"] = False
            bank_df.loc[idx, "Matched"] = True
            matched_bank_keys.add(bank_record_key)
            match_found = True
            break
    if not match_found:
        unmatched_record = {**adj_row.to_dict(), 'Reason': 'No amount match in bank statement'}
        unmatched_adjustments_list.append(unmatched_record)
    return match_found

def perform_reconciliation_for_mode(fx_df, all_bank_dfs, mode, debug):
    matched_list = []
    unmatched_list = []
    matched_bank_keys = set()
    already_matched_adjustments = set()
    if fx_df.empty:
        st.warning(f"{mode.upper()} FX Data is empty.")
        return
    if not all_bank_dfs:
        st.warning("No Bank Statements processed.")
        return
    progress_bar = st.progress(0)
    total_rows = len(fx_df)
    for idx, (index, row) in enumerate(fx_df.iterrows()):
        reconcile_adjustment_row(row, all_bank_dfs, mode, debug=debug,
                                 matched_adjustments_list=matched_list, unmatched_adjustments_list=unmatched_list,
                                 matched_bank_keys=matched_bank_keys, already_matched_adjustments=already_matched_adjustments)
        progress_bar.progress((idx + 1) / total_rows)
    progress_bar.empty()
    matched_df = pd.DataFrame(matched_list)
    unmatched_df = pd.DataFrame(unmatched_list)
    if not matched_df.empty:
        matched_df = add_unique_ids(matched_df)
        matched_df = add_audit_columns(matched_df)
    if not unmatched_df.empty:
        unmatched_df = add_unique_ids(unmatched_df)
        unmatched_df = add_audit_columns(unmatched_df)
    if mode == 'local':
        st.session_state.df_matched_adjustments_local = matched_df
        st.session_state.df_unmatched_adjustments_local = unmatched_df
        st.session_state.matched_local = matched_df
        st.session_state.unmatched_local = unmatched_df
    else:
        st.session_state.df_matched_adjustments_foreign = matched_df
        st.session_state.df_unmatched_adjustments_foreign = unmatched_df
        st.session_state.matched_foreign = matched_df
        st.session_state.unmatched_foreign = unmatched_df
    st.success(f"Reconciliation for {mode.upper()} FX Data Complete!")
    st.write(f"✅ Matched: {len(matched_df)} | ❌ Unmatched: {len(unmatched_df)}")

def identify_unmatched_bank_records(bank_dfs, matched_bank_keys, unmatched_bank_records_list, debug):
    for bank_key, bank_df in bank_dfs.items():
        if bank_df.empty:
            continue
        bank_df_copy = bank_df.copy()
        bank_df_copy.columns = bank_df_copy.columns.str.strip()
        date_col = 'Date'
        amount_cols = ['Credit', 'Debit']
        description_col = get_description_columns(bank_df_copy.columns.tolist())
        if not date_col or not amount_cols or not description_col:
            continue
        if '_ParsedDate' not in bank_df_copy.columns:
            bank_df_copy['_ParsedDate'] = bank_df_copy[date_col].apply(parse_date)
        for idx, row in bank_df_copy.iterrows():
            row_date = row.get('_ParsedDate')
            if pd.isna(row_date) or not isinstance(row_date, datetime):
                continue
            description = str(row.get(description_col, '')).strip()
            is_matched = False
            for amt_col in amount_cols:
                amt_val = safe_float(row.get(amt_col))
                if amt_val is None or abs(amt_val) < 0.01:
                    continue
                rounded_amt = round(amt_val, 2)
                operation_for_key = 'debit' if 'debit' in amt_col.lower() or amt_val < 0 else 'credit'
                if 'credit' in amt_col.lower():
                    operation_for_key = 'credit'
                bank_record_key = (bank_key, row_date.strftime('%Y-%m-%d'), rounded_amt, operation_for_key)
                if bank_record_key in matched_bank_keys:
                    is_matched = True
                    break
            if not is_matched:
                for amt_col in amount_cols:
                    amt_val = safe_float(row.get(amt_col))
                    if amt_val is not None and abs(amt_val) >= 0.01:
                        unmatched_bank_records_list.append({
                            'Bank_Table': bank_key,
                            'Date': row_date.strftime('%Y-%m-%d'),
                            'Description': description,
                            'Transaction_Type_Column': amt_col,
                            'Amount': round(amt_val, 2),
                            'Original_Row_Index': idx
                        })
                        break

def perform_full_reconciliation(bank_dfs):
    st.subheader("--- Overall Reconciliation Process ---")
    unmatched_bank_records_list_global = []
    matched_bank_keys_global = set()
    if not bank_dfs:
        st.warning("No Bank Statements processed.")
        return
    perform_reconciliation_for_mode(st.session_state.fx_trade_df_local, bank_dfs, 'local', st.session_state.debug_mode)
    perform_reconciliation_for_mode(st.session_state.fx_trade_df_foreign, bank_dfs, 'foreign', st.session_state.debug_mode)
    st.subheader("--- Identifying Global Unmatched Bank Records ---")
    identify_unmatched_bank_records(bank_dfs, matched_bank_keys_global, unmatched_bank_records_list_global, st.session_state.debug_mode)
    st.session_state.df_unmatched_bank_records = pd.DataFrame(unmatched_bank_records_list_global)
    st.session_state.bank_records = st.session_state.df_unmatched_bank_records.copy()
    if not st.session_state.df_unmatched_bank_records.empty:
        st.session_state.df_unmatched_bank_records = add_unique_ids(st.session_state.df_unmatched_bank_records)
        st.session_state.bank_records = add_unique_ids(st.session_state.bank_records)
    st.success("Overall Reconciliation Complete!")
    st.write(f"📄 Total Unmatched Bank Records: {len(st.session_state.df_unmatched_bank_records)}")

def perform_data_analysis_and_visualizations():
    st.subheader("Data Analysis and Visualizations")
    all_empty = (st.session_state.df_matched_adjustments_local.empty and st.session_state.df_unmatched_adjustments_local.empty and
                 st.session_state.df_matched_adjustments_foreign.empty and st.session_state.df_unmatched_adjustments_foreign.empty and
                 st.session_state.df_unmatched_bank_records.empty)
    if all_empty:
        st.warning("No data available for analysis.")
        return
    combined_unmatched = pd.concat([st.session_state.df_unmatched_adjustments_local.assign(Mode='Local FX'),
                                     st.session_state.df_unmatched_adjustments_foreign.assign(Mode='Foreign FX')], ignore_index=True)
    reconciliation_status = pd.DataFrame({
        'Category': ['Matched Local', 'Unmatched Local', 'Matched Foreign', 'Unmatched Foreign', 'Unmatched Bank'],
        'Count': [len(st.session_state.df_matched_adjustments_local), len(st.session_state.df_unmatched_adjustments_local),
                  len(st.session_state.df_matched_adjustments_foreign), len(st.session_state.df_unmatched_adjustments_foreign),
                  len(st.session_state.df_unmatched_bank_records)]
    })
    st.dataframe(reconciliation_status)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Category', y='Count', data=reconciliation_status, palette='viridis', ax=ax)
    ax.set_title('Reconciliation Overview')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    if not combined_unmatched.empty:
        reason_counts = combined_unmatched.groupby(['Mode', 'Reason']).size().reset_index(name='Count')
        st.dataframe(reason_counts)
    if not st.session_state.df_unmatched_bank_records.empty:
        bank_counts = st.session_state.df_unmatched_bank_records['Bank_Table'].value_counts().reset_index()
        bank_counts.columns = ['Bank_Table', 'Count']
        st.dataframe(bank_counts)

# --- Main App Function ---
@require_auth
def fx_reconciliation_app(bank_dfs: dict):
    # FX DataFrames
    if 'fx_trade_df_local' not in st.session_state:
        st.session_state.fx_trade_df_local = pd.DataFrame()
    if 'fx_trade_df_foreign' not in st.session_state:
        st.session_state.fx_trade_df_foreign = pd.DataFrame()
    
    # Column mappings
    if 'fx_column_mappings_local' not in st.session_state:
        st.session_state.fx_column_mappings_local = {
            'Amount': 'Amount',
            'Operation': 'Operation ',
            'Completed At': 'Completed At',
            'Intermediary Account': 'Intermediary Account',
            'Currency': 'Currency',
            'Status': 'Status'
        }
    if 'fx_column_mappings_foreign' not in st.session_state:
        st.session_state.fx_column_mappings_foreign = {
            'Amount': 'Amount',
            'Operation': 'Operation ',
            'Completed At': 'Completed At',
            'Intermediary Account': 'Intermediary Account',
            'Currency': 'Currency',
            'Status': 'Status'
        }
    # Sheet selections
    if 'fx_selected_sheet_local' not in st.session_state:
        st.session_state.fx_selected_sheet_local = None
    if 'fx_selected_sheet_foreign' not in st.session_state:
        st.session_state.fx_selected_sheet_foreign = None
    
    # Raw dataframes
    if 'fx_raw_df_local' not in st.session_state:
        st.session_state.fx_raw_df_local = pd.DataFrame()
    if 'fx_raw_df_foreign' not in st.session_state:
        st.session_state.fx_raw_df_foreign = pd.DataFrame()
    
    # Uploaded file objects
    if 'fx_uploaded_file_obj_local' not in st.session_state:
        st.session_state.fx_uploaded_file_obj_local = None
    if 'fx_uploaded_file_obj_foreign' not in st.session_state:
        st.session_state.fx_uploaded_file_obj_foreign = None
    
    # Sheet names
    if 'fx_sheet_names_local' not in st.session_state:
        st.session_state.fx_sheet_names_local = []
    if 'fx_sheet_names_foreign' not in st.session_state:
        st.session_state.fx_sheet_names_foreign = []

    # Initialize session state
    if 'matched_local' not in st.session_state:
        st.session_state.matched_local = pd.DataFrame()
    if 'unmatched_local' not in st.session_state:
        st.session_state.unmatched_local = pd.DataFrame()
    if 'matched_foreign' not in st.session_state:
        st.session_state.matched_foreign = pd.DataFrame()
    if 'unmatched_foreign' not in st.session_state:
        st.session_state.unmatched_foreign = pd.DataFrame()
    if 'bank_records' not in st.session_state:
        st.session_state.bank_records = pd.DataFrame()
    if 'df_matched_adjustments_local' not in st.session_state:
        st.session_state.df_matched_adjustments_local = pd.DataFrame()
    if 'df_unmatched_adjustments_local' not in st.session_state:
        st.session_state.df_unmatched_adjustments_local = pd.DataFrame()
    if 'df_matched_adjustments_foreign' not in st.session_state:
        st.session_state.df_matched_adjustments_foreign = pd.DataFrame()
    if 'df_unmatched_adjustments_foreign' not in st.session_state:
        st.session_state.df_unmatched_adjustments_foreign = pd.DataFrame()
    if 'df_unmatched_bank_records' not in st.session_state:
        st.session_state.df_unmatched_bank_records = pd.DataFrame()
    if 'fx_trade_df_local' not in st.session_state:
        st.session_state.fx_trade_df_local = pd.DataFrame()
    if 'fx_trade_df_foreign' not in st.session_state:
        st.session_state.fx_trade_df_foreign = pd.DataFrame()
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'moved_stats' not in st.session_state:
        st.session_state.moved_stats = {'total_moved': 0}
    if 'deleted_stats' not in st.session_state:
        st.session_state.deleted_stats = {'total_deleted': 0}
    
    # Moved and deleted dataframes
    for key in ['moved_local_matched', 'moved_local_unmatched', 'moved_foreign_matched', 
                'moved_foreign_unmatched', 'moved_bank_records', 'audit_moves_log',
                'deleted_local_matched', 'deleted_local_unmatched', 'deleted_foreign_matched',
                'deleted_foreign_unmatched', 'deleted_bank_records', 'audit_deletes_log']:
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame()
    
    update_moved_stats_cards()
    update_deleted_stats_cards()
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # # Header
    # st.markdown("""
    # <div class="main-header">
    #     <h1>💱 FX Reconciliation Dashboard</h1>
    #     <p>Match FX adjustments with bank statements, manage exceptions, and track audit history</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # # Data Status Indicators
    # st.markdown("### 📊 Data Status")
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     local_status = "✅" if not st.session_state.fx_trade_df_local.empty else "❌"
    #     st.metric("Local FX Data", f"{local_status} {len(st.session_state.fx_trade_df_local)} records")
    
    # with col2:
    #     foreign_status = "✅" if not st.session_state.fx_trade_df_foreign.empty else "❌"
    #     st.metric("Foreign FX Data", f"{foreign_status} {len(st.session_state.fx_trade_df_foreign)} records")
    
    # with col3:
    #     bank_status = "✅" if bank_dfs else "❌"
    #     st.metric("Bank Statements", f"{bank_status} {len(bank_dfs)} files")
    
    # st.markdown("---")
    
    # ========== FX RECONCILIATION DATA MANAGEMENT SECTION ==========
    st.markdown("### 📅 Data Management")

    available_dates = db.get_available_dates()

    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col1:
        if available_dates:
            selected_load_date = st.selectbox(
                "📅 Select date to load:",
                options=available_dates,
                index=0,
                key="fx_recon_load_date_select"
            )
        else:
            st.selectbox("📅 Select date to load:", options=["No data available"], disabled=True, key="fx_recon_load_date_select")
            selected_load_date = None

    with col2:
        if selected_load_date and available_dates:
            if st.button("📂 Load Data", use_container_width=True, key="load_fx_recon_btn"):
                load_fx_recon_state_from_db(selected_load_date)
                st.rerun()

    with col3:
        current_date = datetime.now().strftime('%Y-%m-%d')
        st.metric("Current Date", current_date)

    with col4:
        if st.button("💾 Save Data", type="primary", use_container_width=True, key="save_fx_recon_btn"):
            save_current_fx_recon_state_to_db()
            st.rerun()

    st.markdown("---")

    # Data Management Action Buttons
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
            
            for table_name in FX_RECON_TABLES.values():
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
    
    # FX Data Upload Section
    with st.expander("📤 Upload FX Trade Data", expanded=False):
        col1, col2 = st.columns(2)
        # Local FX Tracker Upload
        with col1:
            fx_uploaded_file_local = st.file_uploader("Upload Local adjustments", type=["csv", "xlsx"], key="fx_uploader_local")
            
            if fx_uploaded_file_local:
                if st.session_state.get("fx_uploaded_file_obj_local") != fx_uploaded_file_local:
                    st.session_state.fx_uploaded_file_obj_local = fx_uploaded_file_local
                    save_uploaded_file(fx_uploaded_file_local, "local_fx_uploaded." + fx_uploaded_file_local.name.split('.')[-1])
                    st.session_state.fx_sheet_names_local = []
                    st.session_state.fx_selected_sheet_local = None
                    if fx_uploaded_file_local.name.endswith(".xlsx"):
                        st.session_state.fx_sheet_names_local = pd.ExcelFile(fx_uploaded_file_local).sheet_names
                        if st.session_state.fx_sheet_names_local:
                            st.session_state.fx_selected_sheet_local = st.session_state.fx_sheet_names_local[0]
                    else:
                        df_fx_raw_temp = pd.read_csv(fx_uploaded_file_local)
                        st.session_state.fx_raw_df_local = df_fx_raw_temp if not df_fx_raw_temp.empty else pd.DataFrame()
                
                df_fx_raw_local = pd.DataFrame()
                if fx_uploaded_file_local.name.endswith(".xlsx"):
                    selected_sheet_fx_local = st.selectbox("Select Sheet:", st.session_state.fx_sheet_names_local, key="fx_sheet_selector_local",
                        index=st.session_state.fx_sheet_names_local.index(st.session_state.fx_selected_sheet_local) if st.session_state.fx_selected_sheet_local in st.session_state.fx_sheet_names_local else 0)
                    if selected_sheet_fx_local != st.session_state.fx_selected_sheet_local:
                        st.session_state.fx_selected_sheet_local = selected_sheet_fx_local
                    df_fx_raw_local = pd.read_excel(fx_uploaded_file_local, sheet_name=selected_sheet_fx_local)
                    st.session_state.fx_raw_df_local = df_fx_raw_local
                else:
                    df_fx_raw_local = st.session_state.fx_raw_df_local
                
                if not df_fx_raw_local.empty:
                    st.dataframe(df_fx_raw_local.head(3))
                    
                    st.markdown("#### Map Columns")
                    fx_column_mappings_local = {}
                    available_columns_local = [""] + df_fx_raw_local.columns.tolist()
                    
                    for expected_col, default_val in FX_EXPECTED_COLUMNS_RECON.items():
                        initial_selection = st.session_state.fx_column_mappings_local.get(expected_col, default_val if default_val.strip() in [col.strip() for col in df_fx_raw_local.columns] else "")
                        mapped_col = st.selectbox(f"{expected_col}", options=available_columns_local, index=available_columns_local.index(initial_selection) if initial_selection in available_columns_local else 0, key=f"fx_map_local_{expected_col}")
                        fx_column_mappings_local[expected_col] = mapped_col if mapped_col else None
                    
                    if st.button("✅ Process Local Data", key="process_fx_local_btn"):
                        temp_df_fx = df_fx_raw_local.copy()
                        renamed_cols_dict = {mapped: expected for expected, mapped in fx_column_mappings_local.items() if mapped in temp_df_fx.columns}
                        temp_df_fx.rename(columns=renamed_cols_dict, inplace=True)
                        temp_df_fx.columns = temp_df_fx.columns.str.strip()
                        st.session_state.fx_trade_df_local = temp_df_fx
                        st.session_state.fx_column_mappings_local = fx_column_mappings_local
                        st.success("✅ Local Adjustments Processed!")
        
        # Foreign FX Tracker Upload
        with col2:
            fx_uploaded_file_foreign = st.file_uploader("Upload Foreign adjustments", type=["csv", "xlsx"], key="fx_uploader_foreign")
            
            if fx_uploaded_file_foreign:
                if st.session_state.get("fx_uploaded_file_obj_foreign") != fx_uploaded_file_foreign:
                    st.session_state.fx_uploaded_file_obj_foreign = fx_uploaded_file_foreign
                    save_uploaded_file(fx_uploaded_file_foreign, "foreign_fx_uploaded." + fx_uploaded_file_foreign.name.split('.')[-1])
                    st.session_state.fx_sheet_names_foreign = []
                    st.session_state.fx_selected_sheet_foreign = None
                    if fx_uploaded_file_foreign.name.endswith(".xlsx"):
                        st.session_state.fx_sheet_names_foreign = pd.ExcelFile(fx_uploaded_file_foreign).sheet_names
                        if st.session_state.fx_sheet_names_foreign:
                            st.session_state.fx_selected_sheet_foreign = st.session_state.fx_sheet_names_foreign[0]
                    else:
                        df_fx_raw_temp = pd.read_csv(fx_uploaded_file_foreign)
                        st.session_state.fx_raw_df_foreign = df_fx_raw_temp if not df_fx_raw_temp.empty else pd.DataFrame()
                
                df_fx_raw_foreign = pd.DataFrame()
                if fx_uploaded_file_foreign.name.endswith(".xlsx"):
                    selected_sheet_fx_foreign = st.selectbox("Select Sheet:", st.session_state.fx_sheet_names_foreign, key="fx_sheet_selector_foreign",
                        index=st.session_state.fx_sheet_names_foreign.index(st.session_state.fx_selected_sheet_foreign) if st.session_state.fx_selected_sheet_foreign in st.session_state.fx_sheet_names_foreign else 0)
                    if selected_sheet_fx_foreign != st.session_state.fx_selected_sheet_foreign:
                        st.session_state.fx_selected_sheet_foreign = selected_sheet_fx_foreign
                    df_fx_raw_foreign = pd.read_excel(fx_uploaded_file_foreign, sheet_name=selected_sheet_fx_foreign)
                    st.session_state.fx_raw_df_foreign = df_fx_raw_foreign
                else:
                    df_fx_raw_foreign = st.session_state.fx_raw_df_foreign
                
                if not df_fx_raw_foreign.empty:
                    st.dataframe(df_fx_raw_foreign.head(3))
                    
                    st.markdown("#### Map Columns")
                    fx_column_mappings_foreign = {}
                    available_columns_foreign = [""] + df_fx_raw_foreign.columns.tolist()
                    
                    for expected_col, default_val in FX_EXPECTED_COLUMNS_RECON.items():
                        initial_selection = st.session_state.fx_column_mappings_foreign.get(expected_col, default_val if default_val.strip() in [col.strip() for col in df_fx_raw_foreign.columns] else "")
                        mapped_col = st.selectbox(f"{expected_col}", options=available_columns_foreign, index=available_columns_foreign.index(initial_selection) if initial_selection in available_columns_foreign else 0, key=f"fx_map_foreign_{expected_col}")
                        fx_column_mappings_foreign[expected_col] = mapped_col if mapped_col else None
                    
                    if st.button("✅ Process Foreign Data", key="process_fx_foreign_btn"):
                        temp_df_fx = df_fx_raw_foreign.copy()
                        renamed_cols_dict = {mapped: expected for expected, mapped in fx_column_mappings_foreign.items() if mapped in temp_df_fx.columns}
                        temp_df_fx.rename(columns=renamed_cols_dict, inplace=True)
                        temp_df_fx.columns = temp_df_fx.columns.str.strip()
                        st.session_state.fx_trade_df_foreign = temp_df_fx
                        st.session_state.fx_column_mappings_foreign = fx_column_mappings_foreign
                        st.success("✅ Foreign Adjustments Processed!")

    st.markdown("---")
    
    # Reconciliation Section
    if not st.session_state.fx_trade_df_local.empty or not st.session_state.fx_trade_df_foreign.empty:
        st.markdown("### ⚙️ Reconciliation Settings")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        with col2:
            date_tolerance = st.slider("Date Tolerance (days)", 0, 7, 3)
        with col3:
            if st.button("🔄 Run Full Reconciliation", type="primary", use_container_width=True):
                if not bank_dfs:
                    st.error("No bank statements loaded!")
                else:
                    with st.spinner("Running reconciliation..."):
                        perform_full_reconciliation(bank_dfs)
                        st.success("Reconciliation complete!")
                        st.rerun()
        st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Quick Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Local Matched", len(st.session_state.matched_local))
    with col2: st.metric("Local Unmatched", len(st.session_state.unmatched_local))
    with col3: st.metric("Foreign Matched", len(st.session_state.matched_foreign))
    with col4: st.metric("Foreign Unmatched", len(st.session_state.unmatched_foreign))
    with col5: st.metric("Bank Records", len(st.session_state.bank_records))
    
    st.markdown("### 📋 Audit Summary")
    col1, col2 = st.columns(2)
    with col1: st.metric("Total Moved", st.session_state.moved_stats.get('total_moved', 0))
    with col2: st.metric("Total Deleted", st.session_state.deleted_stats.get('total_deleted', 0))
    
    st.markdown("---")
    
    # Move targets
    move_targets_local_matched = {"Local Unmatched": "unmatched_local", "Foreign Matched": "matched_foreign", "Foreign Unmatched": "unmatched_foreign"}
    move_targets_local_unmatched = {"Local Matched": "matched_local", "Foreign Matched": "matched_foreign", "Foreign Unmatched": "unmatched_foreign"}
    move_targets_foreign_matched = {"Local Matched": "matched_local", "Local Unmatched": "unmatched_local", "Foreign Unmatched": "unmatched_foreign"}
    move_targets_foreign_unmatched = {"Local Matched": "matched_local", "Local Unmatched": "unmatched_local", "Foreign Matched": "matched_foreign"}
    move_targets_bank = {"Local Matched": "matched_local", "Local Unmatched": "unmatched_local", "Foreign Matched": "matched_foreign", "Foreign Unmatched": "unmatched_foreign"}
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Local Matched", "⚠️ Local Unmatched", "📋 Foreign Matched", "⚠️ Foreign Unmatched",
        "🏦 Bank Records", "📊 Analysis", "📋 Audit Trail"
    ])
    
    with tab1:
        def update_matched_local(df):
            st.session_state.matched_local = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_local.empty:
                st.session_state.matched_local = add_audit_columns(st.session_state.matched_local)
            st.session_state.df_matched_adjustments_local = st.session_state.matched_local
            update_moved_stats_cards()
            update_deleted_stats_cards()
        render_editable_dataframe(st.session_state.matched_local, "Local Matched Adjustments", "matched_local",
                                  on_data_change=update_matched_local, show_delete=True, show_move=True,
                                  move_targets=move_targets_local_matched)
    
    with tab2:
        def update_unmatched_local(df):
            st.session_state.unmatched_local = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_local.empty:
                st.session_state.unmatched_local = add_audit_columns(st.session_state.unmatched_local)
            st.session_state.df_unmatched_adjustments_local = st.session_state.unmatched_local
            update_moved_stats_cards()
            update_deleted_stats_cards()
        render_editable_dataframe(st.session_state.unmatched_local, "Local Unmatched Adjustments", "unmatched_local",
                                  on_data_change=update_unmatched_local, show_delete=True, show_move=True,
                                  move_targets=move_targets_local_unmatched)
    
    with tab3:
        def update_matched_foreign(df):
            st.session_state.matched_foreign = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_foreign.empty:
                st.session_state.matched_foreign = add_audit_columns(st.session_state.matched_foreign)
            st.session_state.df_matched_adjustments_foreign = st.session_state.matched_foreign
            update_moved_stats_cards()
            update_deleted_stats_cards()
        render_editable_dataframe(st.session_state.matched_foreign, "Foreign Matched Adjustments", "matched_foreign",
                                  on_data_change=update_matched_foreign, show_delete=True, show_move=True,
                                  move_targets=move_targets_foreign_matched)
    
    with tab4:
        def update_unmatched_foreign(df):
            st.session_state.unmatched_foreign = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_foreign.empty:
                st.session_state.unmatched_foreign = add_audit_columns(st.session_state.unmatched_foreign)
            st.session_state.df_unmatched_adjustments_foreign = st.session_state.unmatched_foreign
            update_moved_stats_cards()
            update_deleted_stats_cards()
        render_editable_dataframe(st.session_state.unmatched_foreign, "Foreign Unmatched Adjustments", "unmatched_foreign",
                                  on_data_change=update_unmatched_foreign, show_delete=True, show_move=True,
                                  move_targets=move_targets_foreign_unmatched)
    
    with tab5:
        def update_bank_records(df):
            st.session_state.bank_records = add_unique_ids(df) if not df.empty else df
            if not st.session_state.bank_records.empty:
                st.session_state.bank_records = add_audit_columns(st.session_state.bank_records)
            st.session_state.df_unmatched_bank_records = st.session_state.bank_records
            update_moved_stats_cards()
            update_deleted_stats_cards()
        render_editable_dataframe(st.session_state.bank_records, "Unmatched Bank Records", "bank_records",
                                  on_data_change=update_bank_records, show_delete=True, show_move=False,
                                  move_targets=move_targets_bank)
    
    with tab6:
        if st.button("📈 Generate Analysis Report", use_container_width=True):
            with st.spinner("Generating analysis..."):
                perform_data_analysis_and_visualizations()
    
    with tab7:
        audit_tab1, audit_tab2 = st.tabs(["Moved Records", "Deleted Records"])
        with audit_tab1:
            render_moved_records_tab()
        with audit_tab2:
            render_deleted_records_tab()
    
    # Return dataframes for compatibility
    return (
        st.session_state.matched_local, st.session_state.matched_foreign,
        st.session_state.unmatched_local, st.session_state.unmatched_foreign,
        st.session_state.bank_records,
        st.session_state.moved_local_matched, st.session_state.moved_local_unmatched,
        st.session_state.moved_foreign_matched, st.session_state.moved_foreign_unmatched,
        st.session_state.moved_bank_records,
        st.session_state.deleted_local_matched, st.session_state.deleted_local_unmatched,
        st.session_state.deleted_foreign_matched, st.session_state.deleted_foreign_unmatched,
        st.session_state.deleted_bank_records,
        st.session_state.audit_moves_log, st.session_state.audit_deletes_log,
        pd.DataFrame([st.session_state.moved_stats]) if st.session_state.moved_stats else pd.DataFrame(),
        pd.DataFrame([st.session_state.deleted_stats]) if st.session_state.deleted_stats else pd.DataFrame(),
        st.session_state.df_matched_adjustments_local, st.session_state.df_unmatched_adjustments_local,
        st.session_state.df_matched_adjustments_foreign, st.session_state.df_unmatched_adjustments_foreign,
        st.session_state.df_unmatched_bank_records
    )