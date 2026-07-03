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

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Import authentication functions ---
try:
    from auth_system import (
        get_active_version_id, save_reconciliation_data, load_reconciliation_data, 
        get_all_versions, log_audit, require_auth, load_all_saved_results,
        save_all_reconciliation_results, get_available_data_types, get_available_modules,
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

# --- File Operations ---
def save_uploaded_file(file, filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# --- Database Manager Class for FX Reconciliation ---
# --- Database Manager Class for FX Reconciliation ---
class FXReconDB:
    """Database manager for FX Reconciliation module - preserves ALL original fields"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize all FX Reconciliation database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for each dataframe type with comprehensive column support
        # Matched tables
        matched_table_sql = '''
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
        '''
        cursor.execute(matched_table_sql)
        
        # Create foreign matched table
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
        
        # Unmatched tables
        unmatched_table_sql = '''
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
        '''
        cursor.execute(unmatched_table_sql)
        
        # Create foreign unmatched table
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
        moved_table_sql = '''
            CREATE TABLE IF NOT EXISTS fx_recon_moved_local_matched (
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
        '''
        cursor.execute(moved_table_sql)
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_moved_local_unmatched (
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
            CREATE TABLE IF NOT EXISTS fx_recon_moved_foreign_matched (
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
            CREATE TABLE IF NOT EXISTS fx_recon_moved_foreign_unmatched (
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
            CREATE TABLE IF NOT EXISTS fx_recon_moved_bank_records (
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
        deleted_table_sql = '''
            CREATE TABLE IF NOT EXISTS fx_recon_deleted_local_matched (
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
        '''
        cursor.execute(deleted_table_sql)
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_deleted_local_unmatched (
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_deleted_foreign_matched (
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_deleted_foreign_unmatched (
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_deleted_bank_records (
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
        
        # Analytics/DF tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_df_matched_local (
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
            CREATE TABLE IF NOT EXISTS fx_recon_df_unmatched_local (
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
            CREATE TABLE IF NOT EXISTS fx_recon_df_matched_foreign (
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
            CREATE TABLE IF NOT EXISTS fx_recon_df_unmatched_foreign (
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
            CREATE TABLE IF NOT EXISTS fx_recon_df_unmatched_bank (
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
        
        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fx_recon_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create indexes for better query performance
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_fx_recon_matched_local_date ON fx_recon_matched_local(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_fx_recon_unmatched_local_date ON fx_recon_unmatched_local(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_fx_recon_matched_foreign_date ON fx_recon_matched_foreign(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_fx_recon_unmatched_foreign_date ON fx_recon_unmatched_foreign(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_fx_recon_bank_date ON fx_recon_bank_records(record_date)',
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.debug(f"Index creation skipped: {e}")
        
        conn.commit()
        conn.close()
        logger.info("FX Reconciliation database initialized successfully")
    
    def _get_matched_table_schema(self):
        """Get schema for matched records tables"""
        return '''
            CREATE TABLE IF NOT EXISTS %s (
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
        '''
    
    def _get_unmatched_table_schema(self):
        """Get schema for unmatched records tables"""
        return '''
            CREATE TABLE IF NOT EXISTS %s (
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
        '''
    
    def _get_bank_records_schema(self):
        """Get schema for bank records tables"""
        return '''
            CREATE TABLE IF NOT EXISTS %s (
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
        '''
    
    def _get_moved_schema(self):
        """Get schema for moved records tables"""
        return '''
            CREATE TABLE IF NOT EXISTS %s (
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
        '''
    
    def _get_deleted_schema(self):
        """Get schema for deleted records tables"""
        return '''
            CREATE TABLE IF NOT EXISTS %s (
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
        '''
    
    def _get_audit_moves_schema(self):
        """Get schema for audit moves log table"""
        return '''
            CREATE TABLE IF NOT EXISTS %s (
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
        '''
    
    def _get_audit_deletes_schema(self):
        """Get schema for audit deletes log table"""
        return '''
            CREATE TABLE IF NOT EXISTS %s (
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
        '''
    
    def _serialize_value(self, value):
        """Serialize a value for database storage"""
        if value is None:
            return None
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        return str(value) if not isinstance(value, (float, int)) else value
    
    def save_dataframe(self, table_name, df, record_date=None):
        """Save a dataframe to database - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete all existing records for this date
        try:
            cursor.execute(f"DELETE FROM {table_name} WHERE record_date = ?", (record_date,))
        except Exception as e:
            logger.error(f"Error clearing table {table_name}: {e}")
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all records from {table_name} for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert dataframe to dict for flexible column handling
        records = df.to_dict('records')
        
        for record in records:
            _record_id = str(record.get('_record_id', generate_record_id()))
            
            # Build dynamic insert statement based on available columns
            columns = list(record.keys())
            placeholders = ','.join(['?' for _ in columns])
            columns_str = ','.join([f'"{col}"' for col in columns])
            
            values = []
            for col in columns:
                values.append(self._serialize_value(record.get(col)))
            
            # Add record_date and import_date if not in columns
            if 'record_date' not in columns:
                columns_str += ',"record_date"'
                placeholders += ',?'
                values.append(record_date)
            if 'import_date' not in columns:
                columns_str += ',"import_date"'
                placeholders += ',?'
                values.append(import_date)
            
            try:
                insert_sql = f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                cursor.execute(insert_sql, values)
            except Exception as e:
                logger.error(f"Error inserting into {table_name}: {e}")
                # Try with minimal columns as fallback
                try:
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO {table_name} 
                        (_record_id, record_date, created_at, import_date, last_modified) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (_record_id, record_date, import_date, import_date, import_date))
                except Exception as e2:
                    logger.error(f"Fallback insert also failed: {e2}")
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to {table_name} for date: {record_date}")

    def load_dataframe(self, table_name, target_date=None):
        """Load dataframe from database for a specific date"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
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
    
    def save_metadata(self, key, value):
        """Save metadata to database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO fx_recon_metadata (key, value, updated_at) VALUES (?, ?, ?)',
                    (key, json.dumps(value), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
    
    def load_metadata(self, key, default=None):
        """Load metadata from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT value FROM fx_recon_metadata WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else default
    
    def get_available_dates(self):
        """Get all available dates with FX Recon data"""
        conn = sqlite3.connect(self.db_path)
        dates = set()
        for table_name in FX_RECON_TABLES.values():
            try:
                cursor = conn.execute(f"SELECT DISTINCT record_date FROM {table_name} WHERE record_date IS NOT NULL")
                for row in cursor.fetchall():
                    if row[0]:
                        dates.add(row[0])
            except:
                pass
        conn.close()
        return sorted(list(dates), reverse=True)

# Initialize database
db = FXReconDB()

# --- Helper Functions for Record Management ---
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
    
    if 'local_matched' in source_clean or 'matched_local' in source_clean:
        return 'deleted_local_matched'
    elif 'local_unmatched' in source_clean or 'unmatched_local' in source_clean:
        return 'deleted_local_unmatched'
    elif 'foreign_matched' in source_clean or 'matched_foreign' in source_clean:
        return 'deleted_foreign_matched'
    elif 'foreign_unmatched' in source_clean or 'unmatched_foreign' in source_clean:
        return 'deleted_foreign_unmatched'
    elif 'bank' in source_clean:
        return 'deleted_bank_records'
    else:
        return f"deleted_{source_clean}"

def get_moved_df_name(source_name, target_name):
    """Generate a consistent name for the moved records dataframe"""
    target_clean = target_name.lower().replace(' ', '_')
    
    if 'local_matched' in target_clean or 'matched_local' in target_clean:
        return 'moved_local_matched'
    elif 'local_unmatched' in target_clean or 'unmatched_local' in target_clean:
        return 'moved_local_unmatched'
    elif 'foreign_matched' in target_clean or 'matched_foreign' in target_clean:
        return 'moved_foreign_matched'
    elif 'foreign_unmatched' in target_clean or 'unmatched_foreign' in target_clean:
        return 'moved_foreign_unmatched'
    elif 'bank' in target_clean:
        return 'moved_bank_records'
    else:
        return f"moved_{target_clean}"

def move_records_to_new_df(source_df, selected_record_ids, source_name, target_name, move_reason=""):
    """Move selected records from source to a moved dataframe"""
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
    """Delete selected records and store in deleted dataframe"""
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
    
    # Update audit log
    if 'audit_deletes_log' not in st.session_state:
        st.session_state.audit_deletes_log = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.audit_deletes_log
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log_entries = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log_entries.empty:
            st.session_state.audit_deletes_log = pd.concat([existing_log, new_log_entries[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)
    
    # Save audit log to database
    if not st.session_state.audit_deletes_log.empty:
        db.save_dataframe(FX_RECON_TABLES['audit_deletes_log'], st.session_state.audit_deletes_log)
    
    remaining_source_with_numbers = add_row_numbers(remaining_source)
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_row_numbers(remaining_source.copy())
    
    # Update main dataframe
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
    """Synchronize all display dataframes with their original versions"""
    for key in list(st.session_state.keys()):
        if key.endswith('_display_df'):
            base_key = key.replace('_display_df', '')
            if base_key in st.session_state and not st.session_state[base_key].empty:
                st.session_state[key] = add_row_numbers(st.session_state[base_key].copy())

def clear_selection_state(key_prefix):
    """Clear selection state for a given dataframe"""
    selection_key = f"{key_prefix}_selection_state"
    if selection_key in st.session_state:
        for checkbox_key in list(st.session_state[selection_key].keys()):
            if checkbox_key.startswith(f"{key_prefix}_select_"):
                st.session_state[selection_key][checkbox_key] = False

def refresh_analytics_dataframes():
    """Refresh analytics dataframes from current session state"""
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
    """Update the statistics for moved records cards"""
    moved_counts = {
        'moved_local_matched': 0,
        'moved_local_unmatched': 0,
        'moved_foreign_matched': 0,
        'moved_foreign_unmatched': 0,
        'moved_bank_records': 0,
        'total_moved': 0
    }
    
    for key in moved_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            moved_counts[key] = len(st.session_state[key])
    
    moved_counts['total_moved'] = sum([
        moved_counts['moved_local_matched'],
        moved_counts['moved_local_unmatched'],
        moved_counts['moved_foreign_matched'],
        moved_counts['moved_foreign_unmatched'],
        moved_counts['moved_bank_records']
    ])
    
    st.session_state.moved_stats = moved_counts
    return moved_counts

def update_deleted_stats_cards():
    """Update the statistics for deleted records cards"""
    deleted_counts = {
        'deleted_local_matched': 0,
        'deleted_local_unmatched': 0,
        'deleted_foreign_matched': 0,
        'deleted_foreign_unmatched': 0,
        'deleted_bank_records': 0,
        'total_deleted': 0
    }
    
    for key in deleted_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            deleted_counts[key] = len(st.session_state[key])
    
    deleted_counts['total_deleted'] = sum([
        deleted_counts['deleted_local_matched'],
        deleted_counts['deleted_local_unmatched'],
        deleted_counts['deleted_foreign_matched'],
        deleted_counts['deleted_foreign_unmatched'],
        deleted_counts['deleted_bank_records']
    ])
    
    st.session_state.deleted_stats = deleted_counts
    return deleted_counts

# --- Save/Load Functions for Database ---
def save_current_fx_recon_state_to_db(force_save_all=False, target_date=None):
    """Save all FX Reconciliation dataframes to database"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    saved_count = 0
    saved_items = []
    
    for session_key, table_name in FX_RECON_TABLES.items():
        if session_key in st.session_state:
            df = st.session_state[session_key]
            
            # For stats, save as JSON
            if session_key in ['moved_stats', 'deleted_stats']:
                if force_save_all or session_key not in st.session_state.df_hashes:
                    db.save_metadata(f'fx_recon_{session_key}', df)
                    st.session_state.df_hashes[session_key] = str(hash(str(df)))
                    saved_count += 1
                    saved_items.append(session_key)
            else:
                # Ensure dataframe has record_id and audit columns
                if not df.empty:
                    df = ensure_record_ids(df)
                    df = add_audit_columns(df)
                
                # Save to database
                db.save_dataframe(table_name, df, target_date)
                saved_count += 1
                saved_items.append(session_key)
                
                # Update hash
                if not df.empty:
                    st.session_state.df_hashes[session_key] = str(hash(str(df.values)))
                else:
                    st.session_state.df_hashes[session_key] = None
    
    # Save metadata about the save
    save_summary = {
        'date': target_date,
        'saved_items': saved_items,
        'count': saved_count,
        'saved_by': get_current_user(),
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    db.save_metadata('fx_recon_last_save', save_summary)
    db.save_metadata('fx_recon_last_save_date', target_date)
    
    st.session_state.fx_recon_last_save_date = target_date
    
    if saved_count > 0:
        st.success(f"✅ Saved {saved_count} FX Reconciliation datasets to database for date: {target_date}")
        with st.expander("Saved Items", expanded=False):
            st.write(f"Saved: {', '.join(saved_items[:10])}" + (f" and {len(saved_items)-10} more..." if len(saved_items) > 10 else ""))
    else:
        st.info("No data to save")
    
    return saved_count

def load_fx_recon_state_from_db(target_date=None):
    """Load FX Reconciliation state from database"""
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    loaded_count = 0
    loaded_items = []
    
    for session_key, table_name in FX_RECON_TABLES.items():
        if session_key in ['moved_stats', 'deleted_stats']:
            # Load stats from metadata
            stats = db.load_metadata(f'fx_recon_{session_key}', None)
            if stats is not None:
                st.session_state[session_key] = stats
                loaded_count += 1
                loaded_items.append(session_key)
        else:
            # Load dataframe from database
            df = db.load_dataframe(table_name, target_date)
            if not df.empty or (session_key in st.session_state and not st.session_state[session_key].empty):
                # Ensure record_id and audit columns
                if not df.empty:
                    df = ensure_record_ids(df)
                    df = add_audit_columns(df)
                st.session_state[session_key] = df
                loaded_count += 1
                loaded_items.append(session_key)
    
    # Update stats after loading
    update_moved_stats_cards()
    update_deleted_stats_cards()
    
    # Sync display dataframes
    sync_all_display_dataframes()
    refresh_analytics_dataframes()
    
    st.session_state.fx_recon_current_date = target_date
    
    if loaded_count > 0:
        st.success(f"✅ Loaded {loaded_count} FX Reconciliation datasets from database for date: {target_date}")
        with st.expander("Loaded Items", expanded=False):
            st.write(f"Loaded: {', '.join(loaded_items[:10])}" + (f" and {len(loaded_items)-10} more..." if len(loaded_items) > 10 else ""))
    else:
        st.info(f"No saved data found for date: {target_date}")
    
    return loaded_count

def reset_all_fx_recon_data():
    """Reset all FX Reconciliation module dataframes to empty state"""
    with st.spinner("Resetting all FX Reconciliation data..."):
        for session_key in FX_RECON_KEYS:
            if session_key in st.session_state:
                if session_key in ['moved_stats', 'deleted_stats']:
                    st.session_state[session_key] = {'total_moved': 0, 'total_deleted': 0}
                else:
                    st.session_state[session_key] = pd.DataFrame()
        
        # Clear display dataframes
        display_keys = [key for key in st.session_state.keys() if '_display_df' in key]
        for key in display_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear selection states
        selection_keys = [key for key in st.session_state.keys() if '_selection_state' in key]
        for key in selection_keys:
            if key in st.session_state:
                st.session_state[key] = {}
        
        # Reset change tracking
        st.session_state.df_hashes = {}
        
        # Reset stats
        update_moved_stats_cards()
        update_deleted_stats_cards()
        
    st.success("✅ All FX Reconciliation data has been reset!")
    return True

# --- Constants for expected columns ---
FX_EXPECTED_COLUMNS = {
    "Transaction ID": "trans_id",
    "Amount": "amount",
    "Date": "date",
    "Currency": "currency"
}

# Set Seaborn style for beautiful plots
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6)

# --- Constants and Global Mappings ---
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

PREDEFINED_BANK_CURRENCY_OPTIONS = [
    "Absa KES", "Absa USD", "Absa EUR", "Absa GBP",
    "CBK KES", "CBK USD", "CBK EUR", "CBK GBP",
    "Equity KES", "Equity USD", "Equity EUR", "Equity GBP",
    "I&M KES", "I&M USD", "KCB KES", "KCB USD",
    "Kingdom KES", "Kingdom USD", "NCBA KES", "NCBA USD", "NCBA EUR",
    "SBM KES", "SBM USD", "BAAS Temporary KES", "BAAS Temporary USD",
    "FX Temporary KES", "FX Temporary USD", "Other Temporary KES", "Other Temporary USD",
    "Unclaimed Funds KES", "Unclaimed Funds USD", "Yeepay KES", "Yeepay USD", "Yeepay CNY",
    "UBA KES", "UBA USD", "UBA EUR", "UBA GBP",
]

FX_EXPECTED_COLUMNS_RECON = {
    'Amount': 'Amount',
    'Operation': 'Operation ',
    'Completed At': 'Completed At',
    'Intermediary Account': 'Intermediary Account',
    'Currency': 'Currency',
    'Status': 'Status'
}

BANK_EXPECTED_COLUMNS = {
    'Date': ['Date', 'Transaction Date', 'Value Date', 'Value date'],
    'Credit': ['Credit', 'Credit Amount', 'Money In', 'Deposit', 'Credit amount'],
    'Debit': ['Debit', 'Debit Amount', 'Money Out', 'Withdrawal', 'Debit amount'],
    'Description': ['Description', 'Narrative', 'Transaction Details', 'Customer reference', 'Transaction Remarks:', 'Transaction Details', 'TransactionDetails', 'Transaction\nDetails']
}

def parse_date(date_str_raw):
    """Parses a date string into a datetime object using predefined formats."""
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
    """Safely converts a value to a float, handling commas, non-numeric inputs, and ensuring consistency."""
    if pd.isna(x) or x is None:
        return None
    try:
        cleaned_x = str(x).replace(',', '').strip()
        return abs(float(cleaned_x))
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
        candidates = ['amount', 'value', 'credit', 'deposit', 'debit', 'withdrawal']

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
                 'Transaction Details', 'Detail', 'Transaction Remarks:',
                 'TransactionDetails', 'Description', 'Narrative', 'Remarks']:
        if desc in columns:
            return desc
    return None

def process_uploaded_file(uploaded_file, sheet_name=None):
    """Reads an uploaded file (CSV or Excel) into a DataFrame."""
    uploaded_file.seek(0)
    
    if uploaded_file.name.endswith('.csv'):
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
    uploaded_file.seek(0)
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as e:
        st.error(f"Error getting Excel sheet names: {e}")
        return []

# ========== RENDER FUNCTIONS ==========

def render_moved_records_tab():
    """Render a tab that shows all moved records with audit trail"""
    st.markdown("### 📋 Moved Records - Audit Trail")
    st.markdown("This section shows all records that have been moved between dataframes with their audit trail.")
    
    moved_stats = update_moved_stats_cards()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📋 Moved to Local Matched", moved_stats['moved_local_matched'])
    with col2:
        st.metric("⚠️ Moved to Local Unmatched", moved_stats['moved_local_unmatched'])
    with col3:
        st.metric("📋 Moved to Foreign Matched", moved_stats['moved_foreign_matched'])
    with col4:
        st.metric("⚠️ Moved to Foreign Unmatched", moved_stats['moved_foreign_unmatched'])
    with col5:
        st.metric("🏦 Moved to Bank Records", moved_stats['moved_bank_records'])
    with col6:
        st.metric("📊 Total Moved Records", moved_stats['total_moved'])
    
    st.markdown("---")
    
    moved_df_names = [
        'moved_local_matched', 'moved_local_unmatched', 
        'moved_foreign_matched', 'moved_foreign_unmatched',
        'moved_bank_records'
    ]
    
    moved_dfs = {}
    for df_name in moved_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            df_copy = st.session_state[df_name].copy()
            if 'moved_at' in df_copy.columns:
                df_copy['moved_at'] = pd.to_datetime(df_copy['moved_at'], errors='coerce')
            moved_dfs[df_name] = df_copy
    
    if not moved_dfs:
        st.info("No moved records found. Move records between dataframes to see them here.")
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
                        df_sorted = df.dropna(subset=['moved_at']).copy()
                        if not df_sorted.empty:
                            df_sorted['moved_at'] = pd.to_datetime(df_sorted['moved_at'], errors='coerce')
                            recent = df_sorted.sort_values('moved_at', ascending=False).head(10)
                            display_cols = [col for col in ['moved_at', 'moved_by', 'moved_from', 'move_reason'] if col in recent.columns]
                            st.dataframe(recent[display_cols])
            
            st.markdown("---")
            display_df = df.copy()
            cols_to_drop = ['_record_id', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)

def render_deleted_records_tab():
    """Render a tab that shows all deleted records with audit trail"""
    st.markdown("### 🗑️ Deleted Records - Audit Trail")
    st.markdown("This section shows all records that have been deleted from dataframes with their audit trail.")
    
    deleted_stats = update_deleted_stats_cards()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🗑️ Deleted from Local Matched", deleted_stats['deleted_local_matched'])
    with col2:
        st.metric("🗑️ Deleted from Local Unmatched", deleted_stats['deleted_local_unmatched'])
    with col3:
        st.metric("🗑️ Deleted from Foreign Matched", deleted_stats['deleted_foreign_matched'])
    with col4:
        st.metric("🗑️ Deleted from Foreign Unmatched", deleted_stats['deleted_foreign_unmatched'])
    with col5:
        st.metric("🗑️ Deleted from Bank Records", deleted_stats['deleted_bank_records'])
    with col6:
        st.metric("📊 Total Deleted Records", deleted_stats['total_deleted'])
    
    st.markdown("---")
    
    deleted_df_names = [
        'deleted_local_matched', 'deleted_local_unmatched',
        'deleted_foreign_matched', 'deleted_foreign_unmatched',
        'deleted_bank_records'
    ]
    
    deleted_dfs = {}
    for df_name in deleted_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            df_copy = st.session_state[df_name].copy()
            if 'deleted_at' in df_copy.columns:
                df_copy['deleted_at'] = pd.to_datetime(df_copy['deleted_at'], errors='coerce')
            deleted_dfs[df_name] = df_copy
    
    if not deleted_dfs:
        st.info("No deleted records found. Delete records from dataframes to see them here.")
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
                        df_sorted = df.dropna(subset=['deleted_at']).copy()
                        if not df_sorted.empty:
                            df_sorted['deleted_at'] = pd.to_datetime(df_sorted['deleted_at'], errors='coerce')
                            recent = df_sorted.sort_values('deleted_at', ascending=False).head(10)
                            display_cols = [col for col in ['deleted_at', 'deleted_by', 'deleted_from', 'delete_reason'] if col in recent.columns]
                            st.dataframe(recent[display_cols])
            
            st.markdown("---")
            display_df = df.copy()
            cols_to_drop = ['_record_id', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)

# ========== RECONCILIATION FUNCTIONS ==========

def reconcile_adjustment_row(
    adj_row: pd.Series,
    all_bank_dfs: dict,
    mode: str,
    date_tolerance_days: int = 3,
    amount_tolerance: float = 1.0,
    debug: bool = False,
    matched_adjustments_list: list = None,
    unmatched_adjustments_list: list = None,
    matched_bank_keys: set = None,
    already_matched_adjustments: set = None,
    skipped_bank_records: dict = None
) -> bool:
    """Attempts to reconcile a single adjustment row against all uploaded bank statements."""
    if matched_adjustments_list is None or unmatched_adjustments_list is None or matched_bank_keys is None:
        raise ValueError("Matched/unmatched lists and matched_bank_keys set must be provided.")

    if skipped_bank_records is None:
        skipped_bank_records = {}

    adjustment_id = adj_row.get('Request ID', '')
    if not adjustment_id:
        adjustment_id = f"{adj_row.get('Completed At', '')}_{adj_row.get('Amount', '')}_{adj_row.get('Intermediary Account', '')}"

    if already_matched_adjustments and adjustment_id in already_matched_adjustments:
        if debug:
            st.info(f"⏭️ Skipping already matched adjustment: {adjustment_id}")
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
    bank_match_reason = ""
    for bank_df_key in all_bank_dfs.keys():
        key_parts = bank_df_key.split(' ')
        if len(key_parts) >= 2:
            bank_name_from_key = ' '.join(key_parts[:-1]).lower()
            currency_from_key = key_parts[-1].upper()
        else:
            continue

        bank_name_from_adj_standardized = ""
        matched_bank_name = False
        for long, short in BANK_NAME_MAP.items():
            if expected_bank_name_adj.startswith(long):
                bank_name_from_adj_standardized = short
                matched_bank_name = True
                bank_match_reason = f"Bank name mapped: {long} -> {short}"
                break

        if not matched_bank_name:
            bank_name_from_adj_standardized = expected_bank_name_adj.lower().split(' ')[0]
            bank_match_reason = f"Bank name standardized: {expected_bank_name_adj} -> {bank_name_from_adj_standardized}"

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

    date_matches_df = bank_df[
        (bank_df['_ParsedDate'].notna()) &
        (bank_df['_ParsedDate'].between(
            ref_date - timedelta(days=date_tolerance_days),
            ref_date + timedelta(days=date_tolerance_days)
        ))
    ].copy()

    match_found = False
    matches_count = 0
    
    for idx, bank_row in date_matches_df.iterrows():
        bank_amt_raw = bank_row.get(amount_column)
        bank_amt = safe_float(bank_amt_raw)

        if bank_amt is None or bank_row['_ParsedDate'] is None or abs(bank_amt) < 0.01:
            continue

        amount_diff = abs(abs(bank_amt) - abs(amount))
        if amount_diff <= amount_tolerance:
            matches_count += 1
            
            bank_record_key_operation = 'debit' if 'debit' in amount_column.lower() or bank_amt < 0 else 'credit'
            if 'credit' in amount_column.lower():
                bank_record_key_operation = 'credit'

            bank_record_key = (
                target_bank_df_key,
                bank_row['_ParsedDate'].strftime('%Y-%m-%d'),
                round(amount, 2),
                bank_record_key_operation
            )

            is_already_matched = bank_record_key in matched_bank_keys
            
            match_details = {
                'match_type': 'amount_date_operation',
                'bank_statement_found_in': target_bank_df_key,
                'date_tolerance_days': date_tolerance_days,
                'amount_tolerance': amount_tolerance,
                'actual_amount_difference': amount_diff,
                'date_column_used': date_column,
                'amount_column_used': amount_column,
                'match_sequence_number': matches_count,
                'total_matches_for_adjustment': matches_count,
            }

            if is_already_matched:
                current_skipped = bank_df.loc[idx, "Skipped_By_Adjustments"]
                skipped_list = []
                if current_skipped and current_skipped != "":
                    try:
                        skipped_list = json.loads(current_skipped)
                    except:
                        skipped_list = []
                
                skipped_info = {
                    'adjustment_id': adjustment_id,
                    'adjustment_date': parsed_date.strftime('%Y-%m-%d'),
                    'adjustment_amount': amount,
                    'adjustment_operation': operation,
                    'skipped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'match_details': match_details
                }
                skipped_list.append(skipped_info)
                bank_df.loc[idx, "Skipped_By_Adjustments"] = json.dumps(skipped_list)
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
                'Match_Details': json.dumps(match_details),
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

            if "Matched_Adjustment_Records" not in bank_df.columns:
                bank_df["Matched_Adjustment_Records"] = ""
            
            current_adj_matches_str = bank_df.loc[idx, "Matched_Adjustment_Records"]
            current_adj_matches = []
            if current_adj_matches_str and current_adj_matches_str != "":
                try:
                    current_adj_matches = json.loads(current_adj_matches_str)
                except:
                    current_adj_matches = []
            
            current_adj_matches.append({'adjustment_id': adjustment_id, 'amount': amount, 'date': parsed_date.strftime('%Y-%m-%d')})
            bank_df.loc[idx, "Matched_Adjustment_Records"] = json.dumps(current_adj_matches)

            if "Matched" not in bank_df.columns:
                bank_df["Matched"] = False
            bank_df.loc[idx, "Matched"] = True
            matched_bank_keys.add(bank_record_key)

            match_found = True
            break

    if match_found:
        if already_matched_adjustments is not None:
            already_matched_adjustments.add(adjustment_id)

    if not match_found:
        unmatched_record = {**adj_row.to_dict(), 'Reason': 'No amount match in bank statement'}
        unmatched_adjustments_list.append(unmatched_record)
        
    return match_found

def perform_reconciliation_for_mode(fx_df: pd.DataFrame, all_bank_dfs: dict, mode: str, debug: bool):
    """Performs reconciliation for a specific FX mode (local or foreign)."""
    matched_list = []
    unmatched_list = []
    matched_bank_keys = set()
    already_matched_adjustments = set()
    skipped_bank_records = {}

    if fx_df.empty:
        st.warning(f"{mode.upper()} FX Data is empty. Skipping reconciliation for this mode.")
        return

    if not all_bank_dfs:
        st.warning("No Bank Statements processed. Please upload and process bank data.")
        return

    progress_bar = st.progress(0)
    total_rows = len(fx_df)

    for idx, (index, row) in enumerate(fx_df.iterrows()):
        reconcile_adjustment_row(
            adj_row=row,
            all_bank_dfs=all_bank_dfs,
            mode=mode,
            date_tolerance_days=3,
            amount_tolerance=1.0,
            debug=debug,
            matched_adjustments_list=matched_list,
            unmatched_adjustments_list=unmatched_list,
            matched_bank_keys=matched_bank_keys,
            already_matched_adjustments=already_matched_adjustments,
            skipped_bank_records=skipped_bank_records
        )
        progress_bar.progress((idx + 1) / total_rows)
    
    progress_bar.empty()
    
    # Store results in session state
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
        st.session_state.matched_bank_keys_local = matched_bank_keys
    else:
        st.session_state.df_matched_adjustments_foreign = matched_df
        st.session_state.df_unmatched_adjustments_foreign = unmatched_df
        st.session_state.matched_foreign = matched_df
        st.session_state.unmatched_foreign = unmatched_df
        st.session_state.matched_bank_keys_foreign = matched_bank_keys

    st.success(f"Reconciliation for {mode.upper()} FX Data Complete!")
    st.write(f"✅ Total {mode.upper()} Adjustments Matched: {len(matched_df)}")
    st.write(f"❌ Total {mode.upper()} Adjustments Unmatched: {len(unmatched_df)}")

def identify_unmatched_bank_records(bank_dfs: dict, matched_bank_keys: set, unmatched_bank_records_list: list, debug: bool):
    """Identifies bank records that were not matched by any adjustment."""
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

            if not is_matched_in_any_way:
                final_amt_col = None
                final_amt_val = None
                for amt_col in amount_cols:
                    amt_val = safe_float(row.get(amt_col))
                    if amt_val is not None and abs(amt_val) >= 0.01:
                        final_amt_col = amt_col
                        final_amt_val = round(amt_val, 2)
                        break
                
                if final_amt_val is not None:
                    unmatched_record = {
                        'Bank_Table': bank_key,
                        'Date': row_date.strftime('%Y-%m-%d'),
                        'Description': description,
                        'Transaction_Type_Column': final_amt_col,
                        'Amount': final_amt_val,
                        'Original_Row_Index': idx
                    }
                    unmatched_bank_records_list.append(unmatched_record)

def perform_full_reconciliation(bank_dfs: dict):
    """Main function to perform the reconciliation process for both local and foreign FX data."""
    st.subheader("--- Overall Reconciliation Process ---")

    unmatched_bank_records_list_global = []
    matched_bank_keys_global = set()

    if not bank_dfs:
        st.warning("No Bank Statements processed. Please upload and process bank data in 'Bank Statement Management'.")
        return

    perform_reconciliation_for_mode(
        fx_df=st.session_state.fx_trade_df_local,
        all_bank_dfs=bank_dfs,
        mode='local',
        debug=st.session_state.debug_mode
    )
    matched_bank_keys_global.update(st.session_state.get('matched_bank_keys_local', set()))

    perform_reconciliation_for_mode(
        fx_df=st.session_state.fx_trade_df_foreign,
        all_bank_dfs=bank_dfs,
        mode='foreign',
        debug=st.session_state.debug_mode
    )
    matched_bank_keys_global.update(st.session_state.get('matched_bank_keys_foreign', set()))

    st.subheader("--- Identifying Global Unmatched Bank Records ---")
    identify_unmatched_bank_records(
        bank_dfs=bank_dfs,
        matched_bank_keys=matched_bank_keys_global,
        unmatched_bank_records_list=unmatched_bank_records_list_global,
        debug=st.session_state.debug_mode
    )
    st.session_state.df_unmatched_bank_records = pd.DataFrame(unmatched_bank_records_list_global)
    st.session_state.bank_records = st.session_state.df_unmatched_bank_records.copy()
    
    if not st.session_state.df_unmatched_bank_records.empty:
        st.session_state.df_unmatched_bank_records = add_unique_ids(st.session_state.df_unmatched_bank_records)
        st.session_state.bank_records = add_unique_ids(st.session_state.bank_records)

    st.success("Overall Reconciliation Complete!")
    st.write(f"📄 Total Unmatched Bank Records (Global): {len(st.session_state.df_unmatched_bank_records)}")

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

    combined_unmatched_adjustments = pd.concat([
        st.session_state.df_unmatched_adjustments_local.assign(Mode='Local FX'),
        st.session_state.df_unmatched_adjustments_foreign.assign(Mode='Foreign FX')
    ], ignore_index=True)

    combined_matched_adjustments = pd.concat([
        st.session_state.df_matched_adjustments_local.assign(Mode='Local FX'),
        st.session_state.df_matched_adjustments_foreign.assign(Mode='Foreign FX')
    ], ignore_index=True)

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
    else:
        st.info("No unmatched bank records to analyze.")

    st.success("Data Analysis and Visualizations Complete!")

# ========== RENDER EDITABLE DATAFRAME ==========

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
            
            if col1_check.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed"):
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

# ========== MAIN APP FUNCTION ==========

@require_auth
def fx_reconciliation_app(bank_dfs: dict):
    
    # ========== INITIALIZE ALL SESSION STATE VARIABLES ==========
    
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
    
    # Reconciliation results
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
    
    # Editable dataframe states
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
    
    # Moved records storage
    if 'moved_local_matched' not in st.session_state:
        st.session_state.moved_local_matched = pd.DataFrame()
    if 'moved_local_unmatched' not in st.session_state:
        st.session_state.moved_local_unmatched = pd.DataFrame()
    if 'moved_foreign_matched' not in st.session_state:
        st.session_state.moved_foreign_matched = pd.DataFrame()
    if 'moved_foreign_unmatched' not in st.session_state:
        st.session_state.moved_foreign_unmatched = pd.DataFrame()
    if 'moved_bank_records' not in st.session_state:
        st.session_state.moved_bank_records = pd.DataFrame()
    
    # Deleted records storage
    if 'deleted_local_matched' not in st.session_state:
        st.session_state.deleted_local_matched = pd.DataFrame()
    if 'deleted_local_unmatched' not in st.session_state:
        st.session_state.deleted_local_unmatched = pd.DataFrame()
    if 'deleted_foreign_matched' not in st.session_state:
        st.session_state.deleted_foreign_matched = pd.DataFrame()
    if 'deleted_foreign_unmatched' not in st.session_state:
        st.session_state.deleted_foreign_unmatched = pd.DataFrame()
    if 'deleted_bank_records' not in st.session_state:
        st.session_state.deleted_bank_records = pd.DataFrame()
    
    # Audit logs
    if 'audit_moves_log' not in st.session_state:
        st.session_state.audit_moves_log = pd.DataFrame()
    if 'audit_deletes_log' not in st.session_state:
        st.session_state.audit_deletes_log = pd.DataFrame()
    
    # Stats
    if 'moved_stats' not in st.session_state:
        st.session_state.moved_stats = {
            'moved_local_matched': 0, 'moved_local_unmatched': 0,
            'moved_foreign_matched': 0, 'moved_foreign_unmatched': 0,
            'moved_bank_records': 0, 'total_moved': 0
        }
    if 'deleted_stats' not in st.session_state:
        st.session_state.deleted_stats = {
            'deleted_local_matched': 0, 'deleted_local_unmatched': 0,
            'deleted_foreign_matched': 0, 'deleted_foreign_unmatched': 0,
            'deleted_bank_records': 0, 'total_deleted': 0
        }
    
    # Debug mode
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Hash tracking
    if 'df_hashes' not in st.session_state:
        st.session_state.df_hashes = {}
    
    # Current date tracking
    if 'fx_recon_current_date' not in st.session_state:
        st.session_state.fx_recon_current_date = datetime.now().strftime('%Y-%m-%d')
    if 'fx_recon_last_save_date' not in st.session_state:
        st.session_state.fx_recon_last_save_date = None
    
    # Load any previously saved data from database
    available_dates = db.get_available_dates()
    if available_dates and 'fx_recon_loaded' not in st.session_state:
        latest_date = available_dates[0]
        load_fx_recon_state_from_db(latest_date)
        st.session_state.fx_recon_loaded = True
    
    # Update stats
    update_moved_stats_cards()
    update_deleted_stats_cards()
    
    # ========== SIDEBAR CONTROLS ==========
    
    with st.sidebar:
        st.markdown("## 🏦 FX Reconciliation System")
        
        if 'user' in st.session_state:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <div style="color: white; font-size: 0.8rem;">Logged In User</div>
                <div style="color: white; font-size: 1.2rem; font-weight: bold;">{st.session_state['user']['username']}</div>
                <div style="color: white; font-size: 0.8rem;">Role: {st.session_state['user']['role']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💾 Database Operations")
        
        # Date selection for load/save
        available_dates = db.get_available_dates()
        
        col_db1, col_db2 = st.columns(2)
        
        with col_db1:
            if st.button("💾 Save to DB", use_container_width=True):
                with st.spinner("Saving to database..."):
                    save_current_fx_recon_state_to_db()
                    st.rerun()
        
        with col_db2:
            if available_dates:
                selected_load_date = st.selectbox("Select date:", options=available_dates, key="fx_recon_load_date")
                if st.button("📥 Load from DB", use_container_width=True):
                    with st.spinner(f"Loading data from {selected_load_date}..."):
                        load_fx_recon_state_from_db(selected_load_date)
                        st.rerun()
            else:
                st.info("No saved data found")
        
        st.markdown("---")
        
        # Reset buttons
        st.markdown("### 🔧 Data Management")
        
        col_reset1, col_reset2 = st.columns(2)
        
        with col_reset1:
            if st.button("🗑️ Reset Module", use_container_width=True, help="Reset all current module data"):
                reset_all_fx_recon_data()
                st.rerun()
        
        with col_reset2:
            if st.button("🔄 Refresh Stats", use_container_width=True):
                update_moved_stats_cards()
                update_deleted_stats_cards()
                sync_all_display_dataframes()
                st.success("Stats refreshed!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Current Session Stats")
        
        st.metric("Local Matched", len(st.session_state.matched_local) if not st.session_state.matched_local.empty else 0)
        st.metric("Local Unmatched", len(st.session_state.unmatched_local) if not st.session_state.unmatched_local.empty else 0)
        st.metric("Foreign Matched", len(st.session_state.matched_foreign) if not st.session_state.matched_foreign.empty else 0)
        st.metric("Foreign Unmatched", len(st.session_state.unmatched_foreign) if not st.session_state.unmatched_foreign.empty else 0)
        st.metric("Bank Records", len(st.session_state.bank_records) if not st.session_state.bank_records.empty else 0)
        
        st.markdown("---")
        st.markdown("### 📋 Audit Trail Stats")
        st.metric("Total Moved", st.session_state.moved_stats['total_moved'])
        st.metric("Total Deleted", st.session_state.deleted_stats['total_deleted'])
        
        st.markdown("---")
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
        
        st.markdown("---")
        st.markdown("### 📤 Data Upload")
        
        # Local FX Tracker Upload
        with st.expander("📊 Local Adjustments"):
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
        with st.expander("🌍 Foreign Adjustments"):
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



    # ========== MAIN CONTENT ==========
    
    # CSS for styling
    st.markdown("""
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>💱 FX Reconciliation Dashboard</h1>
        <p>Match FX adjustments with bank statements, manage exceptions, and track audit history</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Metrics Row
    moved_stats = st.session_state.moved_stats
    deleted_stats = st.session_state.deleted_stats
    
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    
    with col1:
        st.metric("✅ Local Matched", len(st.session_state.matched_local) if not st.session_state.matched_local.empty else 0)
    with col2:
        st.metric("⚠️ Local Unmatched", len(st.session_state.unmatched_local) if not st.session_state.unmatched_local.empty else 0)
    with col3:
        st.metric("✅ Foreign Matched", len(st.session_state.matched_foreign) if not st.session_state.matched_foreign.empty else 0)
    with col4:
        st.metric("⚠️ Foreign Unmatched", len(st.session_state.unmatched_foreign) if not st.session_state.unmatched_foreign.empty else 0)
    with col5:
        st.metric("🏦 Bank Records", len(st.session_state.bank_records) if not st.session_state.bank_records.empty else 0)
    with col6:
        st.metric("📋 Moved Records", moved_stats['total_moved'])
    with col7:
        st.metric("🗑️ Deleted Records", deleted_stats['total_deleted'])
    with col8:
        total_fx = (len(st.session_state.matched_local) + len(st.session_state.unmatched_local) + 
                   len(st.session_state.matched_foreign) + len(st.session_state.unmatched_foreign))
        st.metric("💰 Total FX", total_fx)
    
    st.markdown("---")
    
    # Move targets configuration
    move_targets_local_matched = {
        "Local Unmatched": "unmatched_local",
        "Foreign Matched": "matched_foreign",
        "Foreign Unmatched": "unmatched_foreign"
    }
    
    move_targets_local_unmatched = {
        "Local Matched": "matched_local",
        "Foreign Matched": "matched_foreign",
        "Foreign Unmatched": "unmatched_foreign"
    }
    
    move_targets_foreign_matched = {
        "Local Matched": "matched_local",
        "Local Unmatched": "unmatched_local",
        "Foreign Unmatched": "unmatched_foreign"
    }
    
    move_targets_foreign_unmatched = {
        "Local Matched": "matched_local",
        "Local Unmatched": "unmatched_local",
        "Foreign Matched": "matched_foreign"
    }
    
    # Reconciliation button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🔄 Perform Full Reconciliation", use_container_width=True):
            if not bank_dfs:
                st.error("Please upload and process bank statements first!")
            else:
                with st.spinner("Running reconciliation..."):
                    perform_full_reconciliation(bank_dfs)
                    # Update stats after reconciliation
                    update_moved_stats_cards()
                    update_deleted_stats_cards()
                    sync_all_display_dataframes()
                st.success("Reconciliation completed successfully!")
                st.balloons()
                st.rerun()
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Local Matched", "⚠️ Local Unmatched", "📋 Foreign Matched", "⚠️ Foreign Unmatched",
        "🏦 Bank Records", "📊 Analysis", "📋 Moved Records", "🗑️ Deleted Records"
    ])
    
    # Tab 1: Local Matched
    with tab1:
        def update_matched_local(df):
            st.session_state.matched_local = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_local.empty:
                st.session_state.matched_local = add_audit_columns(st.session_state.matched_local)
            st.session_state.df_matched_adjustments_local = st.session_state.matched_local
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe(
            st.session_state.matched_local,
            "Local Matched Adjustments",
            "matched_local",
            on_data_change=update_matched_local,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_local_matched
        )
    
    # Tab 2: Local Unmatched
    with tab2:
        def update_unmatched_local(df):
            st.session_state.unmatched_local = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_local.empty:
                st.session_state.unmatched_local = add_audit_columns(st.session_state.unmatched_local)
            st.session_state.df_unmatched_adjustments_local = st.session_state.unmatched_local
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe(
            st.session_state.unmatched_local,
            "Local Unmatched Adjustments",
            "unmatched_local",
            on_data_change=update_unmatched_local,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_local_unmatched
        )
    
    # Tab 3: Foreign Matched
    with tab3:
        def update_matched_foreign(df):
            st.session_state.matched_foreign = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_foreign.empty:
                st.session_state.matched_foreign = add_audit_columns(st.session_state.matched_foreign)
            st.session_state.df_matched_adjustments_foreign = st.session_state.matched_foreign
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe(
            st.session_state.matched_foreign,
            "Foreign Matched Adjustments",
            "matched_foreign",
            on_data_change=update_matched_foreign,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_foreign_matched
        )
    
    # Tab 4: Foreign Unmatched
    with tab4:
        def update_unmatched_foreign(df):
            st.session_state.unmatched_foreign = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_foreign.empty:
                st.session_state.unmatched_foreign = add_audit_columns(st.session_state.unmatched_foreign)
            st.session_state.df_unmatched_adjustments_foreign = st.session_state.unmatched_foreign
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe(
            st.session_state.unmatched_foreign,
            "Foreign Unmatched Adjustments",
            "unmatched_foreign",
            on_data_change=update_unmatched_foreign,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_foreign_unmatched
        )
    
    # Tab 5: Bank Records
    with tab5:
        def update_bank_records(df):
            st.session_state.bank_records = add_unique_ids(df) if not df.empty else df
            if not st.session_state.bank_records.empty:
                st.session_state.bank_records = add_audit_columns(st.session_state.bank_records)
            st.session_state.df_unmatched_bank_records = st.session_state.bank_records
            update_moved_stats_cards()
            update_deleted_stats_cards()
        
        render_editable_dataframe(
            st.session_state.bank_records,
            "Unmatched Bank Records",
            "bank_records",
            on_data_change=update_bank_records,
            show_delete=True,
            show_move=False
        )
    
    # Tab 6: Analysis
    with tab6:
        if st.button("📈 Generate Analysis Report", use_container_width=True):
            with st.spinner("Generating analysis..."):
                refresh_analytics_dataframes()
                perform_data_analysis_and_visualizations()
    
    # Tab 7: Moved Records
    with tab7:
        render_moved_records_tab()
    
    # Tab 8: Deleted Records
    with tab8:
        render_deleted_records_tab()
    
    # Return all dataframes for compatibility with main_dashboard
    return (
        st.session_state.matched_local if not st.session_state.matched_local.empty else pd.DataFrame(),
        st.session_state.matched_foreign if not st.session_state.matched_foreign.empty else pd.DataFrame(),
        st.session_state.unmatched_local if not st.session_state.unmatched_local.empty else pd.DataFrame(),
        st.session_state.unmatched_foreign if not st.session_state.unmatched_foreign.empty else pd.DataFrame(),
        st.session_state.bank_records if not st.session_state.bank_records.empty else pd.DataFrame(),
        st.session_state.moved_local_matched if not st.session_state.moved_local_matched.empty else pd.DataFrame(),
        st.session_state.moved_local_unmatched if not st.session_state.moved_local_unmatched.empty else pd.DataFrame(),
        st.session_state.moved_foreign_matched if not st.session_state.moved_foreign_matched.empty else pd.DataFrame(),
        st.session_state.moved_foreign_unmatched if not st.session_state.moved_foreign_unmatched.empty else pd.DataFrame(),
        st.session_state.moved_bank_records if not st.session_state.moved_bank_records.empty else pd.DataFrame(),
        st.session_state.deleted_local_matched if not st.session_state.deleted_local_matched.empty else pd.DataFrame(),
        st.session_state.deleted_local_unmatched if not st.session_state.deleted_local_unmatched.empty else pd.DataFrame(),
        st.session_state.deleted_foreign_matched if not st.session_state.deleted_foreign_matched.empty else pd.DataFrame(),
        st.session_state.deleted_foreign_unmatched if not st.session_state.deleted_foreign_unmatched.empty else pd.DataFrame(),
        st.session_state.deleted_bank_records if not st.session_state.deleted_bank_records.empty else pd.DataFrame(),
        st.session_state.audit_moves_log if not st.session_state.audit_moves_log.empty else pd.DataFrame(),
        st.session_state.audit_deletes_log if not st.session_state.audit_deletes_log.empty else pd.DataFrame(),
        pd.DataFrame([st.session_state.moved_stats]) if st.session_state.moved_stats else pd.DataFrame(),
        pd.DataFrame([st.session_state.deleted_stats]) if st.session_state.deleted_stats else pd.DataFrame(),
        st.session_state.df_matched_adjustments_local if not st.session_state.df_matched_adjustments_local.empty else pd.DataFrame(),
        st.session_state.df_unmatched_adjustments_local if not st.session_state.df_unmatched_adjustments_local.empty else pd.DataFrame(),
        st.session_state.df_matched_adjustments_foreign if not st.session_state.df_matched_adjustments_foreign.empty else pd.DataFrame(),
        st.session_state.df_unmatched_adjustments_foreign if not st.session_state.df_unmatched_adjustments_foreign.empty else pd.DataFrame(),
        st.session_state.df_unmatched_bank_records if not st.session_state.df_unmatched_bank_records.empty else pd.DataFrame()
    )