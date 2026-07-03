# combine_match_results_page.py
import streamlit as st
import pandas as pd
import sqlite3
import json
import uuid
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import os

from auth_system import log_audit, require_auth

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Constants ---
CROSS_MATCH_DB_PATH = "data/cross_match_reconciliation.db"
os.makedirs("data", exist_ok=True)

# Custom CSS for better UI
CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #6c5ce7 0%, #a8a4e6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #6c5ce7;
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
    .dataframe-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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

# --- Database Manager Class for Cross Match ---
class CrossMatchReconciliationDB:
    """Database manager for Cross Match Reconciliation data"""
    
    def __init__(self, db_path=CROSS_MATCH_DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables for cross match"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Newly matched bank records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_match_newly_matched_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Date TEXT,
                Bank TEXT,
                Credit REAL,
                Debit REAL,
                Amount REAL,
                Description TEXT,
                Bank_Table_Name TEXT,
                Match_Source TEXT,
                Matched_Index TEXT,
                Match_Reason TEXT,
                Match_Confidence TEXT,
                Bank_Record_Index INTEGER,
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
        
        # Still unmatched bank records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_match_still_unmatched_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Date TEXT,
                Bank TEXT,
                Credit REAL,
                Debit REAL,
                Amount REAL,
                Description TEXT,
                Bank_Table_Name TEXT,
                Mismatch_Reason TEXT,
                Bank_Record_Index INTEGER,
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
        
        # Combined bank records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_match_combined_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Date TEXT,
                Bank TEXT,
                Credit REAL,
                Debit REAL,
                Amount REAL,
                Description TEXT,
                Bank_Table_Name TEXT,
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
        
        # Unique unmatched records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_match_unique_unmatched (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Date TEXT,
                Bank TEXT,
                Credit REAL,
                Debit REAL,
                Amount REAL,
                Description TEXT,
                Bank_Table_Name TEXT,
                Mismatch_Reason TEXT,
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
        
        # Moved records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_match_moved_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                source_table TEXT,
                record_type TEXT,
                original_record_json TEXT,
                moved_by TEXT,
                moved_from TEXT,
                moved_to TEXT,
                moved_at TEXT,
                move_reason TEXT,
                move_type TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Deleted records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_match_deleted_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                source_table TEXT,
                record_type TEXT,
                original_record_json TEXT,
                deleted_by TEXT,
                deleted_at TEXT,
                delete_reason TEXT,
                deleted_from TEXT,
                source_dataframe TEXT,
                import_date TEXT,
                last_modified TEXT
            )
        ''')
        
        # Audit logs tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_match_audit_moves_log (
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
            CREATE TABLE IF NOT EXISTS cross_match_audit_deletes_log (
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
            CREATE TABLE IF NOT EXISTS cross_match_reconciliation_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_cross_match_newly_matched_date ON cross_match_newly_matched_bank(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_cross_match_still_unmatched_date ON cross_match_still_unmatched_bank(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_cross_match_combined_date ON cross_match_combined_bank(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_cross_match_moved_date ON cross_match_moved_records(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_cross_match_deleted_date ON cross_match_deleted_records(record_date)',
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Cross Match database initialized successfully")
    
    
    def _serialize_value(self, value):
        if value is None:
            return None
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        return str(value) if not isinstance(value, (float, int)) else value
    
    def _deserialize_boolean(self, value):
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes']
        return False
    
    def save_newly_matched_df(self, df, record_date=None):
        """Save newly matched bank records"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_newly_matched_bank WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_cross_match_record_id()))
            cursor.execute('''
                INSERT INTO cross_match_newly_matched_bank (
                    _record_id, record_date, sys_created_at, Date, Bank, Credit, Debit, Amount,
                    Description, Bank_Table_Name, Match_Source, Matched_Index, Match_Reason,
                    Match_Confidence, Bank_Record_Index, deleted_by, deleted_at, delete_reason,
                    moved_by, moved_from, moved_at, move_reason, move_type, moved_to,
                    import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank')),
                self._serialize_value(row.get('Credit', 0)),
                self._serialize_value(row.get('Debit', 0)),
                self._serialize_value(row.get('Amount', 0)),
                self._serialize_value(row.get('Description', '')),
                self._serialize_value(row.get('Bank_Table_Name', '')),
                self._serialize_value(row.get('Match_Source', '')),
                self._serialize_value(row.get('Matched_Index', '')),
                self._serialize_value(row.get('Match_Reason', '')),
                self._serialize_value(row.get('Match_Confidence', '')),
                self._serialize_value(row.get('Bank_Record_Index', 0)),
                self._serialize_value(row.get('deleted_by', '')),
                self._serialize_value(row.get('deleted_at', '')),
                self._serialize_value(row.get('delete_reason', '')),
                self._serialize_value(row.get('moved_by', '')),
                self._serialize_value(row.get('moved_from', '')),
                self._serialize_value(row.get('moved_at', '')),
                self._serialize_value(row.get('move_reason', '')),
                self._serialize_value(row.get('move_type', '')),
                self._serialize_value(row.get('moved_to', '')),
                import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_newly_matched_bank for date: {record_date}")
    
    def save_still_unmatched_df(self, df, record_date=None):
        """Save still unmatched bank records"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_still_unmatched_bank WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_cross_match_record_id()))
            cursor.execute('''
                INSERT INTO cross_match_still_unmatched_bank (
                    _record_id, record_date, sys_created_at, Date, Bank, Credit, Debit, Amount,
                    Description, Bank_Table_Name, Mismatch_Reason, Bank_Record_Index,
                    deleted_by, deleted_at, delete_reason, moved_by, moved_from, moved_at,
                    move_reason, move_type, moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank')),
                self._serialize_value(row.get('Credit', 0)),
                self._serialize_value(row.get('Debit', 0)),
                self._serialize_value(row.get('Amount', 0)),
                self._serialize_value(row.get('Description', '')),
                self._serialize_value(row.get('Bank_Table_Name', '')),
                self._serialize_value(row.get('Mismatch_Reason', '')),
                self._serialize_value(row.get('Bank_Record_Index', 0)),
                self._serialize_value(row.get('deleted_by', '')),
                self._serialize_value(row.get('deleted_at', '')),
                self._serialize_value(row.get('delete_reason', '')),
                self._serialize_value(row.get('moved_by', '')),
                self._serialize_value(row.get('moved_from', '')),
                self._serialize_value(row.get('moved_at', '')),
                self._serialize_value(row.get('move_reason', '')),
                self._serialize_value(row.get('move_type', '')),
                self._serialize_value(row.get('moved_to', '')),
                import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_still_unmatched_bank for date: {record_date}")
    
    def save_combined_df(self, df, record_date=None):
        """Save combined bank records"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_combined_bank WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_cross_match_record_id()))
            cursor.execute('''
                INSERT INTO cross_match_combined_bank (
                    _record_id, record_date, sys_created_at, Date, Bank, Credit, Debit, Amount,
                    Description, Bank_Table_Name, deleted_by, deleted_at, delete_reason,
                    moved_by, moved_from, moved_at, move_reason, move_type, moved_to,
                    import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank')),
                self._serialize_value(row.get('Credit', 0)),
                self._serialize_value(row.get('Debit', 0)),
                self._serialize_value(row.get('Amount', 0)),
                self._serialize_value(row.get('Description', '')),
                self._serialize_value(row.get('Bank_Table_Name', '')),
                self._serialize_value(row.get('deleted_by', '')),
                self._serialize_value(row.get('deleted_at', '')),
                self._serialize_value(row.get('delete_reason', '')),
                self._serialize_value(row.get('moved_by', '')),
                self._serialize_value(row.get('moved_from', '')),
                self._serialize_value(row.get('moved_at', '')),
                self._serialize_value(row.get('move_reason', '')),
                self._serialize_value(row.get('move_type', '')),
                self._serialize_value(row.get('moved_to', '')),
                import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_combined_bank for date: {record_date}")
    
    def save_unique_unmatched_df(self, df, record_date=None):
        """Save unique unmatched records"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_unique_unmatched WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_cross_match_record_id()))
            cursor.execute('''
                INSERT INTO cross_match_unique_unmatched (
                    _record_id, record_date, sys_created_at, Date, Bank, Credit, Debit, Amount,
                    Description, Bank_Table_Name, Mismatch_Reason, deleted_by, deleted_at,
                    delete_reason, moved_by, moved_from, moved_at, move_reason, move_type,
                    moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank')),
                self._serialize_value(row.get('Credit', 0)),
                self._serialize_value(row.get('Debit', 0)),
                self._serialize_value(row.get('Amount', 0)),
                self._serialize_value(row.get('Description', '')),
                self._serialize_value(row.get('Bank_Table_Name', '')),
                self._serialize_value(row.get('Mismatch_Reason', '')),
                self._serialize_value(row.get('deleted_by', '')),
                self._serialize_value(row.get('deleted_at', '')),
                self._serialize_value(row.get('delete_reason', '')),
                self._serialize_value(row.get('moved_by', '')),
                self._serialize_value(row.get('moved_from', '')),
                self._serialize_value(row.get('moved_at', '')),
                self._serialize_value(row.get('move_reason', '')),
                self._serialize_value(row.get('move_type', '')),
                self._serialize_value(row.get('moved_to', '')),
                import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_unique_unmatched for date: {record_date}")
    
    def save_moved_records(self, df, record_date=None):
        """Save moved records"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_moved_records WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_cross_match_record_id()))
            record_dict = row.to_dict()
            original_record_json = json.dumps(record_dict, default=str)
            
            cursor.execute('''
                INSERT INTO cross_match_moved_records (
                    _record_id, record_date, sys_created_at, source_table, record_type,
                    original_record_json, moved_by, moved_from, moved_to, moved_at,
                    move_reason, move_type, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('moved_from', 'unknown')),
                self._serialize_value(row.get('record_type', '')),
                original_record_json,
                self._serialize_value(row.get('moved_by')),
                self._serialize_value(row.get('moved_from')),
                self._serialize_value(row.get('moved_to')),
                self._serialize_value(row.get('moved_at')),
                self._serialize_value(row.get('move_reason')),
                self._serialize_value(row.get('move_type')),
                import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_moved_records for date: {record_date}")
    
    def save_deleted_records(self, df, record_date=None):
        """Save deleted records"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_deleted_records WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_cross_match_record_id()))
            record_dict = row.to_dict()
            original_record_json = json.dumps(record_dict, default=str)
            
            cursor.execute('''
                INSERT INTO cross_match_deleted_records (
                    _record_id, record_date, sys_created_at, source_table, record_type,
                    original_record_json, deleted_by, deleted_at, delete_reason,
                    deleted_from, source_dataframe, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('deleted_from', 'unknown')),
                self._serialize_value(row.get('record_type', '')),
                original_record_json,
                self._serialize_value(row.get('deleted_by')),
                self._serialize_value(row.get('deleted_at')),
                self._serialize_value(row.get('delete_reason')),
                self._serialize_value(row.get('deleted_from')),
                self._serialize_value(row.get('source_dataframe')),
                import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_deleted_records for date: {record_date}")
    
    def save_audit_moves(self, df, record_date=None):
        """Save audit moves log"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_audit_moves_log WHERE import_date LIKE ?", (f"{record_date}%",))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO cross_match_audit_moves_log (
                    _record_id, timestamp, user, record_type, record_id,
                    from_location, to_location, details, import_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self._serialize_value(row.get('_record_id')),
                self._serialize_value(row.get('timestamp')),
                self._serialize_value(row.get('user')),
                self._serialize_value(row.get('record_type')),
                self._serialize_value(row.get('record_id')),
                self._serialize_value(row.get('from_location')),
                self._serialize_value(row.get('to_location')),
                self._serialize_value(row.get('details')),
                import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_audit_moves_log for date: {record_date}")
    
    def save_audit_deletes(self, df, record_date=None):
        """Save audit deletes log"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cross_match_audit_deletes_log WHERE import_date LIKE ?", (f"{record_date}%",))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO cross_match_audit_deletes_log (
                    _record_id, timestamp, user, record_type, record_id,
                    details, deleted_record, import_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self._serialize_value(row.get('_record_id')),
                self._serialize_value(row.get('timestamp')),
                self._serialize_value(row.get('user')),
                self._serialize_value(row.get('record_type')),
                self._serialize_value(row.get('record_id')),
                self._serialize_value(row.get('details')),
                self._serialize_value(row.get('deleted_record')),
                import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to cross_match_audit_deletes_log for date: {record_date}")
    
    def save_cross_match_data(self, target_date=None):
        """Save ALL Cross Match data"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get current session state data
        newly_matched_df = st.session_state.get('cross_match_newly_matched', pd.DataFrame())
        still_unmatched_df = st.session_state.get('cross_match_still_unmatched', pd.DataFrame())
        combined_df = st.session_state.get('cross_match_combined', pd.DataFrame())
        unique_unmatched_df = st.session_state.get('cross_match_unique_unmatched', pd.DataFrame())
        
        # Save each dataframe
        self.save_newly_matched_df(newly_matched_df, target_date)
        self.save_still_unmatched_df(still_unmatched_df, target_date)
        self.save_combined_df(combined_df, target_date)
        self.save_unique_unmatched_df(unique_unmatched_df, target_date)
        
        # Save moved records
        all_moved_records = []
        moved_keys = ['cross_match_moved_newly_matched', 'cross_match_moved_still_unmatched',
                      'cross_match_moved_combined', 'cross_match_moved_unique_unmatched']
        
        for key in moved_keys:
            df = st.session_state.get(key, pd.DataFrame())
            if not df.empty:
                all_moved_records.append(df)
        
        if all_moved_records:
            combined_moved = pd.concat(all_moved_records, ignore_index=True)
            self.save_moved_records(combined_moved, target_date)
        else:
            self.save_moved_records(pd.DataFrame(), target_date)
        
        # Save deleted records
        all_deleted_records = []
        deleted_keys = ['cross_match_deleted_newly_matched', 'cross_match_deleted_still_unmatched',
                        'cross_match_deleted_combined', 'cross_match_deleted_unique_unmatched']
        
        for key in deleted_keys:
            df = st.session_state.get(key, pd.DataFrame())
            if not df.empty:
                all_deleted_records.append(df)
        
        if all_deleted_records:
            combined_deleted = pd.concat(all_deleted_records, ignore_index=True)
            self.save_deleted_records(combined_deleted, target_date)
        else:
            self.save_deleted_records(pd.DataFrame(), target_date)
        
        # Save audit logs
        audit_moves = st.session_state.get('cross_match_audit_moves_log', pd.DataFrame())
        audit_deletes = st.session_state.get('cross_match_audit_deletes_log', pd.DataFrame())
        
        self.save_audit_moves(audit_moves, target_date)
        self.save_audit_deletes(audit_deletes, target_date)
        
        # Save metadata
        self.save_metadata('cross_match_last_save_date', target_date)
        self.save_metadata('cross_match_moved_stats', st.session_state.get('cross_match_moved_stats', {}))
        self.save_metadata('cross_match_deleted_stats', st.session_state.get('cross_match_deleted_stats', {}))
        
        st.session_state.cross_match_last_save_date = target_date
        
        # Show summary
        with st.container():
            st.markdown('<div class="custom-success">', unsafe_allow_html=True)
            st.success(f"✅ Cross Match data saved for date: {target_date}")
            
            summary = []
            if not newly_matched_df.empty:
                summary.append(f"• newly_matched: {len(newly_matched_df)} records")
            if not still_unmatched_df.empty:
                summary.append(f"• still_unmatched: {len(still_unmatched_df)} records")
            if not combined_df.empty:
                summary.append(f"• combined: {len(combined_df)} records")
            if not unique_unmatched_df.empty:
                summary.append(f"• unique_unmatched: {len(unique_unmatched_df)} records")
            
            if summary:
                st.info("Saved data:\n" + "\n".join(summary))
            st.markdown('</div>', unsafe_allow_html=True)
        
        return target_date
    
    def load_cross_match_data(self, target_date=None):
        """Load ALL Cross Match data"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        
        # Load main dataframes
        st.session_state.cross_match_newly_matched = self.load_newly_matched_df(target_date)
        st.session_state.cross_match_still_unmatched = self.load_still_unmatched_df(target_date)
        st.session_state.cross_match_combined = self.load_combined_df(target_date)
        st.session_state.cross_match_unique_unmatched = self.load_unique_unmatched_df(target_date)
        
        # Load moved records
        query = "SELECT * FROM cross_match_moved_records WHERE record_date = ?"
        all_moved = pd.read_sql_query(query, conn, params=(target_date,))
        
        if not all_moved.empty:
            st.session_state.cross_match_moved_newly_matched = all_moved[all_moved['moved_to'].str.contains('Newly Matched', na=False)].copy()
            st.session_state.cross_match_moved_still_unmatched = all_moved[all_moved['moved_to'].str.contains('Still Unmatched', na=False)].copy()
            st.session_state.cross_match_moved_combined = all_moved[all_moved['moved_to'].str.contains('Combined', na=False)].copy()
            st.session_state.cross_match_moved_unique_unmatched = all_moved[all_moved['moved_to'].str.contains('Unique Unmatched', na=False)].copy()
        else:
            st.session_state.cross_match_moved_newly_matched = pd.DataFrame()
            st.session_state.cross_match_moved_still_unmatched = pd.DataFrame()
            st.session_state.cross_match_moved_combined = pd.DataFrame()
            st.session_state.cross_match_moved_unique_unmatched = pd.DataFrame()
        
        # Load deleted records
        query = "SELECT * FROM cross_match_deleted_records WHERE record_date = ?"
        all_deleted = pd.read_sql_query(query, conn, params=(target_date,))
        
        if not all_deleted.empty:
            st.session_state.cross_match_deleted_newly_matched = all_deleted[all_deleted['deleted_from'].str.contains('Newly Matched', na=False)].copy()
            st.session_state.cross_match_deleted_still_unmatched = all_deleted[all_deleted['deleted_from'].str.contains('Still Unmatched', na=False)].copy()
            st.session_state.cross_match_deleted_combined = all_deleted[all_deleted['deleted_from'].str.contains('Combined', na=False)].copy()
            st.session_state.cross_match_deleted_unique_unmatched = all_deleted[all_deleted['deleted_from'].str.contains('Unique Unmatched', na=False)].copy()
        else:
            st.session_state.cross_match_deleted_newly_matched = pd.DataFrame()
            st.session_state.cross_match_deleted_still_unmatched = pd.DataFrame()
            st.session_state.cross_match_deleted_combined = pd.DataFrame()
            st.session_state.cross_match_deleted_unique_unmatched = pd.DataFrame()
        
        # Load audit logs
        query = "SELECT * FROM cross_match_audit_moves_log WHERE import_date LIKE ?"
        audit_moves = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.cross_match_audit_moves_log = audit_moves if not audit_moves.empty else pd.DataFrame()
        
        query = "SELECT * FROM cross_match_audit_deletes_log WHERE import_date LIKE ?"
        audit_deletes = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.cross_match_audit_deletes_log = audit_deletes if not audit_deletes.empty else pd.DataFrame()
        
        conn.close()
        
        # Add unique IDs and audit columns
        for df_name in ['cross_match_newly_matched', 'cross_match_still_unmatched',
                        'cross_match_combined', 'cross_match_unique_unmatched']:
            if not st.session_state[df_name].empty:
                if '_record_id' not in st.session_state[df_name].columns:
                    st.session_state[df_name] = add_cross_match_unique_ids(st.session_state[df_name])
                st.session_state[df_name] = add_cross_match_audit_columns(st.session_state[df_name])
        
        # Recalculate stats
        update_cross_match_moved_stats()
        update_cross_match_deleted_stats()
        
        st.session_state.cross_match_current_date = target_date
        
        with st.container():
            st.markdown('<div class="custom-success">', unsafe_allow_html=True)
            st.success(f"✅ Cross Match data loaded for date: {target_date}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        return target_date
    
    def load_newly_matched_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM cross_match_newly_matched_bank WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_still_unmatched_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM cross_match_still_unmatched_bank WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_combined_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM cross_match_combined_bank WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_unique_unmatched_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM cross_match_unique_unmatched WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def get_available_dates(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT DISTINCT record_date FROM (
                SELECT record_date FROM cross_match_newly_matched_bank
                UNION SELECT record_date FROM cross_match_still_unmatched_bank
                UNION SELECT record_date FROM cross_match_combined_bank
                UNION SELECT record_date FROM cross_match_unique_unmatched
            ) WHERE record_date IS NOT NULL ORDER BY record_date DESC
        ''')
        dates = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return dates
    
    def save_metadata(self, key, value):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO cross_match_reconciliation_metadata (key, value, updated_at) VALUES (?, ?, ?)',
                    (key, json.dumps(value, default=str), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
    
    def load_metadata(self, key, default=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT value FROM cross_match_reconciliation_metadata WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else default


# Initialize database
cross_match_db = CrossMatchReconciliationDB()


def get_available_cross_match_dates():
    """Get all available dates with Cross Match data"""
    return cross_match_db.get_available_dates()


# --- Helper Functions for Record Management ---
def generate_cross_match_record_id():
    return f"cross_{uuid.uuid4()}"


def add_cross_match_unique_ids(df):
    """Add unique record IDs to dataframe"""
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '_record_id' not in df_copy.columns:
        df_copy['_record_id'] = [generate_cross_match_record_id() for _ in range(len(df_copy))]
    return df_copy


def ensure_cross_match_record_ids(df):
    """Ensure dataframe has _record_id column"""
    if df is None or df.empty:
        return df
    if '_record_id' not in df.columns:
        return add_cross_match_unique_ids(df)
    return df


def add_cross_match_audit_columns(df):
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


def add_cross_match_row_numbers(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '#' in df_copy.columns:
        df_copy = df_copy.drop(columns=['#'])
    df_copy.insert(0, '#', range(1, len(df_copy) + 1))
    return df_copy


def remove_cross_match_row_numbers(df):
    if df is None or df.empty:
        return df
    if '#' in df.columns:
        return df.drop(columns=['#'])
    return df


def get_cross_match_current_user():
    if 'user' in st.session_state:
        return st.session_state['user'].get('username', 'unknown')
    return 'unknown_user'


def get_cross_match_deleted_df_name(source_name):
    source_clean = source_name.lower().replace(' ', '_')
    if 'newly_matched' in source_clean:
        return 'cross_match_deleted_newly_matched'
    elif 'still_unmatched' in source_clean:
        return 'cross_match_deleted_still_unmatched'
    elif 'combined' in source_clean:
        return 'cross_match_deleted_combined'
    elif 'unique_unmatched' in source_clean:
        return 'cross_match_deleted_unique_unmatched'
    return f"cross_match_deleted_{source_clean}"


def get_cross_match_moved_df_name(target_name):
    target_clean = target_name.lower().replace(' ', '_')
    if 'newly_matched' in target_clean:
        return 'cross_match_moved_newly_matched'
    elif 'still_unmatched' in target_clean:
        return 'cross_match_moved_still_unmatched'
    elif 'combined' in target_clean:
        return 'cross_match_moved_combined'
    elif 'unique_unmatched' in target_clean:
        return 'cross_match_moved_unique_unmatched'
    return f"cross_match_moved_{target_clean}"


def move_cross_match_records_to_new_df(source_df, selected_record_ids, source_name, target_name, move_reason=""):
    if not selected_record_ids:
        return pd.DataFrame(), source_df
    source_df_copy = source_df.copy() if source_df is not None else pd.DataFrame()
    source_df_copy = ensure_cross_match_record_ids(source_df_copy)
    if '#' in source_df_copy.columns:
        source_df_copy = source_df_copy.drop(columns=['#'])
    selected_records = source_df_copy[source_df_copy['_record_id'].isin(selected_record_ids)].copy()
    remaining_source = source_df_copy[~source_df_copy['_record_id'].isin(selected_record_ids)].reset_index(drop=True)
    if '#' in remaining_source.columns:
        remaining_source = remaining_source.drop(columns=['#'])
    if selected_records.empty:
        return pd.DataFrame(), source_df
    current_user = get_cross_match_current_user()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    selected_records = add_cross_match_audit_columns(selected_records)
    selected_records['moved_by'] = current_user
    selected_records['moved_from'] = source_name
    selected_records['moved_to'] = target_name
    selected_records['moved_at'] = current_time
    selected_records['move_reason'] = move_reason
    selected_records['move_type'] = f"{source_name} → {target_name}"
    return selected_records, remaining_source


def delete_cross_match_records_to_new_df(source_df, selected_record_ids, source_name, delete_reason=""):
    if not selected_record_ids:
        return pd.DataFrame(), source_df
    source_df_copy = source_df.copy() if source_df is not None else pd.DataFrame()
    source_df_copy = ensure_cross_match_record_ids(source_df_copy)
    if '#' in source_df_copy.columns:
        source_df_copy = source_df_copy.drop(columns=['#'])
    selected_records = source_df_copy[source_df_copy['_record_id'].isin(selected_record_ids)].copy()
    remaining_source = source_df_copy[~source_df_copy['_record_id'].isin(selected_record_ids)].reset_index(drop=True)
    if selected_records.empty:
        return pd.DataFrame(), source_df
    current_user = get_cross_match_current_user()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    selected_records = add_cross_match_audit_columns(selected_records)
    selected_records['deleted_by'] = current_user
    selected_records['deleted_at'] = current_time
    selected_records['delete_reason'] = delete_reason
    selected_records['deleted_from'] = source_name
    selected_records['source_dataframe'] = source_name
    return selected_records, remaining_source


def delete_cross_match_selected_rows_with_audit(df, selected_record_ids, source_name, delete_reason="", df_name=None, on_data_change=None):
    if not selected_record_ids:
        return df, 0
    if isinstance(selected_record_ids, str):
        selected_record_ids = [selected_record_ids]
    source_df = df.copy() if df is not None else pd.DataFrame()
    if source_df.empty:
        return df, 0
    source_df = ensure_cross_match_record_ids(source_df)
    if '#' in source_df.columns:
        source_df = source_df.drop(columns=['#'])
    deleted_records, remaining_source = delete_cross_match_records_to_new_df(source_df, selected_record_ids, source_name, delete_reason)
    if deleted_records.empty:
        return df, 0
    deleted_df_name = get_cross_match_deleted_df_name(source_name)
    if deleted_df_name not in st.session_state:
        st.session_state[deleted_df_name] = deleted_records
    else:
        existing = st.session_state[deleted_df_name]
        existing_ids = set(existing['_record_id'].tolist()) if not existing.empty else set()
        new_records = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_records.empty:
            st.session_state[deleted_df_name] = pd.concat([existing, new_records], ignore_index=True)
    
    cross_match_db.save_deleted_records(st.session_state[deleted_df_name])
    
    if 'cross_match_audit_deletes_log' not in st.session_state:
        st.session_state.cross_match_audit_deletes_log = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.cross_match_audit_deletes_log
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log_entries = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log_entries.empty:
            st.session_state.cross_match_audit_deletes_log = pd.concat([existing_log, new_log_entries[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)
    
    cross_match_db.save_audit_deletes(st.session_state.cross_match_audit_deletes_log)
    
    remaining_source_with_numbers = add_cross_match_row_numbers(remaining_source)
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_cross_match_row_numbers(remaining_source.copy())
    
    if on_data_change:
        on_data_change(remaining_source.copy())
    
    update_cross_match_deleted_stats()
    update_cross_match_moved_stats()
    
    return remaining_source_with_numbers, len(selected_record_ids)


def clear_cross_match_selection_state(key_prefix):
    selection_key = f"{key_prefix}_selection_state"
    if selection_key in st.session_state:
        st.session_state[selection_key] = {}


def update_cross_match_moved_stats():
    moved_counts = {
        'cross_match_moved_newly_matched': 0,
        'cross_match_moved_still_unmatched': 0,
        'cross_match_moved_combined': 0,
        'cross_match_moved_unique_unmatched': 0,
        'total_moved': 0
    }
    for key in moved_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            moved_counts[key] = len(st.session_state[key])
    moved_counts['total_moved'] = sum([moved_counts['cross_match_moved_newly_matched'], 
                                       moved_counts['cross_match_moved_still_unmatched'],
                                       moved_counts['cross_match_moved_combined'], 
                                       moved_counts['cross_match_moved_unique_unmatched']])
    st.session_state.cross_match_moved_stats = moved_counts
    return moved_counts


def update_cross_match_deleted_stats():
    deleted_counts = {
        'cross_match_deleted_newly_matched': 0,
        'cross_match_deleted_still_unmatched': 0,
        'cross_match_deleted_combined': 0,
        'cross_match_deleted_unique_unmatched': 0,
        'total_deleted': 0
    }
    for key in deleted_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            deleted_counts[key] = len(st.session_state[key])
    deleted_counts['total_deleted'] = sum([deleted_counts['cross_match_deleted_newly_matched'],
                                           deleted_counts['cross_match_deleted_still_unmatched'],
                                           deleted_counts['cross_match_deleted_combined'],
                                           deleted_counts['cross_match_deleted_unique_unmatched']])
    st.session_state.cross_match_deleted_stats = deleted_counts
    return deleted_counts


def sync_all_cross_match_display_dataframes():
    for key in list(st.session_state.keys()):
        if key.endswith('_display_df'):
            base_key = key.replace('_display_df', '')
            if base_key in st.session_state and not st.session_state[base_key].empty:
                st.session_state[key] = add_cross_match_row_numbers(st.session_state[base_key].copy())


def refresh_cross_match_analytics_dataframes():
    analytics_dataframes = [
        ('cross_match_newly_matched', 'cross_match_newly_matched_analytics'),
        ('cross_match_still_unmatched', 'cross_match_still_unmatched_analytics'),
        ('cross_match_combined', 'cross_match_combined_analytics'),
        ('cross_match_unique_unmatched', 'cross_match_unique_unmatched_analytics')
    ]
    for session_key, df_key in analytics_dataframes:
        if session_key in st.session_state and not st.session_state[session_key].empty:
            st.session_state[df_key] = st.session_state[session_key].copy()


def reset_all_cross_match_dataframes():
    """Reset all Cross Match dataframes to empty state"""
    with st.spinner("Resetting all dataframes..."):
        # Main dataframes
        st.session_state.cross_match_newly_matched = pd.DataFrame()
        st.session_state.cross_match_still_unmatched = pd.DataFrame()
        st.session_state.cross_match_combined = pd.DataFrame()
        st.session_state.cross_match_unique_unmatched = pd.DataFrame()
        
        # Moved records
        st.session_state.cross_match_moved_newly_matched = pd.DataFrame()
        st.session_state.cross_match_moved_still_unmatched = pd.DataFrame()
        st.session_state.cross_match_moved_combined = pd.DataFrame()
        st.session_state.cross_match_moved_unique_unmatched = pd.DataFrame()
        
        # Deleted records
        st.session_state.cross_match_deleted_newly_matched = pd.DataFrame()
        st.session_state.cross_match_deleted_still_unmatched = pd.DataFrame()
        st.session_state.cross_match_deleted_combined = pd.DataFrame()
        st.session_state.cross_match_deleted_unique_unmatched = pd.DataFrame()
        
        # Audit logs
        st.session_state.cross_match_audit_moves_log = pd.DataFrame()
        st.session_state.cross_match_audit_deletes_log = pd.DataFrame()
        
        # Clear display dataframes
        display_keys = [key for key in st.session_state.keys() if key.endswith('_display_df')]
        for key in display_keys:
            st.session_state[key] = pd.DataFrame()
        
        # Clear selection states
        selection_keys = [key for key in st.session_state.keys() if key.endswith('_selection_state')]
        for key in selection_keys:
            st.session_state[key] = {}
        
        # Reset statistics
        st.session_state.cross_match_moved_stats = {
            'cross_match_moved_newly_matched': 0, 'cross_match_moved_still_unmatched': 0,
            'cross_match_moved_combined': 0, 'cross_match_moved_unique_unmatched': 0,
            'total_moved': 0
        }
        st.session_state.cross_match_deleted_stats = {
            'cross_match_deleted_newly_matched': 0, 'cross_match_deleted_still_unmatched': 0,
            'cross_match_deleted_combined': 0, 'cross_match_deleted_unique_unmatched': 0,
            'total_deleted': 0
        }
        
        logger.info("All Cross Match module dataframes have been reset")
    
    return True


def initialize_cross_match_session_state():
    """Initialize all Cross Match related session state variables"""
    
    # Main dataframes
    if 'cross_match_newly_matched' not in st.session_state:
        st.session_state.cross_match_newly_matched = pd.DataFrame()
    if 'cross_match_still_unmatched' not in st.session_state:
        st.session_state.cross_match_still_unmatched = pd.DataFrame()
    if 'cross_match_combined' not in st.session_state:
        st.session_state.cross_match_combined = pd.DataFrame()
    if 'cross_match_unique_unmatched' not in st.session_state:
        st.session_state.cross_match_unique_unmatched = pd.DataFrame()
    
    # Moved records
    if 'cross_match_moved_newly_matched' not in st.session_state:
        st.session_state.cross_match_moved_newly_matched = pd.DataFrame()
    if 'cross_match_moved_still_unmatched' not in st.session_state:
        st.session_state.cross_match_moved_still_unmatched = pd.DataFrame()
    if 'cross_match_moved_combined' not in st.session_state:
        st.session_state.cross_match_moved_combined = pd.DataFrame()
    if 'cross_match_moved_unique_unmatched' not in st.session_state:
        st.session_state.cross_match_moved_unique_unmatched = pd.DataFrame()
    
    # Deleted records
    if 'cross_match_deleted_newly_matched' not in st.session_state:
        st.session_state.cross_match_deleted_newly_matched = pd.DataFrame()
    if 'cross_match_deleted_still_unmatched' not in st.session_state:
        st.session_state.cross_match_deleted_still_unmatched = pd.DataFrame()
    if 'cross_match_deleted_combined' not in st.session_state:
        st.session_state.cross_match_deleted_combined = pd.DataFrame()
    if 'cross_match_deleted_unique_unmatched' not in st.session_state:
        st.session_state.cross_match_deleted_unique_unmatched = pd.DataFrame()
    
    # Audit logs
    if 'cross_match_audit_moves_log' not in st.session_state:
        st.session_state.cross_match_audit_moves_log = pd.DataFrame()
    if 'cross_match_audit_deletes_log' not in st.session_state:
        st.session_state.cross_match_audit_deletes_log = pd.DataFrame()
    
    # Statistics
    if 'cross_match_moved_stats' not in st.session_state:
        st.session_state.cross_match_moved_stats = {
            'cross_match_moved_newly_matched': 0, 'cross_match_moved_still_unmatched': 0,
            'cross_match_moved_combined': 0, 'cross_match_moved_unique_unmatched': 0,
            'total_moved': 0
        }
    if 'cross_match_deleted_stats' not in st.session_state:
        st.session_state.cross_match_deleted_stats = {
            'cross_match_deleted_newly_matched': 0, 'cross_match_deleted_still_unmatched': 0,
            'cross_match_deleted_combined': 0, 'cross_match_deleted_unique_unmatched': 0,
            'total_deleted': 0
        }
    
    # Current date tracking
    if 'cross_match_current_date' not in st.session_state:
        st.session_state.cross_match_current_date = datetime.now().strftime('%Y-%m-%d')
    if 'cross_match_last_save_date' not in st.session_state:
        st.session_state.cross_match_last_save_date = None


def clean_cross_match_dataframe_for_arrow(df):
    """Clean dataframe to make it Arrow-compatible"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if col in ['_record_id', '#']:
            continue
            
        if df_copy[col].dtype == 'object':
            try:
                numeric_series = pd.to_numeric(df_copy[col], errors='coerce')
                if numeric_series.notna().sum() > len(df_copy[col]) * 0.8:
                    df_copy[col] = numeric_series
                else:
                    df_copy[col] = df_copy[col].astype(str)
            except:
                df_copy[col] = df_copy[col].astype(str)
        
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].fillna(0)
            df_copy[col] = df_copy[col].replace([float('inf'), float('-inf')], 0)
    
    return df_copy


# --- Render Functions for Cross Match ---
def render_cross_match_editable_dataframe(df, title, key_prefix, on_data_change=None, show_delete=True, show_move=True, move_targets=None):
    """Render a single editable dataframe with full functionality"""
    
    logger.debug(f"Rendering {title} with {len(df) if df is not None else 0} records")
    
    if df is None or df.empty:
        st.info(f"No {title} to display.")
        return df if df is not None else pd.DataFrame()
    
    st.markdown(f"### {title}")
    st.markdown(f"**Total Records: {len(df)}**")
    
    df = ensure_cross_match_record_ids(df)
    df = add_cross_match_audit_columns(df)
    df = clean_cross_match_dataframe_for_arrow(df)
    
    display_df_key = f"{key_prefix}_cross_match_display_df"
    original_df_key = f"{key_prefix}_cross_match_original_df"
    
    should_initialize = False
    
    if display_df_key not in st.session_state:
        should_initialize = True
    elif st.session_state[display_df_key].empty and not df.empty:
        should_initialize = True
    elif original_df_key in st.session_state and not st.session_state[original_df_key].empty:
        if len(st.session_state[original_df_key]) != len(df):
            should_initialize = True
    
    if should_initialize:
        df_with_ids = ensure_cross_match_record_ids(df)
        df_with_audit = add_cross_match_audit_columns(df_with_ids)
        st.session_state[display_df_key] = add_cross_match_row_numbers(df_with_audit)
        st.session_state[original_df_key] = remove_cross_match_row_numbers(df_with_audit.copy())
        logger.info(f"Created new display dataframe for {key_prefix} with {len(df)} records")
    
    action_reason = st.text_input(
        "Action Reason (optional):",
        key=f"{key_prefix}_action_reason_input",
        placeholder="Enter reason for moving or deleting these records..."
    )
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.markdown("**Select rows to delete/move:**")
    
    with col2:
        if show_delete and st.button(f"🗑️ Delete Selected", key=f"{key_prefix}_delete_button"):
            selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
            selected_record_ids = [
                record_id for record_id, is_selected in selection_state.items() 
                if is_selected and record_id.startswith(f"{key_prefix}_select_")
            ]
            selected_ids = [rid.replace(f"{key_prefix}_select_", "") for rid in selected_record_ids]
            
            if selected_ids:
                source_df = st.session_state[display_df_key].copy()
                
                updated_df, deleted_count = delete_cross_match_selected_rows_with_audit(
                    source_df, selected_ids, title, action_reason,
                    df_name=display_df_key, on_data_change=on_data_change
                )
                
                if original_df_key in st.session_state:
                    original_updated = remove_cross_match_row_numbers(updated_df.copy())
                    st.session_state[original_df_key] = original_updated
                
                sync_all_cross_match_display_dataframes()
                clear_cross_match_selection_state(key_prefix)
                refresh_cross_match_analytics_dataframes()
                update_cross_match_deleted_stats()
                
                st.success(f"✅ Deleted {deleted_count} record(s) - Audit trail created")
                st.rerun()
            else:
                st.warning("No rows selected for deletion")
    
    with col3:
        if show_move and move_targets:
            if st.button(f"➡️ Move Selected", key=f"{key_prefix}_move_button"):
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
                        source_key = f"{key_prefix}_cross_match_original_df"
                        source_df = st.session_state.get(source_key, pd.DataFrame()).copy()
                        source_df = ensure_cross_match_record_ids(source_df)
                        
                        moved_records, new_source = move_cross_match_records_to_new_df(
                            source_df, selected_ids, title, selected_target, action_reason
                        )
                        
                        if not moved_records.empty:
                            moved_df_name = get_cross_match_moved_df_name(selected_target)
                            
                            if moved_df_name not in st.session_state:
                                st.session_state[moved_df_name] = moved_records
                            else:
                                existing = st.session_state[moved_df_name]
                                existing_ids = set(existing['_record_id'].tolist()) if not existing.empty else set()
                                new_records = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_records.empty:
                                    st.session_state[moved_df_name] = pd.concat([existing, new_records], ignore_index=True)
                            
                            if 'cross_match_audit_moves_log' not in st.session_state:
                                st.session_state.cross_match_audit_moves_log = moved_records[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type']].copy()
                            else:
                                existing_log = st.session_state.cross_match_audit_moves_log
                                existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
                                new_log_entries = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_log_entries.empty:
                                    st.session_state.cross_match_audit_moves_log = pd.concat([existing_log, new_log_entries[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type']]], ignore_index=True)
                            
                            st.session_state[source_key] = new_source
                            st.session_state[display_df_key] = add_cross_match_row_numbers(new_source)
                            
                            if on_data_change:
                                on_data_change(new_source)
                            
                            clear_cross_match_selection_state(key_prefix)
                            refresh_cross_match_analytics_dataframes()
                            update_cross_match_moved_stats()
                            
                            st.success(f"✅ Moved {len(selected_ids)} record(s) to {selected_target}")
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
        
        df_download = clean_cross_match_dataframe_for_arrow(df_download)
        
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_button"
        )
    
    with col5:
        if st.button(f"🔄 Refresh", key=f"{key_prefix}_refresh_button"):
            df_source = st.session_state.get(f"{key_prefix}_cross_match_original_df", df)
            if not df_source.empty:
                df_with_ids = ensure_cross_match_record_ids(df_source)
                df_with_audit = add_cross_match_audit_columns(df_with_ids)
                st.session_state[display_df_key] = add_cross_match_row_numbers(df_with_audit)
                st.session_state[original_df_key] = remove_cross_match_row_numbers(df_with_audit.copy())
            clear_cross_match_selection_state(key_prefix)
            st.rerun()
    
    with st.container():
        st.markdown("---")
        st.markdown("### Edit Data Directly")
        st.info("💡 Tip: Double-click any cell to edit its content. Use checkboxes below for batch operations.")
        
        df_for_edit = st.session_state[display_df_key].copy()
        df_for_edit = clean_cross_match_dataframe_for_arrow(df_for_edit)
        
        columns_to_drop = []
        if '#' in df_for_edit.columns:
            columns_to_drop.append('#')
        if '_record_id' in df_for_edit.columns:
            columns_to_drop.append('_record_id')
        
        if columns_to_drop:
            df_for_edit_for_display = df_for_edit.drop(columns=columns_to_drop)
        else:
            df_for_edit_for_display = df_for_edit
        
        for col in df_for_edit_for_display.columns:
            if df_for_edit_for_display[col].dtype == 'object':
                df_for_edit_for_display[col] = df_for_edit_for_display[col].astype(str)
        
        edited_df = st.data_editor(
            df_for_edit_for_display,
            use_container_width=True,
            height=min(400, len(df_for_edit_for_display) * 35 + 38),
            key=f"{key_prefix}_data_editor_{datetime.now().timestamp()}",
            num_rows="dynamic"
        )
        
        if not edited_df.equals(df_for_edit_for_display):
            edited_with_ids = ensure_cross_match_record_ids(edited_df.copy())
            edited_with_audit = add_cross_match_audit_columns(edited_with_ids)
            updated_with_numbers = add_cross_match_row_numbers(edited_with_audit)
            st.session_state[display_df_key] = updated_with_numbers
            
            if original_df_key in st.session_state:
                st.session_state[original_df_key] = remove_cross_match_row_numbers(edited_with_audit.copy())
            
            if on_data_change:
                on_data_change(remove_cross_match_row_numbers(edited_with_audit.copy()))
            
            refresh_cross_match_analytics_dataframes()
            st.success("✅ Data updated!")
            st.rerun()
        
        st.markdown("### Select Rows for Batch Operations")
        
        if show_move and move_targets:
            st.markdown("#### Move Target Selection")
            target_options = list(move_targets.keys())
            selected_target = st.selectbox(
                "Select target dataframe for moving records:",
                options=target_options,
                key=f"{key_prefix}_target_select"
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
        df_for_selection = clean_cross_match_dataframe_for_arrow(df_for_selection)
        
        if '_record_id' not in df_for_selection.columns:
            df_for_selection = ensure_cross_match_record_ids(df_for_selection)
            st.session_state[display_df_key] = add_cross_match_row_numbers(df_for_selection)
            if original_df_key in st.session_state:
                st.session_state[original_df_key] = remove_cross_match_row_numbers(df_for_selection.copy())
        
        record_ids = df_for_selection['_record_id'].tolist() if '_record_id' in df_for_selection.columns else []
        
        if not record_ids:
            st.warning("No record IDs found. Please refresh the page.")
            return df
        
        rows_container = st.container()
        
        with rows_container:
            for idx in range(len(df_for_selection)):
                col_check, col_content = st.columns([0.05, 0.95])
                
                with col_check:
                    row_num = df_for_selection.iloc[idx]['#'] if '#' in df_for_selection.columns else idx + 1
                    record_id = record_ids[idx] if idx < len(record_ids) else f"temp_{idx}"
                    checkbox_key = f"{key_prefix}_select_{record_id}"
                    
                    is_selected = st.session_state[selection_key].get(checkbox_key, False)
                    
                    if st.checkbox(f"Select row {row_num}", value=is_selected, key=checkbox_key, 
                                   label_visibility="collapsed"):
                        st.session_state[selection_key][checkbox_key] = True
                    else:
                        st.session_state[selection_key][checkbox_key] = False
                
                with col_content:
                    row_summary = []
                    display_cols = [col for col in df_for_selection.columns if col not in ['#', '_record_id']][:5]
                    
                    for col in display_cols:
                        val = df_for_selection.iloc[idx][col]
                        if pd.notna(val) and str(val).strip():
                            str_val = str(val)
                            if len(str_val) > 40:
                                str_val = str_val[:37] + "..."
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
                current_target = st.session_state.get(f"{key_prefix}_target_select", "Not selected")
                st.info(f"📌 These rows will be moved to: **{current_target}**")
    
    result_df = st.session_state[display_df_key].copy()
    if '_record_id' in result_df.columns and '#' in result_df.columns:
        result_df = result_df.drop(columns=['_record_id', '#'])
    elif '_record_id' in result_df.columns:
        result_df = result_df.drop(columns=['_record_id'])
    elif '#' in result_df.columns:
        result_df = result_df.drop(columns=['#'])
    
    return result_df


def render_cross_match_moved_records_tab():
    st.markdown("### 📋 Moved Records - Audit Trail")
    moved_stats = update_cross_match_moved_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Newly Matched Moved", moved_stats.get('cross_match_moved_newly_matched', 0))
    with col2:
        st.metric("📋 Still Unmatched Moved", moved_stats.get('cross_match_moved_still_unmatched', 0))
    with col3:
        st.metric("📋 Combined Moved", moved_stats.get('cross_match_moved_combined', 0))
    with col4:
        st.metric("📊 Total Moved", moved_stats.get('total_moved', 0))
    
    st.markdown("---")
    
    moved_df_names = ['cross_match_moved_newly_matched', 'cross_match_moved_still_unmatched',
                      'cross_match_moved_combined', 'cross_match_moved_unique_unmatched']
    moved_dfs = {}
    for df_name in moved_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            moved_dfs[df_name] = st.session_state[df_name].copy()
    
    if not moved_dfs:
        st.info("No moved records found.")
        return
    
    tabs = st.tabs([name.replace('cross_match_moved_', '').replace('_', ' ').title() for name in moved_dfs.keys()])
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
            cols_to_drop = ['_record_id', 'id', 'sys_created_at', 'import_date', 'last_modified', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)


def render_cross_match_deleted_records_tab():
    st.markdown("### 🗑️ Deleted Records - Audit Trail")
    deleted_stats = update_cross_match_deleted_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🗑️ Newly Matched Deleted", deleted_stats.get('cross_match_deleted_newly_matched', 0))
    with col2:
        st.metric("🗑️ Still Unmatched Deleted", deleted_stats.get('cross_match_deleted_still_unmatched', 0))
    with col3:
        st.metric("🗑️ Combined Deleted", deleted_stats.get('cross_match_deleted_combined', 0))
    with col4:
        st.metric("📊 Total Deleted", deleted_stats.get('total_deleted', 0))
    
    st.markdown("---")
    
    deleted_df_names = ['cross_match_deleted_newly_matched', 'cross_match_deleted_still_unmatched',
                        'cross_match_deleted_combined', 'cross_match_deleted_unique_unmatched']
    deleted_dfs = {}
    for df_name in deleted_df_names:
        if df_name in st.session_state and not st.session_state[df_name].empty:
            deleted_dfs[df_name] = st.session_state[df_name].copy()
    
    if not deleted_dfs:
        st.info("No deleted records found.")
        return
    
    tabs = st.tabs([name.replace('cross_match_deleted_', '').replace('_', ' ').title() for name in deleted_dfs.keys()])
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
            cols_to_drop = ['_record_id', 'id', 'sys_created_at', 'import_date', 'last_modified', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)


def render_cross_match_full_statistics_dashboard():
    """Render comprehensive statistics dashboard"""
    st.markdown("### 📊 Cross Match Statistics Dashboard")
    
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            update_cross_match_moved_stats()
            update_cross_match_deleted_stats()
            st.rerun()
    
    newly_matched = st.session_state.get('cross_match_newly_matched', pd.DataFrame())
    still_unmatched = st.session_state.get('cross_match_still_unmatched', pd.DataFrame())
    combined = st.session_state.get('cross_match_combined', pd.DataFrame())
    unique_unmatched = st.session_state.get('cross_match_unique_unmatched', pd.DataFrame())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Newly Matched", len(newly_matched) if not newly_matched.empty else 0)
    with col2:
        st.metric("⚠️ Still Unmatched", len(still_unmatched) if not still_unmatched.empty else 0)
    with col3:
        st.metric("📊 Combined Records", len(combined) if not combined.empty else 0)
    with col4:
        st.metric("🔄 Unique Unmatched", len(unique_unmatched) if not unique_unmatched.empty else 0)
    
    if not newly_matched.empty:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Match_Source' in newly_matched.columns:
                match_stats = newly_matched['Match_Source'].value_counts()
                fig = px.pie(values=match_stats.values, names=match_stats.index, title='Match Sources Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Match_Confidence' in newly_matched.columns:
                confidence_stats = newly_matched['Match_Confidence'].value_counts()
                fig = px.bar(x=confidence_stats.index, y=confidence_stats.values, title='Match Confidence Levels',
                            color=confidence_stats.index, color_discrete_sequence=['#28a745', '#ffc107', '#dc3545'])
                st.plotly_chart(fig, use_container_width=True)
    
    # Moved and Deleted Records Summary
    st.markdown("---")
    st.markdown("### 📦 Audit Summary")
    moved_stats = update_cross_match_moved_stats()
    deleted_stats = update_cross_match_deleted_stats()
    
    if moved_stats.get('total_moved', 0) > 0 or deleted_stats.get('total_deleted', 0) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if moved_stats.get('total_moved', 0) > 0:
                moved_df = pd.DataFrame([
                    {'Category': 'Newly Matched', 'Count': moved_stats.get('cross_match_moved_newly_matched', 0)},
                    {'Category': 'Still Unmatched', 'Count': moved_stats.get('cross_match_moved_still_unmatched', 0)},
                    {'Category': 'Combined', 'Count': moved_stats.get('cross_match_moved_combined', 0)},
                    {'Category': 'Unique Unmatched', 'Count': moved_stats.get('cross_match_moved_unique_unmatched', 0)}
                ])
                fig = px.bar(moved_df, x='Category', y='Count', title='Moved Records by Category',
                            color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if deleted_stats.get('total_deleted', 0) > 0:
                deleted_df = pd.DataFrame([
                    {'Category': 'Newly Matched', 'Count': deleted_stats.get('cross_match_deleted_newly_matched', 0)},
                    {'Category': 'Still Unmatched', 'Count': deleted_stats.get('cross_match_deleted_still_unmatched', 0)},
                    {'Category': 'Combined', 'Count': deleted_stats.get('cross_match_deleted_combined', 0)},
                    {'Category': 'Unique Unmatched', 'Count': deleted_stats.get('cross_match_deleted_unique_unmatched', 0)}
                ])
                fig = px.bar(deleted_df, x='Category', y='Count', title='Deleted Records by Category',
                            color='Count', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)


# --- Main Cross Match Analysis Function ---
def run_cross_match_analysis(
    df_matched_adjustments_local: pd.DataFrame,
    df_matched_adjustments_foreign: pd.DataFrame,
    df_matched_counterparty: pd.DataFrame,
    df_matched_choice: pd.DataFrame,
    df_matched_intermediary_credit: pd.DataFrame,
    df_matched_intermediary_debit: pd.DataFrame,
    df_matched_interfund: pd.DataFrame,
    df_bank_dfs: dict,
    debug_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a cross-reconciliation check by matching bank records against all matched data
    from the two different reconciliation apps to find potential missed matches.
    """
    st.header("Cross-Match Analysis")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --- Step 1: Handle Bank Records Dictionary ---
    if debug_mode:
        st.write(f"DEBUG: Type of df_bank_dfs: {type(df_bank_dfs)}")
        st.write(f"DEBUG: Keys in df_bank_dfs: {list(df_bank_dfs.keys()) if df_bank_dfs else 'Empty'}")

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
        
        bank_df_copy = bank_df.copy()
        bank_df_copy['Bank_Table_Name'] = bank_name
        all_bank_records.append(bank_df_copy)

    if all_bank_records:
        combined_bank_df = pd.concat(all_bank_records, ignore_index=True)
    else:
        st.error("No valid bank records found in the dictionary.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if debug_mode:
        st.write(f"DEBUG: Combined bank records shape: {combined_bank_df.shape}")

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
    
    available_columns = [col.lower() for col in df_bank_records.columns]
    
    for expected_col in column_mapping.keys():
        if expected_col.lower() in available_columns:
            actual_col = [col for col in df_bank_records.columns if col.lower() == expected_col.lower()][0]
            column_mapping[expected_col] = actual_col
        else:
            matches = [col for col in df_bank_records.columns if expected_col.lower() in col.lower()]
            if matches:
                column_mapping[expected_col] = matches[0]
    
    if debug_mode:
        st.write(f"DEBUG: Column mapping results: {column_mapping}")

    missing_essential = []
    for col in ['Date']:
        if column_mapping[col] is None:
            missing_essential.append(col)
    
    if missing_essential:
        st.error(f"Could not find essential columns in bank records: {missing_essential}")
        st.info(f"Available columns: {list(df_bank_records.columns)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    rename_dict = {}
    for expected_col, actual_col in column_mapping.items():
        if actual_col:
            rename_dict[actual_col] = expected_col
    
    df_bank_records_processed = df_bank_records.rename(columns=rename_dict)
    
    if 'Bank' not in df_bank_records_processed.columns and 'Bank_Table_Name' in df_bank_records_processed.columns:
        df_bank_records_processed['Bank'] = df_bank_records_processed['Bank_Table_Name']
    
    if 'Credit' not in df_bank_records_processed.columns and 'Amount' in df_bank_records_processed.columns:
        df_bank_records_processed['Credit'] = df_bank_records_processed['Amount'].apply(lambda x: abs(x) if abs(x) > 0 else 0)
        df_bank_records_processed['Debit'] = df_bank_records_processed['Amount'].apply(lambda x: abs(x) if abs(x) > 0 else 0)
    else:
        if 'Credit' in df_bank_records_processed.columns:
            df_bank_records_processed['Credit'] = pd.to_numeric(df_bank_records_processed['Credit'], errors='coerce').fillna(0)
        else:
            df_bank_records_processed['Credit'] = 0
        
        if 'Debit' in df_bank_records_processed.columns:
            df_bank_records_processed['Debit'] = pd.to_numeric(df_bank_records_processed['Debit'], errors='coerce').fillna(0)
        else:
            df_bank_records_processed['Debit'] = 0
    
    df_bank_records_processed['Date'] = pd.to_datetime(df_bank_records_processed['Date'], errors='coerce')
    if 'Bank' in df_bank_records_processed.columns:
        df_bank_records_processed['Bank'] = df_bank_records_processed['Bank'].astype(str)
    else:
        df_bank_records_processed['Bank'] = 'Unknown'

    # Add unique IDs
    df_bank_records_processed = add_cross_match_unique_ids(df_bank_records_processed)
    df_bank_records_processed = add_cross_match_audit_columns(df_bank_records_processed)

    # --- Step 3: Prepare Matched Data Sources ---
    if debug_mode:
        st.write("DEBUG: Matched data sources information:")
        st.write(f"  - Local Adjustments: {len(df_matched_adjustments_local)} records")
        st.write(f"  - Foreign Adjustments: {len(df_matched_adjustments_foreign)} records")
        st.write(f"  - Counterparty Trades: {len(df_matched_counterparty)} records")
        st.write(f"  - Choice Trades: {len(df_matched_choice)} records")
        st.write(f"  - Intermediary Credit: {len(df_matched_intermediary_credit)} records")
        st.write(f"  - Intermediary Debit: {len(df_matched_intermediary_debit)} records")
        st.write(f"  - Interfund: {len(df_matched_interfund)} records")

    # --- Step 4: Define Matching Functions ---
    def amounts_match(a, b, tolerance=0.01):
        if a is None or b is None:
            return False
        try:
            return abs(abs(float(a)) - abs(float(b))) <= tolerance
        except:
            return False

    def match_adjustments_local(bank_row: pd.Series, matched_df: pd.DataFrame, already_matched_indices: set = None) -> Dict[str, Any]:
        if already_matched_indices is None:
            already_matched_indices = set()
            
        for adj_index, adj_row in matched_df.iterrows():
            if adj_index in already_matched_indices:
                continue
                
            try:
                adj_date = pd.to_datetime(adj_row.get('Adjustment_Date'), errors='coerce')
                if pd.isna(adj_date) or pd.isna(bank_row['Date']) or adj_date.date() != bank_row['Date'].date():
                    continue
                    
                adj_bank_table = str(adj_row.get('Bank_Table', ''))
                if adj_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                adj_amount = float(adj_row.get('Adjustment_Amount', 0))
                adj_operation = str(adj_row.get('Adjustment_Operation', '')).lower()
                
                if adj_operation == 'credit' and amounts_match(bank_row['Credit'], adj_amount):
                    already_matched_indices.add(adj_index)
                    return {'matched': True, 'source': 'Local Adjustments', 'matched_index': adj_index,
                           'match_reason': f"Credit amount {adj_amount} matches", 'confidence': 'high'}
                elif adj_operation == 'debit' and amounts_match(bank_row['Debit'], adj_amount):
                    already_matched_indices.add(adj_index)
                    return {'matched': True, 'source': 'Local Adjustments', 'matched_index': adj_index,
                           'match_reason': f"Debit amount {adj_amount} matches", 'confidence': 'high'}
            except Exception as e:
                continue
        
        return {'matched': False, 'reason': 'No match found in local adjustments'}

    def match_adjustments_foreign(bank_row: pd.Series, matched_df: pd.DataFrame, already_matched_indices: set = None) -> Dict[str, Any]:
        if already_matched_indices is None:
            already_matched_indices = set()
            
        for adj_index, adj_row in matched_df.iterrows():
            if adj_index in already_matched_indices:
                continue
                
            try:
                adj_date = pd.to_datetime(adj_row.get('Adjustment_Date'), errors='coerce')
                if pd.isna(adj_date) or pd.isna(bank_row['Date']) or adj_date.date() != bank_row['Date'].date():
                    continue
                    
                adj_bank_table = str(adj_row.get('Bank_Table', ''))
                if adj_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                adj_amount = float(adj_row.get('Adjustment_Amount', 0))
                adj_operation = str(adj_row.get('Adjustment_Operation', '')).lower()
                
                if adj_operation == 'credit' and amounts_match(bank_row['Credit'], adj_amount):
                    already_matched_indices.add(adj_index)
                    return {'matched': True, 'source': 'Foreign Adjustments', 'matched_index': adj_index,
                           'match_reason': f"Credit amount {adj_amount} matches", 'confidence': 'high'}
                elif adj_operation == 'debit' and amounts_match(bank_row['Debit'], adj_amount):
                    already_matched_indices.add(adj_index)
                    return {'matched': True, 'source': 'Foreign Adjustments', 'matched_index': adj_index,
                           'match_reason': f"Debit amount {adj_amount} matches", 'confidence': 'high'}
            except Exception as e:
                continue
        
        return {'matched': False, 'reason': 'No match found in foreign adjustments'}

    def match_counterparty(bank_row: pd.Series, matched_df: pd.DataFrame, already_matched_indices: set = None) -> Dict[str, Any]:
        if already_matched_indices is None:
            already_matched_indices = set()
            
        for trade_index, trade_row in matched_df.iterrows():
            if trade_index in already_matched_indices:
                continue
                
            try:
                trade_date = pd.to_datetime(trade_row.get('Date'), errors='coerce')
                if pd.isna(trade_date) or pd.isna(bank_row['Date']) or trade_date.date() != bank_row['Date'].date():
                    continue
                    
                trade_bank_table = str(trade_row.get('Bank_Table', trade_row.get('Bank Table', '')))
                if trade_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                trade_amount = float(trade_row.get('Trade Amount', trade_row.get('Amount', 0)))
                matched_column = str(trade_row.get('Matched In Column', trade_row.get('Matched Column', ''))).lower()
                
                if matched_column == 'credit' and amounts_match(bank_row['Credit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Counterparty Trades', 'matched_index': trade_index,
                           'match_reason': f"Credit amount {trade_amount} matches", 'confidence': 'high'}
                elif matched_column == 'debit' and amounts_match(bank_row['Debit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Counterparty Trades', 'matched_index': trade_index,
                           'match_reason': f"Debit amount {trade_amount} matches", 'confidence': 'high'}
                elif (not matched_column or matched_column == '') and amounts_match(bank_row['Credit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Counterparty Trades', 'matched_index': trade_index,
                           'match_reason': f"Credit amount {trade_amount} matches (auto-detected)", 'confidence': 'medium'}
                elif (not matched_column or matched_column == '') and amounts_match(bank_row['Debit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Counterparty Trades', 'matched_index': trade_index,
                           'match_reason': f"Debit amount {trade_amount} matches (auto-detected)", 'confidence': 'medium'}
            except Exception as e:
                continue
        
        return {'matched': False, 'reason': 'No match found in counterparty trades'}

    def match_choice(bank_row: pd.Series, matched_df: pd.DataFrame, already_matched_indices: set = None) -> Dict[str, Any]:
        if already_matched_indices is None:
            already_matched_indices = set()
            
        for trade_index, trade_row in matched_df.iterrows():
            if trade_index in already_matched_indices:
                continue
                
            try:
                trade_date = pd.to_datetime(trade_row.get('Date'), errors='coerce')
                if pd.isna(trade_date) or pd.isna(bank_row['Date']) or trade_date.date() != bank_row['Date'].date():
                    continue
                    
                trade_bank_table = str(trade_row.get('Bank_Table', trade_row.get('Bank Table', '')))
                if trade_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                trade_amount = float(trade_row.get('Trade Amount', trade_row.get('Amount', 0)))
                matched_column = str(trade_row.get('Matched In Column', trade_row.get('Matched Column', ''))).lower()
                
                if matched_column == 'credit' and amounts_match(bank_row['Credit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Choice Trades', 'matched_index': trade_index,
                           'match_reason': f"Credit amount {trade_amount} matches", 'confidence': 'high'}
                elif matched_column == 'debit' and amounts_match(bank_row['Debit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Choice Trades', 'matched_index': trade_index,
                           'match_reason': f"Debit amount {trade_amount} matches", 'confidence': 'high'}
                elif (not matched_column or matched_column == '') and amounts_match(bank_row['Credit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Choice Trades', 'matched_index': trade_index,
                           'match_reason': f"Credit amount {trade_amount} matches (auto-detected)", 'confidence': 'medium'}
                elif (not matched_column or matched_column == '') and amounts_match(bank_row['Debit'], trade_amount):
                    already_matched_indices.add(trade_index)
                    return {'matched': True, 'source': 'Choice Trades', 'matched_index': trade_index,
                           'match_reason': f"Debit amount {trade_amount} matches (auto-detected)", 'confidence': 'medium'}
            except Exception as e:
                continue
        
        return {'matched': False, 'reason': 'No match found in choice trades'}

    def match_intermediary_credit(bank_row: pd.Series, matched_df: pd.DataFrame, already_matched_indices: set = None) -> Dict[str, Any]:
        if already_matched_indices is None:
            already_matched_indices = set()
            
        for intermediary_index, intermediary_row in matched_df.iterrows():
            if intermediary_index in already_matched_indices:
                continue
                
            try:
                intermediary_date = pd.to_datetime(intermediary_row.get('Date'), errors='coerce')
                if pd.isna(intermediary_date) or pd.isna(bank_row['Date']) or intermediary_date.date() != bank_row['Date'].date():
                    continue
                    
                intermediary_bank_table = str(intermediary_row.get('Bank_Table', intermediary_row.get('Bank Table', '')))
                if intermediary_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                intermediary_amount = float(intermediary_row.get('Intermediary Amount', intermediary_row.get('Amount', 0)))
                
                if amounts_match(bank_row['Debit'], intermediary_amount):
                    already_matched_indices.add(intermediary_index)
                    return {'matched': True, 'source': 'Intermediary Credit', 'matched_index': intermediary_index,
                           'match_reason': f"Debit amount {intermediary_amount} matches with Intermediary Credit", 'confidence': 'high'}
                elif amounts_match(bank_row['Credit'], intermediary_amount):
                    already_matched_indices.add(intermediary_index)
                    return {'matched': True, 'source': 'Intermediary Credit', 'matched_index': intermediary_index,
                           'match_reason': f"Credit amount {intermediary_amount} matches with Intermediary Credit", 'confidence': 'medium'}
            except Exception as e:
                continue
        
        return {'matched': False, 'reason': 'No match found in intermediary credit records'}

    def match_intermediary_debit(bank_row: pd.Series, matched_df: pd.DataFrame, already_matched_indices: set = None) -> Dict[str, Any]:
        if already_matched_indices is None:
            already_matched_indices = set()
            
        for intermediary_index, intermediary_row in matched_df.iterrows():
            if intermediary_index in already_matched_indices:
                continue
                
            try:
                intermediary_date = pd.to_datetime(intermediary_row.get('Date'), errors='coerce')
                if pd.isna(intermediary_date) or pd.isna(bank_row['Date']) or intermediary_date.date() != bank_row['Date'].date():
                    continue
                    
                intermediary_bank_table = str(intermediary_row.get('Bank_Table', intermediary_row.get('Bank Table', '')))
                if intermediary_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                intermediary_amount = float(intermediary_row.get('Intermediary Amount', intermediary_row.get('Amount', 0)))
                
                if amounts_match(bank_row['Credit'], intermediary_amount):
                    already_matched_indices.add(intermediary_index)
                    return {'matched': True, 'source': 'Intermediary Debit', 'matched_index': intermediary_index,
                           'match_reason': f"Credit amount {intermediary_amount} matches with Intermediary Debit", 'confidence': 'high'}
                elif amounts_match(bank_row['Debit'], intermediary_amount):
                    already_matched_indices.add(intermediary_index)
                    return {'matched': True, 'source': 'Intermediary Debit', 'matched_index': intermediary_index,
                           'match_reason': f"Debit amount {intermediary_amount} matches with Intermediary Debit", 'confidence': 'medium'}
            except Exception as e:
                continue
        
        return {'matched': False, 'reason': 'No match found in intermediary debit records'}

    def match_interfund(bank_row: pd.Series, matched_df: pd.DataFrame, already_matched_indices: set = None) -> Dict[str, Any]:
        if already_matched_indices is None:
            already_matched_indices = set()
            
        for interfund_index, interfund_row in matched_df.iterrows():
            if interfund_index in already_matched_indices:
                continue
                
            try:
                interfund_bank_table = str(interfund_row.get('Bank_Table', interfund_row.get('Bank Table', '')))
                if interfund_bank_table.lower() != str(bank_row['Bank']).lower():
                    continue
                
                interfund_amount = float(interfund_row.get('Interfund Amount', interfund_row.get('Amount', 0)))
                
                if amounts_match(bank_row['Debit'], interfund_amount):
                    already_matched_indices.add(interfund_index)
                    return {'matched': True, 'source': 'Interfund', 'matched_index': interfund_index,
                           'match_reason': f"Debit amount {interfund_amount} matches with Interfund", 'confidence': 'high'}
                elif amounts_match(bank_row['Credit'], interfund_amount):
                    already_matched_indices.add(interfund_index)
                    return {'matched': True, 'source': 'Interfund', 'matched_index': interfund_index,
                           'match_reason': f"Credit amount {interfund_amount} matches with Interfund", 'confidence': 'medium'}
            except Exception as e:
                continue
        
        return {'matched': False, 'reason': 'No match found in interfund records'}

    # --- Step 5: Perform Cross-Matching ---
    st.subheader("Cross-Matching Bank Records Against All Matched Data")
    
    already_matched_local = set()
    already_matched_foreign = set()
    already_matched_counterparty = set()
    already_matched_choice = set()
    already_matched_intermediary_credit = set()
    already_matched_intermediary_debit = set()
    already_matched_interfund = set()
    
    newly_matched_unmatched_bank_records = []
    still_unmatched_bank_records = []

    total_records = len(df_bank_records_processed)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for bank_index, bank_row in df_bank_records_processed.iterrows():
        if total_records > 0:
            progress = (bank_index + 1) / total_records
            progress_bar.progress(progress)
            status_text.text(f"Processing record {bank_index + 1} of {total_records}")

        if pd.isna(bank_row['Date']):
            unmatched_record = bank_row.to_dict()
            unmatched_record.update({'Bank_Record_Index': bank_index, 'Mismatch_Reason': 'Invalid date'})
            still_unmatched_bank_records.append(unmatched_record)
            continue

        match_results = []
        
        if not df_matched_adjustments_local.empty:
            match_results.append(match_adjustments_local(bank_row, df_matched_adjustments_local, already_matched_local))
        
        if not df_matched_adjustments_foreign.empty:
            match_results.append(match_adjustments_foreign(bank_row, df_matched_adjustments_foreign, already_matched_foreign))
        
        if not df_matched_counterparty.empty:
            match_results.append(match_counterparty(bank_row, df_matched_counterparty, already_matched_counterparty))
        
        if not df_matched_choice.empty:
            match_results.append(match_choice(bank_row, df_matched_choice, already_matched_choice))
        
        if not df_matched_intermediary_credit.empty:
            match_results.append(match_intermediary_credit(bank_row, df_matched_intermediary_credit, already_matched_intermediary_credit))
        
        if not df_matched_intermediary_debit.empty:
            match_results.append(match_intermediary_debit(bank_row, df_matched_intermediary_debit, already_matched_intermediary_debit))
        
        if not df_matched_interfund.empty:
            match_results.append(match_interfund(bank_row, df_matched_interfund, already_matched_interfund))

        successful_matches = [result for result in match_results if result.get('matched', False)]
        
        if successful_matches:
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
        else:
            unmatched_record = bank_row.to_dict()
            reasons = [result.get('reason', 'Unknown') for result in match_results if not result.get('matched', False)]
            mismatch_reason = ' | '.join(reasons) if reasons else 'No matches in any source'
            unmatched_record.update({'Bank_Record_Index': bank_index, 'Mismatch_Reason': mismatch_reason})
            still_unmatched_bank_records.append(unmatched_record)

    progress_bar.empty()
    status_text.empty()

    # Convert to DataFrames and add unique IDs
    newly_matched_df = pd.DataFrame(newly_matched_unmatched_bank_records)
    still_unmatched_df = pd.DataFrame(still_unmatched_bank_records)
    
    if not newly_matched_df.empty:
        newly_matched_df = add_cross_match_unique_ids(newly_matched_df)
        newly_matched_df = add_cross_match_audit_columns(newly_matched_df)
    
    if not still_unmatched_df.empty:
        still_unmatched_df = add_cross_match_unique_ids(still_unmatched_df)
        still_unmatched_df = add_cross_match_audit_columns(still_unmatched_df)
    
    # Store in session state
    st.session_state.cross_match_newly_matched = newly_matched_df
    st.session_state.cross_match_still_unmatched = still_unmatched_df
    st.session_state.cross_match_combined = df_bank_records_processed
    st.session_state.cross_match_unique_unmatched = still_unmatched_df.drop_duplicates(
        subset=['Bank', 'Date', 'Credit', 'Debit'], keep='first'
    ).copy() if not still_unmatched_df.empty else pd.DataFrame()

    # --- Step 6: Display Match Statistics ---
    st.markdown("---")
    st.subheader("Match Statistics by Source")

    if not newly_matched_df.empty:
        match_stats = newly_matched_df['Match_Source'].value_counts()
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        sources = ['Local Adjustments', 'Foreign Adjustments', 'Counterparty Trades', 
                   'Choice Trades', 'Intermediary Credit', 'Intermediary Debit', 'Interfund']
        
        for i, source in enumerate(sources):
            count = match_stats.get(source, 0)
            with [col1, col2, col3, col4, col5, col6, col7][i]:
                st.metric(f"Matches in {source}", count)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(match_stats.values, labels=match_stats.index, autopct='%1.1f%%', startangle=90)
        ax.set_title("Distribution of Matches by Source")
        ax.axis('equal')
        st.pyplot(fig)
        plt.close()

    # --- Step 7: Display Overall Results ---
    st.markdown("---")
    st.subheader("Cross-Match Results Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bank Records", total_records)
    with col2:
        st.metric("Newly Matched", len(newly_matched_df))
    with col3:
        st.metric("Still Unmatched", len(still_unmatched_df))

    return (
        newly_matched_df,
        still_unmatched_df,
        pd.DataFrame(),  # newly_matched_unmatched_adjustments_df (placeholder)
        pd.DataFrame(),  # still_unmatched_adjustments_df (placeholder)
        df_bank_records_processed  # combined_unmatched_bank_records_df
    )


# --- Main App Function ---
def cross_match_analysis_app():
    """Main cross match analysis app with data management and interactive editing"""
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_cross_match_session_state()
    
    # ========== DATA MANAGEMENT SECTION ==========
    st.markdown("### 📅 Data Management")
    
    available_dates = get_available_cross_match_dates()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        if available_dates:
            selected_load_date = st.selectbox(
                "📅 Select date to load:",
                options=available_dates,
                index=0,
                key="cross_match_load_date_select"
            )
        else:
            st.selectbox("📅 Select date to load:", options=["No data available"], disabled=True, key="cross_match_load_date_select")
            selected_load_date = None
    
    with col2:
        if selected_load_date and available_dates:
            if st.button("📂 Load Data", use_container_width=True, key="load_cross_match_btn"):
                cross_match_db.load_cross_match_data(selected_load_date)
                st.rerun()
    
    with col3:
        current_date = datetime.now().strftime('%Y-%m-%d')
        st.metric("Current Date", current_date)
    
    with col4:
        if st.button("💾 Save Data", type="primary", use_container_width=True, key="save_cross_match_btn"):
            cross_match_db.save_cross_match_data()
            st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Reset Current Module Data", type="secondary", use_container_width=True, key="reset_module_cross_match_btn"):
            reset_all_cross_match_dataframes()
            st.success("✅ All current module dataframes have been reset!")
            st.balloons()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Reset All Data (Including Saved)", type="secondary", use_container_width=True, key="reset_all_cross_match_btn"):
            target_date = datetime.now().strftime('%Y-%m-%d')
            reset_all_cross_match_dataframes()
            
            conn = sqlite3.connect(CROSS_MATCH_DB_PATH)
            cursor = conn.cursor()
            
            tables_to_clear = [
                'cross_match_newly_matched_bank', 'cross_match_still_unmatched_bank',
                'cross_match_combined_bank', 'cross_match_unique_unmatched',
                'cross_match_moved_records', 'cross_match_deleted_records',
                'cross_match_audit_moves_log', 'cross_match_audit_deletes_log'
            ]
            
            for table in tables_to_clear:
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE record_date = ? OR import_date LIKE ?", 
                                (target_date, f"{target_date}%"))
                except:
                    pass
            
            conn.commit()
            conn.close()
            
            st.success("✅ All data (session and database) has been reset!")
            st.balloons()
            st.rerun()
    
    with col3:
        if st.button("📊 Refresh Dashboard", type="primary", use_container_width=True, key="refresh_cross_match_dashboard_btn"):
            update_cross_match_moved_stats()
            update_cross_match_deleted_stats()
            refresh_cross_match_analytics_dataframes()
            st.success("✅ Dashboard refreshed!")
            st.rerun()
    
    st.markdown("---")
    
    # Get data from session state
    newly_matched = st.session_state.get('cross_match_newly_matched', pd.DataFrame())
    still_unmatched = st.session_state.get('cross_match_still_unmatched', pd.DataFrame())
    combined = st.session_state.get('cross_match_combined', pd.DataFrame())
    unique_unmatched = st.session_state.get('cross_match_unique_unmatched', pd.DataFrame())
    
    # Check if we have data
    if newly_matched.empty and still_unmatched.empty and combined.empty:
        st.info("No cross-match data available. Please run the cross-match analysis first.")
        
        # Button to run analysis (would need to be connected to main dashboard)
        if st.button("Run Cross-Match Analysis", type="primary"):
            st.warning("Please ensure all reconciliation modules have been run before cross-match analysis.")
        
        return
    
    # Move targets configuration
    move_targets_newly_matched = {
        "Still Unmatched": "cross_match_still_unmatched",
        "Combined": "cross_match_combined",
        "Unique Unmatched": "cross_match_unique_unmatched"
    }
    
    move_targets_still_unmatched = {
        "Newly Matched": "cross_match_newly_matched",
        "Combined": "cross_match_combined",
        "Unique Unmatched": "cross_match_unique_unmatched"
    }
    
    move_targets_combined = {
        "Newly Matched": "cross_match_newly_matched",
        "Still Unmatched": "cross_match_still_unmatched",
        "Unique Unmatched": "cross_match_unique_unmatched"
    }
    
    move_targets_unique_unmatched = {
        "Newly Matched": "cross_match_newly_matched",
        "Still Unmatched": "cross_match_still_unmatched",
        "Combined": "cross_match_combined"
    }
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✅ Newly Matched Records",
        "⚠️ Still Unmatched Records",
        "📊 Combined Bank Records",
        "🔄 Unique Unmatched Records",
        "📋 Audit Trail"
    ])
    
    with tab1:
        def update_newly_matched(df):
            st.session_state.cross_match_newly_matched = add_cross_match_unique_ids(df) if not df.empty else df
            if not st.session_state.cross_match_newly_matched.empty:
                st.session_state.cross_match_newly_matched = add_cross_match_audit_columns(st.session_state.cross_match_newly_matched)
            update_cross_match_moved_stats()
            update_cross_match_deleted_stats()
        
        render_cross_match_editable_dataframe(
            newly_matched, "Newly Matched Bank Records", "newly_matched",
            on_data_change=update_newly_matched, show_delete=True,
            show_move=True, move_targets=move_targets_newly_matched
        )
    
    with tab2:
        def update_still_unmatched(df):
            st.session_state.cross_match_still_unmatched = add_cross_match_unique_ids(df) if not df.empty else df
            if not st.session_state.cross_match_still_unmatched.empty:
                st.session_state.cross_match_still_unmatched = add_cross_match_audit_columns(st.session_state.cross_match_still_unmatched)
            update_cross_match_moved_stats()
            update_cross_match_deleted_stats()
        
        render_cross_match_editable_dataframe(
            still_unmatched, "Still Unmatched Bank Records", "still_unmatched",
            on_data_change=update_still_unmatched, show_delete=True,
            show_move=True, move_targets=move_targets_still_unmatched
        )
    
    with tab3:
        def update_combined(df):
            st.session_state.cross_match_combined = add_cross_match_unique_ids(df) if not df.empty else df
            if not st.session_state.cross_match_combined.empty:
                st.session_state.cross_match_combined = add_cross_match_audit_columns(st.session_state.cross_match_combined)
            update_cross_match_moved_stats()
            update_cross_match_deleted_stats()
        
        render_cross_match_editable_dataframe(
            combined, "Combined Bank Records", "combined",
            on_data_change=update_combined, show_delete=True,
            show_move=True, move_targets=move_targets_combined
        )
    
    with tab4:
        def update_unique_unmatched(df):
            st.session_state.cross_match_unique_unmatched = add_cross_match_unique_ids(df) if not df.empty else df
            if not st.session_state.cross_match_unique_unmatched.empty:
                st.session_state.cross_match_unique_unmatched = add_cross_match_audit_columns(st.session_state.cross_match_unique_unmatched)
            update_cross_match_moved_stats()
            update_cross_match_deleted_stats()
        
        render_cross_match_editable_dataframe(
            unique_unmatched, "Unique Unmatched Records", "unique_unmatched",
            on_data_change=update_unique_unmatched, show_delete=True,
            show_move=True, move_targets=move_targets_unique_unmatched
        )
    
    with tab5:
        audit_tab1, audit_tab2 = st.tabs(["📋 Moved Records", "🗑️ Deleted Records"])
        with audit_tab1:
            render_cross_match_moved_records_tab()
        with audit_tab2:
            render_cross_match_deleted_records_tab()
    
    # Statistics Dashboard
    st.markdown("---")
    render_cross_match_full_statistics_dashboard()


if __name__ == '__main__':
    st.title("Cross-Match Analysis App")
    st.warning("This file is intended to be imported by main_dashboard.py. This is a placeholder.")
    cross_match_analysis_app()