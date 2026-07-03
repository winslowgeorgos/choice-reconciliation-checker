# pages/business_fx_reconciliation_page.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from fuzzywuzzy import fuzz, process
import re
import io
from io import BytesIO
import os
import pickle
import json
import uuid
import sqlite3
import logging
import plotly.express as px
import plotly.graph_objects as go

from auth_system import get_active_version_id, log_audit

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
DB_PATH = "data/business_fx_reconciliation.db"  # Separate database for business FX
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Custom CSS for better UI
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Card styling */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Success/Info boxes */
    .custom-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Dataframe styling */
    .dataframe-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Tab styling */
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

# --- Database Manager Class for Business FX ---
class BusinessFXReconciliationDB:
    """Database manager for Business FX reconciliation data"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables for business FX"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Final business dataframe table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_final_df (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Created_At TEXT,
                Reference_number TEXT,
                Deal_type TEXT,
                Client_Name TEXT,
                Amount REAL,
                Rate REAL,
                KES_equivalent REAL,
                Collection_bank TEXT,
                Payee_bank TEXT,
                Client_Account_Number TEXT,
                Status TEXT,
                KES_Equivalent_Matched INTEGER,
                Other_Currency_Matched INTEGER,
                Mismatch_Type TEXT,
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
        
        # Matched buy table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_matched_buy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Business_index INTEGER,
                Deal_Type TEXT,
                Business_Amount REAL,
                Business_KES_eq REAL,
                Collection_Bank TEXT,
                Payee_Bank TEXT,
                Match_Type TEXT,
                CounterParty_Match INTEGER,
                Choice_Match INTEGER,
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
        
        # Matched sell table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_matched_sell (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Business_index INTEGER,
                Deal_Type TEXT,
                Business_Amount REAL,
                Business_KES_eq REAL,
                Collection_Bank TEXT,
                Payee_Bank TEXT,
                Match_Type TEXT,
                Choice_Match INTEGER,
                CounterParty_Match INTEGER,
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
        
        # Unmatched buy table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_unmatched_buy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Business_index INTEGER,
                Deal_Type TEXT,
                Business_Amount REAL,
                Business_KES_eq REAL,
                Status TEXT,
                Reasons TEXT,
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
        
        # Unmatched sell table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_unmatched_sell (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Business_index INTEGER,
                Deal_Type TEXT,
                Business_Amount REAL,
                Business_KES_eq REAL,
                Status TEXT,
                Reasons TEXT,
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
        
        # Unmatched business table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_unmatched_business (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                Created_At TEXT,
                Reference_number TEXT,
                Deal_type TEXT,
                Client_Name TEXT,
                Amount REAL,
                Rate REAL,
                KES_equivalent REAL,
                Collection_bank TEXT,
                Payee_bank TEXT,
                Client_Account_Number TEXT,
                Status TEXT,
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
            CREATE TABLE IF NOT EXISTS business_moved_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                source_table TEXT,
                record_type TEXT,
                original_record_json TEXT,
                Date TEXT,
                Action_Type TEXT,
                Amount REAL,
                Bank_Table TEXT,
                Status TEXT,
                Vendor_Name TEXT,
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
            CREATE TABLE IF NOT EXISTS business_deleted_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                sys_created_at TEXT,
                source_table TEXT,
                record_type TEXT,
                original_record_json TEXT,
                Date TEXT,
                Action_Type TEXT,
                Amount REAL,
                Bank_Table TEXT,
                Status TEXT,
                Vendor_Name TEXT,
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
            CREATE TABLE IF NOT EXISTS business_audit_moves_log (
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
            CREATE TABLE IF NOT EXISTS business_audit_deletes_log (
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
            CREATE TABLE IF NOT EXISTS business_reconciliation_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_business_final_date ON business_final_df(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_business_matched_buy_date ON business_matched_buy(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_business_matched_sell_date ON business_matched_sell(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_business_unmatched_buy_date ON business_unmatched_buy(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_business_unmatched_sell_date ON business_unmatched_sell(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_business_moved_date ON business_moved_records(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_business_deleted_date ON business_deleted_records(record_date)',
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Business FX database initialized successfully")
        
    
    
    def _serialize_value(self, value):
        if value is None:
            return None
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        return str(value) if not isinstance(value, (float, int)) else value
    
    def save_final_business_df(self, df, record_date=None):
        """Save final business df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_final_df WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all business_final_df records for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            cursor.execute('''
                INSERT INTO business_final_df (
                    _record_id, record_date, sys_created_at, Created_At, Reference_number,
                    Deal_type, Client_Name, Amount, Rate, KES_equivalent,
                    Collection_bank, Payee_bank, Client_Account_Number, Status,
                    KES_Equivalent_Matched, Other_Currency_Matched, Mismatch_Type,
                    deleted_by, deleted_at, delete_reason, moved_by, moved_from,
                    moved_at, move_reason, move_type, moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Created At')),
                self._serialize_value(row.get('Reference number')),
                self._serialize_value(row.get('Deal type')),
                self._serialize_value(row.get('Client Name')),
                self._serialize_value(row.get('Amount')),
                self._serialize_value(row.get('Rate')),
                self._serialize_value(row.get('KES equivalent')),
                self._serialize_value(row.get('Collection bank')),
                self._serialize_value(row.get('Payee bank')),
                self._serialize_value(row.get('Client Account Number')),
                self._serialize_value(row.get('Status')),
                1 if row.get('KES_Equivalent_Matched') else 0,
                1 if row.get('Other_Currency_Matched') else 0,
                self._serialize_value(row.get('Mismatch_Type', '')),
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
        logger.info(f"Saved {len(df)} records to business_final_df for date: {record_date}")
    
    def save_matched_buy_df(self, df, record_date=None):
        """Save matched buy df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_matched_buy WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            cursor.execute('''
                INSERT INTO business_matched_buy (
                    _record_id, record_date, sys_created_at, Business_index, Deal_Type,
                    Business_Amount, Business_KES_eq, Collection_Bank, Payee_Bank,
                    Match_Type, CounterParty_Match, Choice_Match, deleted_by, deleted_at,
                    delete_reason, moved_by, moved_from, moved_at, move_reason, move_type,
                    moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Business index')),
                self._serialize_value(row.get('Deal Type')),
                self._serialize_value(row.get('Business Amount')),
                self._serialize_value(row.get('Business KES eq')),
                self._serialize_value(row.get('Collection Bank')),
                self._serialize_value(row.get('Payee Bank')),
                self._serialize_value(row.get('Match Type')),
                1 if row.get('CounterParty Match') else 0,
                1 if row.get('Choice Match') else 0,
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
        logger.info(f"Saved {len(df)} records to business_matched_buy for date: {record_date}")
    
    def save_matched_sell_df(self, df, record_date=None):
        """Save matched sell df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_matched_sell WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            cursor.execute('''
                INSERT INTO business_matched_sell (
                    _record_id, record_date, sys_created_at, Business_index, Deal_Type,
                    Business_Amount, Business_KES_eq, Collection_Bank, Payee_Bank,
                    Match_Type, Choice_Match, CounterParty_Match, deleted_by, deleted_at,
                    delete_reason, moved_by, moved_from, moved_at, move_reason, move_type,
                    moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Business index')),
                self._serialize_value(row.get('Deal Type')),
                self._serialize_value(row.get('Business Amount')),
                self._serialize_value(row.get('Business KES eq')),
                self._serialize_value(row.get('Collection Bank')),
                self._serialize_value(row.get('Payee Bank')),
                self._serialize_value(row.get('Match Type')),
                1 if row.get('Choice Match') else 0,
                1 if row.get('CounterParty Match') else 0,
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
        logger.info(f"Saved {len(df)} records to business_matched_sell for date: {record_date}")
    
    def save_unmatched_buy_df(self, df, record_date=None):
        """Save unmatched buy df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_unmatched_buy WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            cursor.execute('''
                INSERT INTO business_unmatched_buy (
                    _record_id, record_date, sys_created_at, Business_index, Deal_Type,
                    Business_Amount, Business_KES_eq, Status, Reasons,
                    deleted_by, deleted_at, delete_reason, moved_by, moved_from,
                    moved_at, move_reason, move_type, moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Business index')),
                self._serialize_value(row.get('Deal Type')),
                self._serialize_value(row.get('Business Amount')),
                self._serialize_value(row.get('Business KES eq')),
                self._serialize_value(row.get('Status')),
                self._serialize_value(row.get('Reasons')),
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
        logger.info(f"Saved {len(df)} records to business_unmatched_buy for date: {record_date}")
    
    def save_unmatched_sell_df(self, df, record_date=None):
        """Save unmatched sell df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_unmatched_sell WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            cursor.execute('''
                INSERT INTO business_unmatched_sell (
                    _record_id, record_date, sys_created_at, Business_index, Deal_Type,
                    Business_Amount, Business_KES_eq, Status, Reasons,
                    deleted_by, deleted_at, delete_reason, moved_by, moved_from,
                    moved_at, move_reason, move_type, moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Business index')),
                self._serialize_value(row.get('Deal Type')),
                self._serialize_value(row.get('Business Amount')),
                self._serialize_value(row.get('Business KES eq')),
                self._serialize_value(row.get('Status')),
                self._serialize_value(row.get('Reasons')),
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
        logger.info(f"Saved {len(df)} records to business_unmatched_sell for date: {record_date}")
    
    def save_unmatched_business_df(self, df, record_date=None):
        """Save unmatched business df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_unmatched_business WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            cursor.execute('''
                INSERT INTO business_unmatched_business (
                    _record_id, record_date, sys_created_at, Created_At, Reference_number,
                    Deal_type, Client_Name, Amount, Rate, KES_equivalent,
                    Collection_bank, Payee_bank, Client_Account_Number, Status,
                    deleted_by, deleted_at, delete_reason, moved_by, moved_from,
                    moved_at, move_reason, move_type, moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Created At')),
                self._serialize_value(row.get('Reference number')),
                self._serialize_value(row.get('Deal type')),
                self._serialize_value(row.get('Client Name')),
                self._serialize_value(row.get('Amount')),
                self._serialize_value(row.get('Rate')),
                self._serialize_value(row.get('KES equivalent')),
                self._serialize_value(row.get('Collection bank')),
                self._serialize_value(row.get('Payee bank')),
                self._serialize_value(row.get('Client Account Number')),
                self._serialize_value(row.get('Status')),
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
        logger.info(f"Saved {len(df)} records to business_unmatched_business for date: {record_date}")
    
    def save_moved_records(self, df, record_date=None):
        """Save moved records - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_moved_records WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all business_moved_records for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            record_dict = row.to_dict()
            original_record_json = json.dumps(record_dict, default=str)
            source_table = row.get('moved_from', 'unknown')
            
            cursor.execute('''
                INSERT INTO business_moved_records (
                    _record_id, record_date, sys_created_at, source_table, record_type,
                    original_record_json, Date, Action_Type, Amount, Bank_Table,
                    Status, Vendor_Name, moved_by, moved_from, moved_to, moved_at,
                    move_reason, move_type, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date, source_table, source_table,
                original_record_json,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Trade Amount', row.get('Amount', 0))),
                self._serialize_value(row.get('Bank Table', row.get('Bank Table (Expected)'))),
                self._serialize_value(row.get('Status')),
                self._serialize_value(row.get('Vendor Name')),
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
        logger.info(f"Saved {len(df)} records to business_moved_records for date: {record_date}")
    
    def save_deleted_records(self, df, record_date=None):
        """Save deleted records - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_deleted_records WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all business_deleted_records for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_business_record_id()))
            record_dict = row.to_dict()
            original_record_json = json.dumps(record_dict, default=str)
            source_table = row.get('deleted_from', row.get('source_dataframe', 'unknown'))
            
            cursor.execute('''
                INSERT INTO business_deleted_records (
                    _record_id, record_date, sys_created_at, source_table, record_type,
                    original_record_json, Date, Action_Type, Amount, Bank_Table,
                    Status, Vendor_Name, deleted_by, deleted_at, delete_reason,
                    deleted_from, source_dataframe, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date, source_table, source_table,
                original_record_json,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Trade Amount', row.get('Amount', 0))),
                self._serialize_value(row.get('Bank Table', row.get('Bank Table (Expected)'))),
                self._serialize_value(row.get('Status')),
                self._serialize_value(row.get('Vendor Name')),
                self._serialize_value(row.get('deleted_by')),
                self._serialize_value(row.get('deleted_at')),
                self._serialize_value(row.get('delete_reason')),
                self._serialize_value(row.get('deleted_from')),
                self._serialize_value(row.get('source_dataframe')),
                import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} records to business_deleted_records for date: {record_date}")
    
    def save_audit_moves(self, df, record_date=None):
        """Save audit moves log - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_audit_moves_log WHERE import_date LIKE ?", (f"{record_date}%",))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO business_audit_moves_log (
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
        logger.info(f"Saved {len(df)} records to business_audit_moves_log for date: {record_date}")
    
    def save_audit_deletes(self, df, record_date=None):
        """Save audit deletes log - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM business_audit_deletes_log WHERE import_date LIKE ?", (f"{record_date}%",))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO business_audit_deletes_log (
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
        logger.info(f"Saved {len(df)} records to business_audit_deletes_log for date: {record_date}")
    
    def save_business_data_only(self, target_date=None):
        """Save ALL Business FX Reconciliation data - REPLACES all data for the date"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get current session state data
        final_business_df = st.session_state.get('final_business_df', pd.DataFrame())
        matched_buy_df = st.session_state.get('matched_buy_df', pd.DataFrame())
        matched_sell_df = st.session_state.get('matched_sell_df', pd.DataFrame())
        unmatched_buy_df = st.session_state.get('unmatched_buy_df', pd.DataFrame())
        unmatched_sell_df = st.session_state.get('unmatched_sell_df', pd.DataFrame())
        unmatched_business_df = st.session_state.get('unmatched_business_df', pd.DataFrame())
        
        # Save each dataframe (each will DELETE old data first)
        self.save_final_business_df(final_business_df, target_date)
        self.save_matched_buy_df(matched_buy_df, target_date)
        self.save_matched_sell_df(matched_sell_df, target_date)
        self.save_unmatched_buy_df(unmatched_buy_df, target_date)
        self.save_unmatched_sell_df(unmatched_sell_df, target_date)
        self.save_unmatched_business_df(unmatched_business_df, target_date)
        
        # Save moved records
        all_moved_records = []
        moved_keys = ['moved_final_business', 'moved_matched_buy', 'moved_matched_sell',
                      'moved_unmatched_buy', 'moved_unmatched_sell', 'moved_unmatched_business']
        
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
        deleted_keys = ['deleted_final_business', 'deleted_matched_buy', 'deleted_matched_sell',
                        'deleted_unmatched_buy', 'deleted_unmatched_sell', 'deleted_unmatched_business']
        
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
        audit_moves = st.session_state.get('audit_moves_log_business', pd.DataFrame())
        audit_deletes = st.session_state.get('audit_deletes_log_business', pd.DataFrame())
        
        self.save_audit_moves(audit_moves, target_date)
        self.save_audit_deletes(audit_deletes, target_date)
        
        # Save metadata
        self.save_metadata('business_last_save_date', target_date)
        self.save_metadata('business_moved_stats', st.session_state.get('moved_stats_business', {}))
        self.save_metadata('business_deleted_stats', st.session_state.get('deleted_stats_business', {}))
        
        # Save summary of what was saved
        save_summary = {
            'final_business_count': len(final_business_df),
            'matched_buy_count': len(matched_buy_df),
            'matched_sell_count': len(matched_sell_df),
            'unmatched_buy_count': len(unmatched_buy_df),
            'unmatched_sell_count': len(unmatched_sell_df),
            'unmatched_business_count': len(unmatched_business_df),
            'moved_count': len(combined_moved) if all_moved_records else 0,
            'deleted_count': len(combined_deleted) if all_deleted_records else 0,
            'audit_moves_count': len(audit_moves),
            'audit_deletes_count': len(audit_deletes)
        }
        self.save_metadata('business_save_summary', save_summary)
        
        st.session_state.business_last_save_date = target_date
        
        # Show summary of saved data
        with st.container():
            st.markdown('<div class="custom-success">', unsafe_allow_html=True)
            st.success(f"✅ Business FX Reconciliation data saved for date: {target_date}")
            
            summary = []
            if not final_business_df.empty:
                summary.append(f"• final_business_df: {len(final_business_df)} records")
            if not matched_buy_df.empty:
                summary.append(f"• matched_buy_df: {len(matched_buy_df)} records")
            if not matched_sell_df.empty:
                summary.append(f"• matched_sell_df: {len(matched_sell_df)} records")
            if not unmatched_buy_df.empty:
                summary.append(f"• unmatched_buy_df: {len(unmatched_buy_df)} records")
            if not unmatched_sell_df.empty:
                summary.append(f"• unmatched_sell_df: {len(unmatched_sell_df)} records")
            if not unmatched_business_df.empty:
                summary.append(f"• unmatched_business_df: {len(unmatched_business_df)} records")
            if all_moved_records:
                summary.append(f"• moved_records: {len(combined_moved)} records")
            if all_deleted_records:
                summary.append(f"• deleted_records: {len(combined_deleted)} records")
            if not audit_moves.empty:
                summary.append(f"• audit_moves_log: {len(audit_moves)} records")
            if not audit_deletes.empty:
                summary.append(f"• audit_deletes_log: {len(audit_deletes)} records")
            
            if summary:
                st.info("Saved data:\n" + "\n".join(summary))
            st.markdown('</div>', unsafe_allow_html=True)
        
        return target_date
    
    def load_business_data_only(self, target_date=None):
        """Load ALL Business FX Reconciliation data from database including audit trails"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        
        # Load main dataframes directly from their specific tables
        st.session_state.final_business_df = self.load_final_business_df(target_date)
        st.session_state.matched_buy_df = self.load_matched_buy_df(target_date)
        st.session_state.matched_sell_df = self.load_matched_sell_df(target_date)
        st.session_state.unmatched_buy_df = self.load_unmatched_buy_df(target_date)
        st.session_state.unmatched_sell_df = self.load_unmatched_sell_df(target_date)
        st.session_state.unmatched_business_df = self.load_unmatched_business_df(target_date)
        
        # Load all moved records from moved_records table
        query = "SELECT * FROM business_moved_records WHERE record_date = ?"
        all_moved = pd.read_sql_query(query, conn, params=(target_date,))
        
        if not all_moved.empty:
            # Split moved records by their moved_to category
            st.session_state.moved_final_business = all_moved[all_moved['moved_to'].str.contains('Final Business', na=False)].copy()
            st.session_state.moved_matched_buy = all_moved[all_moved['moved_to'].str.contains('Matched Buy', na=False)].copy()
            st.session_state.moved_matched_sell = all_moved[all_moved['moved_to'].str.contains('Matched Sell', na=False)].copy()
            st.session_state.moved_unmatched_buy = all_moved[all_moved['moved_to'].str.contains('Unmatched Buy', na=False)].copy()
            st.session_state.moved_unmatched_sell = all_moved[all_moved['moved_to'].str.contains('Unmatched Sell', na=False)].copy()
            st.session_state.moved_unmatched_business = all_moved[all_moved['moved_to'].str.contains('Unmatched Business', na=False)].copy()
        else:
            st.session_state.moved_final_business = pd.DataFrame()
            st.session_state.moved_matched_buy = pd.DataFrame()
            st.session_state.moved_matched_sell = pd.DataFrame()
            st.session_state.moved_unmatched_buy = pd.DataFrame()
            st.session_state.moved_unmatched_sell = pd.DataFrame()
            st.session_state.moved_unmatched_business = pd.DataFrame()
        
        # Load all deleted records from deleted_records table
        query = "SELECT * FROM business_deleted_records WHERE record_date = ?"
        all_deleted = pd.read_sql_query(query, conn, params=(target_date,))
        
        if not all_deleted.empty:
            # Split deleted records by their deleted_from category
            st.session_state.deleted_final_business = all_deleted[all_deleted['deleted_from'].str.contains('Final Business', na=False)].copy()
            st.session_state.deleted_matched_buy = all_deleted[all_deleted['deleted_from'].str.contains('Matched Buy', na=False)].copy()
            st.session_state.deleted_matched_sell = all_deleted[all_deleted['deleted_from'].str.contains('Matched Sell', na=False)].copy()
            st.session_state.deleted_unmatched_buy = all_deleted[all_deleted['deleted_from'].str.contains('Unmatched Buy', na=False)].copy()
            st.session_state.deleted_unmatched_sell = all_deleted[all_deleted['deleted_from'].str.contains('Unmatched Sell', na=False)].copy()
            st.session_state.deleted_unmatched_business = all_deleted[all_deleted['deleted_from'].str.contains('Unmatched Business', na=False)].copy()
        else:
            st.session_state.deleted_final_business = pd.DataFrame()
            st.session_state.deleted_matched_buy = pd.DataFrame()
            st.session_state.deleted_matched_sell = pd.DataFrame()
            st.session_state.deleted_unmatched_buy = pd.DataFrame()
            st.session_state.deleted_unmatched_sell = pd.DataFrame()
            st.session_state.deleted_unmatched_business = pd.DataFrame()
        
        # Load audit logs
        query = "SELECT * FROM business_audit_moves_log WHERE import_date LIKE ?"
        audit_moves = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.audit_moves_log_business = audit_moves if not audit_moves.empty else pd.DataFrame()
        
        query = "SELECT * FROM business_audit_deletes_log WHERE import_date LIKE ?"
        audit_deletes = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.audit_deletes_log_business = audit_deletes if not audit_deletes.empty else pd.DataFrame()
        
        conn.close()
        
        # Add unique IDs and audit columns to main dataframes if missing
        for df_name in ['final_business_df', 'matched_buy_df', 'matched_sell_df', 
                        'unmatched_buy_df', 'unmatched_sell_df', 'unmatched_business_df']:
            if not st.session_state[df_name].empty:
                if '_record_id' not in st.session_state[df_name].columns:
                    st.session_state[df_name] = add_business_unique_ids(st.session_state[df_name])
                st.session_state[df_name] = add_business_audit_columns(st.session_state[df_name])
        
        # Recalculate stats from loaded data
        update_business_moved_stats()
        update_business_deleted_stats()
        
        st.session_state.business_current_date = target_date
        
        # Get saved summary for verification
        save_summary = self.load_metadata('business_save_summary', {})
        
        with st.container():
            st.markdown('<div class="custom-success">', unsafe_allow_html=True)
            st.success(f"✅ Business FX Reconciliation data loaded for date: {target_date}")
            
            # Show summary of loaded data
            summary = []
            if not st.session_state.final_business_df.empty:
                count = len(st.session_state.final_business_df)
                summary.append(f"• final_business_df: {count} records")
            if not st.session_state.matched_buy_df.empty:
                count = len(st.session_state.matched_buy_df)
                summary.append(f"• matched_buy_df: {count} records")
            if not st.session_state.matched_sell_df.empty:
                count = len(st.session_state.matched_sell_df)
                summary.append(f"• matched_sell_df: {count} records")
            if not st.session_state.unmatched_buy_df.empty:
                count = len(st.session_state.unmatched_buy_df)
                summary.append(f"• unmatched_buy_df: {count} records")
            if not st.session_state.unmatched_sell_df.empty:
                count = len(st.session_state.unmatched_sell_df)
                summary.append(f"• unmatched_sell_df: {count} records")
            if not st.session_state.unmatched_business_df.empty:
                count = len(st.session_state.unmatched_business_df)
                summary.append(f"• unmatched_business_df: {count} records")
            
            if summary:
                st.info("Loaded data:\n" + "\n".join(summary))
            st.markdown('</div>', unsafe_allow_html=True)
        
        return target_date
    
    def load_final_business_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM business_final_df WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        
        # Convert boolean columns back
        if 'KES_Equivalent_Matched' in df.columns:
            df['KES_Equivalent_Matched'] = df['KES_Equivalent_Matched'].astype(bool)
        if 'Other_Currency_Matched' in df.columns:
            df['Other_Currency_Matched'] = df['Other_Currency_Matched'].astype(bool)
        
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_matched_buy_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM business_matched_buy WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        
        # Convert boolean columns back
        if 'CounterParty_Match' in df.columns:
            df['CounterParty_Match'] = df['CounterParty_Match'].astype(bool)
        if 'Choice_Match' in df.columns:
            df['Choice_Match'] = df['Choice_Match'].astype(bool)
        
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_matched_sell_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM business_matched_sell WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        
        # Convert boolean columns back
        if 'Choice_Match' in df.columns:
            df['Choice_Match'] = df['Choice_Match'].astype(bool)
        if 'CounterParty_Match' in df.columns:
            df['CounterParty_Match'] = df['CounterParty_Match'].astype(bool)
        
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_unmatched_buy_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM business_unmatched_buy WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_unmatched_sell_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM business_unmatched_sell WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        cols_to_drop = ['id', 'sys_created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_unmatched_business_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM business_unmatched_business WHERE record_date = ? ORDER BY id'
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
                SELECT record_date FROM business_final_df
                UNION SELECT record_date FROM business_matched_buy
                UNION SELECT record_date FROM business_matched_sell
                UNION SELECT record_date FROM business_unmatched_buy
                UNION SELECT record_date FROM business_unmatched_sell
                UNION SELECT record_date FROM business_unmatched_business
            ) WHERE record_date IS NOT NULL ORDER BY record_date DESC
        ''')
        dates = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return dates
    
    def save_metadata(self, key, value):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO business_reconciliation_metadata (key, value, updated_at) VALUES (?, ?, ?)',
                    (key, json.dumps(value, default=str), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
    
    def load_metadata(self, key, default=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT value FROM business_reconciliation_metadata WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else default


# Initialize database
db = BusinessFXReconciliationDB()


def get_available_business_dates():
    """Get all available dates with Business FX data"""
    return db.get_available_dates()


# --- Helper Functions for Record Management ---
def generate_business_record_id():
    return f"bus_{uuid.uuid4()}"


def add_business_unique_ids(df):
    """Add unique record IDs to dataframe"""
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '_record_id' not in df_copy.columns:
        df_copy['_record_id'] = [generate_business_record_id() for _ in range(len(df_copy))]
    return df_copy


def ensure_business_record_ids(df):
    """Ensure dataframe has _record_id column"""
    if df is None or df.empty:
        return df
    if '_record_id' not in df.columns:
        return add_business_unique_ids(df)
    return df


def add_business_audit_columns(df):
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


def add_business_row_numbers(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '#' in df_copy.columns:
        df_copy = df_copy.drop(columns=['#'])
    df_copy.insert(0, '#', range(1, len(df_copy) + 1))
    return df_copy


def remove_business_row_numbers(df):
    if df is None or df.empty:
        return df
    if '#' in df.columns:
        return df.drop(columns=['#'])
    return df


def get_business_current_user():
    if 'user' in st.session_state:
        return st.session_state['user'].get('username', 'unknown')
    return 'unknown_user'


def get_business_deleted_df_name(source_name):
    source_clean = source_name.lower().replace(' ', '_')
    if 'final' in source_clean:
        return 'deleted_final_business'
    elif 'matched_buy' in source_clean:
        return 'deleted_matched_buy'
    elif 'matched_sell' in source_clean:
        return 'deleted_matched_sell'
    elif 'unmatched_buy' in source_clean:
        return 'deleted_unmatched_buy'
    elif 'unmatched_sell' in source_clean:
        return 'deleted_unmatched_sell'
    elif 'unmatched_business' in source_clean:
        return 'deleted_unmatched_business'
    return f"deleted_{source_clean}"


def get_business_moved_df_name(source_name, target_name):
    target_clean = target_name.lower().replace(' ', '_')
    if 'final_business' in target_clean:
        return 'moved_final_business'
    elif 'matched_buy' in target_clean:
        return 'moved_matched_buy'
    elif 'matched_sell' in target_clean:
        return 'moved_matched_sell'
    elif 'unmatched_buy' in target_clean:
        return 'moved_unmatched_buy'
    elif 'unmatched_sell' in target_clean:
        return 'moved_unmatched_sell'
    elif 'unmatched_business' in target_clean:
        return 'moved_unmatched_business'
    return f"moved_{target_clean}"


def move_business_records_to_new_df(source_df, selected_record_ids, source_name, target_name, move_reason=""):
    if not selected_record_ids:
        return pd.DataFrame(), source_df
    source_df_copy = source_df.copy() if source_df is not None else pd.DataFrame()
    source_df_copy = ensure_business_record_ids(source_df_copy)
    if '#' in source_df_copy.columns:
        source_df_copy = source_df_copy.drop(columns=['#'])
    selected_records = source_df_copy[source_df_copy['_record_id'].isin(selected_record_ids)].copy()
    remaining_source = source_df_copy[~source_df_copy['_record_id'].isin(selected_record_ids)].reset_index(drop=True)
    if '#' in remaining_source.columns:
        remaining_source = remaining_source.drop(columns=['#'])
    if selected_records.empty:
        return pd.DataFrame(), source_df
    current_user = get_business_current_user()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    selected_records = add_business_audit_columns(selected_records)
    selected_records['moved_by'] = current_user
    selected_records['moved_from'] = source_name
    selected_records['moved_to'] = target_name
    selected_records['moved_at'] = current_time
    selected_records['move_reason'] = move_reason
    selected_records['move_type'] = f"{source_name} → {target_name}"
    return selected_records, remaining_source


def delete_business_records_to_new_df(source_df, selected_record_ids, source_name, delete_reason=""):
    if not selected_record_ids:
        return pd.DataFrame(), source_df
    source_df_copy = source_df.copy() if source_df is not None else pd.DataFrame()
    source_df_copy = ensure_business_record_ids(source_df_copy)
    if '#' in source_df_copy.columns:
        source_df_copy = source_df_copy.drop(columns=['#'])
    selected_records = source_df_copy[source_df_copy['_record_id'].isin(selected_record_ids)].copy()
    remaining_source = source_df_copy[~source_df_copy['_record_id'].isin(selected_record_ids)].reset_index(drop=True)
    if selected_records.empty:
        return pd.DataFrame(), source_df
    current_user = get_business_current_user()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    selected_records = add_business_audit_columns(selected_records)
    selected_records['deleted_by'] = current_user
    selected_records['deleted_at'] = current_time
    selected_records['delete_reason'] = delete_reason
    selected_records['deleted_from'] = source_name
    selected_records['source_dataframe'] = source_name
    return selected_records, remaining_source


def delete_business_selected_rows_with_audit(df, selected_record_ids, source_name, delete_reason="", df_name=None, on_data_change=None):
    if not selected_record_ids:
        return df, 0
    if isinstance(selected_record_ids, str):
        selected_record_ids = [selected_record_ids]
    source_df = df.copy() if df is not None else pd.DataFrame()
    if source_df.empty:
        return df, 0
    source_df = ensure_business_record_ids(source_df)
    if '#' in source_df.columns:
        source_df = source_df.drop(columns=['#'])
    deleted_records, remaining_source = delete_business_records_to_new_df(source_df, selected_record_ids, source_name, delete_reason)
    if deleted_records.empty:
        return df, 0
    deleted_df_name = get_business_deleted_df_name(source_name)
    if deleted_df_name not in st.session_state:
        st.session_state[deleted_df_name] = deleted_records
    else:
        existing = st.session_state[deleted_df_name]
        existing_ids = set(existing['_record_id'].tolist()) if not existing.empty else set()
        new_records = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_records.empty:
            st.session_state[deleted_df_name] = pd.concat([existing, new_records], ignore_index=True)
    
    db.save_deleted_records(st.session_state[deleted_df_name])
    
    if 'audit_deletes_log_business' not in st.session_state:
        st.session_state.audit_deletes_log_business = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.audit_deletes_log_business
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log_entries = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log_entries.empty:
            st.session_state.audit_deletes_log_business = pd.concat([existing_log, new_log_entries[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)
    
    db.save_audit_deletes(st.session_state.audit_deletes_log_business)
    
    remaining_source_with_numbers = add_business_row_numbers(remaining_source)
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_business_row_numbers(remaining_source.copy())
    
    # Update the main dataframe mapping
    main_df_mapping = {
        'Final Business': 'final_business_df',
        'Matched Buy': 'matched_buy_df',
        'Matched Sell': 'matched_sell_df',
        'Unmatched Buy': 'unmatched_buy_df',
        'Unmatched Sell': 'unmatched_sell_df',
        'Unmatched Business': 'unmatched_business_df'
    }
    if source_name in main_df_mapping:
        main_key = main_df_mapping[source_name]
        if main_key in st.session_state:
            st.session_state[main_key] = remove_business_row_numbers(remaining_source.copy())
    
    if on_data_change:
        on_data_change(remaining_source.copy())
    
    # Force update all stats
    update_business_deleted_stats()
    update_business_moved_stats()
    
    return remaining_source_with_numbers, len(selected_record_ids)


def clear_business_selection_state(key_prefix):
    selection_key = f"{key_prefix}_selection_state"
    if selection_key in st.session_state:
        st.session_state[selection_key] = {}


def update_business_moved_stats():
    moved_counts = {
        'moved_final_business': 0,
        'moved_matched_buy': 0,
        'moved_matched_sell': 0,
        'moved_unmatched_buy': 0,
        'moved_unmatched_sell': 0,
        'moved_unmatched_business': 0,
        'total_moved': 0
    }
    for key in moved_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            moved_counts[key] = len(st.session_state[key])
    moved_counts['total_moved'] = sum([moved_counts['moved_final_business'], moved_counts['moved_matched_buy'],
                                       moved_counts['moved_matched_sell'], moved_counts['moved_unmatched_buy'],
                                       moved_counts['moved_unmatched_sell'], moved_counts['moved_unmatched_business']])
    st.session_state.moved_stats_business = moved_counts
    return moved_counts


def update_business_deleted_stats():
    deleted_counts = {
        'deleted_final_business': 0,
        'deleted_matched_buy': 0,
        'deleted_matched_sell': 0,
        'deleted_unmatched_buy': 0,
        'deleted_unmatched_sell': 0,
        'deleted_unmatched_business': 0,
        'total_deleted': 0
    }
    for key in deleted_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            deleted_counts[key] = len(st.session_state[key])
    deleted_counts['total_deleted'] = sum([deleted_counts['deleted_final_business'], deleted_counts['deleted_matched_buy'],
                                           deleted_counts['deleted_matched_sell'], deleted_counts['deleted_unmatched_buy'],
                                           deleted_counts['deleted_unmatched_sell'], deleted_counts['deleted_unmatched_business']])
    st.session_state.deleted_stats_business = deleted_counts
    return deleted_counts


def sync_all_business_display_dataframes():
    for key in list(st.session_state.keys()):
        if key.endswith('_display_df'):
            base_key = key.replace('_display_df', '')
            if base_key in st.session_state and not st.session_state[base_key].empty:
                st.session_state[key] = add_business_row_numbers(st.session_state[base_key].copy())


def refresh_business_analytics_dataframes():
    analytics_dataframes = [
        ('final_business_df', 'final_business_analytics'),
        ('matched_buy_df', 'matched_buy_analytics'),
        ('matched_sell_df', 'matched_sell_analytics'),
        ('unmatched_buy_df', 'unmatched_buy_analytics'),
        ('unmatched_sell_df', 'unmatched_sell_analytics'),
        ('unmatched_business_df', 'unmatched_business_analytics')
    ]
    for session_key, df_key in analytics_dataframes:
        if session_key in st.session_state and not st.session_state[session_key].empty:
            st.session_state[df_key] = st.session_state[session_key].copy()


def reset_all_business_dataframes():
    """Reset all Business FX module dataframes to empty state"""
    with st.spinner("Resetting all dataframes..."):
        # Main dataframes
        st.session_state.final_business_df = pd.DataFrame()
        st.session_state.matched_buy_df = pd.DataFrame()
        st.session_state.matched_sell_df = pd.DataFrame()
        st.session_state.unmatched_buy_df = pd.DataFrame()
        st.session_state.unmatched_sell_df = pd.DataFrame()
        st.session_state.unmatched_business_df = pd.DataFrame()
        
        # Moved records dataframes
        st.session_state.moved_final_business = pd.DataFrame()
        st.session_state.moved_matched_buy = pd.DataFrame()
        st.session_state.moved_matched_sell = pd.DataFrame()
        st.session_state.moved_unmatched_buy = pd.DataFrame()
        st.session_state.moved_unmatched_sell = pd.DataFrame()
        st.session_state.moved_unmatched_business = pd.DataFrame()
        
        # Deleted records dataframes
        st.session_state.deleted_final_business = pd.DataFrame()
        st.session_state.deleted_matched_buy = pd.DataFrame()
        st.session_state.deleted_matched_sell = pd.DataFrame()
        st.session_state.deleted_unmatched_buy = pd.DataFrame()
        st.session_state.deleted_unmatched_sell = pd.DataFrame()
        st.session_state.deleted_unmatched_business = pd.DataFrame()
        
        # Audit logs
        st.session_state.audit_moves_log_business = pd.DataFrame()
        st.session_state.audit_deletes_log_business = pd.DataFrame()
        
        # Clear display dataframes
        display_keys = [key for key in st.session_state.keys() if key.endswith('_display_df')]
        for key in display_keys:
            st.session_state[key] = pd.DataFrame()
        
        # Clear selection states
        selection_keys = [key for key in st.session_state.keys() if key.endswith('_selection_state')]
        for key in selection_keys:
            st.session_state[key] = {}
        
        # Reset statistics
        st.session_state.moved_stats_business = {
            'moved_final_business': 0, 'moved_matched_buy': 0, 'moved_matched_sell': 0,
            'moved_unmatched_buy': 0, 'moved_unmatched_sell': 0, 'moved_unmatched_business': 0,
            'total_moved': 0
        }
        st.session_state.deleted_stats_business = {
            'deleted_final_business': 0, 'deleted_matched_buy': 0, 'deleted_matched_sell': 0,
            'deleted_unmatched_buy': 0, 'deleted_unmatched_sell': 0, 'deleted_unmatched_business': 0,
            'total_deleted': 0
        }
        
        logger.info("All Business FX module dataframes have been reset")
    
    return True


# --- Session State Initialization ---
def initialize_business_session_state():
    """Initialize all Business FX related session state variables"""
    
    # Main dataframes
    if 'final_business_df' not in st.session_state:
        st.session_state.final_business_df = pd.DataFrame()
    if 'matched_buy_df' not in st.session_state:
        st.session_state.matched_buy_df = pd.DataFrame()
    if 'matched_sell_df' not in st.session_state:
        st.session_state.matched_sell_df = pd.DataFrame()
    if 'unmatched_buy_df' not in st.session_state:
        st.session_state.unmatched_buy_df = pd.DataFrame()
    if 'unmatched_sell_df' not in st.session_state:
        st.session_state.unmatched_sell_df = pd.DataFrame()
    if 'unmatched_business_df' not in st.session_state:
        st.session_state.unmatched_business_df = pd.DataFrame()
    
    # Moved records dataframes
    if 'moved_final_business' not in st.session_state:
        st.session_state.moved_final_business = pd.DataFrame()
    if 'moved_matched_buy' not in st.session_state:
        st.session_state.moved_matched_buy = pd.DataFrame()
    if 'moved_matched_sell' not in st.session_state:
        st.session_state.moved_matched_sell = pd.DataFrame()
    if 'moved_unmatched_buy' not in st.session_state:
        st.session_state.moved_unmatched_buy = pd.DataFrame()
    if 'moved_unmatched_sell' not in st.session_state:
        st.session_state.moved_unmatched_sell = pd.DataFrame()
    if 'moved_unmatched_business' not in st.session_state:
        st.session_state.moved_unmatched_business = pd.DataFrame()
    
    # Deleted records dataframes
    if 'deleted_final_business' not in st.session_state:
        st.session_state.deleted_final_business = pd.DataFrame()
    if 'deleted_matched_buy' not in st.session_state:
        st.session_state.deleted_matched_buy = pd.DataFrame()
    if 'deleted_matched_sell' not in st.session_state:
        st.session_state.deleted_matched_sell = pd.DataFrame()
    if 'deleted_unmatched_buy' not in st.session_state:
        st.session_state.deleted_unmatched_buy = pd.DataFrame()
    if 'deleted_unmatched_sell' not in st.session_state:
        st.session_state.deleted_unmatched_sell = pd.DataFrame()
    if 'deleted_unmatched_business' not in st.session_state:
        st.session_state.deleted_unmatched_business = pd.DataFrame()
    
    # Audit logs
    if 'audit_moves_log_business' not in st.session_state:
        st.session_state.audit_moves_log_business = pd.DataFrame()
    if 'audit_deletes_log_business' not in st.session_state:
        st.session_state.audit_deletes_log_business = pd.DataFrame()
    
    # Statistics
    if 'moved_stats_business' not in st.session_state:
        st.session_state.moved_stats_business = {
            'moved_final_business': 0, 'moved_matched_buy': 0, 'moved_matched_sell': 0,
            'moved_unmatched_buy': 0, 'moved_unmatched_sell': 0, 'moved_unmatched_business': 0,
            'total_moved': 0
        }
    if 'deleted_stats_business' not in st.session_state:
        st.session_state.deleted_stats_business = {
            'deleted_final_business': 0, 'deleted_matched_buy': 0, 'deleted_matched_sell': 0,
            'deleted_unmatched_buy': 0, 'deleted_unmatched_sell': 0, 'deleted_unmatched_business': 0,
            'total_deleted': 0
        }
    
    # Current date tracking
    if 'business_current_date' not in st.session_state:
        st.session_state.business_current_date = datetime.now().strftime('%Y-%m-%d')
    if 'business_last_save_date' not in st.session_state:
        st.session_state.business_last_save_date = None


def safe_get_business_dataframe(df_name, default=pd.DataFrame()):
    """Safely get a dataframe from session state with default if not exists or empty"""
    if df_name in st.session_state and st.session_state[df_name] is not None:
        return st.session_state[df_name]
    return default

def clean_dataframe_for_arrow(df):
    """Clean dataframe to make it Arrow-compatible"""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Skip _record_id and # columns from type conversion
        if col in ['_record_id', '#']:
            continue
            
        # Convert mixed-type columns to string
        if df_copy[col].dtype == 'object':
            # Check if column contains mixed types
            try:
                # Try to see if it's mostly numeric
                numeric_series = pd.to_numeric(df_copy[col], errors='coerce')
                if numeric_series.notna().sum() > len(df_copy[col]) * 0.8:  # 80% numeric
                    df_copy[col] = numeric_series
                else:
                    # Convert to string
                    df_copy[col] = df_copy[col].astype(str)
            except:
                df_copy[col] = df_copy[col].astype(str)
        
        # Handle NaN/inf values in numeric columns
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].fillna(0)
            df_copy[col] = df_copy[col].replace([float('inf'), float('-inf')], 0)
    
    return df_copy


# --- Render Functions for Business FX ---
def render_business_editable_dataframe(df, title, key_prefix, on_data_change=None, show_delete=True, show_move=True, move_targets=None):
    """Render a single editable dataframe with full functionality"""
    
    # Diagnostic logging
    logger.debug(f"Rendering {title} with {len(df) if df is not None else 0} records")
    
    if df is None or df.empty:
        st.info(f"No {title} to display.")
        return df if df is not None else pd.DataFrame()
    
    st.markdown(f"### {title}")
    st.markdown(f"**Total Records: {len(df)}**")
    
    # Ensure dataframe has required columns
    df = ensure_business_record_ids(df)
    df = add_business_audit_columns(df)
    
    # Clean data types to make Arrow compatible
    df = clean_dataframe_for_arrow(df)
    
    # IMPORTANT FIX: Use UNIQUE keys per tab to prevent cross-contamination
    # Each tab gets its own isolated storage in session state
    display_df_key = f"{key_prefix}_business_display_df"
    original_df_key = f"{key_prefix}_business_original_df"
    
    # Check if we need to initialize or if the stored data doesn't match the current df
    should_initialize = False
    
    if display_df_key not in st.session_state:
        should_initialize = True
        logger.info(f"Initializing new display dataframe for {key_prefix}")
    elif st.session_state[display_df_key].empty and not df.empty:
        should_initialize = True
        logger.info(f"Display dataframe empty but source has {len(df)} records, reinitializing for {key_prefix}")
    elif original_df_key in st.session_state and not st.session_state[original_df_key].empty:
        # Check if the stored dataframe has the same number of rows as current df
        if len(st.session_state[original_df_key]) != len(df):
            should_initialize = True
            logger.info(f"Row count mismatch: stored={len(st.session_state[original_df_key])}, current={len(df)}, reinitializing for {key_prefix}")
    
    if should_initialize:
        # Create a fresh display version from the current df
        df_with_ids = ensure_business_record_ids(df)
        df_with_audit = add_business_audit_columns(df_with_ids)
        st.session_state[display_df_key] = add_business_row_numbers(df_with_audit)
        st.session_state[original_df_key] = remove_business_row_numbers(df_with_audit.copy())
        logger.info(f"Created new display dataframe for {key_prefix} with {len(df)} records")
    
    # Also ensure the stored original df is in sync with the current df if they have the same records
    if original_df_key in st.session_state and not st.session_state[original_df_key].empty:
        if len(st.session_state[original_df_key]) == len(df):
            # Same size, just update any changed data
            stored_ids = set(st.session_state[original_df_key]['_record_id'].tolist()) if '_record_id' in st.session_state[original_df_key].columns else set()
            current_ids = set(df['_record_id'].tolist()) if '_record_id' in df.columns else set()
            
            if stored_ids != current_ids:
                # IDs don't match, reinitialize
                df_with_ids = ensure_business_record_ids(df)
                df_with_audit = add_business_audit_columns(df_with_ids)
                st.session_state[display_df_key] = add_business_row_numbers(df_with_audit)
                st.session_state[original_df_key] = remove_business_row_numbers(df_with_audit.copy())
                logger.info(f"ID mismatch, reinitialized {key_prefix}")
    
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
                
                updated_df, deleted_count = delete_business_selected_rows_with_audit(
                    source_df, selected_ids, title, action_reason,
                    df_name=display_df_key, on_data_change=on_data_change
                )
                
                if original_df_key in st.session_state:
                    original_updated = remove_business_row_numbers(updated_df.copy())
                    st.session_state[original_df_key] = original_updated
                
                # Also update the main dataframe in session state
                if on_data_change:
                    on_data_change(original_updated)
                
                sync_all_business_display_dataframes()
                clear_business_selection_state(key_prefix)
                refresh_business_analytics_dataframes()
                update_business_deleted_stats()
                
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
                        source_key = f"{key_prefix}_business_original_df"
                        source_df = st.session_state.get(source_key, pd.DataFrame()).copy()
                        source_df = ensure_business_record_ids(source_df)
                        
                        moved_records, new_source = move_business_records_to_new_df(
                            source_df, selected_ids, title, selected_target, action_reason
                        )
                        
                        if not moved_records.empty:
                            moved_df_name = get_business_moved_df_name(title, selected_target)
                            
                            if moved_df_name not in st.session_state:
                                st.session_state[moved_df_name] = moved_records
                            else:
                                existing = st.session_state[moved_df_name]
                                existing_ids = set(existing['_record_id'].tolist()) if not existing.empty else set()
                                new_records = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_records.empty:
                                    st.session_state[moved_df_name] = pd.concat([existing, new_records], ignore_index=True)
                            
                            if 'audit_moves_log_business' not in st.session_state:
                                st.session_state.audit_moves_log_business = moved_records[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type']].copy() if 'move_type' in moved_records.columns else moved_records[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason']].copy()
                            else:
                                existing_log = st.session_state.audit_moves_log_business
                                existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
                                new_log_entries = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_log_entries.empty:
                                    st.session_state.audit_moves_log_business = pd.concat([existing_log, new_log_entries[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type'] if 'move_type' in new_log_entries.columns else ['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason']]], ignore_index=True)
                            
                            # Update the stored original dataframe
                            st.session_state[source_key] = new_source
                            st.session_state[display_df_key] = add_business_row_numbers(new_source)
                            
                            # Update the main dataframe in session state
                            if on_data_change:
                                on_data_change(new_source)
                            
                            clear_business_selection_state(key_prefix)
                            refresh_business_analytics_dataframes()
                            update_business_moved_stats()
                            
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
        
        # Clean download dataframe
        df_download = clean_dataframe_for_arrow(df_download)
        
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
            # Force refresh from source dataframe
            df_source = st.session_state.get(f"{key_prefix}_business_original_df", df)
            if not df_source.empty:
                df_with_ids = ensure_business_record_ids(df_source)
                df_with_audit = add_business_audit_columns(df_with_ids)
                st.session_state[display_df_key] = add_business_row_numbers(df_with_audit)
                st.session_state[original_df_key] = remove_business_row_numbers(df_with_audit.copy())
            clear_business_selection_state(key_prefix)
            st.rerun()
    
    with st.container():
        st.markdown("---")
        st.markdown("### Edit Data Directly")
        st.info("💡 Tip: Double-click any cell to edit its content. Use checkboxes below for batch operations.")
        
        df_for_edit = st.session_state[display_df_key].copy()
        
        # Clean for Arrow compatibility
        df_for_edit = clean_dataframe_for_arrow(df_for_edit)
        
        columns_to_drop = []
        if '#' in df_for_edit.columns:
            columns_to_drop.append('#')
        if '_record_id' in df_for_edit.columns:
            columns_to_drop.append('_record_id')
        
        if columns_to_drop:
            df_for_edit_for_display = df_for_edit.drop(columns=columns_to_drop)
        else:
            df_for_edit_for_display = df_for_edit
        
        # Convert problematic columns to string for display
        for col in df_for_edit_for_display.columns:
            if df_for_edit_for_display[col].dtype == 'object':
                # Convert mixed type columns to string
                df_for_edit_for_display[col] = df_for_edit_for_display[col].astype(str)
        
        edited_df = st.data_editor(
            df_for_edit_for_display,
            use_container_width=True,
            height=min(400, len(df_for_edit_for_display) * 35 + 38),
            key=f"{key_prefix}_data_editor_{datetime.now().timestamp()}",
            num_rows="dynamic"
        )
        
        if not edited_df.equals(df_for_edit_for_display):
            edited_with_ids = ensure_business_record_ids(edited_df.copy())
            edited_with_audit = add_business_audit_columns(edited_with_ids)
            updated_with_numbers = add_business_row_numbers(edited_with_audit)
            st.session_state[display_df_key] = updated_with_numbers
            
            if original_df_key in st.session_state:
                st.session_state[original_df_key] = remove_business_row_numbers(edited_with_audit.copy())
            
            if on_data_change:
                on_data_change(remove_business_row_numbers(edited_with_audit.copy()))
            
            refresh_business_analytics_dataframes()
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
        df_for_selection = clean_dataframe_for_arrow(df_for_selection)
        
        # Ensure _record_id exists
        if '_record_id' not in df_for_selection.columns:
            df_for_selection = ensure_business_record_ids(df_for_selection)
            st.session_state[display_df_key] = add_business_row_numbers(df_for_selection)
            if original_df_key in st.session_state:
                st.session_state[original_df_key] = remove_business_row_numbers(df_for_selection.copy())
        
        record_ids = df_for_selection['_record_id'].tolist() if '_record_id' in df_for_selection.columns else []
        
        if not record_ids:
            st.warning("No record IDs found. Please refresh the page.")
            return df
        
        # Use a container with fixed height for better performance with large datasets
        rows_container = st.container()
        
        with rows_container:
            # Display rows in a more efficient way for better performance
            for idx in range(len(df_for_selection)):
                # Use columns with proper widths
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
                    # Create a summary of the row for quick reference
                    row_summary = []
                    display_cols = [col for col in df_for_selection.columns if col not in ['#', '_record_id']][:5]  # Show first 5 columns
                    
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

def render_business_moved_records_tab():
    st.markdown("### 📋 Moved Records - Audit Trail")
    moved_stats = update_business_moved_stats()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("📋 Final Business Moved", moved_stats['moved_final_business'])
    with col2:
        st.metric("📋 Matched Buy Moved", moved_stats['moved_matched_buy'])
    with col3:
        st.metric("📋 Matched Sell Moved", moved_stats['moved_matched_sell'])
    with col4:
        st.metric("⚠️ Unmatched Buy Moved", moved_stats['moved_unmatched_buy'])
    with col5:
        st.metric("⚠️ Unmatched Sell Moved", moved_stats['moved_unmatched_sell'])
    with col6:
        st.metric("📊 Total Moved", moved_stats['total_moved'])
    
    st.markdown("---")
    
    moved_df_names = ['moved_final_business', 'moved_matched_buy', 'moved_matched_sell',
                      'moved_unmatched_buy', 'moved_unmatched_sell', 'moved_unmatched_business']
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
            cols_to_drop = ['_record_id', 'id', 'sys_created_at', 'import_date', 'last_modified', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)


def render_business_deleted_records_tab():
    st.markdown("### 🗑️ Deleted Records - Audit Trail")
    deleted_stats = update_business_deleted_stats()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("🗑️ Final Business Deleted", deleted_stats['deleted_final_business'])
    with col2:
        st.metric("🗑️ Matched Buy Deleted", deleted_stats['deleted_matched_buy'])
    with col3:
        st.metric("🗑️ Matched Sell Deleted", deleted_stats['deleted_matched_sell'])
    with col4:
        st.metric("🗑️ Unmatched Buy Deleted", deleted_stats['deleted_unmatched_buy'])
    with col5:
        st.metric("🗑️ Unmatched Sell Deleted", deleted_stats['deleted_unmatched_sell'])
    with col6:
        st.metric("📊 Total Deleted", deleted_stats['total_deleted'])
    
    st.markdown("---")
    
    deleted_df_names = ['deleted_final_business', 'deleted_matched_buy', 'deleted_matched_sell',
                        'deleted_unmatched_buy', 'deleted_unmatched_sell', 'deleted_unmatched_business']
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
            cols_to_drop = ['_record_id', 'id', 'sys_created_at', 'import_date', 'last_modified', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)


def render_business_full_statistics_dashboard():
    """Render comprehensive statistics dashboard with safe access"""
    st.markdown("### 📊 Comprehensive Statistics Dashboard")
    
    # Add refresh button
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            update_business_moved_stats()
            update_business_deleted_stats()
            st.rerun()
    
    # Get current data with safe defaults
    final_business_df = safe_get_business_dataframe('final_business_df')
    matched_buy_df = safe_get_business_dataframe('matched_buy_df')
    matched_sell_df = safe_get_business_dataframe('matched_sell_df')
    unmatched_buy_df = safe_get_business_dataframe('unmatched_buy_df')
    unmatched_sell_df = safe_get_business_dataframe('unmatched_sell_df')
    unmatched_business_df = safe_get_business_dataframe('unmatched_business_df')
    
    # Calculate current statistics
    final_business_count = len(final_business_df) if not final_business_df.empty else 0
    buy_matched_count = len(matched_buy_df) if not matched_buy_df.empty else 0
    buy_unmatched_count = len(unmatched_buy_df) if not unmatched_buy_df.empty else 0
    sell_matched_count = len(matched_sell_df) if not matched_sell_df.empty else 0
    sell_unmatched_count = len(unmatched_sell_df) if not unmatched_sell_df.empty else 0
    business_unmatched_count = len(unmatched_business_df) if not unmatched_business_df.empty else 0
    
    total_fx = buy_matched_count + buy_unmatched_count + sell_matched_count + sell_unmatched_count
    total_matched = buy_matched_count + sell_matched_count
    match_rate = (total_matched / total_fx * 100) if total_fx > 0 else 0
    
    # Create metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Final Business", final_business_count)
    with col2:
        st.metric("✅ Buy Matched", buy_matched_count)
    with col3:
        st.metric("⚠️ Buy Unmatched", buy_unmatched_count)
    with col4:
        st.metric("✅ Sell Matched", sell_matched_count)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚠️ Sell Unmatched", sell_unmatched_count)
    with col2:
        st.metric("🏦 Business Unmatched", business_unmatched_count)
    with col3:
        st.metric("💰 Total FX Trades", total_fx)
    with col4:
        st.metric("📈 Match Rate", f"{match_rate:.1f}%")
    
    # Create charts for better visualization
    if total_fx > 0 or final_business_count > 0:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if total_fx > 0:
                match_data = pd.DataFrame({
                    'Status': ['Matched', 'Unmatched'],
                    'Count': [total_matched, total_fx - total_matched]
                })
                fig = px.pie(match_data, values='Count', names='Status', title='Match Status Distribution', 
                            color_discrete_sequence=['#28a745', '#dc3545'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if total_fx > 0:
                trade_data = pd.DataFrame({
                    'Type': ['Buy Matched', 'Buy Unmatched', 'Sell Matched', 'Sell Unmatched'],
                    'Count': [buy_matched_count, buy_unmatched_count, sell_matched_count, sell_unmatched_count]
                })
                fig = px.bar(trade_data, x='Type', y='Count', title='Trade Distribution by Type', 
                            color='Type', color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
        
        # Moved and Deleted Records Summary
        st.markdown("### 📦 Audit Summary")
        moved_stats = update_business_moved_stats()
        deleted_stats = update_business_deleted_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if moved_stats and moved_stats.get('total_moved', 0) > 0:
                st.markdown("#### Moved Records")
                moved_df = pd.DataFrame([
                    {'Category': 'Final Business', 'Count': moved_stats.get('moved_final_business', 0)},
                    {'Category': 'Matched Buy', 'Count': moved_stats.get('moved_matched_buy', 0)},
                    {'Category': 'Matched Sell', 'Count': moved_stats.get('moved_matched_sell', 0)},
                    {'Category': 'Unmatched Buy', 'Count': moved_stats.get('moved_unmatched_buy', 0)},
                    {'Category': 'Unmatched Sell', 'Count': moved_stats.get('moved_unmatched_sell', 0)},
                    {'Category': 'Unmatched Business', 'Count': moved_stats.get('moved_unmatched_business', 0)}
                ])
                fig = px.bar(moved_df, x='Category', y='Count', title='Moved Records by Category', 
                            color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if deleted_stats and deleted_stats.get('total_deleted', 0) > 0:
                st.markdown("#### Deleted Records")
                deleted_df = pd.DataFrame([
                    {'Category': 'Final Business', 'Count': deleted_stats.get('deleted_final_business', 0)},
                    {'Category': 'Matched Buy', 'Count': deleted_stats.get('deleted_matched_buy', 0)},
                    {'Category': 'Matched Sell', 'Count': deleted_stats.get('deleted_matched_sell', 0)},
                    {'Category': 'Unmatched Buy', 'Count': deleted_stats.get('deleted_unmatched_buy', 0)},
                    {'Category': 'Unmatched Sell', 'Count': deleted_stats.get('deleted_unmatched_sell', 0)},
                    {'Category': 'Unmatched Business', 'Count': deleted_stats.get('deleted_unmatched_business', 0)}
                ])
                fig = px.bar(deleted_df, x='Category', y='Count', title='Deleted Records by Category', 
                            color='Count', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)


# --- Core Reconciliation Function ---
def reconcile_fx_with_business_updated(
    df_matched_counterparty: pd.DataFrame,
    df_matched_choice: pd.DataFrame,
    business_df_or_path,
    pct_tolerance: float = 0.001,
    debug: bool = False
):
    # --- helpers ---
    def order_reasons(reasons: list[str]) -> str:
        priority = {
            "deal side": 1,
            "currency": 2,
            "primary amount": 3,
            "primary bank": 4,
            "secondary amount": 5,
            "secondary bank": 6,
        }
        ordered = sorted(
            reasons,
            key=lambda r: min([priority[k] for k in priority if k in r.lower()], default=99)
        )
        return "; ".join(ordered)

    # --- load business file ---
    if isinstance(business_df_or_path, str):
        business_df = pd.read_excel(business_df_or_path, sheet_name=0)
    else:
        business_df = business_df_or_path.copy()

    business_df.columns = [c.strip() for c in business_df.columns]
    if "x" in business_df.columns:
        business_df = business_df.rename(columns={"x": "Created At"})
    if "Created At" in business_df.columns:
        business_df["Created At"] = business_df["Created At"].apply(parse_date)

    # --- Filter rows ---
    if "Status" in business_df.columns:
        mask = business_df["Status"].astype(str).str.lower().str.strip().apply(
            lambda s: (
                s == "c"
                or s.startswith("completed on")
                or "to receive" in s
            )
        )
        business_df = business_df[mask].copy()

    if "Collection bank" in business_df.columns and "Payee bank" in business_df.columns:
        business_df = business_df[~(
            business_df["Collection bank"].astype(str).str.lower().str.contains("baas|app", na=False)
            & business_df["Payee bank"].astype(str).str.lower().str.contains("baas|app", na=False)
        )].copy()

    # --- new tracking cols ---
    business_df["_matched"] = False
    business_df["KES_Equivalent_Matched"] = False
    business_df["Other_Currency_Matched"] = False
    business_df["Mismatch_Type"] = "Not evaluated"

    matched_buy, matched_sell, unmatched_buy, unmatched_sell = [], [], [], []

    # --- NEW: Matching logic using MatchedCounterParty and MatchedChoicePayment ---
    for idx, business_row in business_df.iterrows():
        if business_df.at[idx, "_matched"]:
            continue

        deal_type = str(business_row.get("Deal type", "")).strip()
        deal_currency, deal_side = parse_deal_type(deal_type)
        
        if not deal_side:
            continue

        business_amount = safe_float(business_row.get("Amount"))
        business_kes = safe_float(business_row.get("KES equivalent"))
        collection_bank = normalize_bank_key(business_row.get("Collection bank", ""))
        payee_bank = normalize_bank_key(business_row.get("Payee bank", ""))
        
        found_match = False
        amount_matched = False
        kes_eq_matched = False
        currency_matched = False
        bank_matched = False

        if deal_side == "buy":
            # For BUY deals:
            # - Amount should match with MatchedCounterPartyPayment (Trade Amount)
            # - KES Equivalent should match with MatchedChoicePayment (Trade Amount)
            
            # Match primary amount with CounterParty
            counterparty_match = False
            if business_amount is not None and deal_currency:
                # Filter CounterParty by bank and currency
                counterparty_filtered = df_matched_counterparty[
                    (df_matched_counterparty["Bank Table"].str.contains(collection_bank, na=False)) &
                    (df_matched_counterparty["Trade Currency"] == deal_currency.upper())
                ]
                
                for _, cp_row in counterparty_filtered.iterrows():
                    cp_amount = safe_float(cp_row.get("Trade Amount"))
                    if amounts_match(business_amount, cp_amount, pct_tolerance):
                        counterparty_match = True
                        amount_matched = True
                        currency_matched = True
                        bank_matched = True
                        break
            
            # Match KES equivalent with ChoicePayment
            choice_match = False
            if business_kes is not None:
                # Filter Choice by bank (KES currency)
                choice_filtered = df_matched_choice[
                    (df_matched_choice["Bank Table"].str.contains(payee_bank, na=False)) &
                    (df_matched_choice["Trade Currency"] == "KES")
                ]
                
                for _, ch_row in choice_filtered.iterrows():
                    ch_amount = safe_float(ch_row.get("Trade Amount"))
                    if amounts_match(business_kes, ch_amount, pct_tolerance):
                        choice_match = True
                        kes_eq_matched = True
                        break

            # Determine match type
            if counterparty_match and choice_match:
                business_df.at[idx, "_matched"] = True
                business_df.at[idx, "KES_Equivalent_Matched"] = True
                business_df.at[idx, "Other_Currency_Matched"] = True
                business_df.at[idx, "Mismatch_Type"] = "None"
                match_type = "Full"
            elif counterparty_match:
                business_df.at[idx, "_matched"] = True
                business_df.at[idx, "KES_Equivalent_Matched"] = False
                business_df.at[idx, "Other_Currency_Matched"] = True
                business_df.at[idx, "Mismatch_Type"] = "KES only mismatch"
                match_type = "Primary only"
            elif choice_match:
                business_df.at[idx, "_matched"] = True
                business_df.at[idx, "KES_Equivalent_Matched"] = True
                business_df.at[idx, "Other_Currency_Matched"] = False
                business_df.at[idx, "Mismatch_Type"] = "Other currency only mismatch"
                match_type = "Secondary only"
            else:
                business_df.at[idx, "_matched"] = False
                match_type = "No match"

            if counterparty_match or choice_match:
                record = {
                    "Business index": idx,
                    "Deal Type": deal_type,
                    "Business Amount": business_amount,
                    "Business KES eq": business_kes,
                    "Collection Bank": collection_bank,
                    "Payee Bank": payee_bank,
                    "Match Type": match_type,
                    "CounterParty Match": counterparty_match,
                    "Choice Match": choice_match,
                }
                matched_buy.append(record)
                found_match = True

        elif deal_side == "sell":
            # For SELL deals:
            # - Amount should match with MatchedChoicePayment (Trade Amount)
            # - KES Equivalent should match with MatchedCounterPartyPayment (Trade Amount)
            
            # Match primary amount with ChoicePayment
            choice_match = False
            if business_amount is not None and deal_currency:
                # Filter Choice by bank and currency
                choice_filtered = df_matched_choice[
                    (df_matched_choice["Bank Table"].str.contains(payee_bank, na=False)) &
                    (df_matched_choice["Trade Currency"] == deal_currency.upper())
                ]
                
                for _, ch_row in choice_filtered.iterrows():
                    ch_amount = safe_float(ch_row.get("Trade Amount"))
                    if amounts_match(business_amount, ch_amount, pct_tolerance):
                        choice_match = True
                        amount_matched = True
                        currency_matched = True
                        bank_matched = True
                        break
            
            # Match KES equivalent with CounterParty
            counterparty_match = False
            if business_kes is not None:
                # Filter CounterParty by bank (KES currency)
                counterparty_filtered = df_matched_counterparty[
                    (df_matched_counterparty["Bank Table"].str.contains(collection_bank, na=False)) &
                    (df_matched_counterparty["Trade Currency"] == "KES")
                ]
                
                for _, cp_row in counterparty_filtered.iterrows():
                    cp_amount = safe_float(cp_row.get("Trade Amount"))
                    if amounts_match(business_kes, cp_amount, pct_tolerance):
                        counterparty_match = True
                        kes_eq_matched = True
                        break

            # Determine match type
            if choice_match and counterparty_match:
                business_df.at[idx, "_matched"] = True
                business_df.at[idx, "KES_Equivalent_Matched"] = True
                business_df.at[idx, "Other_Currency_Matched"] = True
                business_df.at[idx, "Mismatch_Type"] = "None"
                match_type = "Full"
            elif choice_match:
                business_df.at[idx, "_matched"] = True
                business_df.at[idx, "KES_Equivalent_Matched"] = False
                business_df.at[idx, "Other_Currency_Matched"] = True
                business_df.at[idx, "Mismatch_Type"] = "KES only mismatch"
                match_type = "Primary only"
            elif counterparty_match:
                business_df.at[idx, "_matched"] = True
                business_df.at[idx, "KES_Equivalent_Matched"] = True
                business_df.at[idx, "Other_Currency_Matched"] = False
                business_df.at[idx, "Mismatch_Type"] = "Other currency only mismatch"
                match_type = "Secondary only"
            else:
                business_df.at[idx, "_matched"] = False
                match_type = "No match"

            if choice_match or counterparty_match:
                record = {
                    "Business index": idx,
                    "Deal Type": deal_type,
                    "Business Amount": business_amount,
                    "Business KES eq": business_kes,
                    "Collection Bank": collection_bank,
                    "Payee Bank": payee_bank,
                    "Match Type": match_type,
                    "Choice Match": choice_match,
                    "CounterParty Match": counterparty_match,
                }
                matched_sell.append(record)
                found_match = True

        # Handle unmatched records
        if not found_match:
            reasons = []
            if not amount_matched:
                reasons.append("No primary amount match found")
            if not kes_eq_matched:
                reasons.append("No KES equivalent match found")
            if not currency_matched:
                reasons.append("No currency match found")
            if not bank_matched:
                reasons.append("No bank match found")

            rec = {
                "Business index": idx,
                "Deal Type": deal_type,
                "Business Amount": business_amount,
                "Business KES eq": business_kes,
                "Status": "No match",
                "Reasons": order_reasons(reasons) if reasons else ["No eligible matches"]
            }
            if deal_side == "buy":
                unmatched_buy.append(rec)
            else:
                unmatched_sell.append(rec)

    # --- build return dict ---
    unmatched_business_df = business_df[business_df["_matched"] == False].drop(columns=["_matched"])
    return {
        "matched_buy_df": pd.DataFrame(matched_buy),
        "matched_sell_df": pd.DataFrame(matched_sell),
        "unmatched_buy_df": pd.DataFrame(unmatched_buy),
        "unmatched_sell_df": pd.DataFrame(unmatched_sell),
        "unmatched_business_df": unmatched_business_df,
        "final_business_df": business_df.drop(columns=["_matched"]),  # downloadable full version
    }


# --- Helper Functions ---
def save_uploaded_file(file, filename):
    print("saving fx uploaded data : ", filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path


def save_dataframe(df, filename):
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


# --- Restore session state for FX Trade Tracker ---
st.session_state.fx_trade_tracker_df = load_dataframe("fx_trade_tracker_df.pkl")
st.session_state.fx_trade_tracker_sheet = load_object("fx_trade_tracker_sheet.pkl")
st.session_state.fx_trade_tracker_col_mapping = load_object("fx_trade_tracker_col_mapping.pkl", {
    'Action Type': 'Action Type',
    'Status': 'Status',
    'Created At': 'Created At',
    'Buy Currency Amount': 'Buy Currency Amount',
    'Buy Trade Info': 'Buy Trade Info',
    'Sell Currency Amount': 'Sell Currency Amount',
    'Sell Trade Info': 'Sell Trade Info',
    'Vendor ID': 'Vendor ID',
    'Vendor Name': 'Vendor Name',
    'Counterparty Dealer': 'Counterparty Dealer',
})


# --- Config ---
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6)

OUT_MATCHED_BUY = "MatchedBuy_business.csv"
OUT_MATCHED_SELL = "MatchedSell_business.csv"
OUT_UNMATCHED_BUY = "UnmatchedBuy_business.csv"
OUT_UNMATCHED_SELL = "UnmatchedSell_business.csv"
OUT_UNMATCHED_business = "Unmatchedbusiness.csv"

# --- FX Rates (demo only, update with real rates if needed) ---
FX_RATES = {
    "USDKES": 145.0,
    "EURKES": 155.0,
    "GBPUSD": 1.25,
    "USDGBP": 0.8,
    "EURUSD": 1.08,
    "USDEUR": 0.92,
    "KESUSD": 1 / 145.0,
    "KESEUR": 1 / 155.0,
}

DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d", "%d-%m-%Y", "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f", # ISO format with microseconds
]

FUZZY_MATCH_THRESHOLD = 70


# --- Helpers ---
def safe_float(x):
    if pd.isna(x) or x is None:
        return None
    try:
        s = str(x).replace(",", "").strip()
        return abs(float(s))
    except Exception:
        return None


def get_fx_rate(from_currency, to_currency, date=None):
    if not from_currency or not to_currency:
        return 1.0
    f = from_currency.upper(); t = to_currency.upper()
    if f == t: return 1.0
    pair = f + t
    if pair in FX_RATES: return FX_RATES[pair]
    inv = t + f
    if inv in FX_RATES: return 1.0 / FX_RATES[inv]
    return 1.0


def convert_currency(amount, from_currency, to_currency, date=None):
    if amount is None: return None
    rate = get_fx_rate(from_currency, to_currency, date)
    return amount * rate


def parse_date(maybe_date):
    if pd.isna(maybe_date):
        return None
    try:
        return pd.to_datetime(maybe_date, infer_datetime_format=True, errors="coerce")
    except Exception:
        try:
            s = str(maybe_date).strip()
            for fmt in DATE_FORMATS:
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    pass
        except Exception:
            return None
    return None


def normalize_bank_key(raw_key):
    if pd.isna(raw_key) or raw_key is None:
        return ""
    s = str(raw_key).strip()
    replacements = {
        "ncba bank kenya plc": "NCBA", "ncba bank": "NCBA",
        "equity bank": "Equity", "i&m bank": "I&M",
        "central bank of kenya": "CBK", "kenya commercial bank": "KCB",
        "kcb bank": "KCB", "sbm bank (kenya) limited": "SBM", "sbm bank": "SBM",
        "absa bank": "Absa", "kingdom bank": "Kingdom", "uba": "UBA", "yeepay": "Yeepay"
    }
    low = s.lower()
    for long, short in replacements.items():
        if low == long or low.startswith(long):
            return short
    # fuzzy fallback
    choices = list(replacements.values()) + [k.title() for k in replacements.keys()]
    match = process.extractOne(low, choices, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_MATCH_THRESHOLD:
        return match[0]
    return s.title()


def extract_bank_and_currency_from_trade_info(info):
    if pd.isna(info) or info is None:
        return (None, None)
    parts = str(info).split("-")
    if len(parts) < 2:
        return (parts[0].strip(), None) if parts else (None, None)
    bank = parts[0].strip()
    currency = parts[1].strip().upper()
    return (bank, currency)


def parse_deal_type(deal_type_str):
    if pd.isna(deal_type_str) or deal_type_str is None:
        return (None, None)
    parts = str(deal_type_str).strip().split()
    if len(parts) >= 2:
        return (parts[0].upper(), parts[1].strip().lower())
    if len(parts) == 1:
        token = parts[0]
        if "sell" in token.lower(): return (None, "sell")
        if "buy" in token.lower(): return (None, "buy")
        return (token.upper(), None)
    return (None, None)


def amounts_match(a, b, pct_tolerance=0.001, abs_min=0.05):
    """
    Compare two amounts with both percentage and absolute tolerance.

    Args:
        a, b: Numeric values (or strings convertible to float).
        pct_tolerance (float): Relative tolerance (default 0.1%).
        abs_min (float): Minimum absolute tolerance (default 0.05).

    Returns:
        bool: True if amounts are considered matching, else False.
    """
    if a is None or b is None:
        return False

    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False

    tol = max(abs_min, pct_tolerance * max(abs(b), 1.0))
    return abs(abs(a) - abs(b)) <= tol


# utility to ensure the DataFrame has all target columns in order (creates empty cols if missing)
def ensure_columns_and_order(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    # reorder
    out = out[[c for c in cols]]
    return out


def df_to_excel_bytes(df: pd.DataFrame, summary_df: pd.DataFrame | None = None) -> bytes:
    """
    Write the dataframe plus optional summary table into an Excel bytes buffer,
    preserving the TARGET_COLUMNS order and appending the summary block below.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # write main sheet
        final_df_to_write = ensure_columns_and_order(df, TARGET_COLUMNS)
        final_df_to_write.to_excel(writer, index=False, sheet_name="Final-business-with-match-status")
        # append summary on same sheet (start row after content + 2)
        if summary_df is not None:
            startrow = len(final_df_to_write) + 2
            # summary_df should be a small dataframe or Series -> write as frame
            summary_df.to_excel(writer, index=False, sheet_name="Final-business-with-match-status", startrow=startrow)
    return output.getvalue()


# --- UI / Display helpers ---
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6)

OUT_MATCHED_BUY = "MatchedBuy_business.csv"
OUT_MATCHED_SELL = "MatchedSell_business.csv"
OUT_UNMATCHED_BUY = "UnmatchedBuy_business.csv"
OUT_UNMATCHED_SELL = "UnmatchedSell_business.csv"
OUT_UNMATCHED_business = "Unmatchedbusiness.csv"

# The column layout observed in your example final report (keeps the same order).
TARGET_COLUMNS = [
    "Created At",
    "Reference number",
    "Deal type",
    "Client Name",
    "Amount",
    "Rate",
    "KES equivalent",
    "Collection bank",
    "Payee bank",
    "Client Account Number",
    "Status",
    "KES_Equivalent_Matched",
    "Other_Currency_Matched",
    "Mismatch_Type"
]


def build_summary_from_final(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary grouped by Deal type:
      - Sum of Amount (non-empty)
      - Sum of Adjusted KES equivalent (non-empty)

    Adjustment:
      If Status is "completed on <date>" AND Client Account Name contains a numeric value,
      then subtract that numeric value from KES equivalent before summation.
    """
    df = final_df.copy()

    # Ensure numeric (empty/blank -> NaN)
    df["Amount_numeric"] = pd.to_numeric(df.get("Amount", ""), errors="coerce")
    df["KES_numeric"] = pd.to_numeric(df.get("KES equivalent", ""), errors="coerce")

    # Convert Client Account Name to numeric if possible
    df["Client_numeric"] = pd.to_numeric(df.get("Client Account Number", ""), errors="coerce")

    # --- Adjustment logic ---
    cond = (
        df["Status"].astype(str).str.lower().str.startswith("completed on")
        & df["Client_numeric"].notna()
    )

    # Create Adjusted_KES tracker column
    df["Adjusted_KES"] = df["KES_numeric"]

    # Apply adjustment: subtract Client_numeric from KES_numeric
    df.loc[cond, "Adjusted_KES"] = df.loc[cond, "KES_numeric"] - df.loc[cond, "Client_numeric"]

    # Drop rows where both are NaN (nothing to contribute)
    df = df.dropna(subset=["Amount_numeric", "Adjusted_KES"], how="all")

    # Group by deal type using Adjusted_KES
    # check if df.column contains Deal type and if not create it with default value "Unknown"
    if "Deal type" not in df.columns:
        df["Deal type"] = "Unknown"

    summary = (
        df.groupby("Deal type", dropna=False)
        .agg(
            Total_Amount=("Amount_numeric", "sum"),
            Total_KES=("Adjusted_KES", "sum"),
        )
        .reset_index()
    )

    # Clean up numeric formatting
    summary["Total_Amount"] = summary["Total_Amount"].round(2)
    summary["Total_KES"] = summary["Total_KES"].round(2)

    return summary  # return both summary and df with tracker column


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --- Updated UI function ---
def business_reconciliation_app(matched_counterparty, matched_choice, debug_mode):
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_business_session_state()
    
    # Header
    # st.markdown("""
    # <div class="main-header">
    #     <h1>🏦 Business FX Reconciliation</h1>
    #     <p>Reconcile business FX transactions with bank statements</p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # ========== DATA MANAGEMENT SECTION ==========
    st.markdown("### 📅 Data Management")
    
    available_dates = get_available_business_dates()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        if available_dates:
            selected_load_date = st.selectbox(
                "📅 Select date to load:",
                options=available_dates,
                index=0,
                key="business_load_date_select"
            )
        else:
            st.selectbox("📅 Select date to load:", options=["No data available"], disabled=True, key="business_load_date_select")
            selected_load_date = None
    
    with col2:
        if selected_load_date and available_dates:
            if st.button("📂 Load Data", use_container_width=True, key="load_business_btn"):
                db.load_business_data_only(selected_load_date)
                st.rerun()
    
    with col3:
        current_date = datetime.now().strftime('%Y-%m-%d')
        st.metric("Current Date", current_date)
    
    with col4:
        if st.button("💾 Save Data", type="primary", use_container_width=True, key="save_business_btn"):
            db.save_business_data_only()
            st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Reset Current Module Data", type="secondary", use_container_width=True, key="reset_module_business_btn"):
            reset_all_business_dataframes()
            st.success("✅ All current module dataframes have been reset!")
            st.balloons()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Reset All Data (Including Saved)", type="secondary", use_container_width=True, key="reset_all_business_btn"):
            target_date = datetime.now().strftime('%Y-%m-%d')
            
            # Reset session state
            reset_all_business_dataframes()
            
            # Also clear database for current date
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            tables_to_clear = [
                'business_final_df', 'business_matched_buy', 'business_matched_sell',
                'business_unmatched_buy', 'business_unmatched_sell', 'business_unmatched_business',
                'business_moved_records', 'business_deleted_records',
                'business_audit_moves_log', 'business_audit_deletes_log'
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
        if st.button("📊 Refresh Dashboard", type="primary", use_container_width=True, key="refresh_business_dashboard_btn"):
            # Update all stats without resetting data
            update_business_moved_stats()
            update_business_deleted_stats()
            refresh_business_analytics_dataframes()
            st.success("✅ Dashboard refreshed!")
            st.rerun()
    
    st.markdown("---")
    
    # Check if bank statements are processed
    # if not st.session_state.get('bank_dfs'):
    #     st.warning("Please go to 'Bank Statement Management' to upload and process bank statements first.")
    #     return
    
    # Get the matched dataframes from session state
    df_matched_counterparty = (
        matched_counterparty 
        if matched_counterparty is not None 
        else st.session_state.get('df_matched_counterparty', pd.DataFrame())
    )

    df_matched_choice = (
        matched_choice 
        if matched_choice is not None 
        else st.session_state.get('df_matched_choice', pd.DataFrame())
    )
    
    if df_matched_counterparty.empty or df_matched_choice.empty:
        st.warning("Matched counterparty and choice payment data not available. Please process bank statements first.")
        return

    # Mode selection
    mode = st.radio("Select Mode:", ["Interactive Final Report Mode", "Standard Mode"], 
                   help="Standard Mode: Basic reconciliation with downloads. Interactive Mode: Edit final report with row management.")

    try:
        uploaded_business_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

        if uploaded_business_file is not None:
            file_name = uploaded_business_file.name.lower()

            try:
                if file_name.endswith(".csv"):
                    business_df = pd.read_csv(uploaded_business_file)
                    st.success("CSV file loaded successfully!")

                else:
                    # Read all sheets first
                    xls = pd.ExcelFile(uploaded_business_file)
                    sheet_names = xls.sheet_names

                    # If multiple sheets, let user choose
                    sheet_choice = st.selectbox("Select a sheet to load", sheet_names)

                    # Load selected sheet
                    business_df = pd.read_excel(uploaded_business_file, sheet_name=sheet_choice)
                    st.success(f"Excel file loaded successfully! (Sheet: {sheet_choice})")

                # Show preview
                st.write("### Preview of uploaded data", business_df.head())

            except Exception as e:
                st.error(f"Error loading business file: {e}")
                return
        else:
            st.info("Upload the business file to continue.")
            # return

        # --- Data Cleaning & Preparation ---
        # 1. Clean Amount column (drop null, empty, n/a, 0)
        business_df['Amount'] = business_df['Amount'].replace(
            ["", " ", "n/a", "N/A", None], pd.NA
        )
        business_df['Amount'] = pd.to_numeric(business_df['Amount'], errors='coerce')
        business_df = business_df.dropna(subset=['Amount'])
        business_df = business_df[business_df['Amount'] != 0]

        # --- Clean Rate column ---
        if 'Rate' in business_df.columns:
            business_df['Rate'] = business_df['Rate'].replace(
                ["", " ", "n/a", "N/A", None], pd.NA
            )
            business_df['Rate'] = pd.to_numeric(business_df['Rate'], errors='coerce')

        # 2. Create completed_date column
        date_col = "x" if "x" in business_df.columns else business_df.columns[0]

        def extract_completed_date(row):
            status = str(row.get("Status", ""))
            match = re.search(r"completed on (\d{1,2}\.\d{1,2}\.\d{4})", status, re.IGNORECASE)
            if match:
                return pd.to_datetime(match.group(1), format="%d.%m.%Y", errors="coerce")
            return row[date_col]

        if "Status" in business_df.columns:
            business_df["completed_date"] = business_df.apply(extract_completed_date, axis=1)
            business_df["completed_date"] = pd.to_datetime(
                business_df["completed_date"], errors="coerce"
            )

        # Ensure x column is datetime
        if date_col in business_df.columns:
            business_df[date_col] = pd.to_datetime(business_df[date_col], errors="coerce")

        # 3. Add KES equivalent if columns exist
        if 'Amount' in business_df.columns and 'Rate' in business_df.columns:
            business_df['KES equivalent'] = business_df['Amount'] * business_df['Rate']

        # --- Filtering by (x OR completed_date) ---
        st.subheader("📅 Filter by Date Range (x OR completed_date)")

        # Drop rows where both are NaT
        df_with_dates = business_df.dropna(subset=[date_col, "completed_date"], how="all").copy()

        if not df_with_dates.empty:
            min_date = pd.concat([
                df_with_dates[date_col].dropna(),
                df_with_dates["completed_date"].dropna()
            ]).min().date()

            max_date = pd.concat([
                df_with_dates[date_col].dropna(),
                df_with_dates["completed_date"].dropna()
            ]).max().date()

            start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

            if start_date > end_date:
                st.warning("⚠️ Start date cannot be after End date")
            else:
                filtered_df = df_with_dates[
                    ((df_with_dates[date_col].dt.date >= start_date) &
                    (df_with_dates[date_col].dt.date <= end_date)) |
                    ((df_with_dates["completed_date"].dt.date >= start_date) &
                    (df_with_dates["completed_date"].dt.date <= end_date))
                ]

                st.write(
                    f"Showing records where **x OR completed_date** "
                    f"is between **{start_date}** and **{end_date}**"
                )
                st.dataframe(filtered_df)

                # --- Download filtered results ---
                st.subheader("⬇️ Download Filtered Data")

                # CSV
                csv = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="filtered_fx_transactions.csv",
                    mime="text/csv",
                )

                # Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name="Filtered Data")
                st.download_button(
                    label="Download as Excel",
                    data=output.getvalue(),
                    file_name="filtered_fx_transactions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                business_df = filtered_df
        else:
            st.info("No valid dates found in x or completed_date for filtering.")

    except Exception as e:
        st.error(f"Error loading business file: {e}")
        # return

    debug = st.checkbox("Enable Debug Mode", value=False)
    tolerance = st.slider("Matching tolerance (percentage):", 0.01, 1.0, 0.1, step=0.01)

    if st.button("Run Reconciliation"):
        with st.spinner("Reconciling..."):
            results = reconcile_fx_with_business_updated(
                df_matched_counterparty, 
                df_matched_choice, 
                business_df,
                pct_tolerance=tolerance / 100.0,
                debug=debug
            )
        st.success("✅ Reconciliation complete!")

        if results:
            # Store results in session state with unique IDs and audit columns
            matched_buy_df = results.get("matched_buy_df", pd.DataFrame())
            matched_sell_df = results.get("matched_sell_df", pd.DataFrame())
            unmatched_buy_df = results.get("unmatched_buy_df", pd.DataFrame())
            unmatched_sell_df = results.get("unmatched_sell_df", pd.DataFrame())
            unmatched_business_df = results.get("unmatched_business_df", pd.DataFrame())
            final_business_df = results.get("final_business_df", pd.DataFrame()).copy()
            
            # Add unique IDs and audit columns
            for df_name, df_value in [
                ('matched_buy_df', matched_buy_df),
                ('matched_sell_df', matched_sell_df),
                ('unmatched_buy_df', unmatched_buy_df),
                ('unmatched_sell_df', unmatched_sell_df),
                ('unmatched_business_df', unmatched_business_df),
                ('final_business_df', final_business_df)
            ]:
                if not df_value.empty:
                    df_value = add_business_unique_ids(df_value)
                    df_value = add_business_audit_columns(df_value)
                    st.session_state[df_name] = df_value
                else:
                    st.session_state[df_name] = df_value
            
            # Refresh analytics
            refresh_business_analytics_dataframes()
            update_business_moved_stats()
            update_business_deleted_stats()

    # Extract results
    matched_buy_df = st.session_state.get("matched_buy_df", pd.DataFrame())
    matched_sell_df = st.session_state.get("matched_sell_df", pd.DataFrame())
    unmatched_buy_df = st.session_state.get("unmatched_buy_df", pd.DataFrame())
    unmatched_sell_df = st.session_state.get("unmatched_sell_df", pd.DataFrame())
    unmatched_business_df = st.session_state.get("unmatched_business_df", pd.DataFrame())
    final_business_df = st.session_state.get("final_business_df", pd.DataFrame()).copy()

    # Ensure the right logic columns exist
    for flag in ("Other_Currency_Matched", "KES_Equivalent_Matched", "Mismatch_Type"):
        if flag not in final_business_df.columns:
            final_business_df[flag] = False if flag != "Mismatch_Type" else ""

    # Show result summary counts & visualization
    st.header("Results Summary")
    st.write(f"✅ Matched Buy: {len(matched_buy_df)}")
    st.write(f"❌ Unmatched Buy: {len(unmatched_buy_df)}")
    st.write(f"✅ Matched Sell: {len(matched_sell_df)}")
    st.write(f"❌ Unmatched Sell: {len(unmatched_sell_df)}")
    st.write(f"📤 Unmatched business: {len(unmatched_business_df)}")

    st.subheader("Visualization")
    summary_counts = pd.DataFrame({
        "Category": ["Matched Buy", "Unmatched Buy", "Matched Sell", "Unmatched Sell", "Unmatched business"],
        "Count": [len(matched_buy_df), len(unmatched_buy_df), len(matched_sell_df), len(unmatched_sell_df), len(unmatched_business_df)]
    })
    fig, ax = plt.subplots()
    sns.barplot(x="Category", y="Count", data=summary_counts, ax=ax)
    ax.set_title("Reconciliation Status Overview")
    st.pyplot(fig)

    # Standard Mode - Basic downloads
    if mode == "Standard Mode":
        def df_to_csv(df): 
            return df.to_csv(index=False).encode("utf-8")

        st.subheader("Download Results")
        if not matched_buy_df.empty:
            st.download_button("Download Matched Buy", df_to_csv(matched_buy_df), OUT_MATCHED_BUY, "text/csv")
        if not unmatched_buy_df.empty:
            st.download_button("Download Unmatched Buy", df_to_csv(unmatched_buy_df), OUT_UNMATCHED_BUY, "text/csv")
        if not matched_sell_df.empty:
            st.download_button("Download Matched Sell", df_to_csv(matched_sell_df), OUT_MATCHED_SELL, "text/csv")
        if not unmatched_sell_df.empty:
            st.download_button("Download Unmatched Sell", df_to_csv(unmatched_sell_df), OUT_UNMATCHED_SELL, "text/csv")
        if not unmatched_business_df.empty:
            st.download_button("Download Unmatched business", df_to_csv(unmatched_business_df), OUT_UNMATCHED_business, "text/csv")
        if not final_business_df.empty:
            st.download_button("Download Full business with Match Status", df_to_csv(final_business_df), 
                                "Final_business_with_Match_Status.csv", "text/csv")

    # Interactive Mode - Advanced editing capabilities
    else:
        # Move targets configuration
        move_targets_final_business = {
            "Matched Buy": "matched_buy_df",
            "Matched Sell": "matched_sell_df",
            "Unmatched Buy": "unmatched_buy_df",
            "Unmatched Sell": "unmatched_sell_df",
            "Unmatched Business": "unmatched_business_df"
        }
        
        move_targets_matched_buy = {
            "Final Business": "final_business_df",
            "Matched Sell": "matched_sell_df",
            "Unmatched Buy": "unmatched_buy_df",
            "Unmatched Sell": "unmatched_sell_df",
            "Unmatched Business": "unmatched_business_df"
        }
        
        move_targets_matched_sell = {
            "Final Business": "final_business_df",
            "Matched Buy": "matched_buy_df",
            "Unmatched Buy": "unmatched_buy_df",
            "Unmatched Sell": "unmatched_sell_df",
            "Unmatched Business": "unmatched_business_df"
        }
        
        move_targets_unmatched_buy = {
            "Final Business": "final_business_df",
            "Matched Buy": "matched_buy_df",
            "Matched Sell": "matched_sell_df",
            "Unmatched Sell": "unmatched_sell_df",
            "Unmatched Business": "unmatched_business_df"
        }
        
        move_targets_unmatched_sell = {
            "Final Business": "final_business_df",
            "Matched Buy": "matched_buy_df",
            "Matched Sell": "matched_sell_df",
            "Unmatched Buy": "unmatched_buy_df",
            "Unmatched Business": "unmatched_business_df"
        }
        
        move_targets_unmatched_business = {
            "Final Business": "final_business_df",
            "Matched Buy": "matched_buy_df",
            "Matched Sell": "matched_sell_df",
            "Unmatched Buy": "unmatched_buy_df",
            "Unmatched Sell": "unmatched_sell_df"
        }
        
        # Create tabs for main content
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Final Business",
            "📋 Matched Buy", 
            "⚠️ Unmatched Buy", 
            "📋 Matched Sell", 
            "⚠️ Unmatched Sell",
            "🏦 Unmatched Business",
            "📊 Audit Trail"
        ])
        
        with tab1:
            def update_final_business(df):
                st.session_state.final_business_df = add_business_unique_ids(df) if not df.empty else df
                if not st.session_state.final_business_df.empty:
                    st.session_state.final_business_df = add_business_audit_columns(st.session_state.final_business_df)
                update_business_moved_stats()
                update_business_deleted_stats()
            
            # For Final Business tab, we want to show all data without conditional hiding
            final_display_df = final_business_df.copy()
            if not final_display_df.empty:
                # Ensure boolean columns are proper booleans
                if 'KES_Equivalent_Matched' in final_display_df.columns:
                    final_display_df['KES_Equivalent_Matched'] = final_display_df['KES_Equivalent_Matched'].astype(bool)
                if 'Other_Currency_Matched' in final_display_df.columns:
                    final_display_df['Other_Currency_Matched'] = final_display_df['Other_Currency_Matched'].astype(bool)
                
                # Ensure the dataframe has all required columns in the correct order
                final_display_df = ensure_columns_and_order(final_display_df, TARGET_COLUMNS)
            
            render_business_editable_dataframe(
                final_display_df, 
                "Final Business with Match Status", 
                "final_business", 
                on_data_change=update_final_business, 
                show_delete=True, 
                show_move=True, 
                move_targets=move_targets_final_business
            )
        
        with tab2:
            def update_matched_buy(df):
                st.session_state.matched_buy_df = add_business_unique_ids(df) if not df.empty else df
                if not st.session_state.matched_buy_df.empty:
                    st.session_state.matched_buy_df = add_business_audit_columns(st.session_state.matched_buy_df)
                update_business_moved_stats()
                update_business_deleted_stats()
            render_business_editable_dataframe(matched_buy_df, "Matched Buy Records", "matched_buy", 
                                              on_data_change=update_matched_buy, show_delete=True, 
                                              show_move=True, move_targets=move_targets_matched_buy)
        
        with tab3:
            def update_unmatched_buy(df):
                st.session_state.unmatched_buy_df = add_business_unique_ids(df) if not df.empty else df
                if not st.session_state.unmatched_buy_df.empty:
                    st.session_state.unmatched_buy_df = add_business_audit_columns(st.session_state.unmatched_buy_df)
                update_business_moved_stats()
                update_business_deleted_stats()
            render_business_editable_dataframe(unmatched_buy_df, "Unmatched Buy Records", "unmatched_buy", 
                                              on_data_change=update_unmatched_buy, show_delete=True, 
                                              show_move=True, move_targets=move_targets_unmatched_buy)
        
        with tab4:
            def update_matched_sell(df):
                st.session_state.matched_sell_df = add_business_unique_ids(df) if not df.empty else df
                if not st.session_state.matched_sell_df.empty:
                    st.session_state.matched_sell_df = add_business_audit_columns(st.session_state.matched_sell_df)
                update_business_moved_stats()
                update_business_deleted_stats()
            render_business_editable_dataframe(matched_sell_df, "Matched Sell Records", "matched_sell", 
                                              on_data_change=update_matched_sell, show_delete=True, 
                                              show_move=True, move_targets=move_targets_matched_sell)
        
        with tab5:
            def update_unmatched_sell(df):
                st.session_state.unmatched_sell_df = add_business_unique_ids(df) if not df.empty else df
                if not st.session_state.unmatched_sell_df.empty:
                    st.session_state.unmatched_sell_df = add_business_audit_columns(st.session_state.unmatched_sell_df)
                update_business_moved_stats()
                update_business_deleted_stats()
            render_business_editable_dataframe(unmatched_sell_df, "Unmatched Sell Records", "unmatched_sell", 
                                              on_data_change=update_unmatched_sell, show_delete=True, 
                                              show_move=True, move_targets=move_targets_unmatched_sell)
        
        with tab6:
            def update_unmatched_business(df):
                st.session_state.unmatched_business_df = add_business_unique_ids(df) if not df.empty else df
                if not st.session_state.unmatched_business_df.empty:
                    st.session_state.unmatched_business_df = add_business_audit_columns(st.session_state.unmatched_business_df)
                update_business_moved_stats()
                update_business_deleted_stats()
            render_business_editable_dataframe(unmatched_business_df, "Unmatched Business Records", "unmatched_business", 
                                              on_data_change=update_unmatched_business, show_delete=True, 
                                              show_move=True, move_targets=move_targets_unmatched_business)
        
        with tab7:
            audit_tab1, audit_tab2 = st.tabs(["📋 Moved Records", "🗑️ Deleted Records"])
            with audit_tab1:
                render_business_moved_records_tab()
            with audit_tab2:
                render_business_deleted_records_tab()
        
        # Build summary from final business for interactive mode
        if not final_business_df.empty:
            summary_df = build_summary_from_final(final_business_df)
            st.subheader("SUMMARY")
            st.table(summary_df)
            
            # Download options for interactive mode
            st.subheader("Download / Save final report")
            if st.button("Download as Excel (with SUMMARY)"):
                excel_bytes = df_to_excel_bytes(final_business_df, summary_df=summary_df)
                st.download_button("Download Final business Excel", data=excel_bytes, 
                                    file_name="Final-business-with-match-status.xlsx", 
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            if st.button("Download as CSV (main table only)"):
                st.download_button("Download CSV", df_to_csv_bytes(final_business_df), 
                                    "Final-business-with-match-status.csv", "text/csv")
        
        # Additional download buttons for sub-reports
        def df_to_csv(df): 
            return df.to_csv(index=False).encode("utf-8")
        
        st.subheader("Other downloadable results")
        if not matched_buy_df.empty:
            st.download_button("Download Matched Buy", df_to_csv(matched_buy_df), OUT_MATCHED_BUY, "text/csv")
        if not unmatched_buy_df.empty:
            st.download_button("Download Unmatched Buy", df_to_csv(unmatched_buy_df), OUT_UNMATCHED_BUY, "text/csv")
        if not matched_sell_df.empty:
            st.download_button("Download Matched Sell", df_to_csv(matched_sell_df), OUT_MATCHED_SELL, "text/csv")
        if not unmatched_sell_df.empty:
            st.download_button("Download Unmatched Sell", df_to_csv(unmatched_sell_df), OUT_UNMATCHED_SELL, "text/csv")
        if not unmatched_business_df.empty:
            st.download_button("Download Unmatched business", df_to_csv(unmatched_business_df), OUT_UNMATCHED_business, "text/csv")
        if not final_business_df.empty:
            st.download_button("Download Full business with Match Status (CSV)", df_to_csv(final_business_df), 
                                "Final_business_with_Match_Status.csv", "text/csv")
    
    # Auto-save results to original auth system (keep for backward compatibility)
    if 'authenticated' in st.session_state and st.session_state['authenticated']:
        if any([
            not st.session_state.get('matched_buy_df', pd.DataFrame()).empty,
            not st.session_state.get('matched_sell_df', pd.DataFrame()).empty,
            not st.session_state.get('final_business_df', pd.DataFrame()).empty
        ]):
            version_id = get_active_version_id()
            if version_id:
                user_id = st.session_state['user']['user_id']
                current_date = datetime.now().strftime('%Y-%m-%d')
                
                for df_name in ['matched_buy_df', 'matched_sell_df',
                               'unmatched_buy_df', 'unmatched_sell_df',
                               'unmatched_business_df', 'final_business_df']:
                    df = st.session_state.get(df_name, pd.DataFrame())
                
                log_audit(user_id, 'BUSINESS_FX_RESULTS_SAVED', 
                         f'Saved business FX reconciliation results')


# Run the updated app
if __name__ == "__main__":
    business_reconciliation_app(None, None, False)