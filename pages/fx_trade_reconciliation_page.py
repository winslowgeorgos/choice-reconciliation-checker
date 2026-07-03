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
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
DB_PATH = "data/fx_reconciliation.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Custom CSS for better UI
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        border-left: 4px solid #667eea;
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

# --- Database Manager Class ---
class FXReconciliationDB:
    """Database manager for FX reconciliation data"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matched_buy_df (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table TEXT,
                Action_Type TEXT,
                Trade_Amount REAL,
                Trade_Currency TEXT,
                Bank_Statement_Currency TEXT,
                Converted_Trade_Amount REAL,
                Total_Bank_Matches INTEGER,
                Skipped_Bank_Records INTEGER,
                Matched_Bank_Record_Index INTEGER,
                Matched_Bank_Record_Date TEXT,
                Matched_Bank_Description TEXT,
                Matched_Bank_Debit REAL,
                Matched_Bank_Credit REAL,
                All_Matched_Bank_Records TEXT,
                Skipped_Bank_Records_Info TEXT,
                Vendor_ID TEXT,
                Vendor_Name TEXT,
                Counterparty_Dealer TEXT,
                FX_Trade_ID TEXT,
                FX_Reference TEXT,
                FX_Created_At TEXT,
                FX_Amount REAL,
                Source_Column TEXT,
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
            CREATE TABLE IF NOT EXISTS matched_sell_df (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table TEXT,
                Action_Type TEXT,
                Trade_Amount REAL,
                Trade_Currency TEXT,
                Bank_Statement_Currency TEXT,
                Converted_Trade_Amount REAL,
                Total_Bank_Matches INTEGER,
                Skipped_Bank_Records INTEGER,
                Matched_Bank_Record_Index INTEGER,
                Matched_Bank_Record_Date TEXT,
                Matched_Bank_Description TEXT,
                Matched_Bank_Debit REAL,
                Matched_Bank_Credit REAL,
                All_Matched_Bank_Records TEXT,
                Skipped_Bank_Records_Info TEXT,
                Vendor_ID TEXT,
                Vendor_Name TEXT,
                Counterparty_Dealer TEXT,
                FX_Trade_ID TEXT,
                FX_Reference TEXT,
                FX_Created_At TEXT,
                FX_Amount REAL,
                Source_Column TEXT,
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
            CREATE TABLE IF NOT EXISTS unmatched_buy_df (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table_Expected TEXT,
                Action_Type TEXT,
                Amount REAL,
                Status TEXT,
                Vendor_ID TEXT,
                Vendor_Name TEXT,
                Counterparty_Dealer TEXT,
                FX_Trade_ID TEXT,
                FX_Reference TEXT,
                FX_Created_At TEXT,
                FX_Amount REAL,
                Source_Column TEXT,
                Skipped_Bank_Records TEXT,
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
            CREATE TABLE IF NOT EXISTS unmatched_sell_df (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Date TEXT,
                Bank_Table_Expected TEXT,
                Action_Type TEXT,
                Amount REAL,
                Status TEXT,
                Vendor_ID TEXT,
                Vendor_Name TEXT,
                Counterparty_Dealer TEXT,
                FX_Trade_ID TEXT,
                FX_Reference TEXT,
                FX_Created_At TEXT,
                FX_Amount REAL,
                Source_Column TEXT,
                Skipped_Bank_Records TEXT,
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
            CREATE TABLE IF NOT EXISTS unmatched_bank_trade (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Bank_Table TEXT,
                Date TEXT,
                Description TEXT,
                Transaction_Type_Column TEXT,
                Amount REAL,
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
            CREATE TABLE IF NOT EXISTS moved_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deleted_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_moves_log (
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
            CREATE TABLE IF NOT EXISTS audit_deletes_log (
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reconciliation_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_matched_buy_date ON matched_buy_df(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_matched_sell_date ON matched_sell_df(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_unmatched_buy_date ON unmatched_buy_df(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_unmatched_sell_date ON unmatched_sell_df(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_bank_date ON unmatched_bank_trade(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_moved_date ON moved_records(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_deleted_date ON deleted_records(record_date)',
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def _serialize_value(self, value):
        if value is None:
            return None
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        return str(value) if not isinstance(value, (float, int)) else value
    
    def save_matched_buy_df(self, df, record_date=None):
        """Save matched buy df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete all existing records for this date
        cursor.execute("DELETE FROM matched_buy_df WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all matched_buy_df records for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_record_id()))
            cursor.execute('''
                INSERT INTO matched_buy_df (
                    _record_id, record_date, created_at, Date, Bank_Table, Action_Type,
                    Trade_Amount, Trade_Currency, Bank_Statement_Currency, Converted_Trade_Amount,
                    Total_Bank_Matches, Skipped_Bank_Records, Matched_Bank_Record_Index,
                    Matched_Bank_Record_Date, Matched_Bank_Description, Matched_Bank_Debit,
                    Matched_Bank_Credit, All_Matched_Bank_Records, Skipped_Bank_Records_Info,
                    Vendor_ID, Vendor_Name, Counterparty_Dealer, FX_Trade_ID, FX_Reference,
                    FX_Created_At, FX_Amount, Source_Column, deleted_by, deleted_at,
                    delete_reason, moved_by, moved_from, moved_at, move_reason, move_type,
                    moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank Table')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Trade Amount')),
                self._serialize_value(row.get('Trade Currency')),
                self._serialize_value(row.get('Bank Statement Currency')),
                self._serialize_value(row.get('Converted Trade Amount')),
                self._serialize_value(row.get('Total Bank Matches')),
                self._serialize_value(row.get('Skipped Bank Records')),
                self._serialize_value(row.get('Matched Bank Record Index')),
                self._serialize_value(row.get('Matched Bank Record Date')),
                self._serialize_value(row.get('Matched Bank Description')),
                self._serialize_value(row.get('Matched Bank Debit')),
                self._serialize_value(row.get('Matched Bank Credit')),
                self._serialize_value(row.get('All Matched Bank Records')),
                self._serialize_value(row.get('Skipped Bank Records Info')),
                self._serialize_value(row.get('Vendor ID')),
                self._serialize_value(row.get('Vendor Name')),
                self._serialize_value(row.get('Counterparty Dealer')),
                self._serialize_value(row.get('FX Trade ID')),
                self._serialize_value(row.get('FX Reference')),
                self._serialize_value(row.get('FX Created At')),
                self._serialize_value(row.get('FX Amount')),
                self._serialize_value(row.get('Source Column')),
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
        logger.info(f"Saved {len(df)} records to matched_buy_df for date: {record_date}")
    
    def save_matched_sell_df(self, df, record_date=None):
        """Save matched sell df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM matched_sell_df WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all matched_sell_df records for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_record_id()))
            cursor.execute('''
                INSERT INTO matched_sell_df (
                    _record_id, record_date, created_at, Date, Bank_Table, Action_Type,
                    Trade_Amount, Trade_Currency, Bank_Statement_Currency, Converted_Trade_Amount,
                    Total_Bank_Matches, Skipped_Bank_Records, Matched_Bank_Record_Index,
                    Matched_Bank_Record_Date, Matched_Bank_Description, Matched_Bank_Debit,
                    Matched_Bank_Credit, All_Matched_Bank_Records, Skipped_Bank_Records_Info,
                    Vendor_ID, Vendor_Name, Counterparty_Dealer, FX_Trade_ID, FX_Reference,
                    FX_Created_At, FX_Amount, Source_Column, deleted_by, deleted_at,
                    delete_reason, moved_by, moved_from, moved_at, move_reason, move_type,
                    moved_to, import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank Table')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Trade Amount')),
                self._serialize_value(row.get('Trade Currency')),
                self._serialize_value(row.get('Bank Statement Currency')),
                self._serialize_value(row.get('Converted Trade Amount')),
                self._serialize_value(row.get('Total Bank Matches')),
                self._serialize_value(row.get('Skipped Bank Records')),
                self._serialize_value(row.get('Matched Bank Record Index')),
                self._serialize_value(row.get('Matched Bank Record Date')),
                self._serialize_value(row.get('Matched Bank Description')),
                self._serialize_value(row.get('Matched Bank Debit')),
                self._serialize_value(row.get('Matched Bank Credit')),
                self._serialize_value(row.get('All Matched Bank Records')),
                self._serialize_value(row.get('Skipped Bank Records Info')),
                self._serialize_value(row.get('Vendor ID')),
                self._serialize_value(row.get('Vendor Name')),
                self._serialize_value(row.get('Counterparty Dealer')),
                self._serialize_value(row.get('FX Trade ID')),
                self._serialize_value(row.get('FX Reference')),
                self._serialize_value(row.get('FX Created At')),
                self._serialize_value(row.get('FX Amount')),
                self._serialize_value(row.get('Source Column')),
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
        logger.info(f"Saved {len(df)} records to matched_sell_df for date: {record_date}")
    
    def save_unmatched_buy_df(self, df, record_date=None):
        """Save unmatched buy df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM unmatched_buy_df WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_record_id()))
            cursor.execute('''
                INSERT INTO unmatched_buy_df (
                    _record_id, record_date, created_at, Date, Bank_Table_Expected,
                    Action_Type, Amount, Status, Vendor_ID, Vendor_Name, Counterparty_Dealer,
                    FX_Trade_ID, FX_Reference, FX_Created_At, FX_Amount, Source_Column,
                    Skipped_Bank_Records, deleted_by, deleted_at, delete_reason, moved_by,
                    moved_from, moved_at, move_reason, move_type, moved_to,
                    import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank Table (Expected)')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Amount')),
                self._serialize_value(row.get('Status')),
                self._serialize_value(row.get('Vendor ID')),
                self._serialize_value(row.get('Vendor Name')),
                self._serialize_value(row.get('Counterparty Dealer')),
                self._serialize_value(row.get('FX Trade ID')),
                self._serialize_value(row.get('FX Reference')),
                self._serialize_value(row.get('FX Created At')),
                self._serialize_value(row.get('FX Amount')),
                self._serialize_value(row.get('Source Column')),
                self._serialize_value(row.get('Skipped Bank Records')),
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
        logger.info(f"Saved {len(df)} records to unmatched_buy_df for date: {record_date}")
    
    def save_unmatched_sell_df(self, df, record_date=None):
        """Save unmatched sell df - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM unmatched_sell_df WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_record_id()))
            cursor.execute('''
                INSERT INTO unmatched_sell_df (
                    _record_id, record_date, created_at, Date, Bank_Table_Expected,
                    Action_Type, Amount, Status, Vendor_ID, Vendor_Name, Counterparty_Dealer,
                    FX_Trade_ID, FX_Reference, FX_Created_At, FX_Amount, Source_Column,
                    Skipped_Bank_Records, deleted_by, deleted_at, delete_reason, moved_by,
                    moved_from, moved_at, move_reason, move_type, moved_to,
                    import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Bank Table (Expected)')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Amount')),
                self._serialize_value(row.get('Status')),
                self._serialize_value(row.get('Vendor ID')),
                self._serialize_value(row.get('Vendor Name')),
                self._serialize_value(row.get('Counterparty Dealer')),
                self._serialize_value(row.get('FX Trade ID')),
                self._serialize_value(row.get('FX Reference')),
                self._serialize_value(row.get('FX Created At')),
                self._serialize_value(row.get('FX Amount')),
                self._serialize_value(row.get('Source Column')),
                self._serialize_value(row.get('Skipped Bank Records')),
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
        logger.info(f"Saved {len(df)} records to unmatched_sell_df for date: {record_date}")
    
    def save_unmatched_bank_trade(self, df, record_date=None):
        """Save unmatched bank trade - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM unmatched_bank_trade WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_record_id()))
            cursor.execute('''
                INSERT INTO unmatched_bank_trade (
                    _record_id, record_date, created_at, Bank_Table, Date, Description,
                    Transaction_Type_Column, Amount, deleted_by, deleted_at, delete_reason,
                    moved_by, moved_from, moved_at, move_reason, move_type, moved_to,
                    import_date, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Bank Table')),
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Description')),
                self._serialize_value(row.get('Transaction Type (Column)')),
                self._serialize_value(row.get('Amount')),
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
        logger.info(f"Saved {len(df)} records to unmatched_bank_trade for date: {record_date}")
    
    def save_moved_records(self, df, record_date=None):
        """Save moved records - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM moved_records WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all moved_records for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_record_id()))
            record_dict = row.to_dict()
            original_record_json = json.dumps(record_dict, default=str)
            source_table = row.get('moved_from', 'unknown')
            
            cursor.execute('''
                INSERT INTO moved_records (
                    _record_id, record_date, created_at, source_table, record_type,
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
        logger.info(f"Saved {len(df)} records to moved_records for date: {record_date}")
    
    def save_deleted_records(self, df, record_date=None):
        """Save deleted records - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM deleted_records WHERE record_date = ?", (record_date,))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            logger.info(f"Cleared all deleted_records for date: {record_date}")
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            _record_id = str(row.get('_record_id', generate_record_id()))
            record_dict = row.to_dict()
            original_record_json = json.dumps(record_dict, default=str)
            source_table = row.get('deleted_from', row.get('source_dataframe', 'unknown'))
            
            cursor.execute('''
                INSERT INTO deleted_records (
                    _record_id, record_date, created_at, source_table, record_type,
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
        logger.info(f"Saved {len(df)} records to deleted_records for date: {record_date}")
    
    def save_audit_moves(self, df, record_date=None):
        """Save audit moves log - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM audit_moves_log WHERE import_date LIKE ?", (f"{record_date}%",))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO audit_moves_log (
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
        logger.info(f"Saved {len(df)} records to audit_moves_log for date: {record_date}")
    
    def save_audit_deletes(self, df, record_date=None):
        """Save audit deletes log - REPLACES all data for the given date"""
        if record_date is None:
            record_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM audit_deletes_log WHERE import_date LIKE ?", (f"{record_date}%",))
        
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO audit_deletes_log (
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
        logger.info(f"Saved {len(df)} records to audit_deletes_log for date: {record_date}")
    
    def save_fx_trade_data_only(self, target_date=None):
        """Save ALL FX Trade Reconciliation data - REPLACES all data for the date"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get current session state data
        matched_buy_df = st.session_state.get('matched_buy_df', pd.DataFrame())
        matched_sell_df = st.session_state.get('matched_sell_df', pd.DataFrame())
        unmatched_buy_df = st.session_state.get('unmatched_buy_df', pd.DataFrame())
        unmatched_sell_df = st.session_state.get('unmatched_sell_df', pd.DataFrame())
        unmatched_bank_trade = st.session_state.get('unmatched_bank_trade', pd.DataFrame())
        
        # Save each dataframe (each will DELETE old data first)
        self.save_matched_buy_df(matched_buy_df, target_date)
        self.save_matched_sell_df(matched_sell_df, target_date)
        self.save_unmatched_buy_df(unmatched_buy_df, target_date)
        self.save_unmatched_sell_df(unmatched_sell_df, target_date)
        self.save_unmatched_bank_trade(unmatched_bank_trade, target_date)
        
        # Save moved records - combine all moved dataframes
        all_moved_records = []
        moved_keys = ['moved_buy_matched', 'moved_buy_unmatched', 'moved_sell_matched', 
                      'moved_sell_unmatched', 'moved_bank_records_trade']
        
        for key in moved_keys:
            df = st.session_state.get(key, pd.DataFrame())
            if not df.empty:
                all_moved_records.append(df)
        
        if all_moved_records:
            combined_moved = pd.concat(all_moved_records, ignore_index=True)
            self.save_moved_records(combined_moved, target_date)
        else:
            # Clear moved records for this date
            self.save_moved_records(pd.DataFrame(), target_date)
        
        # Save deleted records - combine all deleted dataframes
        all_deleted_records = []
        deleted_keys = ['deleted_buy_matched', 'deleted_buy_unmatched', 'deleted_sell_matched',
                        'deleted_sell_unmatched', 'deleted_bank_trade']
        
        for key in deleted_keys:
            df = st.session_state.get(key, pd.DataFrame())
            if not df.empty:
                all_deleted_records.append(df)
        
        if all_deleted_records:
            combined_deleted = pd.concat(all_deleted_records, ignore_index=True)
            self.save_deleted_records(combined_deleted, target_date)
        else:
            # Clear deleted records for this date
            self.save_deleted_records(pd.DataFrame(), target_date)
        
        # Save audit logs
        audit_moves = st.session_state.get('audit_moves_log_trade', pd.DataFrame())
        audit_deletes = st.session_state.get('audit_deletes_log_trade', pd.DataFrame())
        
        self.save_audit_moves(audit_moves, target_date)
        self.save_audit_deletes(audit_deletes, target_date)
        
        # Save metadata
        self.save_metadata('fx_trade_last_save_date', target_date)
        self.save_metadata('fx_trade_tracker_col_mapping', st.session_state.get('fx_trade_tracker_col_mapping', {}))
        self.save_metadata('fx_trade_moved_stats', st.session_state.get('moved_stats_trade', {}))
        self.save_metadata('fx_trade_deleted_stats', st.session_state.get('deleted_stats_trade', {}))
        
        # Save summary of what was saved
        save_summary = {
            'matched_buy_count': len(matched_buy_df),
            'matched_sell_count': len(matched_sell_df),
            'unmatched_buy_count': len(unmatched_buy_df),
            'unmatched_sell_count': len(unmatched_sell_df),
            'unmatched_bank_count': len(unmatched_bank_trade),
            'moved_count': len(combined_moved) if all_moved_records else 0,
            'deleted_count': len(combined_deleted) if all_deleted_records else 0,
            'audit_moves_count': len(audit_moves),
            'audit_deletes_count': len(audit_deletes)
        }
        self.save_metadata('fx_trade_save_summary', save_summary)
        
        st.session_state.fx_trade_last_save_date = target_date
        
        # Show summary of saved data
        with st.container():
            st.markdown('<div class="custom-success">', unsafe_allow_html=True)
            st.success(f"✅ FX Trade Reconciliation data saved for date: {target_date}")
            
            summary = []
            if not matched_buy_df.empty:
                summary.append(f"• matched_buy_df: {len(matched_buy_df)} records")
            if not matched_sell_df.empty:
                summary.append(f"• matched_sell_df: {len(matched_sell_df)} records")
            if not unmatched_buy_df.empty:
                summary.append(f"• unmatched_buy_df: {len(unmatched_buy_df)} records")
            if not unmatched_sell_df.empty:
                summary.append(f"• unmatched_sell_df: {len(unmatched_sell_df)} records")
            if not unmatched_bank_trade.empty:
                summary.append(f"• unmatched_bank_trade: {len(unmatched_bank_trade)} records")
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
    
    def load_fx_trade_data_only(self, target_date=None):
        """Load ALL FX Trade Reconciliation data from database including audit trails"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        
        # Load main dataframes directly from their specific tables
        st.session_state.matched_buy_df = self.load_matched_buy_df(target_date)
        st.session_state.matched_sell_df = self.load_matched_sell_df(target_date)
        st.session_state.unmatched_buy_df = self.load_unmatched_buy_df(target_date)
        st.session_state.unmatched_sell_df = self.load_unmatched_sell_df(target_date)
        st.session_state.unmatched_bank_trade = self.load_unmatched_bank_trade(target_date)
        
        # Load all moved records from moved_records table
        query = "SELECT * FROM moved_records WHERE record_date = ?"
        all_moved = pd.read_sql_query(query, conn, params=(target_date,))
        
        if not all_moved.empty:
            # Split moved records by their moved_to category
            st.session_state.moved_buy_matched = all_moved[all_moved['moved_to'].str.contains('Buy Matched', na=False)].copy()
            st.session_state.moved_buy_unmatched = all_moved[all_moved['moved_to'].str.contains('Buy Unmatched', na=False)].copy()
            st.session_state.moved_sell_matched = all_moved[all_moved['moved_to'].str.contains('Sell Matched', na=False)].copy()
            st.session_state.moved_sell_unmatched = all_moved[all_moved['moved_to'].str.contains('Sell Unmatched', na=False)].copy()
            st.session_state.moved_bank_records_trade = all_moved[all_moved['moved_to'].str.contains('Bank', na=False)].copy()
        else:
            st.session_state.moved_buy_matched = pd.DataFrame()
            st.session_state.moved_buy_unmatched = pd.DataFrame()
            st.session_state.moved_sell_matched = pd.DataFrame()
            st.session_state.moved_sell_unmatched = pd.DataFrame()
            st.session_state.moved_bank_records_trade = pd.DataFrame()
        
        # Load all deleted records from deleted_records table
        query = "SELECT * FROM deleted_records WHERE record_date = ?"
        all_deleted = pd.read_sql_query(query, conn, params=(target_date,))
        
        if not all_deleted.empty:
            # Split deleted records by their deleted_from category
            st.session_state.deleted_buy_matched = all_deleted[all_deleted['deleted_from'].str.contains('Buy Matched', na=False)].copy()
            st.session_state.deleted_buy_unmatched = all_deleted[all_deleted['deleted_from'].str.contains('Buy Unmatched', na=False)].copy()
            st.session_state.deleted_sell_matched = all_deleted[all_deleted['deleted_from'].str.contains('Sell Matched', na=False)].copy()
            st.session_state.deleted_sell_unmatched = all_deleted[all_deleted['deleted_from'].str.contains('Sell Unmatched', na=False)].copy()
            st.session_state.deleted_bank_trade = all_deleted[all_deleted['deleted_from'].str.contains('Bank', na=False)].copy()
        else:
            st.session_state.deleted_buy_matched = pd.DataFrame()
            st.session_state.deleted_buy_unmatched = pd.DataFrame()
            st.session_state.deleted_sell_matched = pd.DataFrame()
            st.session_state.deleted_sell_unmatched = pd.DataFrame()
            st.session_state.deleted_bank_trade = pd.DataFrame()
        
        # Load audit logs
        query = "SELECT * FROM audit_moves_log WHERE import_date LIKE ?"
        audit_moves = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.audit_moves_log_trade = audit_moves if not audit_moves.empty else pd.DataFrame()
        
        query = "SELECT * FROM audit_deletes_log WHERE import_date LIKE ?"
        audit_deletes = pd.read_sql_query(query, conn, params=(f"{target_date}%",))
        st.session_state.audit_deletes_log_trade = audit_deletes if not audit_deletes.empty else pd.DataFrame()
        
        conn.close()
        
        # Add unique IDs and audit columns to main dataframes if missing
        for df_name in ['matched_buy_df', 'matched_sell_df', 'unmatched_buy_df', 
                        'unmatched_sell_df', 'unmatched_bank_trade']:
            if not st.session_state[df_name].empty:
                if '_record_id' not in st.session_state[df_name].columns:
                    st.session_state[df_name] = add_unique_ids(st.session_state[df_name])
                st.session_state[df_name] = add_audit_columns(st.session_state[df_name])
        
        # Recalculate stats from loaded data
        update_moved_stats_cards_trade()
        update_deleted_stats_cards_trade()
        
        st.session_state.fx_trade_current_date = target_date
        
        # Load column mapping
        col_mapping = self.load_metadata('fx_trade_tracker_col_mapping', {})
        st.session_state.fx_trade_tracker_col_mapping = col_mapping
        
        # Get saved summary for verification
        save_summary = self.load_metadata('fx_trade_save_summary', {})
        
        with st.container():
            st.markdown('<div class="custom-success">', unsafe_allow_html=True)
            st.success(f"✅ FX Trade Reconciliation data loaded for date: {target_date}")
            
            # Show summary of loaded data
            summary = []
            if not st.session_state.matched_buy_df.empty:
                count = len(st.session_state.matched_buy_df)
                saved_count = save_summary.get('matched_buy_count', '?')
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
            if not st.session_state.unmatched_bank_trade.empty:
                count = len(st.session_state.unmatched_bank_trade)
                summary.append(f"• unmatched_bank_trade: {count} records")
            if not st.session_state.moved_buy_matched.empty:
                summary.append(f"• moved_buy_matched: {len(st.session_state.moved_buy_matched)} records")
            if not st.session_state.moved_buy_unmatched.empty:
                summary.append(f"• moved_buy_unmatched: {len(st.session_state.moved_buy_unmatched)} records")
            if not st.session_state.moved_sell_matched.empty:
                summary.append(f"• moved_sell_matched: {len(st.session_state.moved_sell_matched)} records")
            if not st.session_state.moved_sell_unmatched.empty:
                summary.append(f"• moved_sell_unmatched: {len(st.session_state.moved_sell_unmatched)} records")
            if not st.session_state.moved_bank_records_trade.empty:
                summary.append(f"• moved_bank_records: {len(st.session_state.moved_bank_records_trade)} records")
            
            if summary:
                st.info("Loaded data:\n" + "\n".join(summary))
            st.markdown('</div>', unsafe_allow_html=True)
        
        return target_date
    
    def load_matched_buy_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM matched_buy_df WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        column_mapping = {
            'Bank_Table': 'Bank Table', 'Action_Type': 'Action Type',
            'Trade_Amount': 'Trade Amount', 'Trade_Currency': 'Trade Currency',
            'Bank_Statement_Currency': 'Bank Statement Currency',
            'Converted_Trade_Amount': 'Converted Trade Amount',
            'Total_Bank_Matches': 'Total Bank Matches',
            'Skipped_Bank_Records': 'Skipped Bank Records',
            'Matched_Bank_Record_Index': 'Matched Bank Record Index',
            'Matched_Bank_Record_Date': 'Matched Bank Record Date',
            'Matched_Bank_Description': 'Matched Bank Description',
            'Matched_Bank_Debit': 'Matched Bank Debit',
            'Matched_Bank_Credit': 'Matched Bank Credit',
            'All_Matched_Bank_Records': 'All Matched Bank Records',
            'Skipped_Bank_Records_Info': 'Skipped Bank Records Info',
            'Vendor_ID': 'Vendor ID', 'Vendor_Name': 'Vendor Name',
            'Counterparty_Dealer': 'Counterparty Dealer',
            'FX_Trade_ID': 'FX Trade ID', 'FX_Reference': 'FX Reference',
            'FX_Created_At': 'FX Created At', 'FX_Amount': 'FX Amount',
            'Source_Column': 'Source Column',
        }
        df = df.rename(columns=column_mapping)
        cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_matched_sell_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM matched_sell_df WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        column_mapping = {
            'Bank_Table': 'Bank Table', 'Action_Type': 'Action Type',
            'Trade_Amount': 'Trade Amount', 'Trade_Currency': 'Trade Currency',
            'Bank_Statement_Currency': 'Bank Statement Currency',
            'Converted_Trade_Amount': 'Converted Trade Amount',
            'Total_Bank_Matches': 'Total Bank Matches',
            'Skipped_Bank_Records': 'Skipped Bank Records',
            'Matched_Bank_Record_Index': 'Matched Bank Record Index',
            'Matched_Bank_Record_Date': 'Matched Bank Record Date',
            'Matched_Bank_Description': 'Matched Bank Description',
            'Matched_Bank_Debit': 'Matched Bank Debit',
            'Matched_Bank_Credit': 'Matched Bank Credit',
            'All_Matched_Bank_Records': 'All Matched Bank Records',
            'Skipped_Bank_Records_Info': 'Skipped Bank Records Info',
            'Vendor_ID': 'Vendor ID', 'Vendor_Name': 'Vendor Name',
            'Counterparty_Dealer': 'Counterparty Dealer',
            'FX_Trade_ID': 'FX Trade ID', 'FX_Reference': 'FX Reference',
            'FX_Created_At': 'FX Created At', 'FX_Amount': 'FX Amount',
            'Source_Column': 'Source Column',
        }
        df = df.rename(columns=column_mapping)
        cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_unmatched_buy_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM unmatched_buy_df WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        column_mapping = {
            'Bank_Table_Expected': 'Bank Table (Expected)',
            'Action_Type': 'Action Type', 'Vendor_ID': 'Vendor ID',
            'Vendor_Name': 'Vendor Name', 'Counterparty_Dealer': 'Counterparty Dealer',
            'FX_Trade_ID': 'FX Trade ID', 'FX_Reference': 'FX Reference',
            'FX_Created_At': 'FX Created At', 'FX_Amount': 'FX Amount',
            'Source_Column': 'Source Column', 'Skipped_Bank_Records': 'Skipped Bank Records',
        }
        df = df.rename(columns=column_mapping)
        cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_unmatched_sell_df(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM unmatched_sell_df WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        column_mapping = {
            'Bank_Table_Expected': 'Bank Table (Expected)',
            'Action_Type': 'Action Type', 'Vendor_ID': 'Vendor ID',
            'Vendor_Name': 'Vendor Name', 'Counterparty_Dealer': 'Counterparty Dealer',
            'FX_Trade_ID': 'FX Trade ID', 'FX_Reference': 'FX Reference',
            'FX_Created_At': 'FX Created At', 'FX_Amount': 'FX Amount',
            'Source_Column': 'Source Column', 'Skipped_Bank_Records': 'Skipped Bank Records',
        }
        df = df.rename(columns=column_mapping)
        cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified', 'record_date']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def load_unmatched_bank_trade(self, target_date=None):
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        query = 'SELECT * FROM unmatched_bank_trade WHERE record_date = ? ORDER BY id'
        df = pd.read_sql_query(query, conn, params=(target_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        column_mapping = {
            'Bank_Table': 'Bank Table',
            'Transaction_Type_Column': 'Transaction Type (Column)',
        }
        df = df.rename(columns=column_mapping)
        cols_to_drop = ['id', 'created_at', 'import_date', 'last_modified', 'record_date',
                       'deleted_by', 'deleted_at', 'delete_reason', 'moved_by',
                       'moved_from', 'moved_at', 'move_reason', 'move_type', 'moved_to']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    
    def get_available_dates(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT DISTINCT record_date FROM (
                SELECT record_date FROM matched_buy_df
                UNION SELECT record_date FROM matched_sell_df
                UNION SELECT record_date FROM unmatched_buy_df
                UNION SELECT record_date FROM unmatched_sell_df
                UNION SELECT record_date FROM unmatched_bank_trade
            ) WHERE record_date IS NOT NULL ORDER BY record_date DESC
        ''')
        dates = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return dates
    
    def save_metadata(self, key, value):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO reconciliation_metadata (key, value, updated_at) VALUES (?, ?, ?)',
                    (key, json.dumps(value), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
    
    def load_metadata(self, key, default=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT value FROM reconciliation_metadata WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return json.loads(result[0]) if result else default



# Initialize database
db = FXReconciliationDB()

def get_available_fx_trade_dates():
    """Get all available dates with FX Trade data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('''
        SELECT DISTINCT record_date FROM (
            SELECT record_date FROM matched_buy_df
            UNION SELECT record_date FROM matched_sell_df
            UNION SELECT record_date FROM unmatched_buy_df
            UNION SELECT record_date FROM unmatched_sell_df
            UNION SELECT record_date FROM unmatched_bank_trade
        ) WHERE record_date IS NOT NULL ORDER BY record_date DESC
    ''')
    dates = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return dates

# --- Helper Functions for Record Management ---
def generate_record_id():
    return str(uuid.uuid4())

def add_unique_ids(df):
    """Add unique record IDs to dataframe"""
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '_record_id' not in df_copy.columns:
        df_copy['_record_id'] = [generate_record_id() for _ in range(len(df_copy))]
    return df_copy

def ensure_record_ids(df):
    """Ensure dataframe has _record_id column"""
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
    return f"deleted_{source_clean}"

def get_moved_df_name(source_name, target_name):
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
    return f"moved_{target_clean}"

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
    
    db.save_deleted_records(st.session_state[deleted_df_name])
    
    if 'audit_deletes_log_trade' not in st.session_state:
        st.session_state.audit_deletes_log_trade = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.audit_deletes_log_trade
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log_entries = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log_entries.empty:
            st.session_state.audit_deletes_log_trade = pd.concat([existing_log, new_log_entries[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)
    
    db.save_audit_deletes(st.session_state.audit_deletes_log_trade)
    
    remaining_source_with_numbers = add_row_numbers(remaining_source)
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_row_numbers(remaining_source.copy())
    
    # Update the main dataframe
    main_df_mapping = {
        'Buy Matched Records': 'matched_buy_df',
        'Buy Unmatched Records': 'unmatched_buy_df',
        'Sell Matched Records': 'matched_sell_df',
        'Sell Unmatched Records': 'unmatched_sell_df',
        'Unmatched Bank Records': 'unmatched_bank_trade'
    }
    if source_name in main_df_mapping:
        main_key = main_df_mapping[source_name]
        if main_key in st.session_state:
            st.session_state[main_key] = remove_row_numbers(remaining_source.copy())
    
    if on_data_change:
        on_data_change(remaining_source.copy())
    
    # Force update all stats
    update_deleted_stats_cards_trade()
    update_moved_stats_cards_trade()
    
    return remaining_source_with_numbers, len(selected_record_ids)

def clear_selection_state(key_prefix):
    selection_key = f"{key_prefix}_selection_state"
    if selection_key in st.session_state:
        st.session_state[selection_key] = {}

def update_moved_stats_cards_trade():
    moved_counts = {
        'moved_buy_matched': 0, 'moved_buy_unmatched': 0,
        'moved_sell_matched': 0, 'moved_sell_unmatched': 0,
        'moved_bank_records_trade': 0, 'total_moved': 0
    }
    for key in moved_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            moved_counts[key] = len(st.session_state[key])
    moved_counts['total_moved'] = sum([moved_counts['moved_buy_matched'], moved_counts['moved_buy_unmatched'],
                                       moved_counts['moved_sell_matched'], moved_counts['moved_sell_unmatched'],
                                       moved_counts['moved_bank_records_trade']])
    st.session_state.moved_stats_trade = moved_counts
    return moved_counts

def update_deleted_stats_cards_trade():
    deleted_counts = {
        'deleted_buy_matched': 0, 'deleted_buy_unmatched': 0,
        'deleted_sell_matched': 0, 'deleted_sell_unmatched': 0,
        'deleted_bank_trade': 0, 'total_deleted': 0
    }
    for key in deleted_counts.keys():
        if key in st.session_state and not st.session_state[key].empty:
            deleted_counts[key] = len(st.session_state[key])
    deleted_counts['total_deleted'] = sum([deleted_counts['deleted_buy_matched'], deleted_counts['deleted_buy_unmatched'],
                                           deleted_counts['deleted_sell_matched'], deleted_counts['deleted_sell_unmatched'],
                                           deleted_counts['deleted_bank_trade']])
    st.session_state.deleted_stats_trade = deleted_counts
    return deleted_counts

def sync_all_display_dataframes_trade():
    for key in list(st.session_state.keys()):
        if key.endswith('_display_df'):
            base_key = key.replace('_display_df', '')
            if base_key in st.session_state and not st.session_state[base_key].empty:
                st.session_state[key] = add_row_numbers(st.session_state[base_key].copy())

def refresh_analytics_dataframes_trade():
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

# --- File Operations ---
def save_uploaded_file(file, filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

# --- Data Processing Functions ---
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
    '%d/%m/%Y', '%-d/%-m/%Y', '%-d.%-m/%-Y',
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
    '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S', '%-d/%-m/%Y %H:%M:%S',
    '%-d.%-m.%Y %H:%M:%S', "%d.%m.%Y"
]

FUZZY_MATCH_THRESHOLD = 70

FX_RATES = {
    'USDKES': 145.0, 'EURKES': 155.0, 'GBPUSD': 1.25,
    'USDGBP': 0.8, 'EURUSD': 1.08, 'USDEUR': 0.92,
    'KESUSD': 1/145.0, 'KESEUR': 1/155.0
}

def get_fx_rate(from_currency, to_currency, date=None):
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency == to_currency:
        return 1.0
    pair = f"{from_currency}{to_currency}"
    if pair in FX_RATES:
        return FX_RATES[pair]
    inverse_pair = f"{to_currency}{from_currency}"
    if inverse_pair in FX_RATES:
        return 1 / FX_RATES[inverse_pair]
    return 1.0

def convert_currency(amount, from_currency, to_currency, date=None):
    rate = get_fx_rate(from_currency, to_currency, date)
    return amount * rate if rate else amount

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
    if match and match[1] >= FUZZY_MATCH_THRESHOLD:
        for long, short in replacements.items():
            if match[0].lower() == long.lower():
                return short
            if match[0].lower().startswith(long.lower()):
                return short
        return match[0].title() if match[0].islower() else match[0]
    return str(raw_key).strip().title()

def resolve_amount_column(columns, action_type, bank_statement_currency):
    bank_statement_currency = bank_statement_currency.upper()
    if bank_statement_currency == 'KES':
        if action_type == 'Bank Buy' and 'Debit' in columns:
            return 'Debit'
        elif action_type == 'Bank Sell' and 'Credit' in columns:
            return 'Credit'
    else:
        if action_type == 'Bank Sell' and 'Debit' in columns:
            return 'Debit'
        elif action_type == 'Bank Buy' and 'Credit' in columns:
            return 'Credit'
    columns_lower = [col.lower() for col in columns]
    if 'debit' in columns_lower:
        return 'Debit'
    if 'credit' in columns_lower:
        return 'Credit'
    return None

def get_description_columns(columns):
    for desc in ['Description', 'Narrative', 'Transaction Details', 'Customer reference', 
                 'Transaction Remarks:', 'Transaction Details', 'Transaction\nDetails']:
        if desc in columns:
            return desc
    return None

def process_fx_match(fx_row, all_bank_dfs, unmatched_list, matched_list, action_type,
                     fx_amount_field, bank_currency_info_field, date_tolerance_days=3,
                     debug_mode=False, already_matched_fx_trades=None,
                     skipped_bank_records=None, matched_bank_keys=None):
    if already_matched_fx_trades is None:
        already_matched_fx_trades = set()
    if skipped_bank_records is None:
        skipped_bank_records = {}
    if matched_bank_keys is None:
        matched_bank_keys = set()
    
    fx_trade_id = fx_row.get('Trade ID', '')
    if not fx_trade_id:
        fx_trade_id = f"{fx_row.get('Created At', '')}_{fx_row.get(fx_amount_field, '')}_{fx_row.get(bank_currency_info_field, '')}"
    
    if fx_trade_id in already_matched_fx_trades:
        return None
    
    amount = safe_float(fx_row.get(fx_amount_field))
    if amount is None or action_type not in ['Bank Buy', 'Bank Sell']:
        return None
    
    parsed_date = fx_row.get('Created At')
    if parsed_date and not isinstance(parsed_date, datetime):
        parsed_date = parse_date(str(parsed_date))
    if not isinstance(parsed_date, datetime):
        return None
    
    fx_details = {
        'Vendor ID': fx_row.get('Vendor ID'), 'Vendor Name': fx_row.get('Vendor Name'),
        'Counterparty Dealer': fx_row.get('Counterparty Dealer'), 'FX Trade ID': fx_trade_id,
        'FX Reference': fx_row.get('Reference'),
        'FX Created At': parsed_date.strftime('%Y-%m-%d') if parsed_date else None,
        'FX Amount': amount, 'Source Column': bank_currency_info_field, 'Action Type': action_type
    }
    
    counterparty_raw = str(fx_row.get(bank_currency_info_field, '')).strip()
    parts = counterparty_raw.split('-')
    if len(parts) < 2:
        unmatched_list.append({'Date': parsed_date.strftime('%Y-%m-%d'), 'Bank Table (Expected)': f"N/A ({counterparty_raw})",
                              'Action Type': action_type, 'Amount': amount, 'Status': 'Invalid Bank/Currency Info in FX Trade', **fx_details})
        return None
    
    trade_bank_name_raw = parts[0].strip()
    trade_currency = parts[1].strip().upper()
    normalized_trade_bank_name = normalize_bank_key(trade_bank_name_raw, debug_mode)
    expected_bank_key = f"{normalized_trade_bank_name} {trade_currency}"
    
    if expected_bank_key not in all_bank_dfs:
        unmatched_list.append({'Date': parsed_date.strftime('%Y-%m-%d'), 'Bank Table (Expected)': expected_bank_key,
                              'Action Type': action_type, 'Amount': amount, 'Status': 'No Matching Bank Statement File Found', **fx_details})
        return None
    
    bank_df = all_bank_dfs[expected_bank_key]
    bank_df_columns = bank_df.columns.tolist()
    bank_currency = expected_bank_key.split(' ')[1].upper() if ' ' in expected_bank_key else "UNKNOWN"
    
    if 'Skipped_By_FX_Trades' not in bank_df.columns:
        bank_df['Skipped_By_FX_Trades'] = ""
    
    date_column = 'Date'
    amount_column = resolve_amount_column(bank_df_columns, action_type, bank_currency)
    if date_column not in bank_df.columns or not amount_column or amount_column not in bank_df.columns:
        unmatched_list.append({'Date': parsed_date.strftime('%Y-%m-%d'), 'Bank Table (Expected)': expected_bank_key,
                              'Action Type': action_type, 'Amount': amount, 'Status': 'Missing Required Columns in Bank Statement', **fx_details})
        return None
    
    date_matches = bank_df[bank_df['Date'].dt.date.between(parsed_date.date() - pd.Timedelta(days=date_tolerance_days),
                                                           parsed_date.date() + pd.Timedelta(days=date_tolerance_days))]
    
    matched_records = []
    skipped_records = []
    
    for idx, bank_row in date_matches.iterrows():
        bank_amt = safe_float(bank_row.get(amount_column))
        if bank_amt is None:
            continue
        converted_amount = convert_currency(amount, trade_currency, bank_currency, parsed_date)
        amount_diff = abs(abs(bank_amt) - abs(converted_amount)) if converted_amount is not None else float('inf')
        if converted_amount and abs(converted_amount) > 0.01 and amount_diff < 0.05:
            bank_record_key_operation = 'debit' if 'debit' in amount_column.lower() or bank_amt < 0 else 'credit'
            if 'credit' in amount_column.lower():
                bank_record_key_operation = 'credit'
            bank_record_key = (expected_bank_key, bank_row[date_column].strftime('%Y-%m-%d') if hasattr(bank_row[date_column], 'strftime') else str(bank_row[date_column]), round(bank_amt, 2), bank_record_key_operation)
            is_already_matched = bank_record_key in matched_bank_keys
            if is_already_matched:
                current_skipped = bank_df.loc[idx, "Skipped_By_FX_Trades"]
                skipped_list = []
                if current_skipped and current_skipped != "":
                    try:
                        skipped_list = json.loads(current_skipped)
                    except:
                        skipped_list = []
                skipped_info = {'fx_trade_id': fx_trade_id, 'fx_date': parsed_date.strftime('%Y-%m-%d'), 'fx_amount': amount,
                               'fx_action_type': action_type, 'skipped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               'match_details': {'amount_difference': amount_diff, 'converted_amount': converted_amount,
                                               'bank_amount': bank_amt, 'amount_column': amount_column}}
                skipped_list.append(skipped_info)
                bank_df.loc[idx, "Skipped_By_FX_Trades"] = json.dumps(skipped_list)
                if fx_trade_id not in skipped_bank_records:
                    skipped_bank_records[fx_trade_id] = []
                skipped_records.append({'bank_key': bank_record_key, 'bank_table': expected_bank_key,
                                       'bank_date': bank_row[date_column].strftime('%Y-%m-%d') if hasattr(bank_row[date_column], 'strftime') else str(bank_row[date_column]),
                                       'bank_amount': bank_amt, 'bank_row_index': idx,
                                       'match_details': {'amount_difference': amount_diff, 'converted_amount': converted_amount,
                                                       'bank_amount': bank_amt, 'amount_column': amount_column}})
                continue
            matched_records.append({'Bank Index': idx, 'Bank Date': bank_row.get(date_column).strftime('%Y-%m-%d') if bank_row.get(date_column) else None,
                                   'Description': str(bank_row.get('Description', '')).strip(), 'Debit': safe_float(bank_row.get('Debit')),
                                   'Credit': safe_float(bank_row.get('Credit')), 'Matched Column': amount_column,
                                   'Bank Amount': bank_amt, 'Bank Record Key': bank_record_key,
                                   'Amount Difference': amount_diff, 'Converted Amount': converted_amount})
            bank_df.at[idx, "Matched"] = True
            matched_bank_keys.add(bank_record_key)
    
    if matched_records:
        matched_list.append({'Date': parsed_date.strftime('%Y-%m-%d'), 'Bank Table': expected_bank_key,
                            'Action Type': action_type, 'Trade Amount': amount, 'Trade Currency': trade_currency,
                            'Bank Statement Currency': bank_currency, 'Converted Trade Amount': converted_amount,
                            'Total Bank Matches': len(matched_records), 'Skipped Bank Records': len(skipped_records),
                            'Matched Bank Record Index': matched_records[0]['Bank Index'],
                            'Matched Bank Record Date': matched_records[0]['Bank Date'],
                            'Matched Bank Description': matched_records[0]['Description'],
                            'Matched Bank Debit': matched_records[0]['Debit'], 'Matched Bank Credit': matched_records[0]['Credit'],
                            'All Matched Bank Records': json.dumps(matched_records), 'Skipped Bank Records Info': json.dumps(skipped_records), **fx_details})
        already_matched_fx_trades.add(fx_trade_id)
        return [(expected_bank_key, m['Bank Index']) for m in matched_records]
    
    if skipped_records:
        unmatched_list.append({'Date': parsed_date.strftime('%Y-%m-%d'), 'Bank Table (Expected)': expected_bank_key,
                              'Action Type': action_type, 'Amount': amount, 'Status': 'Potential matches found but already taken',
                              'Skipped Bank Records': json.dumps(skipped_records), **fx_details})
        return None
    
    unmatched_list.append({'Date': parsed_date.strftime('%Y-%m-%d'), 'Bank Table (Expected)': expected_bank_key,
                          'Action Type': action_type, 'Amount': amount, 'Status': 'No Bank Statement Match (Amount or Date Tolerance)', **fx_details})
    return None

# --- Session State Initialization ---
def initialize_session_state_trade():
    """Initialize all FX Trade related session state variables"""
    
    # Main dataframes
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
    
    # Moved records dataframes
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
    
    # Deleted records dataframes
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
    
    # Statistics
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
    
    # Current date tracking
    if 'fx_trade_current_date' not in st.session_state:
        st.session_state.fx_trade_current_date = datetime.now().strftime('%Y-%m-%d')
    if 'fx_trade_last_save_date' not in st.session_state:
        st.session_state.fx_trade_last_save_date = None
    
    # Column mapping
    if 'fx_trade_tracker_col_mapping' not in st.session_state:
        st.session_state.fx_trade_tracker_col_mapping = {}
    
    # Debug mode
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

def safe_get_dataframe(df_name, default=pd.DataFrame()):
    """Safely get a dataframe from session state with default if not exists or empty"""
    if df_name in st.session_state and st.session_state[df_name] is not None:
        return st.session_state[df_name]
    return default

# --- Render Functions ---
def render_editable_dataframe_trade(df, title, key_prefix, on_data_change=None, show_delete=True, show_move=True, move_targets=None):
    """Render editable dataframe with proper display sync to main dataframe"""
    
    # Handle empty dataframe
    if df is None or df.empty:
        st.info(f"No {title} to display.")
        # Clear display dataframe if it exists
        display_df_key = f"{key_prefix}_display_df"
        if display_df_key in st.session_state:
            st.session_state[display_df_key] = pd.DataFrame()
        return df if df is not None else pd.DataFrame()
    
    # Ensure df has required columns
    df = ensure_record_ids(df)
    df = add_audit_columns(df)
    
    st.markdown(f"### {title}")
    st.markdown(f"**Total Records: {len(df)}**")
    
    display_df_key = f"{key_prefix}_display_df"
    original_df_key = key_prefix
    
    # ALWAYS sync display dataframe with main dataframe
    # This ensures display_df always matches the current data
    if '#' not in df.columns:
        display_df = add_row_numbers(df.copy())
    else:
        display_df = df.copy()
    
    # Update session state display dataframe
    st.session_state[display_df_key] = display_df
    st.session_state[original_df_key] = remove_row_numbers(df.copy())
    
    with st.expander("📝 Batch Operations", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            action_reason = st.text_input("Action Reason (optional):", key=f"{key_prefix}_action_reason", placeholder="Enter reason for moving or deleting these records...")
        
        with col2:
            if show_delete and st.button(f"🗑️ Delete Selected", key=f"{key_prefix}_delete_btn", use_container_width=True):
                selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
                selected_record_ids = [rid.replace(f"{key_prefix}_select_", "") for rid in selection_state.keys() if selection_state[rid] and rid.startswith(f"{key_prefix}_select_")]
                if selected_record_ids:
                    source_df = st.session_state[display_df_key].copy()
                    updated_df, deleted_count = delete_selected_rows_with_audit(source_df, selected_record_ids, title, action_reason, df_name=display_df_key, on_data_change=on_data_change)
                    if original_df_key in st.session_state:
                        st.session_state[original_df_key] = remove_row_numbers(updated_df.copy())
                    
                    # Update main dataframe
                    main_df_mapping = {
                        'Buy Matched Records': 'matched_buy_df',
                        'Buy Unmatched Records': 'unmatched_buy_df',
                        'Sell Matched Records': 'matched_sell_df',
                        'Sell Unmatched Records': 'unmatched_sell_df',
                        'Unmatched Bank Records': 'unmatched_bank_trade'
                    }
                    if title in main_df_mapping:
                        main_key = main_df_mapping[title]
                        st.session_state[main_key] = remove_row_numbers(updated_df.copy())
                    
                    sync_all_display_dataframes_trade()
                    clear_selection_state(key_prefix)
                    refresh_analytics_dataframes_trade()
                    update_deleted_stats_cards_trade()
                    st.success(f"✅ Deleted {deleted_count} record(s)")
                    st.rerun()
                else:
                    st.warning("No rows selected for deletion")
        
        with col3:
            if show_move and move_targets:
                selected_target = st.selectbox("Move to:", options=list(move_targets.keys()), key=f"{key_prefix}_selected_target")
                if st.button(f"➡️ Move Selected", key=f"{key_prefix}_move_btn", use_container_width=True):
                    selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
                    selected_record_ids = [rid.replace(f"{key_prefix}_select_", "") for rid in selection_state.keys() if selection_state[rid] and rid.startswith(f"{key_prefix}_select_")]
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
                            
                            # Update source dataframe
                            st.session_state[source_key] = new_source
                            st.session_state[display_df_key] = add_row_numbers(new_source)
                            
                            # Update main dataframe
                            main_df_mapping = {
                                'Buy Matched Records': 'matched_buy_df',
                                'Buy Unmatched Records': 'unmatched_buy_df',
                                'Sell Matched Records': 'matched_sell_df',
                                'Sell Unmatched Records': 'unmatched_sell_df',
                                'Unmatched Bank Records': 'unmatched_bank_trade'
                            }
                            if title in main_df_mapping:
                                main_key = main_df_mapping[title]
                                st.session_state[main_key] = remove_row_numbers(new_source.copy())
                            
                            # Update target main dataframe
                            target_main_mapping = {
                                'Buy Matched': 'matched_buy_df',
                                'Buy Unmatched': 'unmatched_buy_df',
                                'Sell Matched': 'matched_sell_df',
                                'Sell Unmatched': 'unmatched_sell_df'
                            }
                            if selected_target in target_main_mapping:
                                target_key = target_main_mapping[selected_target]
                                target_current = st.session_state.get(target_key, pd.DataFrame()).copy()
                                moved_records_clean = remove_row_numbers(moved_records.copy())
                                st.session_state[target_key] = pd.concat([target_current, moved_records_clean], ignore_index=True)
                            
                            if on_data_change:
                                on_data_change(new_source)
                            
                            clear_selection_state(key_prefix)
                            refresh_analytics_dataframes_trade()
                            update_moved_stats_cards_trade()
                            update_deleted_stats_cards_trade()
                            
                            st.success(f"✅ Moved {len(selected_record_ids)} record(s)")
                            st.rerun()
                    else:
                        st.warning("No rows selected or target not specified")
    
    # Main data editor
    st.markdown("---")
    st.markdown("### 📝 Data Editor")
    st.info("💡 Tip: Double-click any cell to edit its content directly.")
    
    # Get current display dataframe (synced with main)
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
        key=f"{key_prefix}_data_editor", 
        num_rows="dynamic"
    )
    
    if not edited_df.equals(df_for_edit_for_display):
        # Add back the required columns
        edited_with_ids = ensure_record_ids(edited_df.copy())
        edited_with_audit = add_audit_columns(edited_with_ids)
        updated_with_numbers = add_row_numbers(edited_with_audit)
        
        # Update display dataframe
        st.session_state[display_df_key] = updated_with_numbers
        
        # Update main dataframe (without row numbers)
        main_df = remove_row_numbers(edited_with_audit.copy())
        st.session_state[original_df_key] = main_df
        
        # Update the specific main dataframe based on title
        main_df_mapping = {
            'Buy Matched Records': 'matched_buy_df',
            'Buy Unmatched Records': 'unmatched_buy_df',
            'Sell Matched Records': 'matched_sell_df',
            'Sell Unmatched Records': 'unmatched_sell_df',
            'Unmatched Bank Records': 'unmatched_bank_trade'
        }
        if title in main_df_mapping:
            main_key = main_df_mapping[title]
            st.session_state[main_key] = main_df.copy()
        
        if on_data_change:
            on_data_change(main_df)
        
        refresh_analytics_dataframes_trade()
        update_deleted_stats_cards_trade()
        update_moved_stats_cards_trade()
        st.success("✅ Data updated!")
        st.rerun()
    
    # Row selection for batch operations
    st.markdown("---")
    st.markdown("### ☑️ Select Rows for Batch Operations")
    
    selection_key = f"{key_prefix}_selection_state"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}
    
    df_for_selection = st.session_state[display_df_key].copy()
    
    if df_for_selection.empty:
        st.info("No rows available for selection.")
        return df
    
    # Ensure '_record_id' exists
    if '_record_id' not in df_for_selection.columns:
        df_for_selection = ensure_record_ids(df_for_selection)
        st.session_state[display_df_key] = add_row_numbers(df_for_selection)
        st.session_state[original_df_key] = remove_row_numbers(df_for_selection.copy())
    
    record_ids = df_for_selection['_record_id'].tolist() if '_record_id' in df_for_selection.columns else []
    
    if not record_ids:
        st.info("No valid record IDs found.")
        return df
    
    # Display rows with checkboxes
    for idx in range(len(df_for_selection)):
        col1, col2 = st.columns([0.1, 0.9])
        row_num = df_for_selection.iloc[idx]['#'] if '#' in df_for_selection.columns else idx + 1
        record_id = record_ids[idx] if idx < len(record_ids) else str(idx)
        checkbox_key = f"{key_prefix}_select_{record_id}"
        is_selected = st.session_state[selection_key].get(checkbox_key, False)
        
        with col1:
            if st.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed"):
                st.session_state[selection_key][checkbox_key] = True
            else:
                st.session_state[selection_key][checkbox_key] = False
        
        with col2:
            # Create a compact row summary
            row_summary_parts = []
            for col in ['Date', 'Action Type', 'Amount', 'Vendor Name']:
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
                file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                mime="text/csv", 
                key=f"{key_prefix}_download",
                use_container_width=True
            )
    
    result_df = st.session_state[display_df_key].copy()
    for col in ['_record_id', '#']:
        if col in result_df.columns:
            result_df = result_df.drop(columns=[col])
    return result_df



def render_moved_records_tab_trade():
    st.markdown("### 📋 Moved Records - Audit Trail")
    moved_stats = update_moved_stats_cards_trade()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("📋 Buy Matched Moved", moved_stats['moved_buy_matched'])
    with col2:
        st.metric("⚠️ Buy Unmatched Moved", moved_stats['moved_buy_unmatched'])
    with col3:
        st.metric("📋 Sell Matched Moved", moved_stats['moved_sell_matched'])
    with col4:
        st.metric("⚠️ Sell Unmatched Moved", moved_stats['moved_sell_unmatched'])
    with col5:
        st.metric("🏦 Bank Records Moved", moved_stats['moved_bank_records_trade'])
    with col6:
        st.metric("📊 Total Moved", moved_stats['total_moved'])
    
    st.markdown("---")
    
    moved_df_names = ['moved_buy_matched', 'moved_buy_unmatched', 'moved_sell_matched', 'moved_sell_unmatched', 'moved_bank_records_trade']
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
            cols_to_drop = ['_record_id', 'id', 'created_at', 'import_date', 'last_modified', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)

def render_deleted_records_tab_trade():
    st.markdown("### 🗑️ Deleted Records - Audit Trail")
    deleted_stats = update_deleted_stats_cards_trade()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("🗑️ Buy Matched Deleted", deleted_stats['deleted_buy_matched'])
    with col2:
        st.metric("🗑️ Buy Unmatched Deleted", deleted_stats['deleted_buy_unmatched'])
    with col3:
        st.metric("🗑️ Sell Matched Deleted", deleted_stats['deleted_sell_matched'])
    with col4:
        st.metric("🗑️ Sell Unmatched Deleted", deleted_stats['deleted_sell_unmatched'])
    with col5:
        st.metric("🗑️ Bank Records Deleted", deleted_stats['deleted_bank_trade'])
    with col6:
        st.metric("📊 Total Deleted", deleted_stats['total_deleted'])
    
    st.markdown("---")
    
    deleted_df_names = ['deleted_buy_matched', 'deleted_buy_unmatched', 'deleted_sell_matched', 'deleted_sell_unmatched', 'deleted_bank_trade']
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
            cols_to_drop = ['_record_id', 'id', 'created_at', 'import_date', 'last_modified', 'original_record_json']
            display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
            st.dataframe(display_df, use_container_width=True, height=400)

def render_full_statistics_dashboard():
    """Render comprehensive statistics dashboard with safe access"""
    st.markdown("### 📊 Comprehensive Statistics Dashboard")
    
    # Add refresh button
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            update_moved_stats_cards_trade()
            update_deleted_stats_cards_trade()
            st.rerun()
    
    # Get current data with safe defaults
    matched_buy_df = safe_get_dataframe('matched_buy_df')
    matched_sell_df = safe_get_dataframe('matched_sell_df')
    unmatched_buy_df = safe_get_dataframe('unmatched_buy_df')
    unmatched_sell_df = safe_get_dataframe('unmatched_sell_df')
    unmatched_bank_trade = safe_get_dataframe('unmatched_bank_trade')
    
    # Calculate current statistics
    buy_matched_count = len(matched_buy_df) if not matched_buy_df.empty else 0
    buy_unmatched_count = len(unmatched_buy_df) if not unmatched_buy_df.empty else 0
    sell_matched_count = len(matched_sell_df) if not matched_sell_df.empty else 0
    sell_unmatched_count = len(unmatched_sell_df) if not unmatched_sell_df.empty else 0
    bank_unmatched_count = len(unmatched_bank_trade) if not unmatched_bank_trade.empty else 0
    
    total_fx = buy_matched_count + buy_unmatched_count + sell_matched_count + sell_unmatched_count
    total_matched = buy_matched_count + sell_matched_count
    match_rate = (total_matched / total_fx * 100) if total_fx > 0 else 0
    
    # Create metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Buy Matched", buy_matched_count)
    with col2:
        st.metric("⚠️ Buy Unmatched", buy_unmatched_count)
    with col3:
        st.metric("✅ Sell Matched", sell_matched_count)
    with col4:
        st.metric("⚠️ Sell Unmatched", sell_unmatched_count)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏦 Bank Unmatched", bank_unmatched_count)
    with col2:
        st.metric("💰 Total FX Trades", total_fx)
    with col3:
        st.metric("✅ Total Matched", total_matched)
    with col4:
        st.metric("📈 Match Rate", f"{match_rate:.1f}%")
    
    # Create charts for better visualization
    if total_fx > 0:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            match_data = pd.DataFrame({
                'Status': ['Matched', 'Unmatched'],
                'Count': [total_matched, total_fx - total_matched]
            })
            fig = px.pie(match_data, values='Count', names='Status', title='Match Status Distribution', 
                        color_discrete_sequence=['#28a745', '#dc3545'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            trade_data = pd.DataFrame({
                'Type': ['Buy Matched', 'Buy Unmatched', 'Sell Matched', 'Sell Unmatched'],
                'Count': [buy_matched_count, buy_unmatched_count, sell_matched_count, sell_unmatched_count]
            })
            fig = px.bar(trade_data, x='Type', y='Count', title='Trade Distribution by Type', 
                        color='Type', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        # Moved and Deleted Records Summary
        st.markdown("### 📦 Audit Summary")
        moved_stats = update_moved_stats_cards_trade()
        deleted_stats = update_deleted_stats_cards_trade()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if moved_stats and moved_stats.get('total_moved', 0) > 0:
                st.markdown("#### Moved Records")
                moved_df = pd.DataFrame([
                    {'Category': 'Buy Matched', 'Count': moved_stats.get('moved_buy_matched', 0)},
                    {'Category': 'Buy Unmatched', 'Count': moved_stats.get('moved_buy_unmatched', 0)},
                    {'Category': 'Sell Matched', 'Count': moved_stats.get('moved_sell_matched', 0)},
                    {'Category': 'Sell Unmatched', 'Count': moved_stats.get('moved_sell_unmatched', 0)},
                    {'Category': 'Bank Records', 'Count': moved_stats.get('moved_bank_records_trade', 0)}
                ])
                fig = px.bar(moved_df, x='Category', y='Count', title='Moved Records by Category', 
                            color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if deleted_stats and deleted_stats.get('total_deleted', 0) > 0:
                st.markdown("#### Deleted Records")
                deleted_df = pd.DataFrame([
                    {'Category': 'Buy Matched', 'Count': deleted_stats.get('deleted_buy_matched', 0)},
                    {'Category': 'Buy Unmatched', 'Count': deleted_stats.get('deleted_buy_unmatched', 0)},
                    {'Category': 'Sell Matched', 'Count': deleted_stats.get('deleted_sell_matched', 0)},
                    {'Category': 'Sell Unmatched', 'Count': deleted_stats.get('deleted_sell_unmatched', 0)},
                    {'Category': 'Bank Records', 'Count': deleted_stats.get('deleted_bank_trade', 0)}
                ])
                fig = px.bar(deleted_df, x='Category', y='Count', title='Deleted Records by Category', 
                            color='Count', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)

def reset_all_module_dataframes():
    """Reset all FX Trade module dataframes to empty state"""
    with st.spinner("Resetting all dataframes..."):
        # Main dataframes
        st.session_state.matched_buy_df = pd.DataFrame()
        st.session_state.matched_sell_df = pd.DataFrame()
        st.session_state.unmatched_buy_df = pd.DataFrame()
        st.session_state.unmatched_sell_df = pd.DataFrame()
        st.session_state.unmatched_bank_trade = pd.DataFrame()
        st.session_state.fx_trade_df = pd.DataFrame()
        
        # Moved records dataframes
        st.session_state.moved_buy_matched = pd.DataFrame()
        st.session_state.moved_buy_unmatched = pd.DataFrame()
        st.session_state.moved_sell_matched = pd.DataFrame()
        st.session_state.moved_sell_unmatched = pd.DataFrame()
        st.session_state.moved_bank_records_trade = pd.DataFrame()
        
        # Deleted records dataframes
        st.session_state.deleted_buy_matched = pd.DataFrame()
        st.session_state.deleted_buy_unmatched = pd.DataFrame()
        st.session_state.deleted_sell_matched = pd.DataFrame()
        st.session_state.deleted_sell_unmatched = pd.DataFrame()
        st.session_state.deleted_bank_trade = pd.DataFrame()
        
        # Audit logs
        st.session_state.audit_moves_log_trade = pd.DataFrame()
        st.session_state.audit_deletes_log_trade = pd.DataFrame()
        
        # Clear display dataframes
        display_keys = [key for key in st.session_state.keys() if key.endswith('_display_df')]
        for key in display_keys:
            st.session_state[key] = pd.DataFrame()
        
        # Clear selection states
        selection_keys = [key for key in st.session_state.keys() if key.endswith('_selection_state')]
        for key in selection_keys:
            st.session_state[key] = {}
        
        # Reset statistics
        st.session_state.moved_stats_trade = {
            'moved_buy_matched': 0, 'moved_buy_unmatched': 0,
            'moved_sell_matched': 0, 'moved_sell_unmatched': 0,
            'moved_bank_records_trade': 0, 'total_moved': 0
        }
        st.session_state.deleted_stats_trade = {
            'deleted_buy_matched': 0, 'deleted_buy_unmatched': 0,
            'deleted_sell_matched': 0, 'deleted_sell_unmatched': 0,
            'deleted_bank_trade': 0, 'total_deleted': 0
        }
        
        logger.info("All FX Trade module dataframes have been reset")
    
    return True
# --- Main App Function ---
def graphed_analysis_app(all_bank_dfs: dict):
    # Initialize all session state variables
    initialize_session_state_trade()
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header Section

    
    # ========== FX TRADE DATA MANAGEMENT SECTION ==========
    st.markdown("### 📅 Data Management")
    
    available_dates = get_available_fx_trade_dates()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        if available_dates:
            selected_load_date = st.selectbox(
                "📅 Select date to load:",
                options=available_dates,
                index=0,
                key="fx_trade_load_date_select"
            )
        else:
            st.selectbox("📅 Select date to load:", options=["No data available"], disabled=True, key="fx_trade_load_date_select")
            selected_load_date = None
    
    with col2:
        if selected_load_date and available_dates:
            if st.button("📂 Load Data", use_container_width=True, key="load_fx_trade_btn"):
                db.load_fx_trade_data_only(selected_load_date)
                st.rerun()
    
    with col3:
        current_date = datetime.now().strftime('%Y-%m-%d')
        st.metric("Current Date", current_date)
    
    with col4:
        if st.button("💾 Save Data", type="primary", use_container_width=True, key="save_fx_trade_btn"):
            db.save_fx_trade_data_only()
            st.rerun()
    

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Reset Current Module Data", type="secondary", use_container_width=True, key="reset_module_btn"):
            reset_all_module_dataframes()
            st.success("✅ All current module dataframes have been reset!")
            st.balloons()
            st.rerun()

    with col2:
        if st.button("🗑️ Reset All Data (Including Saved)", type="secondary", use_container_width=True, key="reset_all_btn"):
            # This will also clear the database for current date
            target_date = datetime.now().strftime('%Y-%m-%d')
            
            # Reset session state
            reset_all_module_dataframes()
            
            # Also clear database for current date
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            tables_to_clear = [
                'matched_buy_df', 'matched_sell_df', 'unmatched_buy_df', 
                'unmatched_sell_df', 'unmatched_bank_trade', 'moved_records', 
                'deleted_records', 'audit_moves_log', 'audit_deletes_log'
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
        if st.button("📊 Refresh Dashboard", type="primary", use_container_width=True, key="refresh_dashboard_btn"):
            # Update all stats without resetting data
            update_moved_stats_cards_trade()
            update_deleted_stats_cards_trade()
            refresh_analytics_dataframes_trade()
            st.success("✅ Dashboard refreshed!")
            st.rerun()
    st.markdown("---")
    
    # ========== FX TRADE TRACKER UPLOAD SECTION ==========
    with st.expander("📤 Upload FX Trade Tracker", expanded=False):
        fx_uploaded_file = st.file_uploader("Choose FX Trade Tracker file (CSV or Excel)", type=["csv", "xlsx"], key="fx_uploader")
        
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
                st.success(f"✅ File loaded successfully! Found {len(fx_trade_df)} rows and {len(fx_trade_df.columns)} columns")
                
                with st.expander("Preview Data"):
                    st.dataframe(fx_trade_df.head(5), use_container_width=True)
                
                st.markdown("#### Map Columns")
                fx_col_options = ['-- Select Column --'] + fx_trade_df.columns.tolist()
                col_mapping = {}
                fx_required_cols = {
                    'Action Type': 'Action Type', 'Status': 'Status', 'Created At': 'Created At',
                    'Buy Currency Amount': 'Buy Currency Amount', 'Buy Trade Info': 'Buy Trade Info',
                    'Sell Currency Amount': 'Sell Currency Amount', 'Sell Trade Info': 'Sell Trade Info',
                    'Vendor ID': 'Vendor ID', 'Vendor Name': 'Vendor Name', 'Counterparty Dealer': 'Counterparty Dealer',
                }
                
                saved_mapping = db.load_metadata('fx_trade_tracker_col_mapping', {})
                
                col1, col2 = st.columns(2)
                cols_list = list(fx_required_cols.items())
                mid_point = len(cols_list) // 2
                
                with col1:
                    for display_name, suggested_col in cols_list[:mid_point]:
                        initial_selection = saved_mapping.get(display_name, suggested_col if suggested_col in fx_col_options else '-- Select Column --')
                        selected_col = st.selectbox(f"Map '{display_name}'", options=fx_col_options, index=fx_col_options.index(initial_selection) if initial_selection in fx_col_options else 0, key=f"fx_map_{display_name}")
                        col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None
                
                with col2:
                    for display_name, suggested_col in cols_list[mid_point:]:
                        initial_selection = saved_mapping.get(display_name, suggested_col if suggested_col in fx_col_options else '-- Select Column --')
                        selected_col = st.selectbox(f"Map '{display_name}'", options=fx_col_options, index=fx_col_options.index(initial_selection) if initial_selection in fx_col_options else 0, key=f"fx_map_{display_name}")
                        col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None
                
                if st.button("✅ Process Data", type="primary", key="process_fx_btn", use_container_width=True):
                    renamed_cols_dict = {selected: original for original, selected in col_mapping.items() if selected and selected in fx_trade_df.columns}
                    if renamed_cols_dict:
                        cols_to_keep = list(renamed_cols_dict.keys())
                        fx_trade_df = fx_trade_df[cols_to_keep].rename(columns=renamed_cols_dict)
                    st.session_state.fx_trade_df = fx_trade_df
                    db.save_metadata('fx_trade_tracker_col_mapping', col_mapping)
                    st.success("✅ Data processed successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # ========== RECONCILIATION CONTROLS ==========
    if 'fx_trade_df' in st.session_state and not st.session_state.fx_trade_df.empty:
        st.markdown("---")
        st.markdown("### ⚙️ Reconciliation Settings")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.get('debug_mode', False))
            st.session_state.debug_mode = debug_mode
        
        with col2:
            date_tolerance_days = st.slider("Date Tolerance (± days)", min_value=0, max_value=7, value=3, step=1)
        
        with col3:
            if st.button("🔄 Run Reconciliation", type="primary", use_container_width=True):
                if not all_bank_dfs:
                    st.error("❌ No bank statements loaded! Please upload bank statements first.")
                else:
                    with st.spinner("Running reconciliation... Please wait"):
                        fx_trade_df = st.session_state.fx_trade_df
                        if 'Status' in fx_trade_df.columns:
                            active_fx_trades = fx_trade_df[~fx_trade_df['Status'].isin(['cancelled', 'pending'])]
                        else:
                            active_fx_trades = fx_trade_df
                        
                        already_matched_fx_trades = set()
                        matched_bank_keys = set()
                        skipped_bank_records = {}
                        unmatched_buy, matched_buy = [], []
                        unmatched_sell, matched_sell = [], []
                        current_run_bank_dfs = {key: df.copy() for key, df in all_bank_dfs.items()}
                        
                        progress_bar = st.progress(0)
                        total_rows = len(active_fx_trades)
                        
                        for idx, row in active_fx_trades.iterrows():
                            action_type = str(row.get('Action Type', '')).strip()
                            status = str(row.get('Status', '')).strip().lower()
                            if status in ['cancelled', 'pending']:
                                continue
                            process_fx_match(row, current_run_bank_dfs, unmatched_buy, matched_buy, action_type,
                                            'Buy Currency Amount', 'Buy Trade Info', date_tolerance_days, debug_mode,
                                            already_matched_fx_trades, skipped_bank_records, matched_bank_keys)
                            process_fx_match(row, current_run_bank_dfs, unmatched_sell, matched_sell, action_type,
                                            'Sell Currency Amount', 'Sell Trade Info', date_tolerance_days, debug_mode,
                                            already_matched_fx_trades, skipped_bank_records, matched_bank_keys)
                            progress_bar.progress((idx + 1) / total_rows)
                        
                        unmatched_bank_records = []
                        for bank_key, bank_df in current_run_bank_dfs.items():
                            bank_df.columns = bank_df.columns.str.strip()
                            date_col = 'Date'
                            description_col = get_description_columns(bank_df.columns.tolist())
                            credit_col, debit_col = 'Credit', 'Debit'
                            if date_col not in bank_df.columns or description_col not in bank_df.columns or (credit_col not in bank_df.columns and debit_col not in bank_df.columns):
                                continue
                            unmatched_bank_df = bank_df[bank_df.get("Matched", False) == False].copy()
                            for _, row in unmatched_bank_df.iterrows():
                                row_date = row.get(date_col)
                                amount_found, transaction_type_col_name = None, "N/A"
                                credit_amt = safe_float(row.get(credit_col))
                                if credit_amt is not None and abs(credit_amt) > 0.01:
                                    amount_found, transaction_type_col_name = credit_amt, credit_col
                                if amount_found is None:
                                    debit_amt = safe_float(row.get(debit_col))
                                    if debit_amt is not None and abs(debit_amt) > 0.01:
                                        amount_found, transaction_type_col_name = debit_amt, debit_col
                                if amount_found is not None:
                                    unmatched_bank_records.append({'Bank Table': bank_key, 'Date': row_date.strftime('%Y-%m-%d') if row_date else None,
                                                                  'Description': str(row.get(description_col, '')).strip(),
                                                                  'Transaction Type (Column)': transaction_type_col_name, 'Amount': round(amount_found, 2)})
                        
                        st.session_state.matched_buy_df = add_unique_ids(pd.DataFrame(matched_buy)) if matched_buy else pd.DataFrame()
                        st.session_state.matched_sell_df = add_unique_ids(pd.DataFrame(matched_sell)) if matched_sell else pd.DataFrame()
                        st.session_state.unmatched_buy_df = add_unique_ids(pd.DataFrame(unmatched_buy)) if unmatched_buy else pd.DataFrame()
                        st.session_state.unmatched_sell_df = add_unique_ids(pd.DataFrame(unmatched_sell)) if unmatched_sell else pd.DataFrame()
                        st.session_state.unmatched_bank_trade = add_unique_ids(pd.DataFrame(unmatched_bank_records)) if unmatched_bank_records else pd.DataFrame()
                        
                        for df_name in ['matched_buy_df', 'matched_sell_df', 'unmatched_buy_df', 'unmatched_sell_df', 'unmatched_bank_trade']:
                            if not st.session_state[df_name].empty:
                                st.session_state[df_name] = add_audit_columns(st.session_state[df_name])
                        
                        update_moved_stats_cards_trade()
                        update_deleted_stats_cards_trade()
                        progress_bar.empty()
                        st.success("✅ Reconciliation complete!")
                        st.balloons()
                        st.rerun()
        
        st.markdown("---")
        
        # Quick Stats Section
        st.markdown("### 📊 Quick Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Buy Matched", len(st.session_state.get('matched_buy_df', pd.DataFrame())))
        with col2:
            st.metric("⚠️ Buy Unmatched", len(st.session_state.get('unmatched_buy_df', pd.DataFrame())))
        with col3:
            st.metric("📊 Sell Matched", len(st.session_state.get('matched_sell_df', pd.DataFrame())))
        with col4:
            st.metric("⚠️ Sell Unmatched", len(st.session_state.get('unmatched_sell_df', pd.DataFrame())))
        with col5:
            st.metric("🏦 Bank Unmatched", len(st.session_state.get('unmatched_bank_trade', pd.DataFrame())))
        
        st.markdown("---")
        st.markdown("### 📋 Audit Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📦 Total Moved Records", st.session_state.get('moved_stats_trade', {}).get('total_moved', 0))
        with col2:
            st.metric("🗑️ Total Deleted Records", st.session_state.get('deleted_stats_trade', {}).get('total_deleted', 0))
        
        if st.button("🗑️ Reset All Data", use_container_width=True):
            for key in ['matched_buy_df', 'matched_sell_df', 'unmatched_buy_df', 'unmatched_sell_df', 'unmatched_bank_trade']:
                if key in st.session_state:
                    st.session_state[key] = pd.DataFrame()
            st.success("All data reset!")
            st.rerun()
    
    # Main Dashboard
    st.markdown("---")
    render_full_statistics_dashboard()
    
    # Move targets configuration
    move_targets_buy_matched = {"Buy Unmatched": "unmatched_buy_df", "Sell Matched": "matched_sell_df", "Sell Unmatched": "unmatched_sell_df"}
    move_targets_buy_unmatched = {"Buy Matched": "matched_buy_df", "Sell Matched": "matched_sell_df", "Sell Unmatched": "unmatched_sell_df"}
    move_targets_sell_matched = {"Buy Matched": "matched_buy_df", "Buy Unmatched": "unmatched_buy_df", "Sell Unmatched": "unmatched_sell_df"}
    move_targets_sell_unmatched = {"Buy Matched": "matched_buy_df", "Buy Unmatched": "unmatched_buy_df", "Sell Matched": "matched_sell_df"}
    move_targets_bank = {"Buy Matched": "matched_buy_df", "Buy Unmatched": "unmatched_buy_df", "Sell Matched": "matched_sell_df", "Sell Unmatched": "unmatched_sell_df"}
    
    # Create tabs for main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Buy Matched", 
        "⚠️ Buy Unmatched", 
        "📋 Sell Matched", 
        "⚠️ Sell Unmatched", 
        "🏦 Bank Records", 
        "📊 Audit Trail"
    ])
    
    with tab1:
        def update_buy_matched(df):
            st.session_state.matched_buy_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_buy_df.empty:
                st.session_state.matched_buy_df = add_audit_columns(st.session_state.matched_buy_df)
            update_moved_stats_cards_trade()
            update_deleted_stats_cards_trade()
        render_editable_dataframe_trade(st.session_state.matched_buy_df, "Buy Matched Records", "matched_buy", on_data_change=update_buy_matched, show_delete=True, show_move=True, move_targets=move_targets_buy_matched)
    
    with tab2:
        def update_buy_unmatched(df):
            st.session_state.unmatched_buy_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_buy_df.empty:
                st.session_state.unmatched_buy_df = add_audit_columns(st.session_state.unmatched_buy_df)
            update_moved_stats_cards_trade()
            update_deleted_stats_cards_trade()
        render_editable_dataframe_trade(st.session_state.unmatched_buy_df, "Buy Unmatched Records", "unmatched_buy", on_data_change=update_buy_unmatched, show_delete=True, show_move=True, move_targets=move_targets_buy_unmatched)
    
    with tab3:
        def update_sell_matched(df):
            st.session_state.matched_sell_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_sell_df.empty:
                st.session_state.matched_sell_df = add_audit_columns(st.session_state.matched_sell_df)
            update_moved_stats_cards_trade()
            update_deleted_stats_cards_trade()
        render_editable_dataframe_trade(st.session_state.matched_sell_df, "Sell Matched Records", "matched_sell", on_data_change=update_sell_matched, show_delete=True, show_move=True, move_targets=move_targets_sell_matched)
    
    with tab4:
        def update_sell_unmatched(df):
            st.session_state.unmatched_sell_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_sell_df.empty:
                st.session_state.unmatched_sell_df = add_audit_columns(st.session_state.unmatched_sell_df)
            update_moved_stats_cards_trade()
            update_deleted_stats_cards_trade()
        render_editable_dataframe_trade(st.session_state.unmatched_sell_df, "Sell Unmatched Records", "unmatched_sell", on_data_change=update_sell_unmatched, show_delete=True, show_move=True, move_targets=move_targets_sell_unmatched)
    
    with tab5:
        def update_bank_trade(df):
            st.session_state.unmatched_bank_trade = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_bank_trade.empty:
                st.session_state.unmatched_bank_trade = add_audit_columns(st.session_state.unmatched_bank_trade)
            update_moved_stats_cards_trade()
            update_deleted_stats_cards_trade()
        render_editable_dataframe_trade(st.session_state.unmatched_bank_trade, "Unmatched Bank Records", "bank_trade", on_data_change=update_bank_trade, show_delete=True, show_move=True, move_targets=move_targets_bank)
    
    with tab6:
        audit_tab1, audit_tab2 = st.tabs(["📋 Moved Records", "🗑️ Deleted Records"])
        with audit_tab1:
            render_moved_records_tab_trade()
        with audit_tab2:
            render_deleted_records_tab_trade()
    
    # Return all dataframes for compatibility
    return (
        safe_get_dataframe('matched_buy_df'),
        safe_get_dataframe('matched_sell_df'),
        safe_get_dataframe('unmatched_buy_df'),
        safe_get_dataframe('unmatched_sell_df'),
        safe_get_dataframe('unmatched_bank_trade'),
        safe_get_dataframe('moved_buy_matched'),
        safe_get_dataframe('moved_buy_unmatched'),
        safe_get_dataframe('moved_sell_matched'),
        safe_get_dataframe('moved_sell_unmatched'),
        safe_get_dataframe('moved_bank_records_trade'),
        safe_get_dataframe('deleted_buy_matched'),
        safe_get_dataframe('deleted_buy_unmatched'),
        safe_get_dataframe('deleted_sell_matched'),
        safe_get_dataframe('deleted_sell_unmatched'),
        safe_get_dataframe('deleted_bank_trade'),
        safe_get_dataframe('audit_moves_log_trade'),
        safe_get_dataframe('audit_deletes_log_trade'),
        pd.DataFrame([st.session_state.moved_stats_trade]) if st.session_state.moved_stats_trade else pd.DataFrame(),
        pd.DataFrame([st.session_state.deleted_stats_trade]) if st.session_state.deleted_stats_trade else pd.DataFrame(),
        safe_get_dataframe('df_matched_adjustments_local'),
        safe_get_dataframe('df_unmatched_adjustments_local'),
        safe_get_dataframe('df_matched_adjustments_foreign'),
        safe_get_dataframe('df_unmatched_adjustments_foreign'),
        safe_get_dataframe('df_unmatched_bank_records')
    )