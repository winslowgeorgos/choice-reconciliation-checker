# pages/interfund_bank_reconciliation_page.py
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
INTERFUND_DB_PATH = "data/interfund_reconciliation.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        border-left: 4px solid #667eea;
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

# ----------------------------- Database Manager for Interfund -----------------------------
class InterfundReconciliationDB:
    """Database manager for Interfund reconciliation data (separate from FX Trade)"""

    def __init__(self, db_path=INTERFUND_DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main matched/unmatched tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interfund_matched (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Bank_Table TEXT,
                Interfund_Amount REAL,
                Currency TEXT,
                Total_Bank_Matches INTEGER,
                Skipped_Bank_Records INTEGER,
                Matched_Bank_Record_Index INTEGER,
                Matched_Bank_Description TEXT,
                Matched_Bank_Debit REAL,
                Matched_Bank_Credit REAL,
                All_Matched_Bank_Records TEXT,
                Skipped_Bank_Records_Info TEXT,
                Application_ID TEXT,
                Intermediary_Bank_Account TEXT,
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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interfund_unmatched (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Bank_Table_Expected TEXT,
                Amount REAL,
                Currency TEXT,
                Status TEXT,
                Skipped_Bank_Records TEXT,
                Application_ID TEXT,
                Intermediary_Bank_Account TEXT,
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
            CREATE TABLE IF NOT EXISTS interfund_unmatched_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                _record_id TEXT UNIQUE,
                record_date TEXT,
                created_at TEXT,
                Bank_Table TEXT,
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

        # Moved / Deleted / Audit tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interfund_moved_records (
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
            CREATE TABLE IF NOT EXISTS interfund_deleted_records (
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
            CREATE TABLE IF NOT EXISTS interfund_audit_moves_log (
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
            CREATE TABLE IF NOT EXISTS interfund_audit_deletes_log (
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
            CREATE TABLE IF NOT EXISTS interfund_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        ''')

        # Indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_interfund_matched_date ON interfund_matched(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_interfund_unmatched_date ON interfund_unmatched(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_interfund_bank_date ON interfund_unmatched_bank(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_interfund_moved_date ON interfund_moved_records(record_date)',
            'CREATE INDEX IF NOT EXISTS idx_interfund_deleted_date ON interfund_deleted_records(record_date)',
        ]
        for idx_sql in indexes:
            cursor.execute(idx_sql)

        conn.commit()
        conn.close()
        logger.info("Interfund database initialized")

    def _serialize_value(self, value):
        if value is None:
            return None
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        return str(value) if not isinstance(value, (float, int)) else value

    def save_matched(self, df, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM interfund_matched WHERE record_date = ?", (record_date,))
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _, row in df.iterrows():
            _record_id = row.get('_record_id', str(uuid.uuid4()))
            cursor.execute('''
                INSERT INTO interfund_matched (
                    _record_id, record_date, created_at, Bank_Table, Interfund_Amount, Currency,
                    Total_Bank_Matches, Skipped_Bank_Records, Matched_Bank_Record_Index,
                    Matched_Bank_Description, Matched_Bank_Debit, Matched_Bank_Credit,
                    All_Matched_Bank_Records, Skipped_Bank_Records_Info, Application_ID,
                    Intermediary_Bank_Account, Status, deleted_by, deleted_at, delete_reason,
                    moved_by, moved_from, moved_at, move_reason, move_type, moved_to,
                    import_date, last_modified
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Bank Table')),
                self._serialize_value(row.get('Interfund Amount')),
                self._serialize_value(row.get('Currency')),
                self._serialize_value(row.get('Total Bank Matches', 0)),
                self._serialize_value(row.get('Skipped Bank Records', 0)),
                self._serialize_value(row.get('Matched Bank Record Index')),
                self._serialize_value(row.get('Matched Bank Description')),
                self._serialize_value(row.get('Matched Bank Debit')),
                self._serialize_value(row.get('Matched Bank Credit')),
                self._serialize_value(row.get('All Matched Bank Records')),
                self._serialize_value(row.get('Skipped Bank Records Info')),
                self._serialize_value(row.get('Application ID')),
                self._serialize_value(row.get('Intermediary Bank Account')),
                self._serialize_value(row.get('Status')),
                '', '', '', '', '', '', '', '', '', import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} matched interfund records for {record_date}")

    def save_unmatched(self, df, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM interfund_unmatched WHERE record_date = ?", (record_date,))
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _, row in df.iterrows():
            _record_id = row.get('_record_id', str(uuid.uuid4()))
            cursor.execute('''
                INSERT INTO interfund_unmatched (
                    _record_id, record_date, created_at, Bank_Table_Expected, Amount, Currency,
                    Status, Skipped_Bank_Records, Application_ID, Intermediary_Bank_Account,
                    deleted_by, deleted_at, delete_reason, moved_by, moved_from, moved_at,
                    move_reason, move_type, moved_to, import_date, last_modified
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Bank Table (Expected)')),
                self._serialize_value(row.get('Amount')),
                self._serialize_value(row.get('Currency')),
                self._serialize_value(row.get('Status')),
                self._serialize_value(row.get('Skipped Bank Records')),
                self._serialize_value(row.get('Application ID')),
                self._serialize_value(row.get('Intermediary Bank Account')),
                '', '', '', '', '', '', '', '', '', import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} unmatched interfund records for {record_date}")

    def save_unmatched_bank(self, df, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM interfund_unmatched_bank WHERE record_date = ?", (record_date,))
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _, row in df.iterrows():
            _record_id = row.get('_record_id', str(uuid.uuid4()))
            cursor.execute('''
                INSERT INTO interfund_unmatched_bank (
                    _record_id, record_date, created_at, Bank_Table, Description,
                    Transaction_Type_Column, Amount, deleted_by, deleted_at, delete_reason,
                    moved_by, moved_from, moved_at, move_reason, move_type, moved_to,
                    import_date, last_modified
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                _record_id, record_date, import_date,
                self._serialize_value(row.get('Bank Table')),
                self._serialize_value(row.get('Description')),
                self._serialize_value(row.get('Transaction Type (Column)')),
                self._serialize_value(row.get('Amount')),
                '', '', '', '', '', '', '', '', '', import_date, import_date
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(df)} unmatched bank records for {record_date}")

    def save_moved_records(self, df, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM interfund_moved_records WHERE record_date = ?", (record_date,))
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _, row in df.iterrows():
            _record_id = row.get('_record_id', str(uuid.uuid4()))
            original_json = json.dumps(row.to_dict(), default=str)
            cursor.execute('''
                INSERT INTO interfund_moved_records (
                    _record_id, record_date, created_at, source_table, record_type,
                    original_record_json, Date, Action_Type, Amount, Bank_Table, Status,
                    Vendor_Name, moved_by, moved_from, moved_to, moved_at, move_reason,
                    move_type, import_date, last_modified
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                _record_id, record_date, import_date,
                row.get('moved_from', 'unknown'), row.get('record_type', ''),
                original_json,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Interfund Amount', row.get('Amount', 0))),
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
        logger.info(f"Saved {len(df)} moved interfund records for {record_date}")

    def save_deleted_records(self, df, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM interfund_deleted_records WHERE record_date = ?", (record_date,))
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _, row in df.iterrows():
            _record_id = row.get('_record_id', str(uuid.uuid4()))
            original_json = json.dumps(row.to_dict(), default=str)
            cursor.execute('''
                INSERT INTO interfund_deleted_records (
                    _record_id, record_date, created_at, source_table, record_type,
                    original_record_json, Date, Action_Type, Amount, Bank_Table, Status,
                    Vendor_Name, deleted_by, deleted_at, delete_reason, deleted_from,
                    source_dataframe, import_date, last_modified
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                _record_id, record_date, import_date,
                row.get('deleted_from', 'unknown'), row.get('record_type', ''),
                original_json,
                self._serialize_value(row.get('Date')),
                self._serialize_value(row.get('Action Type')),
                self._serialize_value(row.get('Interfund Amount', row.get('Amount', 0))),
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
        logger.info(f"Saved {len(df)} deleted interfund records for {record_date}")

    def save_audit_moves(self, df, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM interfund_audit_moves_log WHERE import_date LIKE ?", (f"{record_date}%",))
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO interfund_audit_moves_log (
                    _record_id, timestamp, user, record_type, record_id,
                    from_location, to_location, details, import_date
                ) VALUES (?,?,?,?,?,?,?,?,?)
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

    def save_audit_deletes(self, df, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM interfund_audit_deletes_log WHERE import_date LIKE ?", (f"{record_date}%",))
        if df is None or df.empty:
            conn.commit()
            conn.close()
            return
        import_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO interfund_audit_deletes_log (
                    _record_id, timestamp, user, record_type, record_id,
                    details, deleted_record, import_date
                ) VALUES (?,?,?,?,?,?,?,?)
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

    def load_matched(self, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM interfund_matched WHERE record_date = ?", conn, params=(record_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        rename_map = {
            'Bank_Table': 'Bank Table',
            'Interfund_Amount': 'Interfund Amount',
            'Currency': 'Currency',
            'Total_Bank_Matches': 'Total Bank Matches',
            'Skipped_Bank_Records': 'Skipped Bank Records',
            'Matched_Bank_Record_Index': 'Matched Bank Record Index',
            'Matched_Bank_Description': 'Matched Bank Description',
            'Matched_Bank_Debit': 'Matched Bank Debit',
            'Matched_Bank_Credit': 'Matched Bank Credit',
            'All_Matched_Bank_Records': 'All Matched Bank Records',
            'Skipped_Bank_Records_Info': 'Skipped Bank Records Info',
            'Application_ID': 'Application ID',
            'Intermediary_Bank_Account': 'Intermediary Bank Account',
            'Status': 'Status'
        }
        df = df.rename(columns=rename_map)
        drop_cols = ['id', 'created_at', 'import_date', 'last_modified', 'record_date', 'deleted_by', 'deleted_at', 'delete_reason', 'moved_by', 'moved_from', 'moved_at', 'move_reason', 'move_type', 'moved_to']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df

    def load_unmatched(self, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM interfund_unmatched WHERE record_date = ?", conn, params=(record_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        rename_map = {
            'Bank_Table_Expected': 'Bank Table (Expected)',
            'Amount': 'Amount',
            'Currency': 'Currency',
            'Status': 'Status',
            'Skipped_Bank_Records': 'Skipped Bank Records',
            'Application_ID': 'Application ID',
            'Intermediary_Bank_Account': 'Intermediary Bank Account'
        }
        df = df.rename(columns=rename_map)
        drop_cols = ['id', 'created_at', 'import_date', 'last_modified', 'record_date', 'deleted_by', 'deleted_at', 'delete_reason', 'moved_by', 'moved_from', 'moved_at', 'move_reason', 'move_type', 'moved_to']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df

    def load_unmatched_bank(self, record_date=None):
        record_date = record_date or datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM interfund_unmatched_bank WHERE record_date = ?", conn, params=(record_date,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        rename_map = {
            'Bank_Table': 'Bank Table',
            'Description': 'Description',
            'Transaction_Type_Column': 'Transaction Type (Column)',
            'Amount': 'Amount'
        }
        df = df.rename(columns=rename_map)
        drop_cols = ['id', 'created_at', 'import_date', 'last_modified', 'record_date', 'deleted_by', 'deleted_at', 'delete_reason', 'moved_by', 'moved_from', 'moved_at', 'move_reason', 'move_type', 'moved_to']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df

    def get_available_dates(self):
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT DISTINCT record_date FROM (
                SELECT record_date FROM interfund_matched
                UNION SELECT record_date FROM interfund_unmatched
                UNION SELECT record_date FROM interfund_unmatched_bank
            ) WHERE record_date IS NOT NULL ORDER BY record_date DESC
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df['record_date'].tolist() if not df.empty else []

    def save_all_data(self, target_date=None):
        """Save all current session state dataframes to DB for the target date"""
        target_date = target_date or datetime.now().strftime('%Y-%m-%d')
        self.save_matched(st.session_state.get('matched_interfund_df', pd.DataFrame()), target_date)
        self.save_unmatched(st.session_state.get('unmatched_interfund_df', pd.DataFrame()), target_date)
        self.save_unmatched_bank(st.session_state.get('unmatched_bank_interfund', pd.DataFrame()), target_date)

        moved_dfs = []
        for key in ['moved_interfund_matched', 'moved_interfund_unmatched', 'moved_interfund_bank']:
            df = st.session_state.get(key, pd.DataFrame())
            if not df.empty:
                moved_dfs.append(df)
        if moved_dfs:
            self.save_moved_records(pd.concat(moved_dfs, ignore_index=True), target_date)
        else:
            self.save_moved_records(pd.DataFrame(), target_date)

        deleted_dfs = []
        for key in ['deleted_interfund_matched', 'deleted_interfund_unmatched', 'deleted_interfund_bank']:
            df = st.session_state.get(key, pd.DataFrame())
            if not df.empty:
                deleted_dfs.append(df)
        if deleted_dfs:
            self.save_deleted_records(pd.concat(deleted_dfs, ignore_index=True), target_date)
        else:
            self.save_deleted_records(pd.DataFrame(), target_date)

        self.save_audit_moves(st.session_state.get('audit_moves_log_interfund', pd.DataFrame()), target_date)
        self.save_audit_deletes(st.session_state.get('audit_deletes_log_interfund', pd.DataFrame()), target_date)

        st.session_state.interfund_last_save_date = target_date
        st.success(f"✅ Interfund data saved for {target_date}")

    def load_all_data(self, target_date=None):
        """Load all data from DB for a given date into session state"""
        target_date = target_date or datetime.now().strftime('%Y-%m-%d')
        st.session_state.matched_interfund_df = self.load_matched(target_date)
        st.session_state.unmatched_interfund_df = self.load_unmatched(target_date)
        st.session_state.unmatched_bank_interfund = self.load_unmatched_bank(target_date)

        conn = sqlite3.connect(self.db_path)
        moved_df = pd.read_sql_query("SELECT * FROM interfund_moved_records WHERE record_date = ?", conn, params=(target_date,))
        conn.close()
        if not moved_df.empty:
            st.session_state.moved_interfund_matched = moved_df[moved_df['moved_to'].str.contains('Matched', na=False)].copy()
            st.session_state.moved_interfund_unmatched = moved_df[moved_df['moved_to'].str.contains('Unmatched', na=False)].copy()
            st.session_state.moved_interfund_bank = moved_df[moved_df['moved_to'].str.contains('Bank', na=False)].copy()
        else:
            st.session_state.moved_interfund_matched = pd.DataFrame()
            st.session_state.moved_interfund_unmatched = pd.DataFrame()
            st.session_state.moved_interfund_bank = pd.DataFrame()

        conn = sqlite3.connect(self.db_path)
        deleted_df = pd.read_sql_query("SELECT * FROM interfund_deleted_records WHERE record_date = ?", conn, params=(target_date,))
        conn.close()
        if not deleted_df.empty:
            st.session_state.deleted_interfund_matched = deleted_df[deleted_df['deleted_from'].str.contains('Matched', na=False)].copy()
            st.session_state.deleted_interfund_unmatched = deleted_df[deleted_df['deleted_from'].str.contains('Unmatched', na=False)].copy()
            st.session_state.deleted_interfund_bank = deleted_df[deleted_df['deleted_from'].str.contains('Bank', na=False)].copy()
        else:
            st.session_state.deleted_interfund_matched = pd.DataFrame()
            st.session_state.deleted_interfund_unmatched = pd.DataFrame()
            st.session_state.deleted_interfund_bank = pd.DataFrame()

        conn = sqlite3.connect(self.db_path)
        audit_moves = pd.read_sql_query("SELECT * FROM interfund_audit_moves_log WHERE import_date LIKE ?", conn, params=(f"{target_date}%",))
        audit_deletes = pd.read_sql_query("SELECT * FROM interfund_audit_deletes_log WHERE import_date LIKE ?", conn, params=(f"{target_date}%",))
        conn.close()
        st.session_state.audit_moves_log_interfund = audit_moves if not audit_moves.empty else pd.DataFrame()
        st.session_state.audit_deletes_log_interfund = audit_deletes if not audit_deletes.empty else pd.DataFrame()

        st.session_state.interfund_current_date = target_date
        st.success(f"✅ Interfund data loaded for {target_date}")


# Initialize DB
interfund_db = InterfundReconciliationDB()

# ----------------------------- Helper Functions for Record Management -----------------------------
# --- Helper Functions ---
def save_uploaded_file(file, filename):
    print("saving interfund bank uploaded data : ", filename)
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

def generate_record_id():
    return str(uuid.uuid4())

def add_unique_ids(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '_record_id' not in df_copy.columns:
        df_copy['_record_id'] = [generate_record_id() for _ in range(len(df_copy))]
    return df_copy

def add_audit_columns(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    audit_cols = ['deleted_by', 'deleted_at', 'delete_reason', 'source_dataframe', 'deleted_from',
                  'moved_by', 'moved_from', 'moved_at', 'move_reason', 'move_type', 'moved_to']
    for col in audit_cols:
        if col not in df_copy.columns:
            df_copy[col] = ''
    return df_copy

def add_row_numbers(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if '#' in df_copy.columns:
        df_copy = df_copy.drop(columns=['#'])
    df_copy.insert(0, '#', range(1, len(df_copy)+1))
    return df_copy

def remove_row_numbers(df):
    if df is None or df.empty:
        return df
    if '#' in df.columns:
        return df.drop(columns=['#'])
    return df

def get_current_user():
    return st.session_state.get('user', {}).get('username', 'unknown_user')

def get_moved_df_name(source_name, target_name):
    target_clean = target_name.lower().replace(' ', '_')
    if 'matched' in target_clean:
        return 'moved_interfund_matched'
    elif 'unmatched' in target_clean:
        return 'moved_interfund_unmatched'
    elif 'bank' in target_clean:
        return 'moved_interfund_bank'
    return f"moved_interfund_{target_clean}"

def get_deleted_df_name(source_name):
    source_clean = source_name.lower().replace(' ', '_')
    if 'matched' in source_clean:
        return 'deleted_interfund_matched'
    elif 'unmatched' in source_clean:
        return 'deleted_interfund_unmatched'
    elif 'bank' in source_clean:
        return 'deleted_interfund_bank'
    return f"deleted_interfund_{source_clean}"

def move_records_to_new_df(source_df, selected_record_ids, source_name, target_name, move_reason=""):
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

    if 'audit_deletes_log_interfund' not in st.session_state:
        st.session_state.audit_deletes_log_interfund = deleted_records[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']].copy()
    else:
        existing_log = st.session_state.audit_deletes_log_interfund
        existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
        new_log = deleted_records[~deleted_records['_record_id'].isin(existing_ids)]
        if not new_log.empty:
            st.session_state.audit_deletes_log_interfund = pd.concat([existing_log, new_log[['_record_id', 'deleted_by', 'deleted_from', 'deleted_at', 'delete_reason']]], ignore_index=True)

    remaining_source_with_numbers = add_row_numbers(remaining_source)
    if df_name and df_name in st.session_state:
        st.session_state[df_name] = remaining_source_with_numbers
        original_df_name = df_name.replace('_display_df', '')
        if original_df_name in st.session_state:
            st.session_state[original_df_name] = remove_row_numbers(remaining_source.copy())

    main_mapping = {
        'Matched Interfund Records': 'matched_interfund_df',
        'Unmatched Interfund Records': 'unmatched_interfund_df',
        'Unmatched Bank Records (Interfund)': 'unmatched_bank_interfund'
    }
    if source_name in main_mapping:
        main_key = main_mapping[source_name]
        if main_key in st.session_state:
            st.session_state[main_key] = remove_row_numbers(remaining_source.copy())

    if on_data_change:
        on_data_change(remaining_source.copy())

    update_interfund_stats()
    return remaining_source_with_numbers, len(selected_record_ids)

def ensure_record_ids(df):
    if df is None or df.empty:
        return df
    if '_record_id' not in df.columns:
        return add_unique_ids(df)
    return df

def update_interfund_stats():
    moved_stats = {
        'matched': len(st.session_state.get('moved_interfund_matched', pd.DataFrame())),
        'unmatched': len(st.session_state.get('moved_interfund_unmatched', pd.DataFrame())),
        'bank': len(st.session_state.get('moved_interfund_bank', pd.DataFrame())),
        'total': 0
    }
    moved_stats['total'] = moved_stats['matched'] + moved_stats['unmatched'] + moved_stats['bank']
    st.session_state.interfund_moved_stats = moved_stats

    deleted_stats = {
        'matched': len(st.session_state.get('deleted_interfund_matched', pd.DataFrame())),
        'unmatched': len(st.session_state.get('deleted_interfund_unmatched', pd.DataFrame())),
        'bank': len(st.session_state.get('deleted_interfund_bank', pd.DataFrame())),
        'total': 0
    }
    deleted_stats['total'] = deleted_stats['matched'] + deleted_stats['unmatched'] + deleted_stats['bank']
    st.session_state.interfund_deleted_stats = deleted_stats

def reset_interfund_module():
    keys_to_reset = [
        'matched_interfund_df', 'unmatched_interfund_df', 'unmatched_bank_interfund',
        'moved_interfund_matched', 'moved_interfund_unmatched', 'moved_interfund_bank',
        'deleted_interfund_matched', 'deleted_interfund_unmatched', 'deleted_interfund_bank',
        'audit_moves_log_interfund', 'audit_deletes_log_interfund'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = pd.DataFrame()
    for key in list(st.session_state.keys()):
        if key.endswith('_display_df'):
            st.session_state[key] = pd.DataFrame()
        if key.endswith('_selection_state'):
            st.session_state[key] = {}
    update_interfund_stats()
    st.success("✅ Interfund module data has been reset.")

def render_editable_dataframe_interfund(df, title, key_prefix, on_data_change=None, show_delete=True, show_move=True, move_targets=None):
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
        st.session_state[display_df_key] = add_row_numbers(df.copy())
        if original_df_key not in st.session_state:
            st.session_state[original_df_key] = remove_row_numbers(df.copy())

    action_reason = st.text_input("Action Reason (optional):", key=f"{key_prefix}_action_reason", placeholder="Reason for moving/deleting...")

    col1, col2, col3, col4, col5 = st.columns([2,1,1,1,1])
    with col1:
        st.markdown("**Select rows to delete/move:**")
    with col2:
        if show_delete and st.button(f"🗑️ Delete Selected", key=f"{key_prefix}_delete_btn"):
            selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
            selected_ids = [rid.replace(f"{key_prefix}_select_", "") for rid, sel in selection_state.items() if sel and rid.startswith(f"{key_prefix}_select_")]
            if selected_ids:
                source_df = st.session_state[display_df_key].copy()
                updated_df, deleted_count = delete_selected_rows_with_audit(
                    source_df, selected_ids, title, action_reason,
                    df_name=display_df_key, on_data_change=on_data_change
                )
                if original_df_key in st.session_state:
                    st.session_state[original_df_key] = remove_row_numbers(updated_df.copy())
                st.success(f"✅ Deleted {deleted_count} record(s) - Audit trail created")
                st.rerun()
            else:
                st.warning("No rows selected")
    with col3:
        if show_move and move_targets:
            if st.button(f"➡️ Move Selected", key=f"{key_prefix}_move_btn"):
                selection_state = st.session_state.get(f"{key_prefix}_selection_state", {})
                selected_ids = [rid.replace(f"{key_prefix}_select_", "") for rid, sel in selection_state.items() if sel and rid.startswith(f"{key_prefix}_select_")]
                if selected_ids:
                    target_key = st.session_state.get(f"{key_prefix}_selected_target", None)
                    if target_key and target_key in move_targets:
                        source_df = st.session_state[original_df_key].copy()
                        source_df = ensure_record_ids(source_df)
                        moved_records, new_source = move_records_to_new_df(
                            source_df, selected_ids, title, target_key, action_reason
                        )
                        if not moved_records.empty:
                            moved_df_name = get_moved_df_name(title, target_key)
                            if moved_df_name not in st.session_state:
                                st.session_state[moved_df_name] = moved_records
                            else:
                                existing = st.session_state[moved_df_name]
                                existing_ids = set(existing['_record_id'].tolist()) if not existing.empty else set()
                                new_records = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_records.empty:
                                    st.session_state[moved_df_name] = pd.concat([existing, new_records], ignore_index=True)
                            if 'audit_moves_log_interfund' not in st.session_state:
                                st.session_state.audit_moves_log_interfund = moved_records[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type']].copy()
                            else:
                                existing_log = st.session_state.audit_moves_log_interfund
                                existing_ids = set(existing_log['_record_id'].tolist()) if not existing_log.empty else set()
                                new_log = moved_records[~moved_records['_record_id'].isin(existing_ids)]
                                if not new_log.empty:
                                    st.session_state.audit_moves_log_interfund = pd.concat([existing_log, new_log[['_record_id', 'moved_by', 'moved_from', 'moved_to', 'moved_at', 'move_reason', 'move_type']]], ignore_index=True)
                            st.session_state[original_df_key] = new_source
                            st.session_state[display_df_key] = add_row_numbers(new_source)
                            if on_data_change:
                                on_data_change(new_source)
                            st.success(f"✅ Moved {len(selected_ids)} record(s) to {moved_df_name}")
                            st.rerun()
                    else:
                        st.warning("Please select a target from the dropdown")
                else:
                    st.warning("No rows selected")
    with col4:
        df_download = st.session_state[display_df_key].copy()
        if '#' in df_download.columns:
            df_download = df_download.drop(columns=['#'])
        if '_record_id' in df_download.columns:
            df_download = df_download.drop(columns=['_record_id'])
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name=f"{key_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    with col5:
        if st.button(f"🔄 Refresh", key=f"{key_prefix}_refresh"):
            st.rerun()

    with st.container():
        st.markdown("---")
        st.markdown("### Edit Data Directly")
        st.info("💡 Tip: Double-click any cell to edit. Use checkboxes below for batch operations.")

        df_for_edit = st.session_state[display_df_key].copy()
        cols_to_drop = ['#', '_record_id']
        df_display = df_for_edit.drop(columns=[c for c in cols_to_drop if c in df_for_edit.columns])
        edited_df = st.data_editor(df_display, use_container_width=True, height=min(400, len(df_display)*35+38), key=f"{key_prefix}_editor")

        if not edited_df.equals(df_display):
            edited_with_ids = ensure_record_ids(edited_df.copy())
            edited_with_audit = add_audit_columns(edited_with_ids)
            updated_with_numbers = add_row_numbers(edited_with_audit)
            st.session_state[display_df_key] = updated_with_numbers
            st.session_state[original_df_key] = remove_row_numbers(edited_with_audit)
            if on_data_change:
                on_data_change(remove_row_numbers(edited_with_audit))
            st.success("✅ Data updated!")
            st.rerun()

    st.markdown("### Select Rows for Batch Operations")
    if show_move and move_targets:
        target_options = list(move_targets.keys())
        selected_target = st.selectbox("Select target for moving:", options=target_options, key=f"{key_prefix}_selected_target")
        st.info(f"📌 Moving to: {selected_target}")

    selection_key = f"{key_prefix}_selection_state"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}

    df_sel = st.session_state[display_df_key].copy()
    if '_record_id' not in df_sel.columns:
        df_sel = ensure_record_ids(df_sel)
        st.session_state[display_df_key] = add_row_numbers(df_sel)
        st.session_state[original_df_key] = remove_row_numbers(df_sel)
    record_ids = df_sel['_record_id'].tolist()

    for idx, row in df_sel.iterrows():
        col1, col2 = st.columns([0.1, 0.9])
        row_num = row['#'] if '#' in df_sel.columns else idx+1
        checkbox_key = f"{key_prefix}_select_{record_ids[idx]}"
        is_selected = st.session_state[selection_key].get(checkbox_key, False)
        if col1.checkbox("", value=is_selected, key=checkbox_key):
            st.session_state[selection_key][checkbox_key] = True
        else:
            st.session_state[selection_key][checkbox_key] = False
        with col2:
            summary = []
            for col in df_sel.columns:
                if col not in ['#', '_record_id']:
                    val = row[col]
                    if pd.notna(val):
                        s = str(val)[:50]
                        summary.append(f"**{col}:** {s}")
            st.markdown(f"**Row {row_num}:** " + " | ".join(summary[:3]))
            if len(summary) > 3:
                with st.expander(f"Show all columns for row {row_num}"):
                    for item in summary:
                        st.markdown(item)

    selected_count = sum(1 for v in st.session_state[selection_key].values() if v)
    if selected_count:
        st.success(f"✅ {selected_count} row(s) selected")

    result = st.session_state[display_df_key].copy()
    if '_record_id' in result.columns:
        result = result.drop(columns=['_record_id'])
    if '#' in result.columns:
        result = result.drop(columns=['#'])
    return result

def render_moved_records_tab():
    st.markdown("### 📋 Moved Records - Audit Trail")
    stats = st.session_state.get('interfund_moved_stats', {'matched':0, 'unmatched':0, 'bank':0, 'total':0})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Matched Moved", stats['matched'])
    col2.metric("⚠️ Unmatched Moved", stats['unmatched'])
    col3.metric("🏦 Bank Moved", stats['bank'])
    col4.metric("📊 Total Moved", stats['total'])

    moved_dfs = {
        'Matched Records': st.session_state.get('moved_interfund_matched', pd.DataFrame()),
        'Unmatched Records': st.session_state.get('moved_interfund_unmatched', pd.DataFrame()),
        'Bank Records': st.session_state.get('moved_interfund_bank', pd.DataFrame())
    }
    tabs = st.tabs(list(moved_dfs.keys()))
    for tab, (name, df) in zip(tabs, moved_dfs.items()):
        with tab:
            if df.empty:
                st.info(f"No moved {name.lower()}")
            else:
                st.dataframe(df.drop(columns=['_record_id', 'original_record_json'], errors='ignore'), use_container_width=True)

def render_deleted_records_tab():
    st.markdown("### 🗑️ Deleted Records - Audit Trail")
    stats = st.session_state.get('interfund_deleted_stats', {'matched':0, 'unmatched':0, 'bank':0, 'total':0})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Matched Deleted", stats['matched'])
    col2.metric("⚠️ Unmatched Deleted", stats['unmatched'])
    col3.metric("🏦 Bank Deleted", stats['bank'])
    col4.metric("📊 Total Deleted", stats['total'])

    deleted_dfs = {
        'Matched Records': st.session_state.get('deleted_interfund_matched', pd.DataFrame()),
        'Unmatched Records': st.session_state.get('deleted_interfund_unmatched', pd.DataFrame()),
        'Bank Records': st.session_state.get('deleted_interfund_bank', pd.DataFrame())
    }
    tabs = st.tabs(list(deleted_dfs.keys()))
    for tab, (name, df) in zip(tabs, deleted_dfs.items()):
        with tab:
            if df.empty:
                st.info(f"No deleted {name.lower()}")
            else:
                st.dataframe(df.drop(columns=['_record_id', 'original_record_json'], errors='ignore'), use_container_width=True)

def render_full_statistics_dashboard_interfund():
    st.markdown("### 📊 Interfund Reconciliation Dashboard")
    matched = st.session_state.get('matched_interfund_df', pd.DataFrame())
    unmatched = st.session_state.get('unmatched_interfund_df', pd.DataFrame())
    bank_unmatched = st.session_state.get('unmatched_bank_interfund', pd.DataFrame())

    total = len(matched) + len(unmatched)
    match_rate = (len(matched)/total*100) if total>0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Matched", len(matched))
    col2.metric("❌ Unmatched", len(unmatched))
    col3.metric("🏦 Bank Unmatched", len(bank_unmatched))
    col4.metric("📈 Match Rate", f"{match_rate:.1f}%")

    if total > 0:
        fig = px.pie(values=[len(matched), len(unmatched)], names=['Matched', 'Unmatched'], title='Interfund Match Status')
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Core Reconciliation Functions (from original) -----------------------------
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
    date_formats = [
        '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
        '%d/%m/%Y', '%-d/%-m/%Y', '%-d.%-m/%-Y',
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
        '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%-d/%-m/%Y %H:%M:%S',
        '%-d.%-m.%Y %H:%M:%S', "%d.%m.%Y"
    ]
    date_str = str(date_str_raw).strip()
    for fmt in date_formats:
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

def process_interfund_match(
    interfund_row: pd.Series,
    all_bank_dfs: dict,
    unmatched_list: list,
    matched_list: list,
    date_tolerance_days: int = 3,
    debug_mode: bool = False,
    already_matched_interfund_records: set = None,
    skipped_bank_records: dict = None,
    matched_bank_keys: set = None
) -> list or None:
    if already_matched_interfund_records is None:
        already_matched_interfund_records = set()
    if skipped_bank_records is None:
        skipped_bank_records = {}
    if matched_bank_keys is None:
        matched_bank_keys = set()

    application_id = interfund_row.get('Application ID', '')
    if not application_id:
        application_id = f"{interfund_row.get('Amount (In Debit Account Currency)', '')}_{interfund_row.get('Intermediary Bank Account', '')}"
    record_id = f"{application_id}_INTERFUND"

    if record_id in already_matched_interfund_records:
        return None

    amount = safe_float(interfund_row.get('Amount (In Debit Account Currency)'))
    if amount is None:
        return None

    status = str(interfund_row.get('Status', '')).strip().lower()
    if status in ['declined', 'rejected', 'pending', 'not completed']:
        return None

    currency = str(interfund_row.get('Credit Account Currency', '')).strip().upper()
    if not currency:
        return None

    bank_info_raw = interfund_row.get('Intermediary Bank Account', '')
    normalized_bank_name = extract_bank_info_from_intermediary_column(bank_info_raw)
    if not normalized_bank_name:
        unmatched_list.append({
            'Bank Table (Expected)': f"N/A ({bank_info_raw})",
            'Amount': amount,
            'Currency': currency,
            'Status': 'Invalid Bank Info in Interfund Record',
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        })
        return None

    expected_bank_key = f"{normalized_bank_name} {currency}"
    if expected_bank_key not in all_bank_dfs:
        unmatched_list.append({
            'Bank Table (Expected)': expected_bank_key,
            'Amount': amount,
            'Currency': currency,
            'Status': 'No Matching Bank Statement File Found',
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        })
        return None

    bank_df = all_bank_dfs[expected_bank_key]
    bank_df_columns = bank_df.columns.tolist()
    if 'Skipped_By_Interfund' not in bank_df.columns:
        bank_df['Skipped_By_Interfund'] = ""

    bank_amount_column = 'Debit'
    if bank_amount_column not in bank_df.columns:
        unmatched_list.append({
            'Bank Table (Expected)': expected_bank_key,
            'Amount': amount,
            'Currency': currency,
            'Status': 'Missing Debit Column in Bank Statement',
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        })
        return None

    amount_matches = bank_df.copy()
    matched_records = []
    skipped_records = []

    for idx, bank_row in amount_matches.iterrows():
        bank_amt = safe_float(bank_row.get(bank_amount_column))
        if bank_amt is None:
            continue
        amount_diff = abs(abs(bank_amt) - abs(amount))
        if amount_diff < 0.05:
            bank_record_key = (expected_bank_key, round(abs(bank_amt), 2), 'debit')
            is_already_matched = bank_record_key in matched_bank_keys
            if is_already_matched:
                current_skipped = bank_df.loc[idx, "Skipped_By_Interfund"]
                skipped_list = []
                if current_skipped and current_skipped != "":
                    try:
                        skipped_list = json.loads(current_skipped)
                    except:
                        skipped_list = []
                skipped_info = {
                    'interfund_id': record_id,
                    'interfund_amount': amount,
                    'interfund_currency': currency,
                    'skipped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'match_details': {
                        'amount_difference': amount_diff,
                        'bank_amount': bank_amt,
                        'amount_column': bank_amount_column
                    }
                }
                skipped_list.append(skipped_info)
                bank_df.loc[idx, "Skipped_By_Interfund"] = json.dumps(skipped_list)
                if record_id not in skipped_bank_records:
                    skipped_bank_records[record_id] = []
                skipped_records.append({
                    'bank_key': bank_record_key,
                    'bank_table': expected_bank_key,
                    'bank_amount': bank_amt,
                    'bank_row_index': idx,
                    'match_details': {
                        'amount_difference': amount_diff,
                        'bank_amount': bank_amt,
                        'amount_column': bank_amount_column
                    }
                })
                continue
            matched_records.append({
                'Bank Index': idx,
                'Description': str(bank_row.get('Description', '')).strip(),
                'Debit': safe_float(bank_row.get('Debit')),
                'Credit': safe_float(bank_row.get('Credit')),
                'Matched Column': bank_amount_column,
                'Bank Amount': bank_amt,
                'Bank Record Key': bank_record_key,
                'Amount Difference': amount_diff
            })
            bank_df.at[idx, "Matched"] = True
            matched_bank_keys.add(bank_record_key)

    if matched_records:
        matched_list.append({
            'Bank Table': expected_bank_key,
            'Interfund Amount': amount,
            'Currency': currency,
            'Total Bank Matches': len(matched_records),
            'Skipped Bank Records': len(skipped_records),
            'Matched Bank Record Index': matched_records[0]['Bank Index'],
            'Matched Bank Description': matched_records[0]['Description'],
            'Matched Bank Debit': matched_records[0]['Debit'],
            'Matched Bank Credit': matched_records[0]['Credit'],
            'All Matched Bank Records': json.dumps(matched_records),
            'Skipped Bank Records Info': json.dumps(skipped_records),
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw,
            'Status': status
        })
        already_matched_interfund_records.add(record_id)
        return [(expected_bank_key, m['Bank Index']) for m in matched_records]

    if skipped_records:
        unmatched_list.append({
            'Bank Table (Expected)': expected_bank_key,
            'Amount': amount,
            'Currency': currency,
            'Status': f'Potential matches found but already taken by other records (skipped: {len(skipped_records)})',
            'Skipped Bank Records': json.dumps(skipped_records),
            'Application ID': application_id,
            'Intermediary Bank Account': bank_info_raw
        })
        return None

    unmatched_list.append({
        'Bank Table (Expected)': expected_bank_key,
        'Amount': amount,
        'Currency': currency,
        'Status': 'No Bank Statement Match (Amount)',
        'Application ID': application_id,
        'Intermediary Bank Account': bank_info_raw
    })
    return None

# ----------------------------- Main App Function -----------------------------
def interfund_bank_reconciliation_app(all_bank_dfs: dict):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    # st.title("🏦 Interfund Bank Reconciliation")
    # st.markdown("This dashboard verifies interfund bank records against bank statements, with full audit trail and data management.")

    # Initialize session state for Interfund module
    if 'interfund_initialized' not in st.session_state:
        st.session_state.matched_interfund_df = pd.DataFrame()
        st.session_state.unmatched_interfund_df = pd.DataFrame()
        st.session_state.unmatched_bank_interfund = pd.DataFrame()
        st.session_state.moved_interfund_matched = pd.DataFrame()
        st.session_state.moved_interfund_unmatched = pd.DataFrame()
        st.session_state.moved_interfund_bank = pd.DataFrame()
        st.session_state.deleted_interfund_matched = pd.DataFrame()
        st.session_state.deleted_interfund_unmatched = pd.DataFrame()
        st.session_state.deleted_interfund_bank = pd.DataFrame()
        st.session_state.audit_moves_log_interfund = pd.DataFrame()
        st.session_state.audit_deletes_log_interfund = pd.DataFrame()
        st.session_state.interfund_moved_stats = {'matched':0, 'unmatched':0, 'bank':0, 'total':0}
        st.session_state.interfund_deleted_stats = {'matched':0, 'unmatched':0, 'bank':0, 'total':0}
        st.session_state.interfund_initialized = True

    # Data Management Section
    st.markdown("### 📅 Data Management")
    available_dates = interfund_db.get_available_dates()
    col1, col2, col3, col4 = st.columns([2,1,1,2])
    with col1:
        if available_dates:
            selected_load_date = st.selectbox("Select date to load:", options=available_dates, key="interfund_load_date")
        else:
            st.selectbox("Select date to load:", options=["No data available"], disabled=True)
            selected_load_date = None
    with col2:
        if selected_load_date and st.button("📂 Load Data", key="load_interfund_btn"):
            interfund_db.load_all_data(selected_load_date)
            update_interfund_stats()
            st.rerun()
    with col3:
        st.metric("Current Date", datetime.now().strftime('%Y-%m-%d'))
    with col4:
        if st.button("💾 Save Data", type="primary", key="save_interfund_btn"):
            interfund_db.save_all_data()
            st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Reset Current Module Data", key="reset_interfund_btn"):
            reset_interfund_module()
            st.rerun()
    with col2:
        if st.button("🗑️ Reset All Data (Including Saved)", key="reset_all_interfund_btn"):
            target_date = datetime.now().strftime('%Y-%m-%d')
            reset_interfund_module()
            conn = sqlite3.connect(INTERFUND_DB_PATH)
            cursor = conn.cursor()
            tables = ['interfund_matched', 'interfund_unmatched', 'interfund_unmatched_bank',
                      'interfund_moved_records', 'interfund_deleted_records',
                      'interfund_audit_moves_log', 'interfund_audit_deletes_log']
            for table in tables:
                try:
                    cursor.execute(f"DELETE FROM {table} WHERE record_date = ? OR import_date LIKE ?", (target_date, f"{target_date}%"))
                except:
                    pass
            conn.commit()
            conn.close()
            st.success("All Interfund data (session and database) reset.")
            st.rerun()
    with col3:
        if st.button("📊 Refresh Dashboard", key="refresh_interfund_dashboard"):
            update_interfund_stats()
            st.rerun()
    st.markdown("---")

    # Upload Section
    with st.expander("📤 Upload Interfund Bank Records", expanded=False):
        uploaded_file = st.file_uploader("Choose Interfund file (CSV or Excel)", type=["csv","xlsx"], key="interfund_upload_main")
        if uploaded_file:
            try:
                file_ext = uploaded_file.name.split('.')[-1]
                file_path = os.path.join(UPLOAD_DIR, f"interfund_uploaded.{file_ext}")
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if file_ext == 'xlsx':
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_names = xls.sheet_names
                    selected_sheet = st.selectbox("Select sheet", sheet_names, key="interfund_sheet")
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    df = pd.read_csv(uploaded_file)
                interfund_df = df.copy()
                df.columns = df.columns.str.strip()
                # st.success(f"Loaded {len(df)} rows")
                # st.dataframe(df.head())

                interfund_df.columns = interfund_df.columns.str.strip()
                st.success("Interfund Bank Records loaded successfully!")
                st.dataframe(interfund_df.head())

                # Initialize column mapping if it doesn't exist
                if 'interfund_bank_col_mapping' not in st.session_state:
                    st.session_state.interfund_bank_col_mapping = {
                        'Application ID': 'Application ID',
                        'Amount (In Debit Account Currency)': 'Amount (In Debit Account Currency)',
                        'Credit Account Currency': 'Credit Account Currency',
                        'Intermediary Bank Account': 'Intermediary Bank Account',
                        'Status': 'Status'
                    }

                # Column mapping for Interfund Bank Records
                st.subheader("Interfund Bank Records Column Mapping")
                interfund_col_options = ['-- Select Column --'] + interfund_df.columns.tolist()
                col_mapping = {}

                # Define the required columns and their default/suggested mappings
                required_cols = {
                    'Application ID': 'Application ID',
                    'Amount (In Debit Account Currency)': 'Amount (In Debit Account Currency)',
                    'Credit Account Currency': 'Credit Account Currency',
                    'Intermediary Bank Account': 'Intermediary Bank Account',
                    'Status': 'Status'
                }

                for display_name, suggested_col in required_cols.items():
                    initial_selection = (
                        st.session_state.interfund_bank_col_mapping.get(display_name)
                        if display_name in st.session_state.interfund_bank_col_mapping
                        else suggested_col if suggested_col in interfund_col_options
                        else '-- Select Column --'
                    )
                    selected_col = st.selectbox(
                        f"Map '{display_name}' to:",
                        options=interfund_col_options,
                        index=interfund_col_options.index(initial_selection) if initial_selection in interfund_col_options else 0,
                        key=f"interfund_map_select_{display_name}"
                    )
                    col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None

                renamed_interfund_df = pd.DataFrame()
                mapped_columns_dict = {selected: original for original, selected in col_mapping.items() if selected and selected in interfund_df.columns}

                if mapped_columns_dict:
                    cols_to_keep = list(mapped_columns_dict.keys())
                    renamed_interfund_df = interfund_df[cols_to_keep].rename(columns=mapped_columns_dict)
                    interfund_df = renamed_interfund_df
                    st.success("Interfund Bank Records columns mapped successfully!")
                    st.dataframe(interfund_df.head())
                else:
                    st.warning("No Interfund Bank Records columns mapped. Proceeding with original column names.")

                st.session_state.interfund_bank_df = interfund_df
                st.session_state.interfund_source_df = interfund_df
                save_dataframe(interfund_df, "interfund_bank_df.pkl")
                st.session_state.interfund_bank_col_mapping = col_mapping
                save_object(col_mapping, "interfund_bank_col_mapping.pkl")

            except Exception as e:
                st.error(f"Error loading Interfund Bank Records: {e}")
    # Reconciliation Settings
    if 'interfund_source_df' in st.session_state and not st.session_state.interfund_source_df.empty:
        st.markdown("### ⚙️ Reconciliation Settings")
        debug_mode = st.checkbox("🐛 Debug Mode", key="interfund_debug")
        date_tolerance = st.slider("Date tolerance (days)", 0, 7, 3, key="interfund_tolerance")
        if st.button("🔄 Run Reconciliation", type="primary", use_container_width=True):
            if not all_bank_dfs:
                st.error("No bank statements loaded. Please upload bank statements first.")
            else:
                with st.spinner("Running reconciliation..."):
                    bank_dfs_copy = {k: df.copy() for k, df in all_bank_dfs.items()}
                    unmatched_list = []
                    matched_list = []
                    for idx, row in st.session_state.interfund_source_df.iterrows():
                        process_interfund_match(
                            row, bank_dfs_copy, unmatched_list, matched_list,
                            date_tolerance_days=date_tolerance,
                            debug_mode=debug_mode
                        )
                    unmatched_bank = []
                    for bank_key, bank_df in bank_dfs_copy.items():
                        if 'Matched' not in bank_df.columns:
                            bank_df['Matched'] = False
                        for _, row in bank_df[bank_df['Matched'] == False].iterrows():
                            if 'Debit' in bank_df.columns and safe_float(row['Debit']) is not None and safe_float(row['Debit']) > 0.01:
                                amount = safe_float(row['Debit'])
                                ttype = 'Debit'
                            elif 'Credit' in bank_df.columns and safe_float(row['Credit']) is not None and safe_float(row['Credit']) > 0.01:
                                amount = safe_float(row['Credit'])
                                ttype = 'Credit'
                            else:
                                continue
                            unmatched_bank.append({
                                'Bank Table': bank_key,
                                'Description': row.get('Description', ''),
                                'Transaction Type (Column)': ttype,
                                'Amount': amount
                            })
                    st.session_state.matched_interfund_df = add_unique_ids(pd.DataFrame(matched_list)) if matched_list else pd.DataFrame()
                    st.session_state.unmatched_interfund_df = add_unique_ids(pd.DataFrame(unmatched_list)) if unmatched_list else pd.DataFrame()
                    st.session_state.unmatched_bank_interfund = add_unique_ids(pd.DataFrame(unmatched_bank)) if unmatched_bank else pd.DataFrame()
                    for df_name in ['matched_interfund_df', 'unmatched_interfund_df', 'unmatched_bank_interfund']:
                        if not st.session_state[df_name].empty:
                            st.session_state[df_name] = add_audit_columns(st.session_state[df_name])
                    update_interfund_stats()
                    st.success("Reconciliation complete!")
                    st.rerun()

    # Dashboard and Results
    st.markdown("---")
    render_full_statistics_dashboard_interfund()

    move_targets_matched = {"Move to Unmatched": "Unmatched Interfund Records", "Move to Bank Unmatched": "Unmatched Bank Records (Interfund)"}
    move_targets_unmatched = {"Move to Matched": "Matched Interfund Records", "Move to Bank Unmatched": "Unmatched Bank Records (Interfund)"}
    move_targets_bank = {"Move to Matched": "Matched Interfund Records", "Move to Unmatched": "Unmatched Interfund Records"}

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ Matched", "❌ Unmatched", "🏦 Bank Unmatched", "📋 Moved Records", "🗑️ Deleted Records"])
    with tab1:
        def update_matched(df):
            st.session_state.matched_interfund_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.matched_interfund_df.empty:
                st.session_state.matched_interfund_df = add_audit_columns(st.session_state.matched_interfund_df)
            update_interfund_stats()
        render_editable_dataframe_interfund(
            st.session_state.get('matched_interfund_df', pd.DataFrame()),
            "Matched Interfund Records",
            "interfund_matched",
            on_data_change=update_matched,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_matched
        )
    with tab2:
        def update_unmatched(df):
            st.session_state.unmatched_interfund_df = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_interfund_df.empty:
                st.session_state.unmatched_interfund_df = add_audit_columns(st.session_state.unmatched_interfund_df)
            update_interfund_stats()
        render_editable_dataframe_interfund(
            st.session_state.get('unmatched_interfund_df', pd.DataFrame()),
            "Unmatched Interfund Records",
            "interfund_unmatched",
            on_data_change=update_unmatched,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_unmatched
        )
    with tab3:
        def update_bank(df):
            st.session_state.unmatched_bank_interfund = add_unique_ids(df) if not df.empty else df
            if not st.session_state.unmatched_bank_interfund.empty:
                st.session_state.unmatched_bank_interfund = add_audit_columns(st.session_state.unmatched_bank_interfund)
            update_interfund_stats()
        render_editable_dataframe_interfund(
            st.session_state.get('unmatched_bank_interfund', pd.DataFrame()),
            "Unmatched Bank Records (Interfund)",
            "interfund_bank",
            on_data_change=update_bank,
            show_delete=True,
            show_move=True,
            move_targets=move_targets_bank
        )
    with tab4:
        render_moved_records_tab()
    with tab5:
        render_deleted_records_tab()

    # Return dataframes for compatibility
    return (
        st.session_state.get('matched_interfund_df', pd.DataFrame()),
        st.session_state.get('unmatched_interfund_df', pd.DataFrame()),
        st.session_state.get('unmatched_bank_interfund', pd.DataFrame())
    )