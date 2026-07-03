# auth_system.py
import streamlit as st
import sqlite3
import pandas as pd
import hashlib
import uuid
from datetime import datetime, timedelta
import os
import json
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path


# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Database path - SINGLE SOURCE OF TRUTH
AUTH_DB_PATH = "data/auth.db"


def get_db_connection():
    """Get database connection - uses auth.db"""
    os.makedirs(os.path.dirname(AUTH_DB_PATH), exist_ok=True)
    return sqlite3.connect(AUTH_DB_PATH)


def init_auth_db():
    """Initialize authentication database with tables for users, sessions, and reconciliation data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'viewer',
            full_name TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expiry_time TIMESTAMP NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    # Reconciliation data versions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reconciliation_versions (
            version_id TEXT PRIMARY KEY,
            version_name TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (created_by) REFERENCES users(user_id)
        )
    ''')
    
    # Reconciliation data storage
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reconciliation_data (
            data_id TEXT PRIMARY KEY,
            version_id TEXT NOT NULL,
            data_type TEXT NOT NULL,
            data_json TEXT NOT NULL,
            data_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by TEXT,
            FOREIGN KEY (version_id) REFERENCES reconciliation_versions(version_id),
            FOREIGN KEY (updated_by) REFERENCES users(user_id)
        )
    ''')
    
    # Audit log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    # Add default admin user if not exists
    admin_exists = cursor.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
    if not admin_exists:
        admin_password = hash_password("admin123")
        admin_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO users (user_id, username, password_hash, role, full_name, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (admin_id, 'admin', admin_password, 'admin', 'System Administrator', 1))
    else:
        admin_id = cursor.execute("SELECT user_id FROM users WHERE username = 'admin'").fetchone()[0]
    
    # Create default version if none exists
    version_exists = cursor.execute("SELECT * FROM reconciliation_versions").fetchone()
    if not version_exists:
        default_version_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO reconciliation_versions (version_id, version_name, created_by, description, is_active)
            VALUES (?, ?, ?, ?, ?)
        ''', (default_version_id, 'Default Version', admin_id, 'Auto-created default version', 1))
    
    conn.commit()
    conn.close()
    print("Authentication database initialized successfully")


def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash


def create_user(username: str, password: str, email: str = None, role: str = 'viewer', full_name: str = None) -> bool:
    """Create a new user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        
        cursor.execute('''
            INSERT INTO users (user_id, username, password_hash, email, role, full_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, username, password_hash, email, role, full_name))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user and return user data"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_id, username, password_hash, role, full_name, is_active
        FROM users WHERE username = ?
    ''', (username,))
    
    user = cursor.fetchone()
    conn.close()
    
    if user and verify_password(password, user[2]) and user[5] == 1:
        return {
            'user_id': user[0],
            'username': user[1],
            'role': user[3],
            'full_name': user[4]
        }
    return None


def create_session(user_id: str, expiry_hours: int = 24) -> str:
    """Create a new session for user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    session_id = str(uuid.uuid4())
    token = str(uuid.uuid4())
    expiry_time = datetime.now() + timedelta(hours=expiry_hours)
    
    cursor.execute('''
        INSERT INTO sessions (session_id, user_id, token, expiry_time)
        VALUES (?, ?, ?, ?)
    ''', (session_id, user_id, token, expiry_time))
    
    # Update last login
    cursor.execute('''
        UPDATE users SET last_login = CURRENT_TIMESTAMP
        WHERE user_id = ?
    ''', (user_id,))
    
    conn.commit()
    conn.close()
    
    return token


def validate_session(token: str) -> Optional[Dict]:
    """Validate session token and return user data"""
    if not token:
        return None
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.user_id, s.token, u.username, u.role, u.full_name, s.expiry_time
        FROM sessions s
        JOIN users u ON s.user_id = u.user_id
        WHERE s.token = ? AND u.is_active = 1
    ''', (token,))
    
    session = cursor.fetchone()
    conn.close()
    
    if session:
        try:
            expiry_time = datetime.strptime(session[5], '%Y-%m-%d %H:%M:%S')
            if datetime.now() < expiry_time:
                return {
                    'user_id': session[0],
                    'username': session[2],
                    'role': session[3],
                    'full_name': session[4]
                }
        except:
            pass
    
    return None


def logout_user(token: str):
    """Invalidate user session"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))
    conn.commit()
    conn.close()
    
    # Clear session state
    if 'session_token' in st.session_state:
        del st.session_state['session_token']
    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']
    if 'user' in st.session_state:
        del st.session_state['user']


def cleanup_expired_sessions():
    """Remove expired sessions from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE expiry_time < ?', (datetime.now(),))
        conn.commit()
        conn.close()
    except:
        pass


def log_audit(user_id: str, action: str, details: str = None, ip_address: str = None):
    """Log user actions for audit trail"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        log_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO audit_log (log_id, user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?, ?)
        ''', (log_id, user_id, action, details, ip_address))
        
        conn.commit()
        conn.close()
    except:
        pass


def get_active_version_id() -> Optional[str]:
    """Get the active version ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version_id FROM reconciliation_versions WHERE is_active = 1
        ''')
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    except:
        return None


def get_all_versions() -> pd.DataFrame:
    """Get all reconciliation versions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.version_id, v.version_name, v.created_at, v.description, v.is_active,
                   u.username as created_by_username
            FROM reconciliation_versions v
            JOIN users u ON v.created_by = u.user_id
            ORDER BY v.created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return pd.DataFrame(results, columns=['version_id', 'version_name', 'created_at', 'description', 'is_active', 'created_by'])
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def get_reconciliation_history(data_type: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get historical reconciliation data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT rd.data_type, rd.data_date, rd.updated_at, rd.updated_by,
                   v.version_name, u.username as updated_by_username
            FROM reconciliation_data rd
            JOIN reconciliation_versions v ON rd.version_id = v.version_id
            LEFT JOIN users u ON rd.updated_by = u.user_id
            WHERE 1=1
        '''
        params = []
        
        if data_type:
            query += ' AND rd.data_type = ?'
            params.append(data_type)
        
        if start_date:
            query += ' AND rd.data_date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND rd.data_date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY rd.data_date DESC, rd.updated_at DESC'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return pd.DataFrame(results, columns=['data_type', 'data_date', 'updated_at', 'updated_by', 'version_name', 'updated_by_username'])
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def get_reconciliation_history_by_date_range(data_type: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Get historical reconciliation data by date range for all modules"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT rd.data_id, rd.data_type, rd.data_date, rd.updated_at, rd.updated_by,
                   v.version_name, v.version_id, v.is_active,
                   u.username as updated_by_username,
                   CASE 
                       WHEN rd.data_type LIKE '%matched%' THEN 'Matched'
                       WHEN rd.data_type LIKE '%unmatched%' THEN 'Unmatched'
                       ELSE 'Other'
                   END as match_status,
                   CASE
                       WHEN rd.data_type LIKE '%adjustments%' OR rd.data_type LIKE '%fx_reconcilliation%' OR rd.data_type LIKE '%matched_local%' OR rd.data_type LIKE '%unmatched_local%' THEN 'FX Adjustments'
                       WHEN rd.data_type LIKE '%counterparty%' OR rd.data_type LIKE '%choice%' OR rd.data_type LIKE '%trade%' OR rd.data_type LIKE '%matched_buy%' OR rd.data_type LIKE '%matched_sell%' THEN 'FX Trade'
                       WHEN rd.data_type LIKE '%intermediary%' THEN 'Intermediary'
                       WHEN rd.data_type LIKE '%interfund%' THEN 'Interfund'
                       WHEN rd.data_type LIKE '%business%' OR rd.data_type LIKE '%final_business%' THEN 'Business FX'
                       WHEN rd.data_type LIKE '%cross%' OR rd.data_type LIKE '%bank_records%' OR rd.data_type LIKE '%newly_matched%' OR rd.data_type LIKE '%still_unmatched%' THEN 'Cross-Match'
                       WHEN rd.data_type LIKE '%mpesa%' THEN 'M-Pesa'
                       ELSE 'Other'
                   END as module
            FROM reconciliation_data rd
            JOIN reconciliation_versions v ON rd.version_id = v.version_id
            LEFT JOIN users u ON rd.updated_by = u.user_id
            WHERE 1=1
        '''
        params = []
        
        if data_type and data_type != 'All':
            query += ' AND rd.data_type = ?'
            params.append(data_type)
        
        if start_date:
            query += ' AND rd.data_date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND rd.data_date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY rd.data_date DESC, rd.updated_at DESC'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return pd.DataFrame(results, columns=['data_id', 'data_type', 'data_date', 'updated_at', 'updated_by', 
                                                   'version_name', 'version_id', 'is_active', 'updated_by_username',
                                                   'match_status', 'module'])
        return pd.DataFrame()
    except Exception as e:
        print(f"Error getting reconciliation history: {e}")
        return pd.DataFrame()


def get_available_data_types() -> List[str]:
    """Get all available data types from reconciliation data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT data_type 
            FROM reconciliation_data 
            WHERE data_type NOT LIKE '%_json'
            ORDER BY data_type
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        data_types = ['All'] + [row[0] for row in results if row[0]]
        return data_types
    except Exception as e:
        print(f"Error getting data types: {e}")
        return ['All']


def get_available_modules() -> List[str]:
    """Get all available modules from reconciliation data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT 
                CASE
                    WHEN data_type LIKE '%adjustments%' OR data_type LIKE '%fx_reconcilliation%' OR data_type LIKE '%matched_local%' OR data_type LIKE '%unmatched_local%' THEN 'FX Adjustments'
                    WHEN data_type LIKE '%counterparty%' OR data_type LIKE '%choice%' OR data_type LIKE '%trade%' OR data_type LIKE '%matched_buy%' OR data_type LIKE '%matched_sell%' THEN 'FX Trade'
                    WHEN data_type LIKE '%intermediary%' THEN 'Intermediary'
                    WHEN data_type LIKE '%interfund%' THEN 'Interfund'
                    WHEN data_type LIKE '%business%' OR data_type LIKE '%final_business%' THEN 'Business FX'
                    WHEN data_type LIKE '%cross%' OR data_type LIKE '%bank_records%' OR data_type LIKE '%newly_matched%' OR data_type LIKE '%still_unmatched%' THEN 'Cross-Match'
                    WHEN data_type LIKE '%mpesa%' THEN 'M-Pesa'
                    ELSE 'Other'
                END as module
            FROM reconciliation_data
            WHERE data_type NOT LIKE '%_json'
            ORDER BY module
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        modules = ['All'] + sorted(list(set([row[0] for row in results if row[0]])))
        return modules
    except Exception as e:
        print(f"Error getting modules: {e}")
        return ['All']


def load_results_by_date(data_date: str, module_filter: str = None) -> Dict[str, Any]:
    """Load all results for a specific date"""
    results = {}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT rd.data_type, rd.data_json, rd.data_date, rd.updated_at, rd.updated_by,
                   v.version_name
            FROM reconciliation_data rd
            JOIN reconciliation_versions v ON rd.version_id = v.version_id
            WHERE rd.data_date = ?
        '''
        params = [data_date]
        
        if module_filter and module_filter != 'All':
            module_condition = """
                AND (
                    (? = 'FX Adjustments' AND (data_type LIKE '%adjustments%' OR data_type LIKE '%fx_reconcilliation%' OR data_type LIKE '%matched_local%' OR data_type LIKE '%unmatched_local%'))
                    OR (? = 'FX Trade' AND (data_type LIKE '%counterparty%' OR data_type LIKE '%choice%' OR data_type LIKE '%trade%' OR data_type LIKE '%matched_buy%' OR data_type LIKE '%matched_sell%'))
                    OR (? = 'Intermediary' AND data_type LIKE '%intermediary%')
                    OR (? = 'Interfund' AND data_type LIKE '%interfund%')
                    OR (? = 'Business FX' AND (data_type LIKE '%business%' OR data_type LIKE '%final_business%'))
                    OR (? = 'Cross-Match' AND (data_type LIKE '%cross%' OR data_type LIKE '%bank_records%' OR data_type LIKE '%newly_matched%' OR data_type LIKE '%still_unmatched%'))
                    OR (? = 'M-Pesa' AND data_type LIKE '%mpesa%')
                )
            """
            params.extend([module_filter] * 7)
            query += module_condition
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            data_type = row[0]
            data_json = row[1]
            clean_key = data_type.replace('_json', '')
            
            try:
                df = pd.read_json(data_json, orient='split')
                results[clean_key] = {
                    'type': 'dataframe',
                    'data': df,
                    'data_date': row[2],
                    'updated_at': row[3],
                    'updated_by': row[4],
                    'version_name': row[5],
                    'original_key': data_type
                }
            except:
                try:
                    data = json.loads(data_json)
                    results[clean_key] = {
                        'type': 'dict',
                        'data': data,
                        'data_date': row[2],
                        'updated_at': row[3],
                        'updated_by': row[4],
                        'version_name': row[5],
                        'original_key': data_type
                    }
                except:
                    pass
        
        return results
    except Exception as e:
        print(f"Error loading results by date: {e}")
        return {}


def load_results_by_date_range(start_date: str, end_date: str, module_filter: str = None) -> Dict[str, List[Dict]]:
    """Load all results for a date range"""
    results = {}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT rd.data_type, rd.data_json, rd.data_date, rd.updated_at, rd.updated_by,
                   v.version_name
            FROM reconciliation_data rd
            JOIN reconciliation_versions v ON rd.version_id = v.version_id
            WHERE rd.data_date BETWEEN ? AND ?
        '''
        params = [start_date, end_date]
        
        if module_filter and module_filter != 'All':
            module_condition = """
                AND (
                    (? = 'FX Adjustments' AND (data_type LIKE '%adjustments%' OR data_type LIKE '%fx_reconcilliation%' OR data_type LIKE '%matched_local%' OR data_type LIKE '%unmatched_local%'))
                    OR (? = 'FX Trade' AND (data_type LIKE '%counterparty%' OR data_type LIKE '%choice%' OR data_type LIKE '%trade%' OR data_type LIKE '%matched_buy%' OR data_type LIKE '%matched_sell%'))
                    OR (? = 'Intermediary' AND data_type LIKE '%intermediary%')
                    OR (? = 'Interfund' AND data_type LIKE '%interfund%')
                    OR (? = 'Business FX' AND (data_type LIKE '%business%' OR data_type LIKE '%final_business%'))
                    OR (? = 'Cross-Match' AND (data_type LIKE '%cross%' OR data_type LIKE '%bank_records%' OR data_type LIKE '%newly_matched%' OR data_type LIKE '%still_unmatched%'))
                    OR (? = 'M-Pesa' AND data_type LIKE '%mpesa%')
                )
            """
            params.extend([module_filter] * 7)
            query += module_condition
        
        query += ' ORDER BY rd.data_date DESC, rd.updated_at DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            data_type = row[0]
            data_json = row[1]
            clean_key = data_type.replace('_json', '')
            
            if clean_key not in results:
                results[clean_key] = []
            
            try:
                df = pd.read_json(data_json, orient='split')
                results[clean_key].append({
                    'type': 'dataframe',
                    'data': df,
                    'data_date': row[2],
                    'updated_at': row[3],
                    'updated_by': row[4],
                    'version_name': row[5],
                    'original_key': data_type
                })
            except:
                try:
                    data = json.loads(data_json)
                    results[clean_key].append({
                        'type': 'dict',
                        'data': data,
                        'data_date': row[2],
                        'updated_at': row[3],
                        'updated_by': row[4],
                        'version_name': row[5],
                        'original_key': data_type
                    })
                except:
                    pass
        
        return results
    except Exception as e:
        print(f"Error loading results by date range: {e}")
        return {}


def restore_session_from_loaded_results(loaded_results: Dict) -> int:
    """Restore session state from loaded results"""
    restored_count = 0
    
    for key, value in loaded_results.items():
        if isinstance(value, dict):
            if value.get('type') == 'dataframe':
                st.session_state[key] = value['data']
                restored_count += 1
                st.session_state[f"{key}_metadata"] = {
                    'data_date': value.get('data_date'),
                    'updated_at': value.get('updated_at'),
                    'updated_by': value.get('updated_by'),
                    'version_name': value.get('version_name')
                }
            elif value.get('type') == 'dict':
                st.session_state[key] = value['data']
                restored_count += 1
    
    return restored_count


def get_user_list() -> pd.DataFrame:
    """Get list of all users"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, email, role, full_name, is_active, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return pd.DataFrame(results, columns=['user_id', 'username', 'email', 'role', 'full_name', 'is_active', 'created_at', 'last_login'])
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def update_user_role(user_id: str, role: str) -> bool:
    """Update user role"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET role = ? WHERE user_id = ?', (role, user_id))
        conn.commit()
        conn.close()
        return True
    except:
        return False


def delete_user(user_id: str) -> bool:
    """Soft delete user (deactivate)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_active = 0 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        return True
    except:
        return False


def login_ui():
    """Display a modern login form with fintech styling."""
    
    # Ensure database is initialized
    init_auth_db()
    
    # Custom CSS for login page
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
    }
    .login-card {
        background: #1e293b;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
    .login-title {
        font-size: 1.75rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #38bdf8, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .login-subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-size: 0.875rem;
    }
    .demo-badge {
        background: #020617;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-top: 1rem;
        font-family: monospace;
        font-size: 0.75rem;
        color: #64748b;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        border: none;
        padding: 0.6rem;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-card">
                <div class="login-title">ChoiceBank</div>
                <div class="login-subtitle">FX Reconciliation Dashboard</div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter your username", key="login_username")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
        
        if st.button("Sign In", use_container_width=True, key="login_btn"):
            if username and password:
                user = authenticate_user(username, password)
                if user:
                    token = create_session(user['user_id'])
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = user
                    st.session_state['session_token'] = token
                    log_audit(user['user_id'], 'LOGIN', f'User {username} logged in')
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
        
        st.markdown(f"""
                <div class="demo-badge">
                    Demo credentials: admin / admin123
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def require_auth(func):
    """Decorator to require authentication for pages - persists across refreshes"""
    def wrapper(*args, **kwargs):
        cleanup_expired_sessions()
        
        if st.session_state.get('authenticated', False):
            return func(*args, **kwargs)
        
        if 'session_token' in st.session_state:
            user = validate_session(st.session_state['session_token'])
            if user:
                st.session_state['authenticated'] = True
                st.session_state['user'] = user
                return func(*args, **kwargs)
            else:
                if 'session_token' in st.session_state:
                    del st.session_state['session_token']
        
        login_ui()
        return None
    return wrapper


def require_role(required_role):
    """Decorator to require specific role"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
                login_ui()
                return None
            user_role = st.session_state['user']['role']
            roles_order = ['admin', 'editor', 'viewer']
            if roles_order.index(user_role) <= roles_order.index(required_role):
                return func(*args, **kwargs)
            else:
                st.error(f"Access denied. {required_role} role or higher required.")
                return None
        return wrapper
    return decorator


def user_management_ui():
    """User management interface for admin users"""
    st.subheader("👥 User Management")
    
    tab1, tab2, tab3 = st.tabs(["Users List", "Add User", "User Activity"])
    
    with tab1:
        users_df = get_user_list()
        if not users_df.empty:
            st.dataframe(users_df[['username', 'email', 'role', 'full_name', 'is_active', 'created_at']], use_container_width=True)
            
            st.subheader("Update User Role")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                selected_user = st.selectbox("Select User", users_df['username'].tolist())
            with col2:
                new_role = st.selectbox("New Role", ['viewer', 'editor', 'admin'])
            with col3:
                if st.button("Update Role"):
                    user_id = users_df[users_df['username'] == selected_user]['user_id'].iloc[0]
                    if update_user_role(user_id, new_role):
                        st.success(f"Updated {selected_user} role to {new_role}")
                        st.rerun()
    
    with tab2:
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_email = st.text_input("Email")
            new_role = st.selectbox("Role", ['viewer', 'editor', 'admin'])
            new_full_name = st.text_input("Full Name")
            
            if st.form_submit_button("Create User"):
                if create_user(new_username, new_password, new_email, new_role, new_full_name):
                    st.success(f"User {new_username} created successfully!")
                    st.rerun()
                else:
                    st.error("Username already exists!")
    
    with tab3:
        st.info("User activity logs would be displayed here")


def date_based_retrieval_ui():
    """Enhanced UI for retrieving reconciliation data by date or date range for all modules"""
    
    st.subheader("📅 Load Historical Reconciliation Data")
    
    st.markdown("""
    This tool allows you to load previously saved reconciliation results from any module 
    by selecting a specific date or date range. You can filter by module and data type.
    """)
    
    # Get available filters
    available_modules = get_available_modules()
    available_data_types = get_available_data_types()
    
    # Search mode selection
    search_mode = st.radio(
        "Search Mode",
        ["Single Date", "Date Range"],
        horizontal=True,
        help="Choose between loading data for a single date or a date range"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        module_filter = st.selectbox(
            "Filter by Module",
            options=available_modules,
            help="Select a specific module or 'All' to see results from all modules"
        )
    
    with col2:
        data_type_filter = st.selectbox(
            "Filter by Data Type",
            options=available_data_types,
            help="Select a specific data type or 'All' to see all types"
        )
    
    if search_mode == "Single Date":
        selected_date = st.date_input(
            "Select Date",
            value=None,
            help="Choose the date of the reconciliation results you want to load"
        )
        
        if selected_date and st.button("Load Results for Selected Date", type="primary"):
            with st.spinner(f"Loading results for {selected_date}..."):
                loaded_results = load_results_by_date(
                    str(selected_date), 
                    module_filter if module_filter != 'All' else None
                )
                
                if loaded_results:
                    st.success(f"✅ Loaded {len(loaded_results)} result datasets for {selected_date}")
                    
                    # Display loaded results in expanders
                    st.subheader("📊 Loaded Results")
                    
                    for data_key, data_info in loaded_results.items():
                        with st.expander(f"📁 {data_key} - {data_info.get('data_date', 'N/A')} (Updated: {data_info.get('updated_at', 'N/A')})", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.caption(f"**Updated By:** {data_info.get('updated_by', 'Unknown')}")
                                st.caption(f"**Version:** {data_info.get('version_name', 'N/A')}")
                            
                            with col2:
                                if data_info.get('type') == 'dataframe':
                                    df = data_info['data']
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label=f"📥 Download {data_key} as CSV",
                                        data=csv,
                                        file_name=f"{data_key}_{selected_date}.csv",
                                        mime="text/csv",
                                        key=f"download_{data_key}_{selected_date}"
                                    )
                            
                            if data_info.get('type') == 'dataframe':
                                st.dataframe(data_info['data'], use_container_width=True)
                                st.caption(f"**Shape:** {data_info['data'].shape[0]} rows × {data_info['data'].shape[1]} columns")
                            elif data_info.get('type') == 'dict':
                                st.json(data_info['data'])
                    
                    # Option to restore to session
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Restore ALL to Current Session", key="restore_all_single"):
                            restored_count = restore_session_from_loaded_results(loaded_results)
                            st.success(f"✅ Restored {restored_count} datasets to current session!")
                            st.info("You can now navigate to the respective reconciliation modules to view the loaded data.")
                            
                            if 'user' in st.session_state:
                                log_audit(
                                    st.session_state['user']['user_id'],
                                    'LOAD_HISTORICAL_DATA',
                                    f"Loaded {restored_count} datasets from {selected_date} with module filter: {module_filter}"
                                )
                    
                    with col2:
                        selected_items = st.multiselect(
                            "Or select specific datasets to restore",
                            options=list(loaded_results.keys()),
                            help="Choose which specific datasets to restore"
                        )
                        if selected_items and st.button("Restore Selected"):
                            selected_results = {k: loaded_results[k] for k in selected_items}
                            restored_count = restore_session_from_loaded_results(selected_results)
                            st.success(f"✅ Restored {restored_count} selected datasets!")
                else:
                    st.warning(f"No results found for {selected_date} with the selected filters.")
    
    else:
        # Date Range mode
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=None)
        
        with col2:
            end_date = st.date_input("End Date", value=None)
        
        if start_date and end_date:
            if start_date > end_date:
                st.error("Start date must be before end date")
            else:
                if st.button("Load Results for Date Range", type="primary"):
                    with st.spinner(f"Loading results from {start_date} to {end_date}..."):
                        loaded_results = load_results_by_date_range(
                            str(start_date), 
                            str(end_date),
                            module_filter if module_filter != 'All' else None
                        )
                        
                        if loaded_results:
                            total_records = sum(len(items) for items in loaded_results.values())
                            st.success(f"✅ Loaded {len(loaded_results)} data types with {total_records} total records")
                            
                            # Display summary
                            with st.expander("📋 Summary of Loaded Results", expanded=True):
                                summary_data = []
                                for data_type, items in loaded_results.items():
                                    for item in items:
                                        summary_data.append({
                                            'Data Type': data_type,
                                            'Date': item.get('data_date', 'N/A'),
                                            'Updated At': item.get('updated_at', 'N/A'),
                                            'Updated By': item.get('updated_by', 'N/A'),
                                            'Version': item.get('version_name', 'N/A')
                                        })
                                
                                if summary_data:
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    csv_summary = summary_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 Download Summary as CSV",
                                        data=csv_summary,
                                        file_name=f"historical_data_summary_{start_date}_to_{end_date}.csv",
                                        mime="text/csv"
                                    )
                            
                            # Display loaded results by date
                            st.subheader("📊 Loaded Results by Date")
                            
                            for data_type, items in loaded_results.items():
                                for item in items:
                                    expander_title = f"📁 {data_type} - {item.get('data_date', 'N/A')} (Updated: {item.get('updated_at', 'N/A')})"
                                    with st.expander(expander_title, expanded=False):
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.caption(f"**Updated By:** {item.get('updated_by', 'Unknown')}")
                                            st.caption(f"**Version:** {item.get('version_name', 'N/A')}")
                                        
                                        with col2:
                                            if item.get('type') == 'dataframe':
                                                df = item['data']
                                                csv = df.to_csv(index=False).encode('utf-8')
                                                st.download_button(
                                                    label=f"📥 Download {data_type} as CSV",
                                                    data=csv,
                                                    file_name=f"{data_type}_{item.get('data_date', 'date')}.csv",
                                                    mime="text/csv",
                                                    key=f"download_{data_type}_{item.get('data_date', 'date')}"
                                                )
                                        
                                        if item.get('type') == 'dataframe':
                                            st.dataframe(item['data'], use_container_width=True)
                                            st.caption(f"**Shape:** {item['data'].shape[0]} rows × {item['data'].shape[1]} columns")
                                        elif item.get('type') == 'dict':
                                            st.json(item['data'])
                            
                            # Option to restore
                            st.subheader("Restore to Session")
                            
                            restore_options = []
                            for data_type, items in loaded_results.items():
                                for i, item in enumerate(items):
                                    option_label = f"{data_type} ({item.get('data_date', 'N/A')})"
                                    restore_options.append({
                                        'label': option_label,
                                        'key': f"{data_type}_{i}",
                                        'data_type': data_type,
                                        'date': item.get('data_date', 'N/A'),
                                        'item': item
                                    })
                            
                            if restore_options:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button("🔄 Restore ALL to Current Session", key="restore_all_range"):
                                        all_items = {}
                                        for opt in restore_options:
                                            all_items[f"{opt['data_type']}_{opt['date']}"] = opt['item']
                                        restored_count = restore_session_from_loaded_results(all_items)
                                        st.success(f"✅ Restored {restored_count} datasets to current session!")
                                        
                                        if 'user' in st.session_state:
                                            log_audit(
                                                st.session_state['user']['user_id'],
                                                'LOAD_HISTORICAL_DATA_RANGE',
                                                f"Loaded {restored_count} datasets from {start_date} to {end_date}"
                                            )
                                
                                with col2:
                                    selected_options = st.multiselect(
                                        "Select specific datasets to restore",
                                        options=[opt['label'] for opt in restore_options]
                                    )
                                    
                                    if selected_options and st.button("Restore Selected"):
                                        selected_items = {}
                                        for opt in restore_options:
                                            if opt['label'] in selected_options:
                                                selected_items[f"{opt['data_type']}_{opt['date']}"] = opt['item']
                                        restored_count = restore_session_from_loaded_results(selected_items)
                                        st.success(f"✅ Restored {restored_count} selected datasets!")
                        else:
                            st.warning(f"No results found from {start_date} to {end_date} with the selected filters.")
    
    st.markdown("---")
    
    # Show recent history
    st.subheader("📊 Recent Reconciliation History (Last 30 Days)")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    history_df = get_reconciliation_history_by_date_range(start_date=start_date, end_date=end_date)
    
    if not history_df.empty:
        summary = history_df.groupby(['data_date', 'module']).size().reset_index(name='record_count')
        summary = summary.sort_values('data_date', ascending=False)
        
        st.dataframe(summary, use_container_width=True)
        
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete History (Last 30 Days)",
            data=csv,
            file_name="reconciliation_history_30days.csv",
            mime="text/csv"
        )
    else:
        st.info("No reconciliation history found in the last 30 days.")


def is_authenticated() -> bool:
    """Check if user is authenticated, with session restoration"""
    if st.session_state.get('authenticated', False):
        return True
    
    if 'session_token' in st.session_state:
        user = validate_session(st.session_state['session_token'])
        if user:
            st.session_state['authenticated'] = True
            st.session_state['user'] = user
            return True
        else:
            if 'session_token' in st.session_state:
                del st.session_state['session_token']
    
    return False


# Import these from other modules
def save_all_reconciliation_results_incremental(version_id, user_id, change_tracker=None):
    """Placeholder - import from main_dashboard.py"""
    from main_dashboard import save_all_reconciliation_results_incremental as _save
    return _save(version_id, user_id, change_tracker)