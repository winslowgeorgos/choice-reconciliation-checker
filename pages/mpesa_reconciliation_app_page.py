# mpesa_gl_reconciliation_page.py (complete fixed version)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import io
import uuid
import os
import pickle
from openpyxl import load_workbook
import warnings
from concurrent.futures import ThreadPoolExecutor
import hashlib
warnings.filterwarnings('ignore')

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Performance Configuration ---
DEBUG_MODE = True
CHUNK_SIZE = 10000  # For processing large files in chunks
DATE_MATCH_TOLERANCE_DAYS = 2  # Allow 2-day tolerance for date matching
AMOUNT_TOLERANCE = 1.0  # 1 KES tolerance
USE_MULTITHREADING = True  # Enable for large datasets
MAX_WORKERS = 4  # Number of threads for parallel processing

def debug_print(message, level="INFO"):
    """Print debug messages with timestamps."""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{level}] {message}")

# --- Optimized Helper Functions ---
def generate_data_hash(data):
    """Generate hash for data to avoid redundant processing."""
    if isinstance(data, pd.DataFrame):
        # Use a combination of shape and first/last rows for hash
        sample = str(data.shape) + str(data.iloc[0].to_dict() if len(data) > 0 else "") + str(data.iloc[-1].to_dict() if len(data) > 0 else "")
        return hashlib.md5(sample.encode()).hexdigest()
    elif isinstance(data, bytes):
        return hashlib.md5(data).hexdigest()
    return hashlib.md5(str(data).encode()).hexdigest()

def safe_float_vectorized(series):
    """Vectorized version of safe_float for better performance."""
    def convert_value(x):
        if pd.isna(x) or x is None:
            return None
        try:
            cleaned = str(x).replace(',', '').strip()
            cleaned = re.sub(r'[^\d\.\-]', '', cleaned)
            if cleaned == '' or cleaned == '-':
                return None
            return abs(float(cleaned))
        except (ValueError, TypeError):
            return None
    
    # Apply in chunks for large series
    if len(series) > CHUNK_SIZE:
        result = []
        for i in range(0, len(series), CHUNK_SIZE):
            chunk = series.iloc[i:i+CHUNK_SIZE]
            result_chunk = chunk.apply(convert_value)
            result.append(result_chunk)
        return pd.concat(result, ignore_index=True)
    else:
        return series.apply(convert_value)

def parse_date_vectorized(date_series):
    """Vectorized date parsing for better performance."""
    date_formats = [
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%Y-%m-%d'
    ]
    
    def parse_single_date(date_str):
        if pd.isna(date_str):
            return None
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        return None
    
    # Apply in chunks for large series
    if len(date_series) > CHUNK_SIZE:
        result = []
        for i in range(0, len(date_series), CHUNK_SIZE):
            chunk = date_series.iloc[i:i+CHUNK_SIZE]
            result_chunk = chunk.apply(parse_single_date)
            result.append(result_chunk)
        return pd.concat(result, ignore_index=True)
    else:
        return date_series.apply(parse_single_date)

def extract_receipts_vectorized(description_series):
    """Vectorized receipt extraction."""
    pattern = re.compile(r'([A-Z]{2,}[A-Z0-9]{6,})')
    
    def extract_single(desc):
        if pd.isna(desc):
            return None
        match = pattern.search(str(desc))
        return match.group(1) if match else None
    
    return description_series.apply(extract_single)

# --- Optimized File Processing ---
def detect_header_row_optimized(df, max_scan_rows=20):
    """Optimized header detection."""
    debug_print(f"detect_header_row_optimized: shape={df.shape}")
    
    if len(df) == 0:
        return 0
    
    # Take sample of rows for faster processing
    sample_size = min(max_scan_rows * 2, len(df))
    df_sample = df.iloc[:sample_size]
    
    keywords = [
        "id", "date", "time", "amount", "account",
        "debit", "credit", "reference", "status",
        "name", "number", "currency", "receipt",
        "completion", "initiation", "transaction", "balance"
    ]
    
    scores = []
    for i in range(min(max_scan_rows, len(df_sample))):
        row = df_sample.iloc[i]
        values = row.astype(str)
        
        # Fast checks
        non_empty_count = values[values.str.strip() != ""].count()
        if non_empty_count < 3:  # Header should have several non-empty values
            scores.append(0)
            continue
        
        # Check for keywords
        keyword_hits = 0
        for val in values:
            val_lower = val.lower()
            for kw in keywords:
                if kw in val_lower:
                    keyword_hits += 1
                    break
        
        # Calculate score
        score = non_empty_count + (keyword_hits * 2)
        scores.append(score)
    
    best_row = np.argmax(scores) if scores else 0
    debug_print(f"Best header row: {best_row} with score {scores[best_row] if scores else 0}")
    return best_row

def load_with_auto_header_optimized(file_content, file_name):
    """Optimized header detection and loading."""
    debug_print(f"load_with_auto_header_optimized: {file_name}")
    
    # Read only first 1000 rows initially for header detection
    if file_name.endswith(".csv"):
        df_sample = pd.read_csv(io.BytesIO(file_content), header=None, nrows=1000)
    else:
        df_sample = pd.read_excel(io.BytesIO(file_content), header=None, nrows=1000)
    
    header_row = detect_header_row_optimized(df_sample)
    
    # Now read full file with proper header
    if file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_content), header=header_row, low_memory=False)
    else:
        df = pd.read_excel(io.BytesIO(file_content), header=header_row)
    
    # Clean up
    df = df.loc[:, ~df.columns.str.lower().str.contains('unnamed')]
    df = df.dropna(axis=1, how='all')
    
    # Convert object columns to appropriate types to reduce memory
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    debug_print(f"Loaded DataFrame shape: {df.shape}")
    return df

# --- Optimized M-Pesa Processing ---
def parse_mpesa_csv_optimized(file_content):
    """Optimized M-Pesa CSV parsing."""
    debug_print("parse_mpesa_csv_optimized")
    
    # Find header quickly
    lines = file_content.split('\n', 50)[:50]  # Only check first 50 lines
    header_line = None
    for i, line in enumerate(lines):
        if 'Receipt No.' in line:
            header_line = i
            break
    
    if header_line is None:
        # Try to find header by scanning more lines
        debug_print("Header not found in first 50 lines, scanning more")
        more_lines = file_content.split('\n', 200)[:200]
        for i, line in enumerate(more_lines):
            if any(col in line for col in ['Receipt', 'Completion', 'Initiation', 'Status']):
                header_line = i
                break
    
    # Read CSV with header
    try:
        df = pd.read_csv(io.StringIO(file_content), skiprows=header_line if header_line else 0, 
                        low_memory=False, encoding='utf-8')
    except:
        # Fallback reading
        df = pd.read_csv(io.StringIO(file_content), low_memory=False, encoding='utf-8')
    
    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        if pd.isna(col):
            continue
        col_str = str(col).lower()
        if 'receipt' in col_str:
            column_mapping[col] = 'Receipt No.'
        elif 'completion' in col_str and 'time' in col_str:
            column_mapping[col] = 'Completion Time'
        elif 'initiation' in col_str and 'time' in col_str:
            column_mapping[col] = 'Initiation Time'
        elif 'status' in col_str and 'transaction' in col_str:
            column_mapping[col] = 'Transaction Status'
        elif 'paid' in col_str and 'in' in col_str:
            column_mapping[col] = 'Paid In'
        elif 'withdrawn' in col_str:
            column_mapping[col] = 'Withdrawn'
        elif 'balance' in col_str and 'confirmed' not in col_str:
            column_mapping[col] = 'Balance'
        elif 'balance' in col_str and 'confirmed' in col_str:
            column_mapping[col] = 'Balance Confirmed'
        elif 'equivalent' in col_str:
            column_mapping[col] = 'KES Equivalent'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Parse dates and amounts
    if 'Completion Time' in df.columns:
        df['Completion_Date'] = parse_date_vectorized(df['Completion Time'])
    
    if 'KES Equivalent' in df.columns:
        df['Amount_Clean'] = safe_float_vectorized(df['KES Equivalent'])
    elif 'Withdrawn' in df.columns:
        df['Amount_Clean'] = safe_float_vectorized(df['Withdrawn'])
    elif 'Paid In' in df.columns:
        df['Amount_Clean'] = safe_float_vectorized(df['Paid In'])
    
    return df

def parse_mpesa_excel(file_content, file_name):
    """Parse M-Pesa Excel file with metadata header."""
    debug_print(f"parse_mpesa_excel called: {file_name}")
    try:
        # Read the Excel file
        xls = pd.ExcelFile(io.BytesIO(file_content))
        debug_print(f"Excel sheets: {xls.sheet_names}")
        
        # Try each sheet
        for sheet_name in xls.sheet_names:
            df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            debug_print(f"Reading sheet '{sheet_name}', raw shape: {df_raw.shape}")
            
            # Find the row containing 'Receipt No.'
            header_row = None
            for idx, row in df_raw.iterrows():
                row_str = ' '.join([str(cell) for cell in row.values if pd.notna(cell)])
                if 'Receipt No.' in row_str:
                    header_row = idx
                    debug_print(f"Found 'Receipt No.' in sheet '{sheet_name}' at row {idx}")
                    break
            
            if header_row is not None:
                # Read again with proper header
                df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
                debug_print(f"Read sheet '{sheet_name}' with header row {header_row}, shape: {df.shape}")
                
                # Clean column names
                df.columns = df.columns.str.strip()
                debug_print(f"Columns after cleaning: {list(df.columns)}")
                
                # Standardize column names
                column_mapping = {}
                for col in df.columns:
                    if pd.isna(col):
                        continue
                    col_str = str(col)
                    col_lower = col_str.lower()
                    if 'receipt' in col_lower:
                        column_mapping[col] = 'Receipt No.'
                    elif 'completion' in col_lower and 'time' in col_lower:
                        column_mapping[col] = 'Completion Time'
                    elif 'initiation' in col_lower and 'time' in col_lower:
                        column_mapping[col] = 'Initiation Time'
                    elif 'status' in col_lower:
                        column_mapping[col] = 'Transaction Status'
                    elif 'paid' in col_lower and 'in' in col_lower:
                        column_mapping[col] = 'Paid In'
                    elif 'withdrawn' in col_lower:
                        column_mapping[col] = 'Withdrawn'
                    elif 'balance' in col_lower and not 'confirmed' in col_lower:
                        column_mapping[col] = 'Balance'
                    elif 'balance' in col_lower and 'confirmed' in col_lower:
                        column_mapping[col] = 'Balance Confirmed'
                    elif 'equivalent' in col_lower:
                        column_mapping[col] = 'KES Equivalent'
                
                if column_mapping:
                    debug_print(f"Column mapping applied: {column_mapping}")
                    df = df.rename(columns=column_mapping)
                
                # Parse dates and amounts for consistency
                if 'Completion Time' in df.columns:
                    df['Completion_Date'] = parse_date_vectorized(df['Completion Time'])
                
                if 'KES Equivalent' in df.columns:
                    df['Amount_Clean'] = safe_float_vectorized(df['KES Equivalent'])
                elif 'Withdrawn' in df.columns:
                    df['Amount_Clean'] = safe_float_vectorized(df['Withdrawn'])
                elif 'Paid In' in df.columns:
                    df['Amount_Clean'] = safe_float_vectorized(df['Paid In'])
                
                debug_print(f"Returning DataFrame from sheet '{sheet_name}', shape: {df.shape}")
                return df
        
        debug_print("No header found in any sheet, trying default read")
        # If no header found, try reading with first row as header
        df = pd.read_excel(xls, sheet_name=0)
        
        # Parse dates and amounts
        if 'Completion Time' in df.columns:
            df['Completion_Date'] = parse_date_vectorized(df['Completion Time'])
        
        if 'KES Equivalent' in df.columns:
            df['Amount_Clean'] = safe_float_vectorized(df['KES Equivalent'])
        elif 'Withdrawn' in df.columns:
            df['Amount_Clean'] = safe_float_vectorized(df['Withdrawn'])
        elif 'Paid In' in df.columns:
            df['Amount_Clean'] = safe_float_vectorized(df['Paid In'])
        
        debug_print(f"Default read successful, shape: {df.shape}")
        return df
        
    except Exception as e:
        debug_print(f"Error parsing M-Pesa Excel: {e}", "ERROR")
        st.error(f"Error parsing M-Pesa Excel: {e}")
        return pd.DataFrame()

# --- Optimized Reconciliation with Date Matching ---
def preprocess_for_matching(df, df_type, system_type):
    """Preprocess DataFrame for efficient matching."""
    debug_print(f"preprocess_for_matching: type={df_type}, system={system_type}, shape={df.shape}")
    
    df_processed = df.copy()
    
    if df_type == 'mpesa':
        # M-Pesa preprocessing
        df_processed['Receipt_Clean'] = df_processed['Receipt No.'].astype(str).str.strip() if 'Receipt No.' in df_processed.columns else None
        
        # Extract date
        if 'Completion Time' in df_processed.columns:
            df_processed['Date_Clean'] = parse_date_vectorized(df_processed['Completion Time'])
        elif 'Completion_Date' in df_processed.columns:
            df_processed['Date_Clean'] = df_processed['Completion_Date']
        
        # Extract amount
        if 'Amount_Clean' not in df_processed.columns:
            if 'KES Equivalent' in df_processed.columns:
                df_processed['Amount_Clean'] = safe_float_vectorized(df_processed['KES Equivalent'])
            elif 'Withdrawn' in df_processed.columns:
                df_processed['Amount_Clean'] = safe_float_vectorized(df_processed['Withdrawn'])
            elif 'Paid In' in df_processed.columns:
                df_processed['Amount_Clean'] = safe_float_vectorized(df_processed['Paid In'])
        
        # Create hash for quick lookups
        if df_processed['Receipt_Clean'].notna().any():
            df_processed['Receipt_Hash'] = df_processed['Receipt_Clean'].apply(lambda x: hash(str(x)) if pd.notna(x) else None)
        
        return df_processed
    
    elif df_type == 'gl':
        # GL preprocessing
        if system_type == 'Choice':
            if 'Description' in df_processed.columns:
                df_processed['Receipt_Extracted'] = extract_receipts_vectorized(df_processed['Description'])
        elif system_type == 'IMT':
            if 'Reference Number' in df_processed.columns:
                df_processed['Receipt_Extracted'] = df_processed['Reference Number'].astype(str).str.strip()
        
        # Extract date
        date_col = None
        for col in ['TX Time', 'Post Time', 'Transaction Date', 'Date']:
            if col in df_processed.columns:
                date_col = col
                break
        
        if date_col:
            df_processed['Date_Clean'] = parse_date_vectorized(df_processed[date_col])
        
        # Extract amount
        amount_col = None
        for col in ['Withdrawn', 'KES Equivalent', 'Amount', 'Transaction Amount']:
            if col in df_processed.columns:
                amount_col = col
                break
        
        if amount_col:
            df_processed['Amount_Clean'] = safe_float_vectorized(df_processed[amount_col])
        
        # Create hash for quick lookups
        if 'Receipt_Extracted' in df_processed.columns and df_processed['Receipt_Extracted'].notna().any():
            df_processed['Receipt_Hash'] = df_processed['Receipt_Extracted'].apply(lambda x: hash(str(x)) if pd.notna(x) else None)
        
        return df_processed
    
    elif df_type == 'transaction':
        # Transaction preprocessing
        if 'Reference Number' in df_processed.columns:
            df_processed['Receipt_Extracted'] = df_processed['Reference Number'].astype(str).str.strip()
        
        # Extract date
        date_col = None
        for col in ['Complete Time', 'Create Time', 'Transaction Date', 'Date']:
            if col in df_processed.columns:
                date_col = col
                break
        
        if date_col:
            df_processed['Date_Clean'] = parse_date_vectorized(df_processed[date_col])
        
        # Extract amount
        if 'Amount' in df_processed.columns:
            df_processed['Amount_Clean'] = safe_float_vectorized(df_processed['Amount'])
        
        # Create hash for quick lookups
        if 'Receipt_Extracted' in df_processed.columns and df_processed['Receipt_Extracted'].notna().any():
            df_processed['Receipt_Hash'] = df_processed['Receipt_Extracted'].apply(lambda x: hash(str(x)) if pd.notna(x) else None)
        
        return df_processed
    
    return df_processed

def create_lookup_index(df, key_columns):
    """Create lookup index for faster matching."""
    index = {}
    for idx, row in df.iterrows():
        keys = []
        for col in key_columns:
            if col in row and pd.notna(row[col]):
                keys.append(str(row[col]).strip().lower())
        
        # Create composite key
        if keys:
            composite_key = '|'.join(keys)
            if composite_key not in index:
                index[composite_key] = []
            index[composite_key].append(idx)
    
    return index

def match_records_batch(gl_batch, mpesa_df, mpesa_index, system_type, match_type='gl'):
    """Match a batch of records."""
    matched = []
    unmatched = []
    
    mpesa_matched_indices = set()
    
    for _, gl_row in gl_batch.iterrows():
        gl_receipt = gl_row.get('Receipt_Extracted')
        gl_amount = gl_row.get('Amount_Clean')
        gl_date = gl_row.get('Date_Clean')
        
        if not gl_receipt or gl_amount is None:
            unmatched.append({
                **gl_row.to_dict(),
                'Reason': 'Missing receipt number or amount',
                'System_Type': system_type
            })
            continue
        
        # Look for matches
        found_match = False
        
        # First try exact receipt match
        receipt_key = str(gl_receipt).strip().lower()
        if receipt_key in mpesa_index:
            for mpesa_idx in mpesa_index[receipt_key]:
                if mpesa_idx in mpesa_matched_indices:
                    continue
                
                mpesa_row = mpesa_df.iloc[mpesa_idx]
                mpesa_amount = mpesa_row.get('Amount_Clean')
                mpesa_date = mpesa_row.get('Date_Clean')
                
                if mpesa_amount is None:
                    continue
                
                # Check amount match
                amount_diff = abs(gl_amount - mpesa_amount)
                if amount_diff > AMOUNT_TOLERANCE:
                    continue
                
                # Check date match if available
                date_match = True
                if gl_date is not None and mpesa_date is not None:
                    date_diff = abs((gl_date - mpesa_date).days)
                    if date_diff > DATE_MATCH_TOLERANCE_DAYS:
                        continue
                
                # Create matched record
                matched_record = create_matched_record(gl_row, mpesa_row, system_type, match_type)
                matched.append(matched_record)
                mpesa_matched_indices.add(mpesa_idx)
                found_match = True
                break
        
        if not found_match:
            unmatched.append({
                **gl_row.to_dict(),
                'Reason': 'No matching M-Pesa transaction found',
                'System_Type': system_type
            })
    
    return matched, unmatched, mpesa_matched_indices

def create_matched_record(source_row, mpesa_row, system_type, match_type):
    """Create a standardized matched record."""
    if match_type == 'gl':
        return {
            'System_Type': system_type,
            'GL_Serial_ID': source_row.get('Serial ID', ''),
            'GL_Request_ID': source_row.get('Request ID', ''),
            'GL_Description': source_row.get('Description', ''),
            'GL_Reference': source_row.get('Reference', '') if system_type == 'Choice' else source_row.get('Reference Number', ''),
            'GL_Amount': source_row.get('Amount_Clean'),
            'GL_Date': source_row.get('Date_Clean'),
            'Mpesa_Receipt_No': mpesa_row.get('Receipt_Clean', ''),
            'Mpesa_Amount': mpesa_row.get('Amount_Clean'),
            'Mpesa_Date': mpesa_row.get('Date_Clean'),
            'Mpesa_Status': mpesa_row.get('Transaction Status', ''),
            'Mpesa_Completion_Time': mpesa_row.get('Completion Time', ''),
            'Mpesa_Details': mpesa_row.get('Details', ''),
            'Match_Type': 'GL_to_Mpesa',
            'Match_Confidence': 'High',
            'Date_Diff_Days': calculate_date_diff(source_row.get('Date_Clean'), mpesa_row.get('Date_Clean')),
            'Amount_Diff': calculate_amount_diff(source_row.get('Amount_Clean'), mpesa_row.get('Amount_Clean'))
        }
    else:  # transaction
        return {
            'System_Type': system_type,
            'Transaction_ID': source_row.get('Transaction ID', ''),
            'Transaction_Reference': source_row.get('Receipt_Extracted', ''),
            'Transaction_Amount': source_row.get('Amount_Clean'),
            'Transaction_Date': source_row.get('Date_Clean'),
            'Transaction_Status': source_row.get('Status', ''),
            'Mpesa_Receipt_No': mpesa_row.get('Receipt_Clean', ''),
            'Mpesa_Amount': mpesa_row.get('Amount_Clean'),
            'Mpesa_Date': mpesa_row.get('Date_Clean'),
            'Mpesa_Status': mpesa_row.get('Transaction Status', ''),
            'Mpesa_Completion_Time': mpesa_row.get('Completion Time', ''),
            'Mpesa_Details': mpesa_row.get('Details', ''),
            'Match_Type': 'Transaction_to_Mpesa',
            'Match_Confidence': 'High',
            'Date_Diff_Days': calculate_date_diff(source_row.get('Date_Clean'), mpesa_row.get('Date_Clean')),
            'Amount_Diff': calculate_amount_diff(source_row.get('Amount_Clean'), mpesa_row.get('Amount_Clean'))
        }

def calculate_date_diff(date1, date2):
    """Calculate date difference in days."""
    if pd.isna(date1) or pd.isna(date2):
        return None
    try:
        return abs((date1 - date2).days)
    except:
        return None

def calculate_amount_diff(amount1, amount2):
    """Calculate amount difference."""
    if amount1 is None or amount2 is None:
        return None
    return abs(amount1 - amount2)

def reconcile_gl_with_mpesa_optimized(gl_df, mpesa_df, system_type='Choice'):
    """Optimized reconciliation with date matching."""
    debug_print(f"reconcile_gl_with_mpesa_optimized: system={system_type}, gl_shape={gl_df.shape}, mpesa_shape={mpesa_df.shape}")
    
    # Preprocess data
    with st.spinner("Preprocessing GL data..."):
        gl_processed = preprocess_for_matching(gl_df, 'gl', system_type)
    
    with st.spinner("Preprocessing M-Pesa data..."):
        mpesa_processed = preprocess_for_matching(mpesa_df, 'mpesa', system_type)
    
    # Create indices for fast lookup
    debug_print("Creating lookup indices...")
    mpesa_index = create_lookup_index(mpesa_processed, ['Receipt_Clean'])
    
    # Process in batches if large
    if len(gl_processed) > CHUNK_SIZE and USE_MULTITHREADING:
        debug_print(f"Processing {len(gl_processed)} records in parallel batches")
        matched_records = []
        unmatched_gl_records = []
        all_mpesa_matched = set()
        
        # Split into batches
        batches = []
        for i in range(0, len(gl_processed), CHUNK_SIZE):
            batch = gl_processed.iloc[i:i+CHUNK_SIZE]
            batches.append(batch)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(
                    match_records_batch, 
                    batch, 
                    mpesa_processed, 
                    mpesa_index, 
                    system_type, 
                    'gl'
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                matched_batch, unmatched_batch, mpesa_matched = future.result()
                matched_records.extend(matched_batch)
                unmatched_gl_records.extend(unmatched_batch)
                all_mpesa_matched.update(mpesa_matched)
        
    else:
        # Process sequentially
        matched_records, unmatched_gl_records, all_mpesa_matched = match_records_batch(
            gl_processed, mpesa_processed, mpesa_index, system_type, 'gl'
        )
    
    # Identify unmatched M-Pesa records
    unmatched_mpesa_records = []
    for idx, row in mpesa_processed.iterrows():
        if idx not in all_mpesa_matched:
            unmatched_mpesa_records.append({
                **row.to_dict(),
                'Reason': 'No matching GL entry found',
                'System_Type': system_type
            })
    
    debug_print(f"Reconciliation complete: Matched={len(matched_records)}, Unmatched GL={len(unmatched_gl_records)}, Unmatched M-Pesa={len(unmatched_mpesa_records)}")
    
    # Sort matched records by date difference (best matches first)
    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        if 'Date_Diff_Days' in matched_df.columns:
            matched_df = matched_df.sort_values('Date_Diff_Days')
            matched_records = matched_df.to_dict('records')
    st.write(f"Matched Records: {len(matched_records)}")
    
    return matched_records, unmatched_gl_records, unmatched_mpesa_records

def reconcile_transactions_with_mpesa_optimized(transactions_df, mpesa_df, system_type='Choice'):
    """Optimized transaction reconciliation with date matching."""
    debug_print(f"reconcile_transactions_with_mpesa_optimized: system={system_type}, trans_shape={transactions_df.shape}, mpesa_shape={mpesa_df.shape}")
    
    # Preprocess data
    with st.spinner("Preprocessing transaction data..."):
        trans_processed = preprocess_for_matching(transactions_df, 'transaction', system_type)
    
    with st.spinner("Preprocessing M-Pesa data..."):
        mpesa_processed = preprocess_for_matching(mpesa_df, 'mpesa', system_type)
    
    # Create indices
    mpesa_index = create_lookup_index(mpesa_processed, ['Receipt_Clean'])
    
    # Process in batches if large
    if len(trans_processed) > CHUNK_SIZE and USE_MULTITHREADING:
        debug_print(f"Processing {len(trans_processed)} transaction records in parallel batches")
        matched_records = []
        unmatched_trans_records = []
        all_mpesa_matched = set()
        
        batches = []
        for i in range(0, len(trans_processed), CHUNK_SIZE):
            batch = trans_processed.iloc[i:i+CHUNK_SIZE]
            batches.append(batch)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(
                    match_records_batch, 
                    batch, 
                    mpesa_processed, 
                    mpesa_index, 
                    system_type, 
                    'transaction'
                )
                futures.append(future)
            
            for future in futures:
                matched_batch, unmatched_batch, mpesa_matched = future.result()
                matched_records.extend(matched_batch)
                unmatched_trans_records.extend(unmatched_batch)
                all_mpesa_matched.update(mpesa_matched)
        
    else:
        matched_records, unmatched_trans_records, all_mpesa_matched = match_records_batch(
            trans_processed, mpesa_processed, mpesa_index, system_type, 'transaction'
        )
    
    # Identify unmatched M-Pesa records
    unmatched_mpesa_records = []
    for idx, row in mpesa_processed.iterrows():
        if idx not in all_mpesa_matched:
            unmatched_mpesa_records.append({
                **row.to_dict(),
                'Reason': 'No matching transaction record found',
                'System_Type': system_type
            })
    
    debug_print(f"Transaction reconciliation complete: Matched={len(matched_records)}, Unmatched Transactions={len(unmatched_trans_records)}, Unmatched M-Pesa={len(unmatched_mpesa_records)}")
    
    # Sort by match quality
    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        if 'Date_Diff_Days' in matched_df.columns:
            matched_df = matched_df.sort_values(['Date_Diff_Days', 'Amount_Diff'])
            matched_records = matched_df.to_dict('records')
    
    return matched_records, unmatched_trans_records, unmatched_mpesa_records

# --- Main App with Performance Optimizations ---
def mpesa_gl_reconciliation_app():
    """Optimized main Streamlit app."""
    debug_print("=" * 80)
    debug_print("STARTING OPTIMIZED M-PESA & GL RECONCILIATION APP")
    debug_print("=" * 80)
    
    # Initialize session state
    session_keys = [
        'mpesa_gl_matched_records',
        'mpesa_gl_unmatched_gl',
        'mpesa_gl_unmatched_mpesa',
        'mpesa_transaction_matched_records',
        'mpesa_transaction_unmatched_trans',
        'mpesa_transaction_unmatched_mpesa',
        'processing_cache'  # For caching processed data
    ]
    
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = [] if 'records' in key or 'unmatched' in key else {}
    
    # st.title("🚀 Optimized M-Pesa & GL Reconciliation Module")
    st.markdown("""
    This module reconciles M-Pesa transactions with GL entries and transaction records 
    for both Choice and IMT systems. **Optimized for large datasets** with date matching.
    """)
    
    # Performance settings
    with st.expander("⚙️ Performance Settings"):
        col1, col2 = st.columns(2)
        with col1:
            global CHUNK_SIZE, USE_MULTITHREADING
            CHUNK_SIZE = st.number_input("Chunk Size", min_value=1000, max_value=50000, value=10000, step=1000)
            DATE_MATCH_TOLERANCE_DAYS = st.number_input("Date Match Tolerance (days)", min_value=0, max_value=30, value=2, step=1)
        with col2:
            USE_MULTITHREADING = st.checkbox("Use Multithreading", value=True)
            AMOUNT_TOLERANCE = st.number_input("Amount Tolerance (KES)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # System Selection
    system_type = st.radio("Select System", ['Choice', 'IMT'], horizontal=True)
    debug_print(f"System type selected: {system_type}")
    
    # File Upload Section with progress tracking
    st.header("📁 File Uploads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"{system_type} System Files")
        gl_file = st.file_uploader(f"Upload {system_type} GL Entries", 
                                  type=['xlsx', 'csv'], 
                                  key=f"gl_{system_type}")
        transaction_file = st.file_uploader(f"Upload {system_type} Transaction Records", 
                                           type=['xlsx', 'csv'], 
                                           key=f"trans_{system_type}")
    
    with col2:
        st.subheader("M-Pesa Files (MMF Account)")
        mpesa_mmf_files = st.file_uploader("Upload M-Pesa MMF Account Files", 
                                          type=['csv', 'xls', 'xlsx'], 
                                          accept_multiple_files=True,
                                          key="mpesa_mmf")
    
    with col3:
        st.subheader("M-Pesa Files (Utility Account)")
        mpesa_utility_files = st.file_uploader("Upload M-Pesa Utility Account Files", 
                                              type=['csv', 'xls', 'xlsx'], 
                                              accept_multiple_files=True,
                                              key="mpesa_utility")
    
    # Process uploaded files with progress bars
    gl_df = pd.DataFrame()
    transaction_df = pd.DataFrame()
    mpesa_mmf_df = pd.DataFrame()
    mpesa_utility_df = pd.DataFrame()
    
    # Check cache first
    cache_key = None
    if gl_file or transaction_file or mpesa_mmf_files or mpesa_utility_files:
        cache_key = generate_data_hash({
            'gl': gl_file.getvalue() if gl_file else '',
            'trans': transaction_file.getvalue() if transaction_file else '',
            'mpesa_mmf': [f.getvalue() for f in mpesa_mmf_files] if mpesa_mmf_files else [],
            'mpesa_utility': [f.getvalue() for f in mpesa_utility_files] if mpesa_utility_files else []
        })
    
    if cache_key in st.session_state.processing_cache:
        debug_print(f"Using cached data for key: {cache_key}")
        cached_data = st.session_state.processing_cache[cache_key]
        gl_df = cached_data.get('gl_df', pd.DataFrame())
        transaction_df = cached_data.get('transaction_df', pd.DataFrame())
        mpesa_mmf_df = cached_data.get('mpesa_mmf_df', pd.DataFrame())
        mpesa_utility_df = cached_data.get('mpesa_utility_df', pd.DataFrame())
    else:
        # Process files
        if gl_file:
            with st.spinner(f"Processing {gl_file.name}..."):
                gl_content = gl_file.getvalue()
                gl_df = load_with_auto_header_optimized(gl_content, gl_file.name)
                if not gl_df.empty:
                    st.success(f"✅ Loaded {system_type} GL Entries: {len(gl_df):,} records")
                    debug_print(f"GL DataFrame shape: {gl_df.shape}")
        
        if transaction_file:
            with st.spinner(f"Processing {transaction_file.name}..."):
                trans_content = transaction_file.getvalue()
                transaction_df = load_with_auto_header_optimized(trans_content, transaction_file.name)
                if not transaction_df.empty:
                    st.success(f"✅ Loaded {system_type} Transaction Records: {len(transaction_df):,} records")
                    debug_print(f"Transaction DataFrame shape: {transaction_df.shape}")
        
        # Process M-Pesa MMF files
        if mpesa_mmf_files:
            mpesa_mmf_records = []
            progress_bar = st.progress(0)
            for i, mpesa_file in enumerate(mpesa_mmf_files):
                progress_bar.progress((i + 1) / len(mpesa_mmf_files), text=f"Processing {mpesa_file.name}")
                file_content = mpesa_file.getvalue()
                if mpesa_file.name.endswith('.csv'):
                    df = parse_mpesa_csv_optimized(file_content.decode('utf-8', errors='ignore'))
                else:
                    df = parse_mpesa_excel(file_content, mpesa_file.name)
                
                if not df.empty:
                    mpesa_mmf_records.append(df)
            
            if mpesa_mmf_records:
                mpesa_mmf_df = pd.concat(mpesa_mmf_records, ignore_index=True)
                st.success(f"✅ Loaded {len(mpesa_mmf_records)} M-Pesa MMF files: {len(mpesa_mmf_df):,} total records")
                debug_print(f"M-Pesa MMF DataFrame shape: {mpesa_mmf_df.shape}")
            progress_bar.empty()
        
        # Process M-Pesa Utility files
        if mpesa_utility_files:
            mpesa_utility_records = []
            progress_bar = st.progress(0)
            for i, mpesa_file in enumerate(mpesa_utility_files):
                progress_bar.progress((i + 1) / len(mpesa_utility_files), text=f"Processing {mpesa_file.name}")
                file_content = mpesa_file.getvalue()
                if mpesa_file.name.endswith('.csv'):
                    df = parse_mpesa_csv_optimized(file_content.decode('utf-8', errors='ignore'))
                else:
                    df = parse_mpesa_excel(file_content, mpesa_file.name)
                
                if not df.empty:
                    mpesa_utility_records.append(df)
            
            if mpesa_utility_records:
                mpesa_utility_df = pd.concat(mpesa_utility_records, ignore_index=True)
                st.success(f"✅ Loaded {len(mpesa_utility_records)} M-Pesa Utility files: {len(mpesa_utility_df):,} total records")
                debug_print(f"M-Pesa Utility DataFrame shape: {mpesa_utility_df.shape}")
            progress_bar.empty()
        
        # Cache processed data
        if cache_key:
            st.session_state.processing_cache[cache_key] = {
                'gl_df': gl_df,
                'transaction_df': transaction_df,
                'mpesa_mmf_df': mpesa_mmf_df,
                'mpesa_utility_df': mpesa_utility_df
            }
            debug_print(f"Cached processed data with key: {cache_key}")
    
    # Combine M-Pesa data
    mpesa_combined_df = pd.concat([mpesa_mmf_df, mpesa_utility_df], ignore_index=True) \
        if not mpesa_mmf_df.empty or not mpesa_utility_df.empty else pd.DataFrame()
    
    # Display data summaries
    st.header("Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("GL Records", f"{len(gl_df):,}" if not gl_df.empty else "0")
    with col2:
        st.metric("Transaction Records", f"{len(transaction_df):,}" if not transaction_df.empty else "0")
    with col3:
        st.metric("M-Pesa MMF Records", f"{len(mpesa_mmf_df):,}" if not mpesa_mmf_df.empty else "0")
    with col4:
        st.metric("M-Pesa Utility Records", f"{len(mpesa_utility_df):,}" if not mpesa_utility_df.empty else "0")
    
    # Reconciliation Section
    st.header("Reconciliation")
    
    if st.button("Run Optimized Reconciliation", type="primary"):
        if gl_df.empty and transaction_df.empty:
            st.warning("Please upload at least one GL Entries or Transaction Records file.")
        elif mpesa_combined_df.empty:
            st.warning("Please upload at least one M-Pesa file.")
        else:
            debug_print(f"Starting optimized reconciliation with {len(gl_df)} GL, {len(transaction_df)} transactions, {len(mpesa_combined_df)} M-Pesa records")
            
            # Create progress containers
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # GL Reconciliation
                if not gl_df.empty:
                    progress_text.text("🔍 Reconciling GL Entries with M-Pesa...")
                    gl_matched, gl_unmatched_gl, gl_unmatched_mpesa = reconcile_gl_with_mpesa_optimized(
                        gl_df, mpesa_combined_df, system_type
                    )
                    st.session_state.mpesa_gl_matched_records = gl_matched
                    st.session_state.mpesa_gl_unmatched_gl = gl_unmatched_gl
                    st.session_state.mpesa_gl_unmatched_mpesa = gl_unmatched_mpesa
                    progress_bar.progress(0.5)
                
                # Transaction Reconciliation
                if not transaction_df.empty:
                    progress_text.text("🔍 Reconciling Transaction Records with M-Pesa...")
                    trans_matched, trans_unmatched_trans, trans_unmatched_mpesa = reconcile_transactions_with_mpesa_optimized(
                        transaction_df, mpesa_combined_df, system_type
                    )
                    st.session_state.mpesa_transaction_matched_records = trans_matched
                    st.session_state.mpesa_transaction_unmatched_trans = trans_unmatched_trans
                    st.session_state.mpesa_transaction_unmatched_mpesa = trans_unmatched_mpesa
                    progress_bar.progress(1.0)
                
                progress_text.text("✅ Reconciliation complete!")
                st.success("✅ Reconciliation complete!")
                
            except Exception as e:
                st.error(f"Error during reconciliation: {str(e)}")
                debug_print(f"Reconciliation error: {e}", "ERROR")
            finally:
                # Clean up progress indicators
                import time
                time.sleep(0.5)
                progress_text.empty()
                progress_bar.empty()
    
    # Results Display with improved visualization
    st.header("📊 Results")
    
    # GL Reconciliation Results
    if st.session_state.mpesa_gl_matched_records:
        st.subheader(f"📋 {system_type} GL Reconciliation Results")
        
        # Calculate match quality statistics
        matched_df = pd.DataFrame(st.session_state.mpesa_gl_matched_records)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Matched Records", len(st.session_state.mpesa_gl_matched_records))
        with col2:
            perfect_matches = matched_df[matched_df['Date_Diff_Days'] == 0].shape[0] if 'Date_Diff_Days' in matched_df.columns else 0
            st.metric("Perfect Date Matches", perfect_matches)
        with col3:
            st.metric("Unmatched GL", len(st.session_state.mpesa_gl_unmatched_gl))
        with col4:
            st.metric("Unmatched M-Pesa", len(st.session_state.mpesa_gl_unmatched_mpesa))
        
        # Display match quality distribution
        with st.expander("📈 Match Quality Analysis"):
            if 'Date_Diff_Days' in matched_df.columns:
                st.subheader("Date Match Distribution")
                date_diff_counts = matched_df['Date_Diff_Days'].value_counts().sort_index()
                st.bar_chart(date_diff_counts.head(10))
            
            if 'Amount_Diff' in matched_df.columns:
                st.subheader("Amount Difference Distribution")
                amount_diff_counts = matched_df['Amount_Diff'].value_counts().sort_index()
                st.bar_chart(amount_diff_counts.head(10))
        
        # Download options
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = pd.DataFrame(st.session_state.mpesa_gl_matched_records).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Matched GL Records",
                data=csv,
                file_name=f"{system_type}_GL_Matched_Records.csv",
                mime="text/csv"
            )
        with col2:
            if st.session_state.mpesa_gl_unmatched_gl:
                csv = pd.DataFrame(st.session_state.mpesa_gl_unmatched_gl).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Unmatched GL",
                    data=csv,
                    file_name=f"{system_type}_GL_Unmatched.csv",
                    mime="text/csv"
                )
        with col3:
            if st.session_state.mpesa_gl_unmatched_mpesa:
                csv = pd.DataFrame(st.session_state.mpesa_gl_unmatched_mpesa).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Unmatched M-Pesa (GL)",
                    data=csv,
                    file_name=f"{system_type}_Mpesa_Unmatched_GL.csv",
                    mime="text/csv"
                )
    
    # Transaction Reconciliation Results
    if st.session_state.mpesa_transaction_matched_records:
        st.subheader(f"📋 {system_type} Transaction Reconciliation Results")
        
        trans_matched_df = pd.DataFrame(st.session_state.mpesa_transaction_matched_records)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Matched Records", len(st.session_state.mpesa_transaction_matched_records))
        with col2:
            perfect_matches = trans_matched_df[trans_matched_df['Date_Diff_Days'] == 0].shape[0] if 'Date_Diff_Days' in trans_matched_df.columns else 0
            st.metric("Perfect Date Matches", perfect_matches)
        with col3:
            st.metric("Unmatched Transactions", len(st.session_state.mpesa_transaction_unmatched_trans))
        with col4:
            st.metric("Unmatched M-Pesa", len(st.session_state.mpesa_transaction_unmatched_mpesa))
        
        # Download options for transactions
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = pd.DataFrame(st.session_state.mpesa_transaction_matched_records).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Matched Transactions",
                data=csv,
                file_name=f"{system_type}_Transaction_Matched.csv",
                mime="text/csv"
            )
        with col2:
            if st.session_state.mpesa_transaction_unmatched_trans:
                csv = pd.DataFrame(st.session_state.mpesa_transaction_unmatched_trans).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Unmatched Transactions",
                    data=csv,
                    file_name=f"{system_type}_Transactions_Unmatched.csv",
                    mime="text/csv"
                )
        with col3:
            if st.session_state.mpesa_transaction_unmatched_mpesa:
                csv = pd.DataFrame(st.session_state.mpesa_transaction_unmatched_mpesa).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Unmatched M-Pesa (Trans)",
                    data=csv,
                    file_name=f"{system_type}_Mpesa_Unmatched_Transactions.csv",
                    mime="text/csv"
                )
    
    # Performance Statistics
    if st.session_state.mpesa_gl_matched_records or st.session_state.mpesa_transaction_matched_records:
        st.header("📊 Performance Summary")
        
        total_processed = (
            len(st.session_state.mpesa_gl_matched_records) +
            len(st.session_state.mpesa_gl_unmatched_gl) +
            len(st.session_state.mpesa_transaction_matched_records) +
            len(st.session_state.mpesa_transaction_unmatched_trans)
        )
        
        total_matched = (
            len(st.session_state.mpesa_gl_matched_records) +
            len(st.session_state.mpesa_transaction_matched_records)
        )
        
        match_rate = (total_matched / total_processed * 100) if total_processed > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Processed", f"{total_processed:,}")
        with col2:
            st.metric("Total Matched", f"{total_matched:,}")
        with col3:
            st.metric("Match Rate", f"{match_rate:.1f}%")
        
        # Cache management
        with st.expander("🗑️ Cache Management"):
            st.write(f"Cached datasets: {len(st.session_state.processing_cache)}")
            if st.button("Clear Cache"):
                st.session_state.processing_cache = {}
                st.success("Cache cleared!")
    
    debug_print("=" * 80)
    debug_print("ENDING OPTIMIZED M-PESA & GL RECONCILIATION APP")
    debug_print("=" * 80)
    
    return (
        st.session_state.mpesa_gl_matched_records,
        st.session_state.mpesa_gl_unmatched_gl,
        st.session_state.mpesa_gl_unmatched_mpesa,
        st.session_state.mpesa_transaction_matched_records,
        st.session_state.mpesa_transaction_unmatched_trans,
        st.session_state.mpesa_transaction_unmatched_mpesa
    )