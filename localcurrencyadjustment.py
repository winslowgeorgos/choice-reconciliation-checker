import mysql.connector
from datetime import datetime
import csv

# Configuring paths for adjustment Adjustment and Bank files
adjustment_table = 'LOCAL_CURRENCY_ACCOUNT_ADJUSTMENT 11th July'
out_csv_path_adjustment_unmatched = r'C:\Users\HP\Desktop\verify_transfers\LocalAdjustment\UnmatchedLocalAdjustment.csv'
out_csv_path_adjustment_matched = r'C:\Users\HP\Desktop\verify_transfers\LocalAdjustment\MatchedLocalAdjustment.csv'

fx_date_str = '11/07/2025'  # The date we're working with

date_formats = [
    '%Y-%m-%d', 
    '%Y/%m/%d', 
    '%d.%m.%Y', 
    '%Y.%m.%d',
    '%d/%m/%Y', 
    '%-d/%-m/%Y', 
    '%-d.%-m.%Y'
]

# Bank name map: maps raw intermediary account name (lowercase) to bank code (short)
bank_name_map = {
    'central bank of kenya': 'cbk',
    'kenya commercial bank': 'kcb',
    'kingdom bank': 'kingdom',
    'ABSA Bank': 'absa',
    'absa bank': 'absa',
    'equity bank': 'equity',
    'i&m bank': 'i&m',
    'ncba bank kenya plc': 'ncba',
    'ncba bank': 'ncba',
}

# Helper Functions
def safe_float(x):
    try:
        return float(str(x).replace(',', '').strip())
    except:
        return None

def normalize_bank_key(raw_key):
    return raw_key.lower().strip()

def resolve_amount_column(columns, operation):
    columns_lower = [col.lower() for col in columns]
    
    if operation == 'credit':
        for col in ['credit', 'deposit']:
            if col in columns_lower:
                return columns[columns_lower.index(col)]
    elif operation == 'debit':
        for col in ['debit', 'withdrawal']:
            if col in columns_lower:
                return columns[columns_lower.index(col)]
    return None

def resolve_date_column(columns):
    for candidate in ['Value Date', 'Transaction Date', 'MyUnknownColumn', 'Transaction date']:
        if candidate in columns:
            return candidate
    return None

# Function to process and compare adjustment Adjustment with bank records
def process_account_adjustment_match(
    cursor, 
    unmatched_adjustments, 
    matched_adjustments, 
    row, 
    amount, 
    date, 
    bank_currency_field, 
    operation
    ):
    
    if not amount:
        return False

    
    date_str = date.split()[0].strip()
    parsed = None
    for fmt in date_formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if not parsed:
        return False

    formatted_slash = parsed.strftime('%d/%m/%Y')
    formatted_dot = parsed.strftime('%d.%m.%Y')

    adjustment_raw = str(row.get(bank_currency_field, '')).strip().lower()
    
    # Skips non-bank intermediary accounts 
    if adjustment_raw in ['baas temporary account', 'unclaimed funds', 'other temporary account']:
        return False

    bank_code = bank_name_map.get(adjustment_raw)
    if not bank_code:
        print(f"Bank name '{adjustment_raw}' not mapped to any bank code. Skipping...")
        return False

    currency = str(row.get('Currency', '')).strip().lower()
    if not currency:
        print("Currency missing in row, skipping...")
        return False

    bank_key = normalize_bank_key(f"{bank_code} {currency}")

    try:
        cursor.execute("SHOW TABLES")
        available_tables = [list(r.values())[0].lower() for r in cursor.fetchall()]
        if bank_key not in available_tables:
            return False

        cursor.execute(f"SHOW COLUMNS FROM `{bank_key}`")
        columns = [col['Field'] for col in cursor.fetchall()]
        date_column = resolve_date_column(columns)
        amount_column = resolve_amount_column(columns, operation)
        if not date_column or not amount_column:
            print(f"Could not find date or amount column in table '{bank_key}'. Skipping...")
            return False

        cursor.execute(f"SELECT * FROM `{bank_key}`")
        all_rows = cursor.fetchall()
        date_matches = [
            r for r in all_rows
            if str(r.get(date_column, '')).strip() in (formatted_slash, formatted_dot)
        ]
        for r in date_matches:
            amt_val = safe_float(r.get(amount_column))
            if amt_val is not None and abs(amt_val - amount) < 1:  # Match found
                matched_adjustments.append({
                    'Date': date,
                    'Bank Table': bank_key,
                    'Amount': amount,
                    'Matched In': amount_column,
                    'Date Column Used': date_column,
                    'Source Column': bank_currency_field
                })
                return True
        unmatched_adjustments.append({
            'Date': date,
            'Bank Table': bank_key,
            'Amount': amount,
            'Expected In': amount_column,
            'Date Column Used': date_column,
            'Status': 'Not Found',
            'Source Column': bank_currency_field
        })
    except Exception as e:
        print(f"⚠️ Failed to process {bank_key}: {e}")
        return False
    return False

# Execution
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Dorothy2015.',
    database='accounts'
)
cursor = conn.cursor(dictionary=True)

print("Using table:", adjustment_table)
cursor.execute(f"SELECT * FROM `{adjustment_table}`")
adjustment_rows = [{k.strip(): v for k, v in row.items()} for row in cursor.fetchall()]
statuses = [str(row.get('Status', '')).strip() for row in adjustment_rows]

# Initialize counters and lists
unmatched_adjustments = []
matched_adjustments = []
matched_set = set()

# Process each row in the adjustment Adjustment table
for row in adjustment_rows:
    status = str(row.get('Status', '')).strip().lower()
    if status != 'successful':  # Only process Successful records
        continue

    date = str(row.get('Completed At', '')).strip()
    amount = safe_float(row.get('Amount'))
    operation = str(row.get('Operation', '')).strip().lower()

    if operation not in ['credit', 'debit']:  # Only process 'credit' and 'debit' operations
        continue


    if process_account_adjustment_match(
        cursor, 
        unmatched_adjustments, 
        matched_adjustments, 
        row, 
        amount, 
        date, 
        'Intermediary Account',
        operation
        ):
        matched_set.add((row.get('Intermediary Account', '').strip().lower(), round(amount, 2)))

# Export unmatched adjustment Adjustments
if unmatched_adjustments:
    with open(out_csv_path_adjustment_unmatched, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=unmatched_adjustments[0].keys())
        writer.writeheader()
        writer.writerows(unmatched_adjustments)

# Export matched adjustment Adjustments
if matched_adjustments:
    with open(out_csv_path_adjustment_matched, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=matched_adjustments[0].keys())
        writer.writeheader()
        writer.writerows(matched_adjustments)

# Summary
print("\n===== SUMMARY =====")
print(f"✅ Adjustments Matched: {len(matched_adjustments)}")
print(f"❌ Adjustments Unmatched: {len(unmatched_adjustments)}")

cursor.close()
conn.close()
