import mysql.connector
from datetime import datetime
import csv

# Configuring the pathsv4
fx_table = 'FX_Trade_Tracker_20250711_20250713'
out_csv_path_buy = r'C:\Users\HP\Desktop\verify_transfers\CounterpartyPayment\UnmatchedCounterpartyPayment.csv'
out_csv_path_sell = r'C:\Users\HP\Desktop\verify_transfers\ChoicePayment\UnmatchedChoicePayment.csv'
out_csv_path_bank = r'C:\Users\HP\Desktop\verify_transfers\unmatched_bank_records.csv'
out_csv_path_buy_matched = r'C:\Users\HP\Desktop\verify_transfers\CounterpartyPayment\MatchedCounterpartyPayment.csv'
out_csv_path_sell_matched = r'C:\Users\HP\Desktop\verify_transfers\ChoicePayment\MatchedChoicePayment.csv'
fx_date_str = '11/07/2025'

# Various Date Formats
date_formats = [
    '%Y-%m-%d', 
    '%Y/%m/%d', 
    '%d.%m.%Y', 
    '%Y.%m.%d',
    '%d/%m/%Y', 
    '%-d/%-m/%Y', 
    '%-d.%-m.%Y'
]

def safe_float(x):
    try:
        return float(str(x).replace(',', '').strip())
    except:
        return None

def normalize_bank_key(raw_key):
    raw_key = raw_key.lower()
    replacements = {
        'ncba bank kenya plc': 'ncba', 
        'ncba bank': 'ncba',
        'equity bank': 'equity', 
        'i&m bank': 'i&m',
        'central bank of kenya': 'cbk', 
        'kenya commercial bank': 'kcb',
        'kcb bank': 'kcb', 
        'sbm bank (kenya) limited': 'sbm',
        'sbm bank': 'sbm',
        'absa bank': 'absa'
    }
    for long, short in replacements.items():
        if raw_key.startswith(long):
            raw_key = raw_key.replace(long, short)
            break
    return raw_key.strip()

def resolve_amount_column(columns, action_type, is_sell_side=False):
    columns_lower = [col.lower() for col in columns]
    if not is_sell_side:
        if action_type == 'Bank Buy':
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['deposit', 'credit']), None)
        elif action_type == 'Bank Sell':
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['credit', 'deposit']), None)
    else:
        if action_type == 'Bank Buy':
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['withdrawal', 'debit']), None)
        elif action_type == 'Bank Sell':
            return next((columns[i] for i, col in enumerate(columns_lower) if col in ['debit', 'withdrawal']), None)
    return None

def resolve_date_column(columns):
    for candidate in ['Value Date', 'Transaction Date', 'MyUnknownColumn','Transaction date']:
        if candidate in columns:
            return candidate
    return None

def get_description_columns(columns):
    for desc in ['Transaction details','Transaction', 'Customer reference','Narration',
                 'Transaction Details', 'Detail',  'Transaction Remarks:', 
                 'TransactionDetails', 'Description', 'Narrative']:
        if desc in columns:
            return desc
    return None
        
def get_amount_columns(columns):
    return [col for col in columns if col.lower() in ['deposit', 'credit', 'withdrawal', 'debit']]

def process_fx_match(cursor, unmatched, matched_list, row, action_type, date, amount, bank_currency_field, is_sell_side):
    if not amount or action_type not in ['Bank Buy', 'Bank Sell']:
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

    counterparty_raw = str(row.get(bank_currency_field, '')).strip()
    parts = counterparty_raw.split('-')
    if len(parts) < 2:
        return False

    bank_name = parts[0].strip().lower()
    currency = parts[1].strip().lower()
    raw_key = f"{bank_name} {currency}"
    bank_key = normalize_bank_key(raw_key)

    try:
        cursor.execute("SHOW TABLES")
        available_tables = [list(r.values())[0].lower() for r in cursor.fetchall()]
        if bank_key not in available_tables:
            return False

        cursor.execute(f"SHOW COLUMNS FROM `{bank_key}`")
        columns = [col['Field'] for col in cursor.fetchall()]
        date_column = resolve_date_column(columns)
        amount_column = resolve_amount_column(columns, action_type, is_sell_side)
        if not date_column or not amount_column:
            return False

        cursor.execute(f"SELECT * FROM `{bank_key}`")
        all_rows = cursor.fetchall()
        date_matches = [
            r for r in all_rows
            if str(r.get(date_column, '')).strip() in (formatted_slash, formatted_dot)
        ]
        for r in date_matches:
            amt_val = safe_float(r.get(amount_column))
            if amt_val is not None and abs(amt_val - amount) < 1:
                matched_list.append({
                    'Date': date,
                    'Bank Table': bank_key,
                    'Action Type': action_type,
                    'Amount': amount,
                    'Matched In': amount_column,
                    'Date Column Used': date_column,
                    'Source Column': bank_currency_field
                })
                return True
        unmatched.append({
            'Date': date,
            'Bank Table': bank_key,
            'Action Type': action_type,
            'Amount': amount,
            'Expected In': amount_column,
            'Date Column Used': date_column,
            'Status': 'Not Found',
            'Source Column': bank_currency_field
        })
    except:
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

print("Using table:", fx_table)
cursor.execute(f"SELECT * FROM `{fx_table}`")
fx_rows = [{k.strip(): v for k, v in row.items()} for row in cursor.fetchall()]

buy_match_count = 0
sell_match_count = 0
unmatched_buy = []
unmatched_sell = []
matched_buy = []
matched_sell = []
matched_set = set()

for row in fx_rows:
    action_type = str(row.get('Action Type', '')).strip()
    date = str(row.get('Created At', '')).strip()
    status = str(row.get('Status', '')).strip().lower()
    if status == 'cancelled':
        continue

    buy_amount = safe_float(row.get('Buy Currency Amount'))
    if process_fx_match(cursor, unmatched_buy, matched_buy, row, action_type, date, buy_amount, 'MyUnknownColumn', False):
        buy_match_count += 1
        bank_info = row.get('MyUnknownColumn')
        if bank_info:
            parts = bank_info.lower().split('-')
            if len(parts) >= 2:
                key = f"{parts[0].strip()} {parts[1].strip()}"
                matched_set.add((key, '10/07/2025', round(buy_amount, 2)))

    sell_amount = safe_float(row.get('Sell Currency Amount'))
    if process_fx_match(cursor, unmatched_sell, matched_sell, row, action_type, date, sell_amount, 'MyUnknownColumn_[2]', True):
        sell_match_count += 1
        bank_info = row.get('MyUnknownColumn_[2]')
        if bank_info:
            parts = bank_info.lower().split('-')
            if len(parts) >= 2:
                key = f"{parts[0].strip()} {parts[1].strip()}"
                matched_set.add((key, '10/07/2025', round(sell_amount, 2)))

# Export unmatched FX
if unmatched_buy:
    with open(out_csv_path_buy, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=unmatched_buy[0].keys())
        writer.writeheader()
        writer.writerows(unmatched_buy)

if unmatched_sell:
    with open(out_csv_path_sell, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=unmatched_sell[0].keys())
        writer.writeheader()
        writer.writerows(unmatched_sell)

# Export matched FX
if matched_buy:
    with open(out_csv_path_buy_matched, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=matched_buy[0].keys())
        writer.writeheader()
        writer.writerows(matched_buy)

if matched_sell:
    with open(out_csv_path_sell_matched, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=matched_sell[0].keys())
        writer.writeheader()
        writer.writerows(matched_sell)

# Scan unmatched bank records
cursor.execute("SHOW TABLES")
bank_tables = [list(row.values())[0]
               for row in cursor.fetchall()
               if not list(row.values())[0].startswith('fx_')]
unmatched_records = []

for table in bank_tables:
    bank_key = table.strip().lower()
    try:
        cursor.execute(f"SHOW COLUMNS FROM `{table}`")
        columns = [col['Field'] for col in cursor.fetchall()]
        date_col = resolve_date_column(columns)
        amount_cols = get_amount_columns(columns)
        description_col = get_description_columns(columns)
        if not date_col or not amount_cols or not description_col:
            continue

        cursor.execute(f"SELECT `{date_col}`, {', '.join(amount_cols)}, `{description_col}` FROM `{table}`")

        for row in cursor.fetchall():
            row_date = str(row.get(date_col, '')).strip()
            if row_date != fx_date_str:
                continue

            for amt_col in amount_cols:
                amt_val = safe_float(row.get(amt_col))
                if amt_val:
                    rounded_amt = round(amt_val, 2)
                    description = str(row.get(description_col, '')).strip()
                    if (bank_key, fx_date_str, rounded_amt) not in matched_set:
                        unmatched_records.append({
                            'Bank Table': bank_key,
                            'Date': fx_date_str,
                            'Description': description,
                            'Transaction Type': amt_col,
                            'Amount': rounded_amt
                        })
    except Exception as e:
        print(f"⚠️ Failed to process {table}: {e}")

if unmatched_records:
    with open(out_csv_path_bank, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=unmatched_records[0].keys())
        writer.writeheader()
        writer.writerows(unmatched_records)

# Summary
print("\n===== SUMMARY =====")
print(f"✅ BUY Side Matches (Counterparty Payment): {buy_match_count}")
print(f"❌ BUY Side Unmatched: {len(unmatched_buy)}")
print(f"✅ SELL Side Matches (Choice Payment): {sell_match_count}")
print(f"❌ SELL Side Unmatched: {len(unmatched_sell)}")
print(f"📤 Bank-only unmatched entries: {len(unmatched_records)}")


cursor.close()
conn.close()
