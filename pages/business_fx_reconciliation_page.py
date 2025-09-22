# pages/business_fx_reconciliation_page.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from fuzzywuzzy import fuzz, process
import io

import os
import pickle

# --- Constants ---
UPLOAD_DIR = "data/uploads"
CACHE_DIR = "data/cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

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
                'Vendor Name' : 'Vendor Name',
                'Counterparty Dealer' : 'Counterparty Dealer',
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
        return float(s)
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
        "absa bank": "Absa", "kingdom bank": "Kingdom", "uba": "UBA"
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
    return abs(a - b) <= tol


# --- Core Reconciliation ---
def reconcile_fx_with_business(
    fx_trade_df: pd.DataFrame,
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

    # --- FX loop ---
    for idx, fx_row in fx_trade_df.iterrows():
        if "Status" in fx_trade_df.columns and str(fx_row.get("Status")).lower() in ("cancelled", "pending"):
            continue

        action_type = str(fx_row.get("Action Type", "")).strip().lower()
        is_sell_action = "sell" in action_type
        is_buy_action = "buy" in action_type

        if is_sell_action:
            fx_primary_amount_field = "Sell Currency Amount"
            fx_primary_trade_info_field = "Sell Trade Info"
            fx_secondary_amount_field = "Buy Currency Amount"
            fx_secondary_trade_info_field = "Buy Trade Info"
            business_side = "sell"
        elif is_buy_action:
            fx_primary_amount_field = "Buy Currency Amount"
            fx_primary_trade_info_field = "Buy Trade Info"
            fx_secondary_amount_field = "Sell Currency Amount"
            fx_secondary_trade_info_field = "Sell Trade Info"
            business_side = "buy"
        else:
            continue

        fx_primary_amount = safe_float(fx_row.get(fx_primary_amount_field))
        fx_secondary_amount = safe_float(fx_row.get(fx_secondary_amount_field))

        primary_info_raw = fx_row.get(fx_primary_trade_info_field, "")
        sec_info_raw = fx_row.get(fx_secondary_trade_info_field, "")
        fx_primary_bank_raw, fx_primary_currency = extract_bank_and_currency_from_trade_info(primary_info_raw)
        fx_secondary_bank_raw, fx_secondary_currency = extract_bank_and_currency_from_trade_info(sec_info_raw)

        fx_primary_bank = normalize_bank_key(fx_primary_bank_raw)
        fx_secondary_bank = normalize_bank_key(fx_secondary_bank_raw)
        found_match = False
        reasons = []

        amount_matched = False
        kes_eq_matched = False
        currency_matched = False
        bank_matched = False

        for p_idx, business_row in business_df[business_df["_matched"] == False].iterrows():
            local_reasons = []
            kes_matched = False
            other_matched = False
            match_type = "Partial"

            # 1. deal side
            deal_currency, deal_side = parse_deal_type(business_row.get("Deal type", ""))
            if deal_side != business_side:
                local_reasons.append("Deal type mismatch")
                continue
            
            # 2. currency
            business_amount = safe_float(business_row.get("Amount"))
            currency_ok = True
            if deal_currency and fx_primary_currency:
                currency_ok = deal_currency.upper() == str(fx_primary_currency).upper()

            if currency_ok:
                currency_matched = True
            else:
                local_reasons.append("Currency mismatch")

            # 3. primary amount
            primary_amounts_ok = (
                fx_primary_amount is not None and
                business_amount is not None and
                amounts_match(fx_primary_amount, business_amount, pct_tolerance)
            )
            if primary_amounts_ok:
                other_matched = True
                amount_matched = True
            else:
                local_reasons.append("Primary amount mismatch")

            # 4. primary bank
            business_primary_bank_raw = (
                business_row.get("Payee bank") if business_side == "sell" else business_row.get("Collection bank")
            )
            business_primary_bank = normalize_bank_key(business_primary_bank_raw)
            banks_ok = (
                fx_primary_bank and
                business_primary_bank and
                fx_primary_bank.lower() == business_primary_bank.lower()
            )

            if banks_ok:
                bank_matched = True
            else:
                local_reasons.append("Primary bank mismatch")

            # Check if primary criteria are met (currency + amount + bank)
            primary_criteria_met = currency_ok and primary_amounts_ok and banks_ok
            
            # 5. secondary (KES equivalent)
            business_kes = safe_float(business_row.get("KES equivalent"))
            sec_amt = fx_secondary_amount
            sec_curr = fx_secondary_currency
            sec_amt_in_kes = (
                convert_currency(sec_amt, sec_curr, "KES")
                if sec_amt is not None and sec_curr and str(sec_curr).upper() != "KES"
                else sec_amt
            )
            secondary_amounts_ok = (
                business_kes is not None and
                sec_amt_in_kes is not None and
                amounts_match(sec_amt_in_kes, business_kes, pct_tolerance)
            )

            if secondary_amounts_ok:
                kes_matched = True
                kes_eq_matched = True
            else:
                local_reasons.append("KES equivalent mismatch")

            # 6. secondary bank
            business_secondary_bank_raw = (
                business_row.get("Collection bank") if business_side == "sell" else business_row.get("Payee bank")
            )
            business_secondary_bank = normalize_bank_key(business_secondary_bank_raw)
            sec_bank_ok = (
                fx_secondary_bank and
                business_secondary_bank and
                fx_secondary_bank.lower() == business_secondary_bank.lower()
            )

            if sec_bank_ok:
                pass  # Secondary bank matched
            else:
                local_reasons.append("Secondary bank mismatch")

            # Check if secondary criteria are met (KES amount + secondary bank)
            secondary_criteria_met = secondary_amounts_ok and sec_bank_ok

            # MARK AS MATCHED IF EITHER PRIMARY OR SECONDARY CRITERIA ARE MET
            if primary_criteria_met or secondary_criteria_met:
                business_df.at[p_idx, "_matched"] = True
                business_df.at[p_idx, "KES_Equivalent_Matched"] = kes_matched
                business_df.at[p_idx, "Other_Currency_Matched"] = other_matched
                
                if primary_criteria_met and secondary_criteria_met:
                    business_df.at[p_idx, "Mismatch_Type"] = "None"
                    match_type = "Full"
                elif primary_criteria_met:
                    business_df.at[p_idx, "Mismatch_Type"] = "KES only mismatch"
                    business_df.at[p_idx, "KES_Equivalent_Matched"] = False
                    match_type = "Primary only"
                elif secondary_criteria_met:
                    business_df.at[p_idx, "Mismatch_Type"] = "Other currency only mismatch"
                    business_df.at[p_idx, "Other_Currency_Matched"] = False
                    match_type = "Secondary only"

                record = {
                    "FX index": idx,
                    "Action Type": fx_row.get("Action Type"),
                    "Primary Amount": fx_primary_amount,
                    "Primary Info": primary_info_raw,
                    "Secondary Amount": fx_secondary_amount,
                    "Secondary Info": sec_info_raw,
                    "business Deal Type": business_row.get("Deal type"),
                    "business Amount": business_amount,
                    "business KES eq": business_kes,
                    "Match Type": match_type,
                    "Reasons": order_reasons(local_reasons),
                }
                if is_sell_action:
                    matched_sell.append(record)
                else:
                    matched_buy.append(record)

                found_match = True
                break
            else:
                # Record partial matches for tracking (even if not fully matched)
                business_df.at[p_idx, "KES_Equivalent_Matched"] = kes_matched
                business_df.at[p_idx, "Other_Currency_Matched"] = other_matched
                
                if not kes_matched and not other_matched:
                    business_df.at[p_idx, "Mismatch_Type"] = "Both mismatch"
                elif not kes_matched:
                    business_df.at[p_idx, "Mismatch_Type"] = "KES only mismatch"
                elif not other_matched:
                    business_df.at[p_idx, "Mismatch_Type"] = "Other currency only mismatch"

        # ---- AFTER LOOP ----
        if not found_match:
            # collect all unmatched categories
            if not amount_matched:
                reasons.append("No primary amount match found")
            if not kes_eq_matched:
                reasons.append("No KES equivalent match found")
            if not currency_matched:
                reasons.append("No currency match found")
            if not bank_matched:
                reasons.append("No bank match found")

            rec = {
                "FX index": idx,
                "Action Type": fx_row.get("Action Type"),
                "Primary Amount": fx_primary_amount,
                "Primary Info": primary_info_raw,
                "Status": "No match",
                "Reasons": order_reasons(reasons) if reasons else ["No eligible business rows"]
            }
            if is_sell_action:
                unmatched_sell.append(rec)
            else:
                unmatched_buy.append(rec)
    st.dataframe(business_df)

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



def business_reconciliation_app_ntr():
    st.title("📑 FX vs business Reconciliation")

    fx_trade_df = st.session_state.get("fx_trade_tracker_df", pd.DataFrame())
    if fx_trade_df.empty:
        st.warning("⚠️ FX Trade Tracker not loaded. Please upload it first on the FX Trade Reconciliation page.")
        return

    uploaded_business_file = st.file_uploader("Upload business FX Transactions (Excel)", type=["xlsx"], key="business_file")
    if not uploaded_business_file:
        st.info("Upload the business file to continue.")
        return

    try:
        business_df = pd.read_excel(uploaded_business_file, sheet_name=0)
        st.success("business FX Transactions file loaded successfully!")
        
        if 'Amount' in business_df.columns and 'Rate' in business_df.columns:
            business_df['KES equivalent'] = business_df['Amount'] * business_df['Rate']
        st.dataframe(business_df.head())
    except Exception as e:
        st.error(f"Error loading business file: {e}")
        return

    debug = st.checkbox("Enable Debug Mode", value=False)
    tolerance = st.slider("Matching tolerance (percentage):", 0.01, 1.0, 0.1, step=0.01)

    if st.button("Run Reconciliation"):
        with st.spinner("Reconciling..."):
            results = reconcile_fx_with_business(fx_trade_df, business_df, pct_tolerance=tolerance / 100.0, debug=debug)

    st.success("✅ Reconciliation complete!")

    matched_buy_df = results["matched_buy_df"]
    matched_sell_df = results["matched_sell_df"]
    unmatched_buy_df = results["unmatched_buy_df"]
    unmatched_sell_df = results["unmatched_sell_df"]
    unmatched_business_df = results["unmatched_business_df"]

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

    def df_to_csv(df): return df.to_csv(index=False).encode("utf-8")

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
    if not results["final_business_df"].empty:
        st.download_button("Download Full business with Match Status", df_to_csv(results["final_business_df"]), "Final_business_with_Match_Status.csv", "text/csv")


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

# Helper functions (keep these the same as in your original code)
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def business_reconciliation_app():
    st.title("📑 FX vs business Reconciliation")
    
    # Mode selection
    mode = st.radio("Select Mode:", [ "Interactive Final Report Mode", "Standard Mode"], 
                   help="Standard Mode: Basic reconciliation with downloads. Interactive Mode: Edit final report with row management.")
    
    fx_trade_df = st.session_state.get("fx_trade_tracker_df", pd.DataFrame())
    if fx_trade_df.empty:
        st.warning("⚠️ FX Trade Tracker not loaded. Please upload it first on the FX Trade Reconciliation page.")
            # FX Trade Tracker Upload
        st.subheader("Upload FX Trade Tracker")
        uploaded_fx_file = st.file_uploader("Choose FX Trade Tracker (CSV or XLSX)", type=["csv", "xlsx"], key="fx_uploader")

            # Initialize or load FX trade data
        fx_trade_df = pd.DataFrame()
        if 'fx_trade_tracker_df' in st.session_state:
            fx_trade_df = st.session_state.fx_trade_tracker_df
        else:
            try:
                loaded_df = load_dataframe('fx_trade_tracker_df.pkl')
                if not loaded_df.empty:
                    fx_trade_df = loaded_df
                    st.session_state.fx_trade_tracker_df = loaded_df
            except Exception as e:
                st.warning(f"Could not load cached FX trade data: {e}")

        if uploaded_fx_file:
            try:
                save_uploaded_file(uploaded_fx_file, "fx_trade_uploaded." + uploaded_fx_file.name.split('.')[-1])
                if uploaded_fx_file.name.endswith('.xlsx'):
                    xls = pd.ExcelFile(uploaded_fx_file)
                    sheet_names = xls.sheet_names
                    
                    # Initialize session state if it doesn't exist
                    if 'fx_trade_tracker_sheet' not in st.session_state:
                        st.session_state.fx_trade_tracker_sheet = sheet_names[0]  # Default to first sheet
                    
                    selected_sheet = st.selectbox(
                        "Select sheet for FX Tracker", 
                        sheet_names, 
                        key="fx_sheet_selector",
                        index=sheet_names.index(st.session_state.fx_trade_tracker_sheet) 
                        if st.session_state.fx_trade_tracker_sheet in sheet_names 
                        else 0
                    )
                    
                    # Update session state if selection changed
                    if selected_sheet != st.session_state.fx_trade_tracker_sheet:
                        st.session_state.fx_trade_tracker_sheet = selected_sheet
                        save_object(selected_sheet, "fx_trade_tracker_sheet.pkl")
                        
                    fx_trade_df = pd.read_excel(uploaded_fx_file, sheet_name=selected_sheet)
                else:
                    fx_trade_df = pd.read_csv(uploaded_fx_file)

                fx_trade_df.columns = fx_trade_df.columns.str.strip()
                st.success("FX Trade Tracker loaded successfully!")
                st.dataframe(fx_trade_df.head())

                # Initialize column mapping if it doesn't exist
                if 'fx_trade_tracker_col_mapping' not in st.session_state:
                    st.session_state.fx_trade_tracker_col_mapping = {
                    'Action Type': 'Action Type',
                    'Status': 'Status',
                    'Created At': 'Created At',
                    'Buy Currency Amount': 'Buy Currency Amount',
                    'Buy Trade Info': 'Buy Trade Info',
                    'Sell Currency Amount': 'Sell Currency Amount',
                    'Sell Trade Info': 'Sell Trade Info',
                    'Vendor ID': 'Vendor ID',
                    'Vendor Name' : 'Vendor Name',
                    'Counterparty Dealer' : 'Counterparty Dealer',
                }

                # Column mapping for FX Trade Tracker
                st.subheader("FX Trade Tracker Column Mapping")
                fx_col_options = ['-- Select Column --'] + fx_trade_df.columns.tolist()
                col_mapping = {}

                # Define the required FX columns and their default/suggested mappings
                fx_required_cols = {
                    'Action Type': 'Action Type',
                    'Status': 'Status',
                    'Created At': 'Created At',
                    'Buy Currency Amount': 'Buy Currency Amount',
                    'Buy Trade Info': 'Buy Trade Info',
                    'Sell Currency Amount': 'Sell Currency Amount',
                    'Sell Trade Info': 'Sell Trade Info',
                    'Vendor ID': 'Vendor ID',
                    'Vendor Name' : 'Vendor Name',
                    'Counterparty Dealer' : 'Counterparty Dealer',
                }

                for display_name, suggested_col in fx_required_cols.items():
                    initial_selection = (
                        st.session_state.fx_trade_tracker_col_mapping.get(display_name)
                        if display_name in st.session_state.fx_trade_tracker_col_mapping
                        else suggested_col if suggested_col in fx_col_options
                        else '-- Select Column --'
                    )
                    selected_col = st.selectbox(
                        f"Map '{display_name}' to:",
                        options=fx_col_options,
                        index=fx_col_options.index(initial_selection) if initial_selection in fx_col_options else 0,
                        key=f"fx_map_select_{display_name}"
                    )
                    col_mapping[display_name] = selected_col if selected_col != '-- Select Column --' else None

                renamed_fx_df = pd.DataFrame()
                mapped_columns_dict = {selected: original for original, selected in col_mapping.items() if selected and selected in fx_trade_df.columns}

                if mapped_columns_dict:
                    cols_to_keep = list(mapped_columns_dict.keys())
                    renamed_fx_df = fx_trade_df[cols_to_keep].rename(columns=mapped_columns_dict)
                    fx_trade_df = renamed_fx_df
                    st.success("FX Trade Tracker columns mapped successfully!")
                    st.dataframe(fx_trade_df.head())
                else:
                    st.warning("No FX Trade Tracker columns mapped. Proceeding with original column names.")

                st.session_state.fx_trade_tracker_df = fx_trade_df
                save_dataframe(fx_trade_df, "fx_trade_tracker_df.pkl")
                st.session_state.fx_trade_tracker_col_mapping = {
                    'Action Type': 'Action Type',
                    'Status': 'Status',
                    'Created At': 'Created At',
                    'Buy Currency Amount': 'Buy Currency Amount',
                    'Buy Trade Info': 'Buy Trade Info',
                    'Sell Currency Amount': 'Sell Currency Amount',
                    'Sell Trade Info': 'Sell Trade Info',
                    'Vendor ID': 'Vendor ID',
                    'Vendor Name' : 'Vendor Name',
                    'Counterparty Dealer' : 'Counterparty Dealer',
                }
                save_object(col_mapping, "fx_trade_tracker_col_mapping.pkl")

            except Exception as e:
                st.error(f"Error loading FX Trade Tracker: {e}")
        return

    uploaded_business_file = st.file_uploader("Upload business FX Transactions (Excel)", type=["xlsx"], 
                                         key="business_file_unified")
    if not uploaded_business_file:
        st.info("Upload the business file to continue.")
        return

    try:
        business_df = pd.read_excel(uploaded_business_file, sheet_name=0)
        st.success("business FX Transactions file loaded successfully!")
        
        if 'Amount' in business_df.columns and 'Rate' in business_df.columns:
            business_df['KES equivalent'] = business_df['Amount'] * business_df['Rate']
        st.dataframe(business_df.head())
    except Exception as e:
        st.error(f"Error loading business file: {e}")
        return

    debug = st.checkbox("Enable Debug Mode", value=False)
    tolerance = st.slider("Matching tolerance (percentage):", 0.01, 1.0, 0.1, step=0.01)

    if st.button("Run Reconciliation"):
        with st.spinner("Reconciling..."):
            results = reconcile_fx_with_business(
                fx_trade_df, business_df,
                pct_tolerance=tolerance / 100.0,
                debug=debug
            )
        st.success("✅ Reconciliation complete!")

        if results:
            # Store results in session state for reuse
            st.session_state["matched_buy_df"] = results.get("matched_buy_df", pd.DataFrame())
            st.session_state["matched_sell_df"] = results.get("matched_sell_df", pd.DataFrame())
            st.session_state["unmatched_buy_df"] = results.get("unmatched_buy_df", pd.DataFrame())
            st.session_state["unmatched_sell_df"] = results.get("unmatched_sell_df", pd.DataFrame())
            st.session_state["unmatched_business_df"] = results.get("unmatched_business_df", pd.DataFrame())
            st.session_state["final_business_df"] = results.get("final_business_df", pd.DataFrame()).copy()


    # Extract results
    matched_buy_df = st.session_state.get("matched_buy_df", pd.DataFrame())
    matched_sell_df = st.session_state.get("matched_sell_df", pd.DataFrame())
    unmatched_buy_df = st.session_state.get("unmatched_buy_df", pd.DataFrame())
    unmatched_sell_df = st.session_state.get("unmatched_sell_df", pd.DataFrame())
    unmatched_business_df = st.session_state.get("unmatched_business_df", pd.DataFrame())
    final_business_df = st.session_state.get("final_business_df", pd.DataFrame()).copy()

    # Ensure the right logic columns exist (from business_reconciliation_app)
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
        # Apply display rules for interactive mode
        display_df = final_business_df.copy()
        # Keep original values for potential reconciliation
        display_df["_orig_Amount"] = display_df.get("Amount", "")
        display_df["_orig_KES_equivalent"] = display_df.get("KES equivalent", "")

        # Format 'Amount' for display - show only if matched
        def conditional_amount(row):
            try:
                if bool(row.get("Other_Currency_Matched", False)):
                    return row.get("Amount")
                return ""
            except Exception:
                return ""

        # Format 'KES equivalent' for display - show only if matched
        def conditional_kes(row):
            try:
                if bool(row.get("KES_Equivalent_Matched", False)):
                    return row.get("KES equivalent")
                return ""
            except Exception:
                return ""

        display_df["Amount"] = display_df.apply(conditional_amount, axis=1)
        display_df["KES equivalent"] = display_df.apply(conditional_kes, axis=1)

        # Put our display df into the TARGET column layout
        display_df = ensure_columns_and_order(display_df, TARGET_COLUMNS)

        # === Interactive Final business editor ===
        st.subheader("Interactive Final business with Match Status (editable)")

        try:
            editor = st.experimental_data_editor(display_df, num_rows="dynamic", 
                                                key="final_business_editor", use_container_width=True)
            edited_df = editor.copy()
        except Exception:
            edited_df = st.data_editor(display_df, use_container_width=True, key="final_business_editor")

        # Row management controls
        cols = st.columns([2, 2, 2])
        with cols[0]:
            if st.button("Add blank row"):
                temp = st.session_state.get("final_business_edits", edited_df)
                blank = {c: "" for c in temp.columns}
                temp = pd.concat([temp, pd.DataFrame([blank])], ignore_index=True)
                st.session_state["final_business_edits"] = temp
                st.rerun()
        
        with cols[1]:
            current_df = st.session_state.get("final_business_edits", edited_df)
            row_indices = list(map(str, list(current_df.index)))
            sel = st.multiselect("Select rows (index) to delete", row_indices, key="delete_row_select")
            if st.button("Delete selected rows"):
                if sel:
                    temp = current_df.copy()
                    to_drop = [int(s) for s in sel]
                    temp = temp.drop(index=to_drop).reset_index(drop=True)
                    st.session_state["final_business_edits"] = temp
                    st.success(f"Deleted {len(to_drop)} rows.")
                    st.rerun()
        
        with cols[2]:
            if st.button("Refresh editor"):
                st.session_state["final_business_edits"] = edited_df
                st.rerun()

        # Persist edited data in session state
        if "final_business_edits" not in st.session_state:
            st.session_state["final_business_edits"] = edited_df
        else:
            if not edited_df.equals(st.session_state["final_business_edits"]):
                st.session_state["final_business_edits"] = edited_df

        final_edited_df = st.session_state["final_business_edits"].copy()

        # Show preview
        st.write("Preview (editable):")
        st.dataframe(final_edited_df.head(20), use_container_width=True)

        # Build summary
        for c in ["Mismatch_Type", "Other_Currency_Matched", "KES_Equivalent_Matched"]:
            if c not in final_edited_df.columns:
                final_edited_df[c] = "" if c == "Mismatch_Type" else False

        summary_df = build_summary_from_final(final_edited_df)

        st.subheader("SUMMARY")
        st.table(summary_df)

        # Download options for interactive mode
        st.subheader("Download / Save final report")
        if st.button("Download as Excel (with SUMMARY)"):
            excel_bytes = df_to_excel_bytes(final_edited_df, summary_df=summary_df)
            st.download_button("Download Final business Excel", data=excel_bytes, 
                                file_name="Final-business-with-match-status.xlsx", 
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        if st.button("Download as CSV (main table only)"):
            st.download_button("Download CSV", df_to_csv_bytes(final_edited_df), 
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
        if not final_edited_df.empty:
            st.download_button("Download Full business with Match Status (CSV)", df_to_csv(final_edited_df), 
                                "Final_business_with_Match_Status.csv", "text/csv")


# Run if launched directly
if __name__ == "__main__":
    business_reconciliation_app()