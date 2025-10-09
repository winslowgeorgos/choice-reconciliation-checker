# main_dashboard.py
"""
Enhanced Finance (FX) Reconciliation Dashboard
- Deep royal purple + red-orange banking theme
- Embedded icons (SVG), animated header and metric cards
- Company logo upload (stores in session state) with default embedded logo (no placeholders)
- All functionality preserved from original dashboard; designed for production-looking UI
"""

import streamlit as st
from io import BytesIO
import pandas as pd
from datetime import datetime
import base64
import io

# ----------------------------
# Placeholder imports / fallbacks
# (kept to ensure the single-file app runs even if sub-pages are absent)
# ----------------------------
try:
    from fx_reconcilliation_app_page import fx_reconciliation_app
    from fx_trade_reconciliation_page import graphed_analysis_app
    from combine_match_results_page import run_cross_match_analysis, cross_match_analysis_app
    from business_fx_reconciliation_page import business_reconciliation_app
except ImportError:
    # Keep safe defaults to allow the UI to run
    def fx_reconciliation_app(bank_dfs): return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    def graphed_analysis_app(bank_dfs): return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    def run_cross_match_analysis(*args, **kwargs): st.error("Required reconciliation function not found.")
    def cross_match_analysis_app(): st.info("Cross-Match results will appear here after analysis.")
    def business_reconciliation_app(*args, **kwargs): st.info("Business Reconciliation analysis goes here.")
    st.warning("One or more reconciliation modules are missing. The main dashboard will run, but advanced pages will show placeholders.")


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="ChoiceBank — FX Reconciliation",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------
# Embedded default company logo (SVG -> PNG fallback encoded)
# This SVG is bundled as default logo so there are no external placeholders.
# ----------------------------
DEFAULT_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="320" height="80" viewBox="0 0 320 80">
  <defs>
    <linearGradient id="g1" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#FF6B35"/>
      <stop offset="1" stop-color="#6B4FA8"/>
    </linearGradient>
  </defs>
  <rect rx="12" width="320" height="80" fill="#0F1422"/>
  <g transform="translate(20,16)" fill="url(#g1)">
    <rect x="0" y="0" width="48" height="48" rx="8" />
    <path d="M14 8 L34 8 L34 28 L14 28 Z" fill="white" opacity="0.08"/>
  </g>
  <g transform="translate(84,22)" fill="#E2E8F0" font-family="Inter, Arial, sans-serif">
    <text x="0" y="14" font-size="18" font-weight="800">CHOICE</text>
    <text x="0" y="38" font-size="14" fill="#94A3B8">BANK • Reconciliation</text>
  </g>
</svg>
"""

def svg_to_data_uri(svg_text: str) -> str:
    b = svg_text.encode('utf-8')
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

DEFAULT_LOGO_DATAURI = svg_to_data_uri(DEFAULT_LOGO_SVG)

# ----------------------------
# Custom CSS (theme, cards, animations, icons)
# ----------------------------
st.markdown(
    f"""
<style>
:root {{
    --royal-purple: #4A2C8F;
    --deep-purple: #3A1C6E;
    --light-purple: #6B4FA8;
    --red-orange: #FF6B35;
    --light-orange: #FF8E5E;
    --dark-bg: #0F1422;
    --card-bg: #1E293B;
    --muted: #94A3B8;
    --text-light: #F8FAFC;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: linear-gradient(180deg, rgba(15,20,34,1) 0%, rgba(8,10,20,1) 100%);
    color: var(--text-light);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}

.stApp {{
    padding-top: 18px;
    padding-bottom: 24px;
}}

/* Sidebar - dark with branding */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(12,15,26,0.98), rgba(16,19,33,0.98));
    border-right: 1px solid rgba(255,255,255,0.03);
    padding-top: 12px;
}}

/* Logo container inside sidebar */
.logo-container {{
    padding: 18px 12px;
    text-align: center;
    margin-bottom: 12px;
    border-radius: 10px;
    background: linear-gradient(90deg, rgba(255,107,53,0.06), rgba(75,43,143,0.04));
    border: 1px solid rgba(255,255,255,0.02);
}}
.logo-container img {{
    height: 56px;
    object-fit: contain;
    margin-bottom: 6px;
    border-radius: 8px;
    box-shadow: 0 6px 18px rgba(74,44,143,0.12);
}}

/* Header */
.main-header {{
    background: linear-gradient(120deg, rgba(74,44,143,0.92), rgba(255,107,53,0.92));
    padding: 22px;
    border-radius: 12px;
    color: white;
    box-shadow: 0 18px 40px rgba(10,10,30,0.6);
    border-top: 4px solid rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
    animation: headerFadeIn 900ms ease-out both;
}}
.header-sub {{
    color: rgba(226,232,240,0.9);
    margin-top: 6px;
}}

/* animated overlay sweep */
.main-header::after {{
    content: "";
    position: absolute;
    left: -120%;
    top: -40%;
    width: 160%;
    height: 200%;
    transform: rotate(25deg);
    background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    animation: sweep 3.2s ease-in-out infinite;
    opacity: 0.6;
}}
@keyframes sweep {{
    0% {{ left: -120%; }}
    50% {{ left: 10%; }}
    100% {{ left: -120%; }}
}}
@keyframes headerFadeIn {{
    0% {{ transform: translateY(-20px); opacity: 0; }}
    100% {{ transform: translateY(0); opacity: 1; }}
}}

/* Section header style */
.section-header {{
    margin: 22px 0 12px 0;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text-light);
    display: flex;
    align-items: center;
    gap: 12px;
}}
.section-header .left-accent {{
    width: 6px;
    height: 36px;
    border-radius: 6px;
    background: linear-gradient(180deg, var(--red-orange), var(--royal-purple));
    box-shadow: 0 6px 20px rgba(74,44,143,0.12);
}}

/* Metric cards */
.metric-card {{
    background: linear-gradient(180deg, rgba(30,41,59,0.9), rgba(18,24,36,0.9));
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(2,6,23,0.6);
    border: 1px solid rgba(255,255,255,0.03);
    transition: transform 300ms cubic-bezier(.2,.9,.2,1), box-shadow 300ms;
    min-height: 120px;
}}
.metric-card:hover {{
    transform: translateY(-8px) scale(1.01);
    box-shadow: 0 18px 50px rgba(74,44,143,0.16);
}}
.card-row {{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap: 12px;
}}
.card-icon {{
    width:56px; height:56px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(180deg, rgba(255,107,53,0.12), rgba(106,61,153,0.08));
    border: 1px solid rgba(255,255,255,0.02);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}}
.stat-highlight {{
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--text-light);
    letter-spacing: 0.4px;
}}
.metric-label {{
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

/* Buttons (streamlit) */
.stButton>button {{
    background: linear-gradient(90deg, var(--royal-purple), var(--light-purple));
    color: white;
    border-radius: 10px;
    padding: 10px 14px;
    border: none;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(74,44,143,0.12);
}}
.stButton>button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 14px 30px rgba(255,107,53,0.16);
}}

/* Status boxes */
.status-box {{
    padding:12px;
    border-radius:10px;
    background: rgba(30,41,59,0.5);
    border-left: 4px solid;
    display:flex;
    align-items:center;
    gap:10px;
    font-weight:700;
}}
.status-success {{ color:#10B981; border-color:#10B981; }}
.status-pending {{ color:var(--red-orange); border-color:var(--red-orange); }}
.status-warning {{ color:#F59E0B; border-color:#F59E0B; }}

/* Dataframe styling fallback */
.stDataFrame>div {{
    background: transparent;
}}

/* Small responsive tweaks */
@media (max-width: 800px) {{
    .card-icon {{ width:48px; height:48px; }}
    .stat-highlight {{ font-size: 1.4rem; }}
}}

</style>
""",
    unsafe_allow_html=True
)


# ----------------------------
# Constants & expected columns (kept from original)
# ----------------------------
DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d', '%d/%m/%Y',
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d.%m.%Y %H:%M:%S',
    '%Y.%m.%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'
]

PREDEFINED_BANK_CURRENCY_OPTIONS = [
    "Absa KES", "Absa USD", "Absa EUR", "Absa GBP",
    "CBK KES", "CBK USD", "CBK EUR", "CBK GBP",
    "Equity KES", "Equity USD", "Equity EUR", "Equity GBP",
    "I&M KES", "I&M USD", "I&M EUR", "I&M GBP",
    "KCB KES", "KCB USD", "KCB EUR", "KCB GBP",
    "Kingdom KES", "Kingdom USD", "Kingdom EUR", "Kingdom GBP",
    "NCBA KES", "NCBA USD", "NCBA EUR", "NCBA GBP",
    "SBM KES", "SBM USD", "SBM EUR", "SBM GBP",
    "UBA KES", "UBA USD", "UBA EUR", "UBA GBP",
    "BAAS Temporary KES", "BAAS Temporary USD", "BAAS Temporary EUR", "BAAS Temporary GBP",
    "FX Temporary KES", "FX Temporary USD", "FX Temporary EUR", "FX Temporary GBP",
    "Other Temporary KES", "Other Temporary USD", "Other Temporary EUR", "Other Temporary GBP",
    "Unclaimed Funds KES", "Unclaimed Funds USD", "Unclaimed Funds EUR", "Unclaimed Funds GBP",
    "Yeepay KES", "Yeepay USD", "Yeepay EUR", "Yeepay GBP"
]

FX_EXPECTED_COLUMNS = {
    'Amount': 'Amount', 'Operation': 'Operation', 'Completed At': 'Completed At',
    'Intermediary Account': 'Intermediary Account', 'Currency': 'Currency', 'Status': 'Status'
}

BANK_EXPECTED_COLUMNS = {
    'Date': ['Date', 'Transaction Date', 'Value Date', 'Value date'],
    'Credit': ['Credit', 'Credit Amount', 'Money In', 'Deposit', 'Credit amount'],
    'Debit': ['Debit', 'Debit Amount', 'Money Out', 'Withdrawal', 'Debit amount'],
    'Description': ['Description', 'Narrative', 'Transaction Details', 'Customer reference', 'Transaction Remarks:', 'Transaction Details', 'TransactionDetails', 'Transaction\\nDetails'],
    'Running Balances': ['Running Balances', 'Running Balance', 'Running Balance (KES)', 'Running Balance (USD)', 'RUNNING BALANCES']
}

# ----------------------------
# Utility helpers (kept from original)
# ----------------------------
def parse_date(date_str_raw):
    if pd.isna(date_str_raw) or date_str_raw == pd.NaT:
        return None
    if isinstance(date_str_raw, datetime):
        return date_str_raw
    if not isinstance(date_str_raw, str):
        date_str_raw = str(date_str_raw)
    date_str = date_str_raw.strip()
    # try many formats
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            # try stripping time portion if present
            try:
                return datetime.strptime(date_str.split()[0], fmt)
            except Exception:
                continue
    return None

def safe_float(x):
    if pd.isna(x) or x is None:
        return None
    try:
        cleaned_x = str(x).replace(',', '').strip()
        return float(cleaned_x)
    except (ValueError, TypeError):
        return None

def process_uploaded_file(uploaded_file, sheet_name=None):
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith('.csv'):
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1', 'windows-1252']
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except Exception:
                continue
        st.error(f"Failed to decode CSV file '{uploaded_file.name}'.")
        return pd.DataFrame()
    elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
        try:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Error reading Excel file '{uploaded_file.name}': {e}")
            return pd.DataFrame()
    else:
        st.error("Unsupported file type. Please upload CSV or Excel files.")
        return pd.DataFrame()

def get_excel_sheet_names(uploaded_file):
    uploaded_file.seek(0)
    try:
        x = pd.ExcelFile(uploaded_file)
        return x.sheet_names
    except Exception as e:
        st.error(f"Error getting sheet names: {e}")
        return []


# ----------------------------
# Session state initialization
# ----------------------------
if 'df_matched_adjustments_local' not in st.session_state: st.session_state.df_matched_adjustments_local = pd.DataFrame()
if 'df_matched_adjustments_foreign' not in st.session_state: st.session_state.df_matched_adjustments_foreign = pd.DataFrame()
if 'df_unmatched_adjustments_local' not in st.session_state: st.session_state.df_unmatched_adjustments_local = pd.DataFrame()
if 'df_unmatched_adjustments_foreign' not in st.session_state: st.session_state.df_unmatched_adjustments_foreign = pd.DataFrame()
if 'df_unmatched_bank_recon' not in st.session_state: st.session_state.df_unmatched_bank_recon = pd.DataFrame()
if 'df_matched_counterparty' not in st.session_state: st.session_state.df_matched_counterparty = pd.DataFrame()
if 'df_matched_choice' not in st.session_state: st.session_state.df_matched_choice = pd.DataFrame()
if 'df_unmatched_counterparty' not in st.session_state: st.session_state.df_unmatched_counterparty = pd.DataFrame()
if 'df_unmatched_choice' not in st.session_state: st.session_state.df_unmatched_choice = pd.DataFrame()
if 'df_unmatched_bank_trade' not in st.session_state: st.session_state.df_unmatched_bank_trade = pd.DataFrame()
if 'df_unmatched_bank_records' not in st.session_state: st.session_state.df_unmatched_bank_records = pd.DataFrame()
if 'debug_mode' not in st.session_state: st.session_state.debug_mode = False
if 'bank_dfs' not in st.session_state: st.session_state.bank_dfs = {}
if 'bank_uploaded_file_objs' not in st.session_state: st.session_state.bank_uploaded_file_objs = []
if 'raw_bank_data_previews' not in st.session_state: st.session_state.raw_bank_data_previews = {}
if 'merged_bank_statement' not in st.session_state: st.session_state.merged_bank_statement = pd.DataFrame()
if "cached_bank_files" not in st.session_state: st.session_state.cached_bank_files = {}
if "company_logo_bytes" not in st.session_state:
    # initialize default from embedded SVG
    st.session_state.company_logo_bytes = base64.b64decode(DEFAULT_LOGO_DATAURI.split(",")[1]) if "," in DEFAULT_LOGO_DATAURI else DEFAULT_LOGO_SVG.encode('utf-8')
if "company_name" not in st.session_state:
    st.session_state.company_name = "Choice Bank"

# ----------------------------
# Sidebar with logo upload + navigation
# ----------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="{DEFAULT_LOGO_DATAURI}" alt="Choice Bank logo" />
            <div style="font-weight:800; color:var(--text-light); margin-top:6px; font-size:14px;">{st.session_state.company_name}</div>
            <div style="color:var(--muted); font-size:12px; margin-top:4px;">FX Reconciliation Console</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.subheader("Branding")
    logo_file = st.sidebar.file_uploader("Upload Company Logo (PNG/SVG)", type=["png", "jpg", "jpeg", "svg"], key="logo_upload", help="Upload your company logo to appear in the header and exports.")
    if logo_file is not None:
        # store uploaded logo bytes so it persists in session
        st.session_state.company_logo_bytes = logo_file.read()
        # also update the display name if filename looks like company name
        name_guess = logo_file.name.split('.')[0].replace('_', ' ').title()
        st.session_state.company_name = st.sidebar.text_input("Company display name", value=name_guess)
    else:
        # allow editing name even if no new logo uploaded
        st.session_state.company_name = st.sidebar.text_input("Company display name", value=st.session_state.company_name)

    st.sidebar.markdown("---")
    page_selection = st.sidebar.radio(
        "Navigate to:",
        ["📊 Dashboard Overview", "🏛️ Bank Statement Management", "🔍 Adjustments Reconciliation",
         "💱 FX Trade Reconciliation", "🏢 Business FX Reconciliation", "🔄 Cross-Match Analysis"],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Built for secure bank teams • Deep royal purple + red-orange visual system")


# Helper to produce data URI for current logo bytes
def logo_bytes_to_data_uri(b: bytes) -> str:
    if not b:
        return DEFAULT_LOGO_DATAURI
    # Try to detect if provided bytes already are svg text
    try:
        text = b.decode('utf-8')
        if text.strip().startswith("<svg"):
            b64 = base64.b64encode(b).decode('utf-8')
            return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        pass
    # else assume raster image (png/jpeg)
    mime = "image/png"
    # quick heuristic: check jpeg magic bytes
    if b[:2] == b'\xff\xd8':
        mime = "image/jpeg"
    b64 = base64.b64encode(b).decode('utf-8')
    return f"data:{mime};base64,{b64}"

CURRENT_LOGO_URI = logo_bytes_to_data_uri(st.session_state.company_logo_bytes)


# ----------------------------
# Inline utility: small SVG icons used in metric cards (embedded, no external fetch)
# ----------------------------
ICON_FOLDER_SVG = {
    "files": """
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none">
  <rect x="3" y="3" width="14" height="14" rx="2" stroke="white" stroke-opacity="0.95" stroke-width="1.6" fill="url(#g)"/>
  <path d="M7 11h10M7 15h6" stroke="white" stroke-opacity="0.95" stroke-width="1.4" stroke-linecap="round"/>
</svg>
""",
    "transactions": """
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none">
  <path d="M12 3v6" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M12 21v-6" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M5 10l7-7 7 7" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
  <path d="M19 14l-7 7-7-7" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
</svg>
""",
    "currency": """
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none">
  <circle cx="12" cy="12" r="8" stroke="white" stroke-width="1.6"/>
  <path d="M8 12h8" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M12 8v8" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
</svg>
""",
    "matched": """
<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none">
  <path d="M20 6L9 17l-5-5" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""
}

def svg_to_img_tag(svg: str, bgcolor: bool = False) -> str:
    # Return an inline SVG wrapped appropriately
    return f'<div style="display:flex; align-items:center; justify-content:center;">{svg}</div>'


# ----------------------------
# Dashboard pages
# ----------------------------
if page_selection == "📊 Dashboard Overview":
    # Header (logo + title + subtitle)
    st.markdown(
        f"""
        <div class="main-header" style="display:flex; gap:18px; align-items:center; justify-content:space-between;">
            <div style="display:flex; gap:14px; align-items:center;">
                <img src="{CURRENT_LOGO_URI}" alt="logo" style="height:64px; border-radius:8px;"/>
                <div>
                    <div style="font-size:20px; font-weight:800;">{st.session_state.company_name} — FX Reconciliation Dashboard</div>
                    <div class="header-sub">Comprehensive Foreign Exchange Transaction Monitoring & Secure Reconciliation</div>
                </div>
            </div>
            <div style="display:flex; gap:12px; align-items:center;">
                <div style="text-align:right; color: rgba(226,232,240,0.9); font-size:13px;">Team • Audit-ready • Secure</div>
                <div style="background: rgba(255,255,255,0.04); padding:8px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.03);">
                    <strong style="letter-spacing:0.6px;">v1.0</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Metrics overview
    st.markdown('<div class="section-header"><div class="left-accent"></div>Reconciliation Metrics Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="large")
    # Metric 1: Processed Bank Files
    with col1:
        count_files = len(st.session_state.bank_dfs)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-row">
                    <div style="display:flex; gap:12px;">
                        <div class="card-icon">{svg_to_img_tag(ICON_FOLDER_SVG['files'])}</div>
                        <div>
                            <div class="stat-highlight">{count_files}</div>
                            <div class="metric-label">Processed Bank Files</div>
                        </div>
                    </div>
                    <div style="text-align:right; color:var(--muted); font-size:12px;">Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Metric 2: Total Bank Records
    with col2:
        total_transactions = sum(len(df) for df in st.session_state.bank_dfs.values()) if st.session_state.bank_dfs else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-row">
                    <div style="display:flex; gap:12px;">
                        <div class="card-icon">{svg_to_img_tag(ICON_FOLDER_SVG['transactions'])}</div>
                        <div>
                            <div class="stat-highlight">{total_transactions:,}</div>
                            <div class="metric-label">Total Bank Records</div>
                        </div>
                    </div>
                    <div style="text-align:right; color:var(--muted); font-size:12px;">Records</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Metric 3: Active Currencies
    with col3:
        currencies = set()
        for bank_name in st.session_state.bank_dfs.keys():
            if ' ' in str(bank_name):
                currencies.add(str(bank_name).split()[-1])
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-row">
                    <div style="display:flex; gap:12px;">
                        <div class="card-icon">{svg_to_img_tag(ICON_FOLDER_SVG['currency'])}</div>
                        <div>
                            <div class="stat-highlight">{len(currencies)}</div>
                            <div class="metric-label">Active Currencies</div>
                        </div>
                    </div>
                    <div style="text-align:right; color:var(--muted); font-size:12px;">{', '.join(sorted(currencies)) if currencies else '—'}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Metric 4: Matched Trades
    with col4:
        total_matched = len(st.session_state.df_matched_adjustments_local) + len(st.session_state.df_matched_counterparty)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="card-row">
                    <div style="display:flex; gap:12px;">
                        <div class="card-icon">{svg_to_img_tag(ICON_FOLDER_SVG['matched'])}</div>
                        <div>
                            <div class="stat-highlight">{total_matched:,}</div>
                            <div class="metric-label">Total Matched Trades/Adjustments</div>
                        </div>
                    </div>
                    <div style="text-align:right; color:var(--muted); font-size:12px;">Matches</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Status summary
    st.markdown('<div class="section-header"><div class="left-accent"></div>Process Status Summary</div>', unsafe_allow_html=True)
    status_col1, status_col2 = st.columns(2, gap="large")
    with status_col1:
        st.subheader("Data Upload & Integrity")
        if st.session_state.bank_dfs:
            st.markdown('<div class="status-box status-success">✅ Bank Statements: PROCESSED & READY</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-warning">⚠️ Bank Statements: AWAITING UPLOAD</div>', unsafe_allow_html=True)
        if not st.session_state.df_matched_adjustments_local.empty:
            st.markdown('<div class="status-box status-success">✅ Adjustments Recon: MATCHES FOUND</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-pending">🔶 Adjustments Recon: PENDING</div>', unsafe_allow_html=True)

    with status_col2:
        st.subheader("Reconciliation Progress")
        if not st.session_state.df_matched_counterparty.empty:
            st.markdown('<div class="status-box status-success">✅ FX Trade Recon: MATCHES FOUND</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-pending">🔶 FX Trade Recon: PENDING</div>', unsafe_allow_html=True)
        if not st.session_state.get("cross_match_complete", False):
            st.markdown('<div class="status-box status-pending">🔶 Cross-Match Analysis: PENDING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-success">✅ Cross-Match Analysis: COMPLETE</div>', unsafe_allow_html=True)

    # Quick actions
    st.markdown('<div class="section-header"><div class="left-accent"></div>Quick Actions</div>', unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns([1,1,1], gap="medium")
    with action_col1:
        if st.button("📥 Upload Bank Statements", use_container_width=True):
            st.experimental_rerun()
    with action_col2:
        if st.session_state.bank_dfs and st.button("🔍 Run Adjustments Reconciliation", use_container_width=True):
            st.session_state.page_redirect = "Adjustments Reconciliation"
            st.experimental_rerun()
    with action_col3:
        if st.session_state.bank_dfs and st.button("💱 Run FX Trade Reconciliation", use_container_width=True):
            st.session_state.page_redirect = "FX Trade Reconciliation"
            st.experimental_rerun()

    # Brief footer
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div style='color:var(--muted); font-size:12px;'>Audit trails, CSV export and secure data handling baked in.</div>", unsafe_allow_html=True)


# ----------------------------
# Bank Statement Management page (kept logic, improved styling)
# ----------------------------
elif page_selection == "🏛️ Bank Statement Management":
    st.markdown(
        f"""
        <div class="main-header" style="padding:18px;">
            <div style="display:flex; gap:14px; align-items:center;">
                <img src="{CURRENT_LOGO_URI}" alt="logo" style="height:54px; border-radius:8px;"/>
                <div>
                    <div style="font-size:18px; font-weight:800;">Bank Statement Management</div>
                    <div style="color: rgba(226,232,240,0.9); margin-top:4px;">Upload, preview, and standardize your statements for reconciliation.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background: rgba(30,41,59,0.6); padding:12px; border-radius:10px; border-left:4px solid var(--red-orange);">
            <strong style="color:var(--red-orange);">Upload Bank Statements</strong>
            <div style="color:var(--muted); font-size:13px;">Supported: CSV, Excel (.xlsx). Use the column mapping to standardize files quickly.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose bank statement files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="bank_uploader_main",
        label_visibility="collapsed"
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.cached_bank_files:
                file_bytes = file.read()
                file_type = file.type
                st.session_state.cached_bank_files[file.name] = {"content": file_bytes, "type": file_type}

    files_to_delete = []
    if st.session_state.cached_bank_files:
        st.markdown('<div class="section-header"><div class="left-accent"></div>Uploaded Bank Statements: Configuration</div>', unsafe_allow_html=True)
        for file_name, file_data in list(st.session_state.cached_bank_files.items()):
            file_key = file_name.lower().replace('.', '_')
            with st.expander(f"🗂️ {file_name} — {len(file_data['content'])/1024:.1f} KB", expanded=False):
                col1, col2 = st.columns([8,2])
                with col2:
                    if st.button("❌ Remove", key=f"remove_{file_name}", use_container_width=True):
                        files_to_delete.append(file_name)
                        continue

                if file_key not in st.session_state.raw_bank_data_previews:
                    fake_file = BytesIO(file_data["content"])
                    fake_file.name = file_name
                    if file_name.lower().endswith(('.xlsx', '.xls')):
                        sheet_names = get_excel_sheet_names(fake_file)
                        selected_sheet = sheet_names[0] if sheet_names else None
                        df = process_uploaded_file(fake_file, sheet_name=selected_sheet)
                    else:
                        sheet_names = []
                        selected_sheet = None
                        df = process_uploaded_file(BytesIO(file_data["content"]))

                    st.session_state.raw_bank_data_previews[file_key] = {
                        'file_obj': fake_file, 'df_raw': df, 'sheet_names': sheet_names,
                        'selected_sheet': selected_sheet, 'column_mappings': {}, 'standardized_name': ""
                    }

                data = st.session_state.raw_bank_data_previews[file_key]
                df_bank_raw = data['df_raw']
                if file_name.lower().endswith(('.xlsx', '.xls')) and data['sheet_names']:
                    current_sheet = st.selectbox(
                        f"Select Sheet for {file_name}:",
                        data['sheet_names'],
                        index=data['sheet_names'].index(data['selected_sheet']) if data['selected_sheet'] in data['sheet_names'] else 0,
                        key=f"bank_sheet_selector_{file_key}",
                    )
                    if current_sheet != data['selected_sheet']:
                        data['selected_sheet'] = current_sheet
                        fake_file = BytesIO(file_data["content"])
                        fake_file.name = file_name
                        df_bank_raw = process_uploaded_file(fake_file, sheet_name=current_sheet)
                        df_bank_raw.columns = df_bank_raw.columns.str.strip()
                        st.info(f"📊 Sheet '{current_sheet}' selected. Columns: {df_bank_raw.columns.tolist()}")
                        data['df_raw'] = df_bank_raw

                selected_standardized_name = st.selectbox(
                    f"Select Standardized Name for {file_name}:",
                    options=[""] + PREDEFINED_BANK_CURRENCY_OPTIONS,
                    index=PREDEFINED_BANK_CURRENCY_OPTIONS.index(data['standardized_name']) + 1 if data['standardized_name'] in PREDEFINED_BANK_CURRENCY_OPTIONS else 0,
                    key=f"standardized_name_selector_{file_key}",
                )
                data['standardized_name'] = selected_standardized_name

                if not df_bank_raw.empty:
                    st.write("**Preview (First 5 Rows):**")
                    st.dataframe(df_bank_raw.head(), use_container_width=True)
                    available_columns = df_bank_raw.columns.tolist()
                    available_columns.insert(0, "")
                    current_mappings = data['column_mappings']
                    st.write("**🔧 Column Mapping:** Map your statement columns to the required fields.")
                    col_map_cols = st.columns(2)
                    for expected_col, default_val_list in BANK_EXPECTED_COLUMNS.items():
                        initial_selection = current_mappings.get(expected_col)
                        if not initial_selection:
                            for default_val in default_val_list:
                                if default_val.strip() in [col.strip() for col in df_bank_raw.columns]:
                                    initial_selection = default_val
                                    break
                        with col_map_cols[0]:
                            st.markdown(f"**{expected_col}**")
                        with col_map_cols[1]:
                            mapped_col = st.selectbox(
                                f"Map '{expected_col}' to:",
                                options=available_columns,
                                index=available_columns.index(initial_selection) if initial_selection and initial_selection in available_columns else 0,
                                key=f"bank_map_{file_key}_{expected_col}",
                                label_visibility="collapsed"
                            )
                            data['column_mappings'][expected_col] = mapped_col if mapped_col else None
                else:
                    st.error(f"❌ Could not load data from {file_name}. Check file structure.")
    # Remove selected files
    for file_name in files_to_delete:
        st.session_state.cached_bank_files.pop(file_name, None)
        file_key = file_name.lower().replace('.', '_')
        st.session_state.raw_bank_data_previews.pop(file_key, None)
        st.success(f"🗑️ Removed: {file_name}")

    # reset current processing outputs until user processes again
    st.session_state.bank_dfs = st.session_state.bank_dfs or {}
    st.session_state.merged_bank_statement = st.session_state.merged_bank_statement or pd.DataFrame()

    # Process button
    if st.button("🚀 Process All Bank Statements", key="process_all_bank_btn_main", use_container_width=True):
        st.session_state.bank_dfs = {}
        all_success = True
        dfs_to_concat = []
        st.session_state.running_balances_col = None

        for file_key, data in st.session_state.raw_bank_data_previews.items():
            st.info(f"🔄 Processing '{data['file_obj'].name}'...")
            if not data['standardized_name']:
                st.error(f"❌ Missing standardized name for '{data['file_obj'].name}'")
                all_success = False
                continue
            if data['standardized_name'] in st.session_state.bank_dfs:
                st.error(f"❌ Duplicate standardized name '{data['standardized_name']}' detected.")
                all_success = False
                continue

            df_to_process = data['df_raw'].copy()
            renamed_cols = {}
            for expected_col, mapped_col in data['column_mappings'].items():
                if mapped_col and mapped_col in df_to_process.columns:
                    renamed_cols[mapped_col] = expected_col
            if renamed_cols:
                df_to_process.rename(columns=renamed_cols, inplace=True)
            df_to_process.columns = df_to_process.columns.str.strip()

            # Validation
            errors = []
            required_cols = ['Date', 'Credit', 'Debit', 'Running Balances']
            missing_cols = [col for col in required_cols if col not in df_to_process.columns]
            if missing_cols:
                st.error(f"❌ Validation failed for '{data['file_obj'].name}'. Missing required columns: {', '.join(missing_cols)}.")
                all_success = False
                continue

            df_to_process['Date'] = df_to_process['Date'].apply(parse_date)
            invalid_dates_mask = df_to_process['Date'].isna()
            if invalid_dates_mask.any():
                num_errors = invalid_dates_mask.sum()
                st.warning(f"⚠️ Warning in '{data['file_obj'].name}': {num_errors} invalid dates found. These rows will be dropped.")
                df_to_process = df_to_process[~invalid_dates_mask].copy()

            df_to_process['Credit'] = df_to_process['Credit'].apply(safe_float)
            df_to_process['Debit'] = df_to_process['Debit'].apply(safe_float)
            df_to_process['Running Balances'] = df_to_process['Running Balances'].apply(safe_float)

            df_to_process["Matched"] = False
            df_to_process['Bank'] = data['standardized_name']
            st.session_state.bank_dfs[data['standardized_name']] = df_to_process
            st.success(f"✅ Processed: {data['file_obj'].name} as '{data['standardized_name']}'")
            dfs_to_concat.append(df_to_process)

        if all_success and dfs_to_concat:
            st.session_state.merged_bank_statement = pd.concat(dfs_to_concat, ignore_index=True)
            st.markdown('<div class="status-box status-success">✅ All bank statements processed and merged successfully!</div>', unsafe_allow_html=True)

            if not st.session_state.merged_bank_statement.empty:
                df_bal = st.session_state.merged_bank_statement.copy()
                rb_col = 'Running Balances'
                df_bal.rename(columns={'Date': 'date', 'Debit': 'debit', 'Credit': 'credit', 'Bank': 'bank'}, inplace=True)
                df_bal["currency"] = df_bal["bank"].apply(lambda x: str(x).split()[-1].upper())
                df_bal = df_bal.sort_values(by=['bank', 'date'])
                per_bank_rows = []
                for bank_name, df_bank in df_bal.groupby("bank"):
                    df_bank = df_bank.sort_values("date").reset_index(drop=True)
                    first_row = df_bank.iloc[0]
                    last_row = df_bank.iloc[-1]
                    currency = str(bank_name).split()[-1].upper()
                    running_balance_first = first_row[rb_col] if rb_col in first_row and pd.notna(first_row[rb_col]) else 0
                    debit_first = first_row["debit"] if "debit" in first_row and pd.notna(first_row["debit"]) else 0
                    credit_first = first_row["credit"] if "credit" in first_row and pd.notna(first_row["credit"]) else 0
                    closing_balance = last_row[rb_col] if rb_col in last_row and pd.notna(last_row[rb_col]) else 0
                    opening_balance = running_balance_first - credit_first + debit_first
                    per_bank_rows.append({"Bank": bank_name, "Currency": currency, "Opening Balance": round(opening_balance, 2), "Closing Balance": round(closing_balance, 2)})
                per_bank_df = pd.DataFrame(per_bank_rows).sort_values(by=["Currency", "Bank"]).reset_index(drop=True)

                # Display balances and charts
                st.markdown('<div class="section-header"><div class="left-accent"></div>Balance Summary</div>', unsafe_allow_html=True)
                bcol1, bcol2 = st.columns(2, gap="large")
                with bcol1:
                    st.subheader("Per-Bank Opening & Closing Balances")
                    st.dataframe(per_bank_df.style.set_properties(**{'background-color': '#0F1422', 'color': '#F8FAFC'}), use_container_width=True)
                    csv_per_bank = per_bank_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Per-Bank Balances CSV", data=csv_per_bank, file_name="per_bank_balances.csv", mime="text/csv", use_container_width=True)
                with bcol2:
                    currency_summary = (per_bank_df.groupby("Currency").agg({"Opening Balance": "sum", "Closing Balance": "sum"}).round(2).reset_index().sort_values(by="Currency").reset_index(drop=True))
                    st.subheader("Currency Summary")
                    st.dataframe(currency_summary.style.set_properties(**{'background-color': '#0F1422', 'color': '#F8FAFC'}), use_container_width=True)
                    csv_summary = currency_summary.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Currency Summary CSV", data=csv_summary, file_name="currency_balance_summary.csv", mime="text/csv", use_container_width=True)

                # Transaction Analytics (bar chart)
                st.markdown("---")
                st.markdown('<div class="section-header"><div class="left-accent"></div>Transaction Analytics</div>', unsafe_allow_html=True)
                st.subheader("Monthly Transaction Volume (Credit vs. Debit)")
                df_chart = st.session_state.merged_bank_statement.copy()
                # safe date coercion
                df_chart['Date'] = pd.to_datetime(df_chart['Date'], errors='coerce')
                df_chart['YearMonth'] = df_chart['Date'].dt.to_period('M').astype(str)
                df_chart['Credit'] = pd.to_numeric(df_chart['Credit'], errors='coerce').fillna(0)
                df_chart['Debit'] = pd.to_numeric(df_chart['Debit'], errors='coerce').fillna(0)
                monthly_volume = df_chart.groupby(['Bank', 'YearMonth']).agg(Total_Credit=('Credit', 'sum'), Total_Debit=('Debit', 'sum')).reset_index()
                # use built-in chart; color param not supported in older streamlit; use columns with values for bar_chart
                chart_df = monthly_volume.groupby('YearMonth').sum().reset_index()
                if not chart_df.empty:
                    st.bar_chart(chart_df.set_index('YearMonth')[['Total_Credit', 'Total_Debit']])
                else:
                    st.info("No transaction volume data available yet.")

        elif all_success and not dfs_to_concat:
            st.info("⚠️ No valid files processed.")
        else:
            st.warning("⚠️ Some files could not be processed. See messages above.")

    st.markdown("---")
    st.markdown('<div class="section-header"><div class="left-accent"></div>Merged Bank Statement</div>', unsafe_allow_html=True)
    if not st.session_state.get("merged_bank_statement", pd.DataFrame()).empty:
        st.write("### Combined Merged Statement:")
        st.dataframe(st.session_state.merged_bank_statement, use_container_width=True)
        csv = st.session_state.merged_bank_statement.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Merged Bank Statement as CSV", data=csv, file_name="merged_bank_statement.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("📭 No merged bank statement available yet.")

# ----------------------------
# Adjustments Reconciliation page
# ----------------------------
elif page_selection == "🔍 Adjustments Reconciliation":
    st.markdown(
        """
        <div class="main-header" style="padding:16px;">
            <div>
                <div style="font-size:18px; font-weight:800;">Adjustments Reconciliation</div>
                <div style="color: rgba(226,232,240,0.9); margin-top:4px;">Local & Foreign adjustments matching analysis.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.bank_dfs:
        st.markdown('<div class="status-box status-warning">⚠️ Please upload & process bank statements first.</div>', unsafe_allow_html=True)
    else:
        st.write("---")
        (st.session_state.df_matched_adjustments_local,
         st.session_state.df_matched_adjustments_foreign,
         st.session_state.df_unmatched_adjustments_local,
         st.session_state.df_unmatched_adjustments_foreign,
         st.session_state.df_unmatched_bank_records) = fx_reconciliation_app(st.session_state.bank_dfs)

# ----------------------------
# FX Trade Reconciliation page
# ----------------------------
elif page_selection == "💱 FX Trade Reconciliation":
    st.markdown(
        """
        <div class="main-header" style="padding:16px;">
            <div>
                <div style="font-size:18px; font-weight:800;">FX Trade Reconciliation</div>
                <div style="color: rgba(226,232,240,0.9); margin-top:4px;">Foreign exchange trades matching & visualization.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.bank_dfs:
        st.markdown('<div class="status-box status-warning">⚠️ Please upload & process bank statements first.</div>', unsafe_allow_html=True)
    else:
        st.write("---")
        (st.session_state.df_matched_counterparty,
         st.session_state.df_matched_choice,
         st.session_state.df_unmatched_counterparty,
         st.session_state.df_unmatched_choice,
         st.session_state.df_unmatched_bank_trade) = graphed_analysis_app(st.session_state.bank_dfs)

# ----------------------------
# Business FX Reconciliation page
# ----------------------------
elif page_selection == "🏢 Business FX Reconciliation":
    st.markdown(
        """
        <div class="main-header" style="padding:16px;">
            <div>
                <div style="font-size:18px; font-weight:800;">Business FX Reconciliation</div>
                <div style="color: rgba(226,232,240,0.9); margin-top:4px;">Business-level FX reconciliation tooling.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.bank_dfs:
        st.markdown('<div class="status-box status-warning">⚠️ Please upload & process bank statements first.</div>', unsafe_allow_html=True)
    else:
        st.write("---")
        business_reconciliation_app(st.session_state.df_matched_counterparty, st.session_state.df_matched_choice, debug_mode=st.session_state.debug_mode)

# ----------------------------
# Cross-Match Analysis page
# ----------------------------
elif page_selection == "🔄 Cross-Match Analysis":
    st.markdown(
        """
        <div class="main-header" style="padding:16px;">
            <div>
                <div style="font-size:18px; font-weight:800;">Cross-Match Analysis</div>
                <div style="color: rgba(226,232,240,0.9); margin-top:4px;">Combine & compare reconciliation outputs to find gaps.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("This section combines results from the reconciliation apps to find potential missed matches.")
    if (st.session_state.df_matched_adjustments_local.empty
            and st.session_state.df_matched_adjustments_foreign.empty
            and st.session_state.df_matched_counterparty.empty
            and st.session_state.df_matched_choice.empty):
        st.markdown('<div class="status-box status-warning">⚠️ Run Adjustments & FX Trade Reconciliation first.</div>', unsafe_allow_html=True)
    else:
        if st.button("🚀 Perform Cross-Match Analysis", use_container_width=True):
            with st.spinner("🔄 Performing cross-match analysis..."):
                run_cross_match_analysis(
                    st.session_state.df_matched_adjustments_local,
                    st.session_state.df_matched_adjustments_foreign,
                    st.session_state.df_matched_counterparty,
                    st.session_state.df_matched_choice,
                    st.session_state.bank_dfs,
                    debug_mode=st.session_state.debug_mode
                )
                st.session_state.cross_match_complete = True
                st.success("✅ Cross-match analysis complete.")
        else:
            st.info("💡 Click the button above to run the cross-match analysis.")
        st.write("---")
        cross_match_analysis_app()

# ----------------------------
# handle page redirect if triggered (keeps UX smooth)
# ----------------------------
if hasattr(st.session_state, 'page_redirect') and st.session_state.page_redirect:
    # set the radio to the selected page and clear
    # (Streamlit persists the sidebar selection automatically on rerun)
    st.session_state.page_redirect = None
