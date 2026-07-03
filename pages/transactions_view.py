import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Excel Matcher & Analyzer", layout="wide")
st.title("📊 Flexible Excel File Matcher & Analyzer")
st.markdown("Upload two Excel files, define matching rules, and analyse matched/unmatched records.")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_excel(file) -> pd.DataFrame:
    """Load Excel file and clean column names."""
    df = pd.read_excel(file)
    df.columns = df.columns.astype(str).str.strip()
    return df

def to_excel_download(df: pd.DataFrame, filename: str) -> BytesIO:
    """Convert DataFrame to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    output.seek(0)
    return output

def apply_case_insensitive(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert selected columns to lower case strings for matching."""
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.lower()
    return df_copy

def merge_with_rules(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     left_on: list, right_on: list,
                     how: str, case_insensitive: bool) -> pd.DataFrame:
    """Perform merge with optional case‑insensitive transformation."""
    df_a_work = df_a.copy()
    df_b_work = df_b.copy()
    if case_insensitive:
        df_a_work = apply_case_insensitive(df_a_work, left_on)
        df_b_work = apply_case_insensitive(df_b_work, right_on)
    return pd.merge(df_a_work, df_b_work,
                    left_on=left_on, right_on=right_on,
                    how=how, indicator=False)

def get_unmatched(df_a: pd.DataFrame, df_b: pd.DataFrame,
                  left_on: list, right_on: list,
                  case_insensitive: bool, which: str):
    """
    Return unmatched records.
    which: 'left_only' (in A but not B) or 'right_only' (in B but not A)
    """
    if case_insensitive:
        df_a_work = apply_case_insensitive(df_a.copy(), left_on)
        df_b_work = apply_case_insensitive(df_b.copy(), right_on)
    else:
        df_a_work = df_a.copy()
        df_b_work = df_b.copy()
    # Create a merge with indicator to find unmatched
    merged = pd.merge(df_a_work, df_b_work,
                      left_on=left_on, right_on=right_on,
                      how='outer', indicator=True)
    if which == 'left_only':
        mask = merged['_merge'] == 'left_only'
        # keep original columns from A only
        cols_a = df_a.columns.tolist()
        result = merged.loc[mask, cols_a].copy()
        # reorder to match original A order
        result = result[df_a.columns]
    else:  # right_only
        mask = merged['_merge'] == 'right_only'
        cols_b = df_b.columns.tolist()
        result = merged.loc[mask, cols_b].copy()
        result = result[df_b.columns]
    return result

# ------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------
if 'rules' not in st.session_state:
    st.session_state.rules = []   # each rule: {'col_a': '', 'col_b': ''}
if 'df_a' not in st.session_state:
    st.session_state.df_a = None
if 'df_b' not in st.session_state:
    st.session_state.df_b = None

# ------------------------------------------------------------
# Sidebar: File uploads
# ------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload Files")
    file_a = st.file_uploader("File A (Excel)", type=["xlsx", "xls"], key="file_a")
    file_b = st.file_uploader("File B (Excel)", type=["xlsx", "xls"], key="file_b")

    if file_a and file_b:
        st.session_state.df_a = load_excel(file_a)
        st.session_state.df_b = load_excel(file_b)
        st.success(f"File A: {st.session_state.df_a.shape[0]} rows, {st.session_state.df_a.shape[1]} cols")
        st.success(f"File B: {st.session_state.df_b.shape[0]} rows, {st.session_state.df_b.shape[1]} cols")
    else:
        st.info("Please upload both files to start.")

# ------------------------------------------------------------
# Main area: Only proceed if both DataFrames exist
# ------------------------------------------------------------
if st.session_state.df_a is not None and st.session_state.df_b is not None:
    df_a = st.session_state.df_a
    df_b = st.session_state.df_b

    # --- Column selection for rules ---
    st.subheader("2. Define Matching Rules")
    cols_a = df_a.columns.tolist()
    cols_b = df_b.columns.tolist()

    # Display existing rules
    for i, rule in enumerate(st.session_state.rules):
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            st.text(f"File A: {rule['col_a']}")
        with col2:
            st.text(f"File B: {rule['col_b']}")
        with col3:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.rules.pop(i)
                st.rerun()

    # Add new rule
    with st.expander("➕ Add a matching rule"):
        col_a_sel = st.selectbox("File A column", cols_a, key="new_col_a")
        col_b_sel = st.selectbox("File B column", cols_b, key="new_col_b")
        if st.button("Add Rule"):
            if col_a_sel and col_b_sel:
                st.session_state.rules.append({'col_a': col_a_sel, 'col_b': col_b_sel})
                st.rerun()
            else:
                st.warning("Please select both columns.")

    # --- Matching options ---
    st.subheader("3. Matching Options")
    match_type = st.radio(
        "Which records do you want?",
        options=["Matched (inner join)", "Unmatched in A only", "Unmatched in B only"],
        horizontal=True
    )
    case_sensitive = st.checkbox("Case‑insensitive matching (strings)", value=True)

    # --- Numeric column selection for summary ---
    # Automatically detect numeric columns in each file
    numeric_cols_a = df_a.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_b = df_b.select_dtypes(include=[np.number]).columns.tolist()
    all_numeric = list(set(numeric_cols_a + numeric_cols_b))

    st.subheader("4. Summary of Numeric Columns")
    selected_numeric = st.multiselect(
        "Choose numeric columns to summarise (sum, mean, etc.)",
        options=all_numeric,
        default=all_numeric[:3] if len(all_numeric) > 0 else []
    )

    # --- Perform matching when rules exist ---
    if len(st.session_state.rules) == 0:
        st.warning("Please add at least one matching rule.")
    else:
        left_on = [r['col_a'] for r in st.session_state.rules]
        right_on = [r['col_b'] for r in st.session_state.rules]

        if match_type == "Matched (inner join)":
            result_df = merge_with_rules(df_a, df_b, left_on, right_on,
                                         how='inner', case_insensitive=case_sensitive)
            st.subheader("✅ Matched Records")
        elif match_type == "Unmatched in A only":
            result_df = get_unmatched(df_a, df_b, left_on, right_on,
                                      case_insensitive=case_sensitive, which='left_only')
            st.subheader("🔍 Records in File A NOT found in File B")
        else:  # Unmatched in B only
            result_df = get_unmatched(df_a, df_b, left_on, right_on,
                                      case_insensitive=case_sensitive, which='right_only')
            st.subheader("🔍 Records in File B NOT found in File A")

        if result_df.empty:
            st.info("No records match the criteria.")
        else:
            st.dataframe(result_df, use_container_width=True)

            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download as CSV", csv,
                                   file_name="result.csv", mime="text/csv")
            with col_dl2:
                excel_bytes = to_excel_download(result_df, "result.xlsx")
                st.download_button("📥 Download as Excel", excel_bytes,
                                   file_name="result.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # --- Numeric summary and analytics ---
        st.subheader("📈 Summary Statistics & Analytics")
        if selected_numeric and not result_df.empty:
            # Summary for the result table (matched or unmatched)
            summary = result_df[selected_numeric].agg(['sum', 'mean', 'count', 'min', 'max']).round(2)
            st.write("**Numeric summary (result dataset):**")
            st.dataframe(summary)

            # Additional analytics: match rate (only meaningful for matched view)
            if match_type == "Matched (inner join)":
                total_a = len(df_a)
                total_b = len(df_b)
                matched = len(result_df)
                st.metric("Match rate (A → matched)", f"{matched / total_a * 100:.1f}%" if total_a else "N/A")
                st.metric("Match rate (B → matched)", f"{matched / total_b * 100:.1f}%" if total_b else "N/A")
                st.write(f"**Total records in A:** {total_a}  |  **Matched:** {matched}  |  **Unmatched in A:** {total_a - matched}")
                st.write(f"**Total records in B:** {total_b}  |  **Matched:** {matched}  |  **Unmatched in B:** {total_b - matched}")
        else:
            st.info("No numeric columns selected or result set is empty.")

        # --- Optionally show column mapping used ---
        with st.expander("🔧 Current matching rules"):
            for i, rule in enumerate(st.session_state.rules, 1):
                st.write(f"{i}. File A: `{rule['col_a']}` ↔ File B: `{rule['col_b']}`")

else:
    st.info("👈 Please upload both Excel files in the sidebar to begin.")