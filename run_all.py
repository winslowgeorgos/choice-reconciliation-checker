import streamlit as st
import pandas as pd
from datetime import datetime
import io

def detect_column_type(series):
    """
    Detect if a pandas series contains datetime objects or strings
    """
    # Check if already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    
    # Check if numeric (not date)
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    
    # Try to convert sample to datetime to check if it contains dates
    sample_size = min(100, len(series))
    sample = series.head(sample_size).dropna()
    
    if len(sample) == 0:
        return 'unknown'
    
    # Try parsing as datetime
    try:
        # Try multiple common date formats
        test_parsed = pd.to_datetime(sample, errors='coerce', format='%m/%d/%Y')
        success_rate = (test_parsed.notna().sum() / len(sample)) * 100
        
        if success_rate > 80:  # If >80% successfully parsed as dates
            return 'date_string'
        else:
            return 'general_string'
    except:
        return 'general_string'

def main():
    st.set_page_config(
        page_title="Date Formatter",
        page_icon="📅",
        layout="wide"
    )
    
    st.title("📅 Transaction Date Formatter")
    st.markdown("Upload a CSV file to convert **'transaction date'** column from `mm/dd/yyyy` to `dd/m/yyyy` format")
    
    # File upload section
    st.header("1. Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="File should contain a 'transaction date' column in mm/dd/yyyy format"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display file info
            st.success(f"✅ File successfully loaded! Shape: {df.shape}")
            
            # Show basic information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'details' in col.lower()]
                st.metric("Relevant Columns", len(date_cols))
            
            # Column detection and selection
            st.header("2. Column Detection & Selection")
            
            # Find potential date columns
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            details_columns = [col for col in df.columns if 'details' in col.lower()]
            
            # Detect column types
            col_types = {}
            for col in df.columns:
                col_types[col] = detect_column_type(df[col])
            
            # Display column type information
            st.subheader("Column Type Analysis")
            type_df = pd.DataFrame({
                'Column': df.columns,
                'Detected Type': [col_types[col] for col in df.columns],
                'Sample Value': [str(df[col].iloc[0]) if len(df[col]) > 0 else 'N/A' for col in df.columns]
            })
            st.dataframe(type_df, use_container_width=True)
            
            # Select target column for transaction date
            target_column = None
            if date_columns:
                target_column = st.selectbox(
                    "Select the transaction date column:",
                    date_columns,
                    help="Columns containing 'date' in their name"
                )
            else:
                st.error("❌ No columns containing 'date' found in the uploaded file!")
                st.write("Available columns:", df.columns.tolist())
                return
            
            # Check transaction details column
            details_column = None
            details_conversion_needed = False
            
            if details_columns:
                details_column = st.selectbox(
                    "Select the transaction details column (optional):",
                    ['None'] + details_columns,
                    help="Columns containing 'details' in their name"
                )
                
                if details_column != 'None':
                    details_type = col_types[details_column]
                    st.write(f"**Detected type for '{details_column}':** `{details_type}`")
                    
                    if details_type in ['datetime', 'date_string']:
                        st.warning(f"⚠️ '{details_column}' appears to contain dates. Conversion will be applied.")
                        details_conversion_needed = True
                    else:
                        st.success(f"✅ '{details_column}' appears to be general text. No conversion needed.")
                        details_conversion_needed = False
            
            # Display original data
            st.header("3. Original Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show original dates
            st.subheader("Original Target Column Values")
            preview_cols = [target_column]
            if details_column and details_column != 'None':
                preview_cols.append(details_column)
            
            st.write(df[preview_cols].head())
            
            # Date formatting section
            st.header("4. Date Formatting Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Format:**")
                st.code("mm/dd/yyyy\nExample: 12/25/2023")
                
                # Show detected type for target column
                target_type = col_types[target_column]
                st.info(f"Target column type: **{target_type}**")
            
            with col2:
                st.markdown("**Target Format:**")
                st.code("dd/m/yyyy\nExample: 25/12/2023")
                
                if details_column and details_column != 'None':
                    details_type = col_types[details_column]
                    st.info(f"Details column type: **{details_type}**")
            
            # Format dates
            if st.button("🔄 Convert Dates", type="primary"):
                with st.spinner("Converting dates..."):
                    try:
                        # Create a copy for processing
                        df_processed = df.copy()
                        conversion_log = []
                        
                        # Convert main transaction date column
                        st.subheader("Conversion Results")
                        
                        # Process target column
                        original_target_type = col_types[target_column]
                        if original_target_type in ['datetime', 'date_string']:
                            df_processed[target_column] = pd.to_datetime(
                                df_processed[target_column], 
                                format='%m/%d/%Y',
                                errors='coerce'
                            )
                            
                            null_dates = df_processed[target_column].isnull().sum()
                            df_processed[f"{target_column}_formatted"] = df_processed[target_column].dt.strftime('%d/%m/%Y')
                            
                            conversion_log.append(f"✅ **{target_column}**: Converted {len(df_processed) - null_dates} dates ({null_dates} errors)")
                        else:
                            conversion_log.append(f"ℹ️ **{target_column}**: No conversion needed (general text)")
                        
                        # Process details column if selected and needs conversion
                        if details_column and details_column != 'None' and details_conversion_needed:
                            original_details_type = col_types[details_column]
                            if original_details_type in ['datetime', 'date_string']:
                                df_processed[details_column] = pd.to_datetime(
                                    df_processed[details_column], 
                                    format='%m/%d/%Y',
                                    errors='coerce'
                                )
                                
                                null_details = df_processed[details_column].isnull().sum()
                                df_processed[f"{details_column}_formatted"] = df_processed[details_column].dt.strftime('%d/%m/%Y')
                                
                                conversion_log.append(f"✅ **{details_column}**: Converted {len(df_processed) - null_details} dates ({null_details} errors)")
                            else:
                                conversion_log.append(f"ℹ️ **{details_column}**: No conversion needed (general text)")
                        
                        # Display conversion log
                        st.markdown("### Conversion Summary")
                        for log_entry in conversion_log:
                            st.write(log_entry)
                        
                        # Show comparison
                        st.subheader("Formatted Results Preview")
                        comparison_cols = [target_column]
                        if f"{target_column}_formatted" in df_processed.columns:
                            comparison_cols.append(f"{target_column}_formatted")
                        
                        if details_column and details_column != 'None' and f"{details_column}_formatted" in df_processed.columns:
                            comparison_cols.extend([details_column, f"{details_column}_formatted"])
                        
                        st.dataframe(df_processed[comparison_cols].head(10), use_container_width=True)
                        
                        # Download section
                        st.header("5. Download Results")
                        
                        # Convert processed dataframe to CSV
                        csv = df_processed.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Formatted CSV",
                            data=csv,
                            file_name="formatted_transactions.csv",
                            mime="text/csv",
                            help="Download the complete dataset with formatted dates"
                        )
                        
                        # Show statistics
                        st.header("6. Conversion Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        total_conversions = sum(1 for col in df_processed.columns if col.endswith('_formatted'))
                        with col1:
                            st.metric("Total Columns Processed", len(preview_cols))
                        with col2:
                            st.metric("Columns Converted", total_conversions)
                        with col3:
                            total_rows = len(df_processed)
                            st.metric("Total Rows", total_rows)
                            
                    except Exception as e:
                        st.error(f"❌ Error during date conversion: {e}")
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        # Example section
        st.header("Example Format")
        st.markdown("""
        Your CSV file should have a structure like this:
        
        | transaction date | transaction details | amount | description |
        |------------------|---------------------|--------|-------------|
        | 12/25/2023       | Payment received    | 100.50 | Purchase    |
        | 01/15/2024       | 01/20/2024          | 75.00  | Refund      |
        """)
        
        # Create example data with mixed types
        example_data = {
            'transaction date': ['12/25/2023', '01/15/2024', '03/08/2024'],
            'transaction details': ['Payment received', '01/20/2024', 'Bank transfer'],
            'amount': [100.50, 75.00, 200.00],
            'description': ['Purchase', 'Refund', 'Transfer']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Detection Logic:**
        - ✅ **Date strings**: Values like '12/25/2023' will be converted
        - ✅ **General strings**: Values like 'Payment received' will be preserved
        - ✅ **Mixed content**: Automatic detection determines if conversion is needed
        """)

if __name__ == "__main__":
    main()