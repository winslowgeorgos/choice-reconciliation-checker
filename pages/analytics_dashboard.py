# analytics_dashboard.py (Upgraded with Fintech UI)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Import UI components
from ui_components import (
     kpi_metric, empty_state, loading_state,
    ag_grid_table, fin_tabs, section_header, subsection_header,
    metrics_row, fin_bar_chart, fin_pie_chart, fin_line_chart,
    fin_gauge_chart, format_currency, format_percentage, status_badge,
    COLORS
)

def analyze_reconciliation_data(loaded_results, historical_data=None):
    """Analyze loaded reconciliation data and return metrics."""
    metrics = {
        'total_transactions': 0,
        'matched_count': 0,
        'unmatched_count': 0,
        'match_rate': 0,
        'by_module': {},
        'by_match_status': {},
        'amount_summary': {'matched': 0, 'unmatched': 0},
        'data_types': [],
        'module_details': {},
        'historical_trends': []
    }
    
    # Process historical trends
    if historical_data:
        for date_key, date_results in historical_data.items():
            date_metrics = {
                'date': date_key,
                'total': 0,
                'matched': 0,
                'unmatched': 0,
                'match_rate': 0
            }
            for key, data_info in date_results.items():
                if data_info.get('type') == 'dataframe':
                    df = data_info['data']
                    date_metrics['total'] += len(df)
                    
                    if 'Mismatch_Type' in df.columns:
                        date_metrics['matched'] += df[df['Mismatch_Type'] == 'None'].shape[0]
                        date_metrics['unmatched'] += df[df['Mismatch_Type'] != 'None'].shape[0]
                    elif 'Matched' in df.columns:
                        if df['Matched'].dtype == bool:
                            date_metrics['matched'] += df[df['Matched'] == True].shape[0]
                        else:
                            date_metrics['matched'] += df[df['Matched'].astype(str).str.lower() == 'true'].shape[0]
                        date_metrics['unmatched'] = date_metrics['total'] - date_metrics['matched']
            
            if date_metrics['total'] > 0:
                date_metrics['match_rate'] = (date_metrics['matched'] / date_metrics['total']) * 100
            metrics['historical_trends'].append(date_metrics)
        
        metrics['historical_trends'] = sorted(metrics['historical_trends'], key=lambda x: x['date'])
    
    for key, data_info in loaded_results.items():
        if data_info.get('type') == 'dataframe':
            df = data_info['data']
            metrics['data_types'].append(key)
            row_count = len(df)
            metrics['total_transactions'] += row_count
            
            module = key.replace('_df', '').replace('_', ' ').title()
            
            # Detect match status
            match_status_col = None
            amount_col = None
            
            status_columns = ['Mismatch_Type', 'Match_Status', 'Status', 'Match_Type', 'Reconciliation_Status']
            for col in status_columns:
                if col in df.columns:
                    match_status_col = col
                    break
            
            amount_columns = ['Amount', 'Trade Amount', 'Intermediary Amount', 'Transaction Amount', 
                             'Adjustment_Amount', 'Bank Amount', 'GL_Amount', 'Mpesa_Amount']
            for col in amount_columns:
                if col in df.columns:
                    amount_col = col
                    break
            
            matched = 0
            unmatched = 0
            
            if match_status_col:
                if match_status_col == 'Mismatch_Type':
                    matched = df[df[match_status_col] == 'None'].shape[0]
                    unmatched = df[df[match_status_col] != 'None'].shape[0]
                elif match_status_col == 'Match_Type':
                    matched = df[df[match_status_col] != 'No match'].shape[0]
                    unmatched = df[df[match_status_col] == 'No match'].shape[0]
                else:
                    matched = df[df[match_status_col].notna() & (df[match_status_col] != '')].shape[0]
                    unmatched = row_count - matched
            else:
                matched_indicator_cols = ['Matched', 'Is_Matched', 'is_matched', 'matched']
                for col in matched_indicator_cols:
                    if col in df.columns:
                        if df[col].dtype == bool:
                            matched = df[df[col] == True].shape[0]
                        else:
                            matched = df[df[col].astype(str).str.lower() == 'true'].shape[0]
                        unmatched = row_count - matched
                        break
                
                if matched == 0 and 'Matched_Bank_Record_Key' in df.columns:
                    matched = df[df['Matched_Bank_Record_Key'].notna()].shape[0]
                    unmatched = row_count - matched
            
            metrics['matched_count'] += matched
            metrics['unmatched_count'] += unmatched
            
            metrics['by_module'][module] = {
                'matched': matched,
                'unmatched': unmatched,
                'total': row_count,
                'match_rate': (matched / row_count * 100) if row_count > 0 else 0
            }
            
            if match_status_col and match_status_col in df.columns:
                status_counts = df[match_status_col].value_counts().to_dict()
                for status, count in status_counts.items():
                    display_status = str(status) if status != 'None' else 'Matched'
                    if display_status not in metrics['by_match_status']:
                        metrics['by_match_status'][display_status] = 0
                    metrics['by_match_status'][display_status] += count
            
            if amount_col and amount_col in df.columns:
                df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                
                if match_status_col and match_status_col in df.columns:
                    if match_status_col == 'Mismatch_Type':
                        matched_amount = df[df[match_status_col] == 'None'][amount_col].sum()
                        unmatched_amount = df[df[match_status_col] != 'None'][amount_col].sum()
                    else:
                        matched_amount = df.iloc[:matched][amount_col].sum() if matched > 0 else 0
                        unmatched_amount = df.iloc[matched:][amount_col].sum() if unmatched > 0 else 0
                else:
                    matched_amount = df.iloc[:matched][amount_col].sum() if matched > 0 else 0
                    unmatched_amount = df.iloc[matched:][amount_col].sum() if unmatched > 0 else 0
                
                metrics['amount_summary']['matched'] += matched_amount if not pd.isna(matched_amount) else 0
                metrics['amount_summary']['unmatched'] += unmatched_amount if not pd.isna(unmatched_amount) else 0
                
                metrics['module_details'][module] = {
                    'matched_amount': matched_amount if not pd.isna(matched_amount) else 0,
                    'unmatched_amount': unmatched_amount if not pd.isna(unmatched_amount) else 0
                }
    
    if metrics['total_transactions'] > 0:
        metrics['match_rate'] = (metrics['matched_count'] / metrics['total_transactions']) * 100
    
    return metrics

def load_historical_data_for_trends(selected_date, days_back=30):
    """Load historical data for trend analysis."""
    from auth_system import load_results_by_date_range
    
    historical_data = {}
    
    try:
        start_date = (datetime.strptime(selected_date, '%Y-%m-%d') - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = selected_date
        
        range_results = load_results_by_date_range(start_date, end_date)
        
        for data_type, items in range_results.items():
            for item in items:
                date_key = item.get('data_date')
                if date_key:
                    if date_key not in historical_data:
                        historical_data[date_key] = {}
                    historical_data[date_key][data_type] = item
        
        return historical_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return {}

def render_analytics_content(loaded_results, selected_date):
    """Render the analytics dashboard with loaded data."""
    
    if not loaded_results:
        empty_state(f"No reconciliation data found for {selected_date}", "📭")
        return
    
    with st.spinner("Loading historical data for trend analysis..."):
        historical_data = load_historical_data_for_trends(selected_date, days_back=30)
    
    metrics = analyze_reconciliation_data(loaded_results, historical_data)
    
    section_header("Analytics Dashboard", f"Reconciliation Analysis for {selected_date}")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        kpi_metric("Total Transactions", f"{metrics['total_transactions']:,}", icon="📋")
    
    with col2:
        match_rate = metrics['match_rate']
        kpi_metric("Match Rate", format_percentage(match_rate, 1), 
                  trend="up" if match_rate > 80 else "down" if match_rate < 50 else None,
                  trend_value=abs(match_rate - 75), icon="🎯")
    
    with col3:
        kpi_metric("Matched", f"{metrics['matched_count']:,}", icon="✅")
    
    with col4:
        matched_amount = metrics['amount_summary']['matched']
        kpi_metric("Matched Amount", format_currency(matched_amount, "KES"), icon="💰")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        fin_gauge_chart(metrics['match_rate'], "Overall Match Rate", height=350)
    
    with col2:
        if metrics['by_match_status']:
            status_data = {k: v for k, v in metrics['by_match_status'].items()}
            fin_pie_chart(list(status_data.keys()), list(status_data.values()), 
                         "Match Status Distribution", height=350)
        else:
            fin_pie_chart(['Matched', 'Unmatched'], 
                         [metrics['matched_count'], metrics['unmatched_count']],
                         "Match Status Distribution", height=350)
    
    # Module Performance
    subsection_header("Module Performance")
    
    if metrics['by_module']:
        module_df = pd.DataFrame([
            {
                'Module': module,
                'Total': stats['total'],
                'Matched': stats['matched'],
                'Unmatched': stats['unmatched'],
                'Match Rate': f"{stats['match_rate']:.1f}%"
            }
            for module, stats in metrics['by_module'].items()
        ]).sort_values('Match Rate', ascending=False)
        
        ag_grid_table(module_df, key="module_performance_table", height=400)
    
    # Historical Trends
    if metrics['historical_trends'] and len(metrics['historical_trends']) > 1:
        subsection_header("Historical Trends")
        
        trends_df = pd.DataFrame(metrics['historical_trends'])
        fin_line_chart(trends_df, 'date', ['match_rate'], "Match Rate Trend", height=400)
    
    # Export Options
    subsection_header("Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as JSON", use_container_width=True):
            export_data = {
                'date': selected_date,
                'metrics': {
                    'total_transactions': metrics['total_transactions'],
                    'matched_count': metrics['matched_count'],
                    'unmatched_count': metrics['unmatched_count'],
                    'match_rate': metrics['match_rate'],
                    'by_module': metrics['by_module'],
                    'amount_summary': metrics['amount_summary']
                },
                'historical_trends': metrics['historical_trends']
            }
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"analytics_report_{selected_date}.json",
                mime="application/json",
                key="export_json_btn"
            )
    
    with col2:
        if metrics['by_module']:
            export_df = pd.DataFrame([
                {
                    'Module': module,
                    'Total': stats['total'],
                    'Matched': stats['matched'],
                    'Unmatched': stats['unmatched'],
                    'Match Rate': stats['match_rate']
                }
                for module, stats in metrics['by_module'].items()
            ])
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Export as CSV",
                data=csv,
                file_name=f"analytics_report_{selected_date}.csv",
                mime="text/csv",
                key="export_csv_btn"
            )
    
    with col3:
        if st.button("📄 Generate HTML Report", use_container_width=True):
            html_report = generate_html_report(selected_date, metrics)
            st.download_button(
                label="Download HTML Report",
                data=html_report,
                file_name=f"analytics_report_{selected_date}.html",
                mime="text/html",
                key="export_html_btn"
            )
    
    # Recommendations
    subsection_header("Insights & Recommendations")
    
    if metrics['match_rate'] >= 80:
        st.success(f"""
        ✅ **Excellent Performance** - {metrics['match_rate']:.1f}% match rate
            
        - Successfully matched {metrics['matched_count']:,} out of {metrics['total_transactions']:,} transactions
        - Review the {metrics['unmatched_count']:,} unmatched transactions for data quality issues
        """)
    elif metrics['match_rate'] >= 50:
        st.warning(f"""
        ⚠️ **Needs Improvement** - {metrics['match_rate']:.1f}% match rate
            
        - {metrics['matched_count']:,} out of {metrics['total_transactions']:,} transactions matched
        - Review data formatting and reference number consistency
        - Check date formats and amount matching thresholds
        """)
    else:
        st.error(f"""
        ❌ **Critical Issues** - {metrics['match_rate']:.1f}% match rate
            
        - Only {metrics['matched_count']:,} out of {metrics['total_transactions']:,} transactions matched
        - Verify bank statement processing and column mappings
        - Check that bank names and currencies are properly standardized
        - Run individual reconciliation modules to identify specific issues
        """)

def generate_html_report(date, metrics):
    """Generate an HTML report for download."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Report - {date}</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 40px;
                background-color: #0f172a;
                color: #e2e8f0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: #1e293b;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }}
            h1 {{ color: #38bdf8; }}
            h2 {{ color: #f1f5f9; margin-top: 30px; border-left: 4px solid #38bdf8; padding-left: 15px; }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #020617;
                padding: 20px;
                border-radius: 12px;
                border-left: 4px solid #38bdf8;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
                color: #38bdf8;
            }}
            .metric-label {{
                font-size: 12px;
                color: #94a3b8;
                text-transform: uppercase;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: #020617;
                border-radius: 8px;
                overflow: hidden;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #334155;
            }}
            th {{
                background-color: #38bdf8;
                color: #020617;
            }}
            .success {{ color: #22c55e; }}
            .warning {{ color: #f59e0b; }}
            .danger {{ color: #ef4444; }}
            hr {{ border-color: #334155; margin: 30px 0; }}
            .footer {{ text-align: center; color: #64748b; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Reconciliation Analytics Report</h1>
            <p><strong>Date:</strong> {date}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Transactions</div>
                    <div class="metric-value">{metrics['total_transactions']:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Match Rate</div>
                    <div class="metric-value">{metrics['match_rate']:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Matched</div>
                    <div class="metric-value">{metrics['matched_count']:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Unmatched</div>
                    <div class="metric-value">{metrics['unmatched_count']:,}</div>
                </div>
            </div>
            
            <h2>Module Performance</h2>
            <table>
                <thead>
                    <tr><th>Module</th><th>Total</th><th>Matched</th><th>Unmatched</th><th>Match Rate</th></tr>
                </thead>
                <tbody>
    """
    
    for module, stats in metrics['by_module'].items():
        rate_class = 'success' if stats['match_rate'] >= 80 else 'warning' if stats['match_rate'] >= 50 else 'danger'
        html += f"""
                    <tr>
                        <td>{module}</td>
                        <td>{stats['total']}</td>
                        <td>{stats['matched']}</td>
                        <td>{stats['unmatched']}</td>
                        <td class="{rate_class}">{stats['match_rate']:.1f}%</td>
                    </tr>
        """
    
    html += f"""
                </tbody>
            </table>
            
            <h2>Amount Summary</h2>
            <table>
                <tr><td><strong>Total Matched Amount</strong></td><td>KES {metrics['amount_summary']['matched']:,.2f}</td></tr>
                <tr><td><strong>Total Unmatched Amount</strong></td><td>KES {metrics['amount_summary']['unmatched']:,.2f}</td></tr>
            </table>
            
            <h2>Historical Trends</h2>
            <table>
                <thead>
                    <tr><th>Date</th><th>Total</th><th>Matched</th><th>Unmatched</th><th>Match Rate</th></tr>
                </thead>
                <tbody>
    """
    
    for trend in metrics['historical_trends']:
        rate_class = 'success' if trend['match_rate'] >= 80 else 'warning' if trend['match_rate'] >= 50 else 'danger'
        html += f"""
                    <tr>
                        <td>{trend['date']}</td>
                        <td>{trend['total']}</td>
                        <td>{trend['matched']}</td>
                        <td>{trend['unmatched']}</td>
                        <td class="{rate_class}">{trend['match_rate']:.1f}%</td>
                    </tr>
        """
    
    html += f"""
                </tbody>
            </table>
            
            <hr>
            <div class="footer">
                <p>Generated by ChoiceBank FX Reconciliation Dashboard</p>
                <p>Report includes data from {len(metrics['data_types'])} reconciliation modules</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def show_analytics_for_date(loaded_results, date_str):
    """Display analytics dashboard for a specific date."""
    st.session_state.selected_analytics_data = loaded_results
    st.session_state.selected_analytics_date = date_str
    st.session_state.show_analytics_modal = True
    st.rerun()

def analytics_dashboard_modal():
    """Display the analytics dashboard modal."""
    if 'show_analytics_modal' not in st.session_state:
        st.session_state.show_analytics_modal = False
    if 'selected_analytics_date' not in st.session_state:
        st.session_state.selected_analytics_date = None
    if 'selected_analytics_data' not in st.session_state:
        st.session_state.selected_analytics_data = None
    
    if st.session_state.show_analytics_modal and st.session_state.selected_analytics_data:
        render_analytics_content(
            st.session_state.selected_analytics_data, 
            st.session_state.selected_analytics_date
        )
        
        if st.button("Close Dashboard", use_container_width=True):
            st.session_state.show_analytics_modal = False
            st.rerun()