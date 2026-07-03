# ui_components.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any, List

# Attempt to import st_aggrid with fallback
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    from st_aggrid.shared import GridUpdateMode
    AG_GRID_AVAILABLE = True
except ImportError:
    AG_GRID_AVAILABLE = False
    # Don't show warning here, will be handled in the function

# --- Custom Color Palette ---
COLORS = {
    'purple': '#4B2D8F',
    'purple-dark': '#3A2070',
    'purple-light': '#6B4DB5',
    'purple-pale': '#F0EBF9',
    'navy': '#0D1F4E',
    'navy-light': '#1A3A7A',
    'wine': '#7D1128',
    'gold-light': '#F5C842',
    'cream': '#FAFAF8',
    'white': '#FFFFFF',
    'muted': '#6B7280',
    'text': '#1A1230',
    'border': '#E2D9F3',
    'border-light': '#EDE8F5',
    'green': '#059669',
    'red': '#DC2626',
    'orange': '#EA580C',
}

# Shadow values as actual CSS values
SHADOWS = {
    'shadow-md': '0 8px 32px rgba(75, 45, 143, 0.12)',
    'shadow-lg': '0 20px 60px rgba(75, 45, 143, 0.16)',
}

# --- Light Mode Theme CSS ---
LIGHT_MODE_CSS = f"""
<style>
    /* Main app background */
    .stApp {{
        background-color: {COLORS['cream']};
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['white']};
        border-right: 1px solid {COLORS['border']};
        box-shadow: {SHADOWS['shadow-md']};
    }}
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {{
        color: {COLORS['text']};
    }}
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLORS['purple']};
    }}
    
    /* Card styling */
    .fin-card {{
        background: {COLORS['white']};
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: {SHADOWS['shadow-md']};
        border: 1px solid {COLORS['border-light']};
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .fin-card:hover {{
        transform: translateY(-2px);
        box-shadow: {SHADOWS['shadow-lg']};
    }}
    
    /* KPI styling */
    .kpi-container {{
        background: linear-gradient(135deg, {COLORS['white']}, {COLORS['purple-pale']});
        padding: 1.25rem;
        border-radius: 1rem;
        border-left: 4px solid {COLORS['purple']};
        margin-bottom: 1rem;
        box-shadow: {SHADOWS['shadow-md']};
    }}
    
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS['purple']};
        line-height: 1.2;
    }}
    
    .kpi-label {{
        font-size: 0.75rem;
        color: {COLORS['muted']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }}
    
    .kpi-trend-up {{
        color: {COLORS['green']};
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }}
    
    .kpi-trend-down {{
        color: {COLORS['red']};
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }}
    
    /* Headers */
    .fin-header {{
        font-size: 1.75rem;
        font-weight: 600;
        color: {COLORS['navy']};
        margin-bottom: 1rem;
        border-left: 4px solid {COLORS['purple']};
        padding-left: 1rem;
    }}
    
    .fin-subheader {{
        font-size: 1.25rem;
        font-weight: 500;
        color: {COLORS['purple-dark']};
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    .fin-section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {COLORS['navy']};
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['purple']}, {COLORS['purple-light']});
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(75, 45, 143, 0.2);
        background: linear-gradient(135deg, {COLORS['purple-dark']}, {COLORS['purple']});
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background-color: {COLORS['white']};
        padding: 0.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid {COLORS['border-light']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 0.5rem;
        color: {COLORS['muted']};
        font-weight: 500;
        padding: 0.5rem 1rem;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['purple']}, {COLORS['purple-light']});
        color: white;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: {COLORS['white']};
        border-radius: 0.5rem;
        color: {COLORS['purple']};
        font-weight: 500;
        border: 1px solid {COLORS['border-light']};
    }}
    
    .streamlit-expanderContent {{
        background-color: {COLORS['purple-pale']};
        border-radius: 0.5rem;
        border: 1px solid {COLORS['border-light']};
        padding: 1rem;
    }}
    
    /* Dataframe wrapper */
    .dataframe-wrapper {{
        border-radius: 0.75rem;
        overflow: auto;
        border: 1px solid {COLORS['border-light']};
        background-color: {COLORS['white']};
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        background-color: {COLORS['white']};
        border: 2px dashed {COLORS['border']};
        border-radius: 0.75rem;
    }}
    
    /* Success/Warning/Error messages */
    .stAlert {{
        border-radius: 0.75rem;
        border-left: 4px solid;
    }}
    
    /* Select boxes and inputs */
    .stSelectbox > div, .stTextInput > div {{
        background-color: {COLORS['white']};
        border-radius: 0.5rem;
        border: 1px solid {COLORS['border']};
    }}
    
    /* Metrics row */
    .metrics-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['border-light']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['purple']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['purple-light']};
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: {COLORS['purple-pale']};
        border-left-color: {COLORS['purple']};
    }}
    
    /* Success boxes */
    .stSuccess {{
        background-color: #ECFDF5;
        border-left-color: {COLORS['green']};
    }}
    
    /* Warning boxes */
    .stWarning {{
        background-color: #FEF3C7;
        border-left-color: {COLORS['orange']};
    }}
    
    /* Error boxes */
    .stError {{
        background-color: #FEE2E2;
        border-left-color: {COLORS['red']};
    }}
    
    /* Dataframe text */
    .stDataFrame {{
        color: {COLORS['text']};
    }}
    
    /* Caption text */
    .stCaption {{
        color: {COLORS['muted']};
    }}
</style>
"""

# --- Dark Mode Theme CSS ---
DARK_MODE_CSS = f"""
<style>
    /* Main app background */
    .stApp {{
        background-color: #0f172a;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }}
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {{
        color: #e2e8f0;
    }}
    
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {COLORS['purple-light']};
    }}
    
    /* Card styling */
    .fin-card {{
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .fin-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }}
    
    /* KPI styling */
    .kpi-container {{
        background: linear-gradient(135deg, #1e293b, #020617);
        padding: 1.25rem;
        border-radius: 1rem;
        border-left: 4px solid {COLORS['purple-light']};
        margin-bottom: 1rem;
    }}
    
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {COLORS['purple-light']}, {COLORS['gold-light']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }}
    
    .kpi-label {{
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }}
    
    .kpi-trend-up {{
        color: {COLORS['green']};
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }}
    
    .kpi-trend-down {{
        color: {COLORS['red']};
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }}
    
    /* Headers */
    .fin-header {{
        font-size: 1.75rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1rem;
        border-left: 4px solid {COLORS['purple-light']};
        padding-left: 1rem;
    }}
    
    .fin-subheader {{
        font-size: 1.25rem;
        font-weight: 500;
        color: #cbd5e1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['purple']}, {COLORS['purple-light']});
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(75, 45, 143, 0.3);
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background-color: #1e293b;
        padding: 0.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 0.5rem;
        color: #94a3b8;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['purple']}, {COLORS['purple-light']});
        color: white;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: #1e293b;
        border-radius: 0.5rem;
        color: {COLORS['purple-light']};
        font-weight: 500;
    }}
    
    .streamlit-expanderContent {{
        background-color: #020617;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }}
    
    /* Dataframe wrapper */
    .dataframe-wrapper {{
        border-radius: 0.75rem;
        overflow: auto;
        border: 1px solid #334155;
        background-color: #020617;
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        background-color: #1e293b;
        border: 1px dashed #334155;
        border-radius: 0.75rem;
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #1e293b;
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['purple']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['purple-light']};
    }}
    
    /* Dataframe text */
    .stDataFrame {{
        color: #e2e8f0;
    }}
</style>
"""

def apply_theme():
    """Apply the current theme based on session state."""
    if st.session_state.get('dark_mode', False):
        st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

def theme_toggle():
    """Display a theme toggle switch in the sidebar."""
    current = "🌙 Dark Mode" if st.session_state.get('dark_mode', False) else "☀️ Light Mode"
    if st.button(current, use_container_width=True, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.get('dark_mode', False)
        st.rerun()

# --- Reusable UI Components ---

def fin_card(content, key=None):
    """Render content inside a styled card."""
    return st.markdown(f'<div class="fin-card">{content}</div>', unsafe_allow_html=True)


def kpi_metric(label: str, value: str, trend: Optional[str] = None, 
               trend_value: Optional[float] = None, icon: Optional[str] = None):
    """Display a KPI metric card with optional trend indicator."""
    trend_html = ""
    if trend and trend_value is not None:
        trend_class = "kpi-trend-up" if trend == "up" else "kpi-trend-down"
        trend_symbol = "↑" if trend == "up" else "↓"
        trend_html = f'<div class="{trend_class}">{trend_symbol} {abs(trend_value):.1f}% vs previous</div>'
    
    icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    
    metric_html = f"""
    <div class="kpi-container">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{icon_html}{value}</div>
                {trend_html}
            </div>
        </div>
    </div>
    """
    return st.markdown(metric_html, unsafe_allow_html=True)


def status_badge(status: str, type: str = "success"):
    """Generate an HTML status badge."""
    badge_colors = {
        "success": COLORS['green'],
        "warning": COLORS['orange'],
        "danger": COLORS['red'],
        "info": COLORS['purple'],
        "primary": COLORS['purple']
    }
    color = badge_colors.get(type, COLORS['muted'])
    badge = f'<span style="background:{color}; padding:0.2rem 0.6rem; border-radius:1rem; color:white; font-size:0.7rem; font-weight:500; white-space: nowrap;">{status}</span>'
    return badge


def empty_state(message: str = "No data available", icon: str = "📭", description: str = None):
    """Display a friendly empty state."""
    text_color = COLORS['text'] if not st.session_state.get('dark_mode', False) else '#f1f5f9'
    desc_html = f'<p style="color: {COLORS["muted"]}; font-size: 0.875rem; margin-top: 0.5rem;">{description}</p>' if description else ""
    st.markdown(f"""
    <div class="fin-card" style="text-align: center;">
        <div style="font-size: 3rem;">{icon}</div>
        <p style="color: {text_color}; margin-top: 0.5rem; font-weight: 500;">{message}</p>
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


def loading_state(message: str = "Loading...", icon: str = "⏳"):
    """Display a loading state."""
    st.markdown(f"""
    <div class="fin-card" style="text-align: center;">
        <div style="font-size: 2rem;">{icon}</div>
        <p style="color: {COLORS['muted']};">{message}</p>
    </div>
    """, unsafe_allow_html=True)


def ag_grid_table(df: pd.DataFrame, key: str, height: int = 400, 
                  fit_columns: bool = True, selection_mode: str = "single",
                  pagination: bool = True, page_size: int = 25):
    """Render a professional, interactive table using st-aggrid."""
    if not AG_GRID_AVAILABLE:
        # Fallback to standard dataframe
        return st.dataframe(df, use_container_width=True)
    
    if df.empty:
        empty_state("No data to display")
        return None
    
    # Build grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        groupable=False,
        value=True,
        enableRowGroup=False,
        aggFunc=None,
        editable=False,
        filterable=True,
        sortable=True,
        resizable=True
    )
    
    # Configure grid layout
    if fit_columns:
        gb.configure_grid_options(domLayout='autoHeight')
    else:
        gb.configure_grid_options(domLayout=f'{height}px')
    
    # Add pagination for large datasets
    if pagination and len(df) > page_size:
        gb.configure_pagination(
            paginationAutoPageSize=False,
            paginationPageSize=page_size
        )
    
    # Configure selection
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=(selection_mode == "multiple")
    )
    
    # Add sidebar filters
    gb.configure_side_bar()
    
    grid_options = gb.build()
    
    try:
        theme = "dark" if st.session_state.get('dark_mode', False) else "streamlit"
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            theme=theme,
            key=key,
            allow_unsafe_jscode=True,
            height=height if not fit_columns else None,
            fit_columns_on_load=fit_columns
        )
        return grid_response
    except Exception as e:
        return st.dataframe(df, use_container_width=True)


def fin_tabs(tab_names: List[str]):
    """Create styled tabs and return the tab objects."""
    return st.tabs(tab_names)


def section_header(title: str, description: Optional[str] = None):
    """Display a styled section header."""
    st.markdown(f'<div class="fin-header">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<p style="color: {COLORS["muted"]}; margin-bottom: 1rem;">{description}</p>', 
                   unsafe_allow_html=True)


def subsection_header(title: str):
    """Display a styled subsection header."""
    st.markdown(f'<div class="fin-subheader">{title}</div>', unsafe_allow_html=True)


def metrics_row(metrics: List[Dict[str, Any]]):
    """Display a row of metrics using columns."""
    cols = st.columns(len(metrics))
    for idx, metric in enumerate(metrics):
        with cols[idx]:
            kpi_metric(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                trend=metric.get('trend'),
                trend_value=metric.get('trend_value'),
                icon=metric.get('icon')
            )


# --- Chart Components ---

def fin_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "",
                  color: str = COLORS['purple'], height: int = 400):
    """Create a styled bar chart."""
    if df.empty:
        empty_state("No data available for chart")
        return None
    
    is_dark = st.session_state.get('dark_mode', False)
    bg_color = "#0f172a" if is_dark else "#FAFAF8"
    paper_bg = "#020617" if is_dark else "#FFFFFF"
    text_color = "#e2e8f0" if is_dark else "#1A1230"
    grid_color = "#334155" if is_dark else "#E2D9F3"
    
    fig = go.Figure(data=[
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=color,
            text=df[y_col].apply(lambda x: f'{x:,.0f}' if isinstance(x, (int, float)) else str(x)),
            textposition='outside',
            textfont=dict(color=text_color)
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=text_color)),
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_bg,
        font=dict(color=text_color),
        xaxis=dict(
            gridcolor=grid_color,
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color)
        ),
        yaxis=dict(
            gridcolor=grid_color,
            title_font=dict(color=text_color),
            tickfont=dict(color=text_color)
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=height,
        hovermode='x unified'
    )
    
    return st.plotly_chart(fig, use_container_width=True)


def fin_pie_chart(labels: List[str], values: List[float], title: str = "",
                  hole: float = 0.4, height: int = 400):
    """Create a styled pie/donut chart."""
    if not labels or not values:
        empty_state("No data available for chart")
        return None
    
    is_dark = st.session_state.get('dark_mode', False)
    bg_color = "#0f172a" if is_dark else "#FAFAF8"
    paper_bg = "#020617" if is_dark else "#FFFFFF"
    text_color = "#e2e8f0" if is_dark else "#1A1230"
    
    chart_colors = [COLORS['purple'], COLORS['purple-light'], COLORS['navy'], 
                    COLORS['gold-light'], COLORS['wine'], COLORS['green']]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            marker=dict(colors=chart_colors[:len(labels)]),
            textinfo='label+percent',
            textposition='auto',
            textfont=dict(color=text_color),
            hoverinfo='label+value+percent'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=text_color)),
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_bg,
        font=dict(color=text_color),
        height=height,
        showlegend=True,
        legend=dict(
            font=dict(color=text_color),
            bgcolor=paper_bg
        )
    )
    
    return st.plotly_chart(fig, use_container_width=True)


def fin_line_chart(df: pd.DataFrame, x_col: str, y_cols: List[str], 
                   title: str = "", height: int = 400):
    """Create a styled line chart for trends."""
    if df.empty:
        empty_state("No data available for chart")
        return None
    
    is_dark = st.session_state.get('dark_mode', False)
    bg_color = "#0f172a" if is_dark else "#FAFAF8"
    paper_bg = "#020617" if is_dark else "#FFFFFF"
    text_color = "#e2e8f0" if is_dark else "#1A1230"
    grid_color = "#334155" if is_dark else "#E2D9F3"
    
    fig = go.Figure()
    chart_colors = [COLORS['purple'], COLORS['purple-light'], COLORS['navy'], 
                    COLORS['gold-light'], COLORS['wine'], COLORS['green']]
    
    for idx, y_col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            name=y_col.replace('_', ' ').title(),
            line=dict(color=chart_colors[idx % len(chart_colors)], width=2),
            marker=dict(size=6, color=chart_colors[idx % len(chart_colors)])
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=text_color)),
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_bg,
        font=dict(color=text_color),
        xaxis=dict(
            gridcolor=grid_color,
            title_font=dict(color=text_color)
        ),
        yaxis=dict(
            gridcolor=grid_color,
            title_font=dict(color=text_color)
        ),
        height=height,
        hovermode='x unified',
        legend=dict(
            font=dict(color=text_color),
            bgcolor=paper_bg,
            bordercolor=grid_color
        )
    )
    
    return st.plotly_chart(fig, use_container_width=True)


def fin_gauge_chart(value: float, title: str = "", min_val: float = 0, 
                    max_val: float = 100, height: int = 300):
    """Create a styled gauge chart for match rates."""
    is_dark = st.session_state.get('dark_mode', False)
    bg_color = "#0f172a" if is_dark else "#FAFAF8"
    paper_bg = "#020617" if is_dark else "#FFFFFF"
    text_color = "#e2e8f0" if is_dark else "#1A1230"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': 80, 'increasing': {'color': COLORS['green']}},
        title={'text': title, 'font': {'color': text_color, 'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': text_color},
            'bar': {'color': COLORS['purple']},
            'bgcolor': bg_color,
            'borderwidth': 1,
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [min_val, 30], 'color': f'rgba(220, 38, 38, 0.2)'},
                {'range': [30, 70], 'color': f'rgba(245, 158, 11, 0.2)'},
                {'range': [70, max_val], 'color': f'rgba(5, 150, 105, 0.2)'}
            ],
            'threshold': {
                'line': {'color': COLORS['red'], 'width': 2},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor=paper_bg,
        font={'color': text_color}
    )
    
    return st.plotly_chart(fig, use_container_width=True)


# --- Utility Functions ---

def format_currency(amount: float, currency: str = "KES") -> str:
    """Format currency for display."""
    if currency.upper() == "KES":
        return f"KSh {amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage for display."""
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 0) -> str:
    """Format large numbers with commas."""
    return f"{value:,.{decimals}f}"