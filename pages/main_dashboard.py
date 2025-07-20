# main_dashboard.py
import streamlit as st
import sys
import os

# Add the 'pages' directory to the Python path
# This allows importing modules directly from the 'pages' directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'pages'))

# Import the functions from your individual app pages
from fx_reconcilliation_app import fx_reconciliation_app
from fx_trade_reconciliation import graphed_analysis_app

st.set_page_config(layout="wide", page_title="Unified FX Dashboard")

# Custom CSS for styling (re-applying the theme from fx_reconciliation_app for consistency)
st.markdown(
    """
    <style>
    /* Define CSS Variables */
    :root {
        --color-white: #FFFFFF;
        --color-secondary: #798088;
        --color-primary: #361371;
        --color-pink-80: #9F6AF8CC; /* 0.8 alpha */
        --color-container: #F0EFEF4D; /* ~30% opacity */
        --color-buy-goods: #F5EFFD;
        --color-green: #2B9973;
        --color-buy-airtime-20: #77CE8780; /* 20% alpha */
        --color-utilities-20: #9F6AF833; /* 20% alpha */
        --color-red: #E85E5D;
        --color-vibrant-pink: #9F6AF8;
        --font-family: 'Inter', sans-serif;
        --color-text-input: '#FFAD6B';
    }

    /* General Body and Text */
    body {
        font-family: var(--font-family);
        color: var(--color-primary);
    }


    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--color-primary);
        font-weight: 600;
    }
    h1 {
        color: var(--color-vibrant-pink);
    }

    /* Markdown text */
    .stMarkdown {
        color: var(--color-secondary);
    }

    /* Buttons */
    .stButton button {
        background-color: var(--color-vibrant-pink);
        color: var(--color-white);
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    .stButton button:hover {
        background-color: var(--color-pink-80);
        color: var(--color-white);
    }

    /* Selectboxes and Text Inputs */
    .stSelectbox > div > div, .stTextInput > div > div > input {

        color: var(--color-text-input);
    }
    .stSelectbox > label, .stTextInput label {
        color: var(--color-secondary);
        font-weight: bold;
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid var(--color-secondary);
        border-radius: 8px;
        overflow: auto;
        background-color: var(--color-container);
    }
    .stDataFrame table {
        color: var(--color-primary);
    }
    .stDataFrame thead th {
        background-color: var(--color-primary);
        color: var(--color-white);
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: rgba(var(--color-primary-rgb), 0.05); /* Light primary tint */
    }

    /* Info, Warning, Error messages */
    .stAlert {
        border-radius: 8px;
    }
    .stAlert.info {
        background-color: var(--color-utilities-20);
        color: var(--color-primary);
    }
    .stAlert.warning {
        background-color: var(--color-pink-80); /* Using pink for warning, could be orange */
        color: var(--color-primary);
    }
    .stAlert.error {
        background-color: var(--color-red);
        color: var(--color-white);
    }
    .stAlert.success {
        background-color: var(--color-buy-airtime-20);
        color: var(--color-green);
    }

    /* Sliders */
    .stSlider .st-bb { /* Track */
        background: var(--color-utilities-20);
    }
    .stSlider .st-bc { /* Fill */
        background: var(--color-vibrant-pink);
    }
    .stSlider .st-bd { /* Thumb */
        background: var(--color-primary);
        border: 2px solid var(--color-vibrant-pink);
    }

    /* Checkbox */
    .stCheckbox span {
        color: var(--color-primary);
    }

    /* Spinner */
    .stSpinner > div > div {
        color: var(--color-vibrant-pink);
    }

    /* Custom RGB for rgba calculations - Streamlit doesn't expose these directly */
    /* This is a workaround, ideally Streamlit would handle alpha better with its theme config */
    /* You would need to manually get RGB values from your hex colors */
    /* For example, #361371 is 54, 19, 113 */
    .stApp {
        --primary-color-rgb: 54, 19, 113;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["FX Reconciliation", "Graphed Analysis"])

if page_selection == "FX Reconciliation":
    fx_reconciliation_app()
elif page_selection == "Graphed Analysis":
    graphed_analysis_app()

