# =============================================================================
# data_fetcher.py
# Module for fetching and cleaning financial data
# Sources: yfinance (public companies) or manual input (private companies)
# BA870/AC820 - Boston University
# Team: Sutikshna Tiwari, Kelvin Nlebemchukwu, Vicente Llinares Llata
# =============================================================================

import yfinance as yf
import pandas as pd
import streamlit as st


def fetch_financials(ticker: str) -> dict:
    """
    Fetches the latest annual financial data for a public company using yfinance.
    Returns a standardized dictionary with key financial line items.
    """
    try:
        # Download company data from Yahoo Finance
        company = yf.Ticker(ticker)

        # Get annual financial statements
        income_stmt = company.financials        # Income statement
        balance_sheet = company.balance_sheet   # Balance sheet
        cash_flow = company.cashflow            # Cash flow statement

        # Check if data exists
        if income_stmt.empty or balance_sheet.empty:
            st.error(f"No financial data found for ticker: {ticker}")
            return None

        # Use the most recent year (first column)
        inc = income_stmt.iloc[:, 0]
        bal = balance_sheet.iloc[:, 0]
        cf = cash_flow.iloc[:, 0]

        # Helper function to safely extract values
        def get_val(series, *keys):
            for key in keys:
                if key in series.index:
                    val = series[key]
                    if pd.notna(val):
                        return float(val)
            return 0.0

        # Build standardized dictionary with all required financial items
        data = {
            # Balance Sheet items
            "Total Assets":          get_val(bal, "Total Assets"),
            "Current Assets":        get_val(bal, "Current Assets"),
            "Current Liabilities":   get_val(bal, "Current Liabilities"),
            "Total Liabilities":     get_val(bal, "Total Liabilities Net Minority Interest",
                                              "Total Liabilities"),
            "Shareholders Equity":   get_val(bal, "Stockholders Equity",
                                              "Common Stock Equity"),
            "Retained Earnings":     get_val(bal, "Retained Earnings"),

            # Income Statement items
            "Sales":                 get_val(inc, "Total Revenue"),
            "EBIT":                  get_val(inc, "EBIT", "Operating Income"),
            "Net Income":            get_val(inc, "Net Income"),
            "Gross Profit":          get_val(inc, "Gross Profit"),
            "SGA":                   get_val(inc, "Selling General Administrative",
                                              "Operating Expense"),
            "Depreciation":          get_val(inc, "Reconciled Depreciation",
                                              "Depreciation Amortization Depletion"),

            # Balance Sheet - additional
            "Accounts Receivable":   get_val(bal, "Accounts Receivable",
                                              "Net Receivables"),

            # Cash Flow items
            "Cash Flow Operations":  get_val(cf, "Operating Cash Flow",
                                              "Cash Flow From Continuing Operating Activities"),

            # Ticker for reference
            "ticker": ticker.upper()
        }

        return data

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def manual_input_form(company_label: str = "Company") -> dict:
    """
    Displays a Streamlit form for manual entry of financial data.
    Used for private companies or when auto-fetch is not available.
    All values should be entered in millions (USD).
    """

    st.subheader(f"📝 Manual Input – {company_label}")
    st.caption("Enter all values in USD millions. Use negative numbers where applicable.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Balance Sheet**")
        total_assets       = st.number_input(f"Total Assets [{company_label}]",
                                              value=0.0, step=1.0, key=f"at_{company_label}")
        current_assets     = st.number_input(f"Current Assets [{company_label}]",
                                              value=0.0, step=1.0, key=f"act_{company_label}")
        current_liab       = st.number_input(f"Current Liabilities [{company_label}]",
                                              value=0.0, step=1.0, key=f"lct_{company_label}")
        total_liab         = st.number_input(f"Total Liabilities [{company_label}]",
                                              value=0.0, step=1.0, key=f"lt_{company_label}")
        equity             = st.number_input(f"Shareholders Equity [{company_label}]",
                                              value=0.0, step=1.0, key=f"ceq_{company_label}")
        retained_earnings  = st.number_input(f"Retained Earnings [{company_label}]",
                                              value=0.0, step=1.0, key=f"re_{company_label}")
        accounts_rec       = st.number_input(f"Accounts Receivable [{company_label}]",
                                              value=0.0, step=1.0, key=f"rec_{company_label}")

    with col2:
        st.markdown("**Income Statement & Cash Flow**")
        sales              = st.number_input(f"Sales / Revenue [{company_label}]",
                                              value=0.0, step=1.0, key=f"sale_{company_label}")
        ebit               = st.number_input(f"EBIT [{company_label}]",
                                              value=0.0, step=1.0, key=f"ebit_{company_label}")
        net_income         = st.number_input(f"Net Income [{company_label}]",
                                              value=0.0, step=1.0, key=f"ni_{company_label}")
        gross_profit       = st.number_input(f"Gross Profit [{company_label}]",
                                              value=0.0, step=1.0, key=f"gp_{company_label}")
        sga                = st.number_input(f"SG&A Expenses [{company_label}]",
                                              value=0.0, step=1.0, key=f"sga_{company_label}")
        depreciation       = st.number_input(f"Depreciation [{company_label}]",
                                              value=0.0, step=1.0, key=f"dep_{company_label}")
        cfo                = st.number_input(f"Cash Flow from Operations [{company_label}]",
                                              value=0.0, step=1.0, key=f"cfo_{company_label}")

    # Return standardized dictionary matching fetch_financials() output
    data = {
        "Total Assets":         total_assets,
        "Current Assets":       current_assets,
        "Current Liabilities":  current_liab,
        "Total Liabilities":    total_liab,
        "Shareholders Equity":  equity,
        "Retained Earnings":    retained_earnings,
        "Sales":                sales,
        "EBIT":                 ebit,
        "Net Income":           net_income,
        "Gross Profit":         gross_profit,
        "SGA":                  sga,
        "Depreciation":         depreciation,
        "Accounts Receivable":  accounts_rec,
        "Cash Flow Operations": cfo,
        "ticker":               "PRIVATE"
    }

    return data