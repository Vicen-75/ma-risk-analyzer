# =============================================================================
# manipulation_detector.py
# Earnings manipulation detection using the Beneish M-Score model
# BA870/AC820 - Boston University
# Team: Sutikshna Tiwari, Kelvin Nlebemchukwu, Vicente Llinares Llata
# =============================================================================


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divides two numbers, returning 0 if denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_m_score_indices(current: dict, previous: dict) -> dict:
    """
    Calculates the 8 indices required for the Beneish M-Score.

    Parameters:
        current:  dictionary with current year financial data
        previous: dictionary with previous year financial data

    Returns: dictionary with all 8 indices
    """

    # Extract current year values
    sale_t    = current.get("Sales", 0)
    rec_t     = current.get("Accounts Receivable", 0)
    gp_t      = current.get("Gross Profit", 0)
    at_t      = current.get("Total Assets", 0)
    ppe_t     = current.get("Total Assets", 0) - current.get("Current Assets", 0)  # PPE proxy
    dep_t     = current.get("Depreciation", 0)
    sga_t     = current.get("SGA", 0)
    lt_t      = current.get("Total Liabilities", 0)
    ni_t      = current.get("Net Income", 0)
    cfo_t     = current.get("Cash Flow Operations", 0)

    # Extract previous year values
    sale_t1   = previous.get("Sales", 0)
    rec_t1    = previous.get("Accounts Receivable", 0)
    gp_t1     = previous.get("Gross Profit", 0)
    at_t1     = previous.get("Total Assets", 0)
    ppe_t1    = previous.get("Total Assets", 0) - previous.get("Current Assets", 0)
    dep_t1    = previous.get("Depreciation", 0)
    sga_t1    = previous.get("SGA", 0)
    lt_t1     = previous.get("Total Liabilities", 0)

    # ── Index 1: DSRI – Days Sales Receivables Index ──────────────────────
    # Rising DSRI may indicate revenue inflation
    dsri = safe_divide(
        safe_divide(rec_t, sale_t),
        safe_divide(rec_t1, sale_t1)
    )

    # ── Index 2: GMI – Gross Margin Index ─────────────────────────────────
    # Deteriorating margins may signal manipulation pressure
    gmi = safe_divide(
        safe_divide(gp_t1, sale_t1),
        safe_divide(gp_t, sale_t)
    )

    # ── Index 3: AQI – Asset Quality Index ────────────────────────────────
    # Increase in non-current, non-PPE assets relative to total assets
    aqi = safe_divide(
        1 - safe_divide(current.get("Current Assets", 0) + ppe_t, at_t),
        1 - safe_divide(previous.get("Current Assets", 0) + ppe_t1, at_t1)
    )

    # ── Index 4: SGI – Sales Growth Index ─────────────────────────────────
    # High sales growth firms are more likely to manipulate
    sgi = safe_divide(sale_t, sale_t1)

    # ── Index 5: DEPI – Depreciation Index ────────────────────────────────
    # Falling depreciation rate may indicate asset life manipulation
    depi = safe_divide(
        safe_divide(dep_t1, dep_t1 + ppe_t1),
        safe_divide(dep_t, dep_t + ppe_t)
    )

    # ── Index 6: SGAI – SG&A Index ────────────────────────────────────────
    # Disproportionate SG&A growth relative to sales
    sgai = safe_divide(
        safe_divide(sga_t, sale_t),
        safe_divide(sga_t1, sale_t1)
    )

    # ── Index 7: TATA – Total Accruals to Total Assets ────────────────────
    # High accruals relative to assets signal earnings manipulation
    tata = safe_divide(ni_t - cfo_t, at_t)

    # ── Index 8: LVGI – Leverage Index ────────────────────────────────────
    # Increasing leverage may motivate earnings manipulation
    lvgi = safe_divide(
        safe_divide(lt_t, at_t),
        safe_divide(lt_t1, at_t1)
    )

    return {
        "DSRI": round(dsri, 4),
        "GMI":  round(gmi, 4),
        "AQI":  round(aqi, 4),
        "SGI":  round(sgi, 4),
        "DEPI": round(depi, 4),
        "SGAI": round(sgai, 4),
        "TATA": round(tata, 4),
        "LVGI": round(lvgi, 4)
    }


def calculate_m_score(current: dict, previous: dict) -> dict:
    """
    Calculates the Beneish M-Score for earnings manipulation detection.

    Formula:
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
            + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Threshold:
        M > -1.78 → Likely Manipulator (red flag)
        M ≤ -1.78 → Non-Manipulator

    Parameters:
        current:  current year financial data dictionary
        previous: previous year financial data dictionary

    Returns: dict with M-Score, flag, color, and all 8 indices
    """

    # Calculate the 8 indices
    idx = calculate_m_score_indices(current, previous)

    # Beneish M-Score formula
    m_score = (-4.84
               + 0.920 * idx["DSRI"]
               + 0.528 * idx["GMI"]
               + 0.404 * idx["AQI"]
               + 0.892 * idx["SGI"]
               + 0.115 * idx["DEPI"]
               - 0.172 * idx["SGAI"]
               + 4.679 * idx["TATA"]
               - 0.327 * idx["LVGI"])

    # Determine manipulation flag
    if m_score > -1.78:
        flag  = "⚠️ Likely Manipulator"
        color = "red"
    else:
        flag  = "✅ Non-Manipulator"
        color = "green"

    return {
        "m_score": round(m_score, 4),
        "flag":    flag,
        "color":   color,
        "indices": idx
    }