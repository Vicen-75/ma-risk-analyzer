# =============================================================================
# distress_models.py
# Bankruptcy and financial distress prediction models
# Implements: Altman Z"-Score and Grover G-Score
# BA870/AC820 - Boston University
# Team: Sutikshna Tiwari, Kelvin Nlebemchukwu, Vicente Llinares Llata
# =============================================================================


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divides two numbers, returning 0 if denominator is zero.
    Prevents ZeroDivisionError throughout the module.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_x_variables(data: dict) -> dict:
    """
    Calculates the four X variables used in both Altman Z" and Grover G models.

    X1 = Working Capital / Total Assets         → liquidity ratio
    X2 = Retained Earnings / Total Assets       → cumulative profitability
    X3 = EBIT / Total Assets                    → operating efficiency
    X4 = Shareholders Equity / Total Liabilities → financial leverage
    """
    at  = data.get("Total Assets", 0)
    act = data.get("Current Assets", 0)
    lct = data.get("Current Liabilities", 0)
    re  = data.get("Retained Earnings", 0)
    ebit = data.get("EBIT", 0)
    ceq  = data.get("Shareholders Equity", 0)
    lt   = data.get("Total Liabilities", 0)

    X1 = safe_divide(act - lct, at)   # Working Capital / Total Assets
    X2 = safe_divide(re, at)          # Retained Earnings / Total Assets
    X3 = safe_divide(ebit, at)        # EBIT / Total Assets
    X4 = safe_divide(ceq, lt)         # Equity / Total Liabilities

    return {"X1": X1, "X2": X2, "X3": X3, "X4": X4}


def calculate_z_score(data: dict) -> dict:
    """
    Calculates the Altman Z"-Score (1983 version, for private/non-manufacturing firms).

    Formula: Z" = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    Zones:
        Z" > 2.60  → Safe (green)
        1.10 < Z" < 2.60 → Grey zone (yellow)
        Z" < 1.10  → Distress (red)

    Returns: dict with score, zone, color, and individual X values
    """
    x = calculate_x_variables(data)

    # Altman Z" formula (1983)
    z_score = (6.56 * x["X1"] +
               3.26 * x["X2"] +
               6.72 * x["X3"] +
               1.05 * x["X4"])

    # Determine zone based on thresholds
    if z_score > 2.60:
        zone  = "Safe"
        color = "green"
    elif z_score > 1.10:
        zone  = "Grey Zone"
        color = "orange"
    else:
        zone  = "Distress"
        color = "red"

    return {
        "score": round(z_score, 4),
        "zone":  zone,
        "color": color,
        "X1": round(x["X1"], 4),
        "X2": round(x["X2"], 4),
        "X3": round(x["X3"], 4),
        "X4": round(x["X4"], 4)
    }


def calculate_g_score(data: dict) -> dict:
    """
    Calculates the Grover G-Score for financial distress prediction.

    Formula: G = 1.65*X1 + 3.40*X2 + 6.50*X3 + 1.00*X4

    Zones:
        G > 0.01  → Healthy (green)
        -0.02 < G < 0.01 → Caution (yellow)
        G < -0.02 → Distress (red)

    Returns: dict with score, zone, color, and individual X values
    """
    x = calculate_x_variables(data)

    # Grover G-Score formula
    g_score = (1.65 * x["X1"] +
               3.40 * x["X2"] +
               6.50 * x["X3"] +
               1.00 * x["X4"])

    # Determine zone based on thresholds
    if g_score > 0.01:
        zone  = "Healthy"
        color = "green"
    elif g_score > -0.02:
        zone  = "Caution"
        color = "orange"
    else:
        zone  = "Distress"
        color = "red"

    return {
        "score": round(g_score, 4),
        "zone":  zone,
        "color": color,
        "X1": round(x["X1"], 4),
        "X2": round(x["X2"], 4),
        "X3": round(x["X3"], 4),
        "X4": round(x["X4"], 4)
    }