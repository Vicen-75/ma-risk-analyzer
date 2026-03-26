# =============================================================================
# probability_models.py
# Logistic regression bankruptcy probability + overall risk scoring
# Coefficients trained on WRDS Compustat North America 2010-2024
# (103,770 observations)
# BA870/AC820 - Boston University
# Team: Sutikshna Tiwari, Kelvin Nlebemchukwu, Vicente Llinares Llata
# =============================================================================

import math


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divides two numbers, returning 0 if denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_logit_probability(data: dict) -> dict:
    """
    Calculates bankruptcy probability using a logistic regression model.

    Coefficients derived from training on WRDS Compustat North America
    data (2010-2024), using an Ohlson-style 5-variable specification.

    Variables:
        X1 = log(Total Assets)                    - firm size
        X2 = Total Liabilities / Total Assets     - leverage
        X3 = Working Capital / Total Assets       - liquidity
        X4 = Current Liabilities / Current Assets - short-term risk
        X5 = Net Income / Total Assets            - profitability (ROA)

    Coefficients from WRDS training (2010-2024):
        Intercept: -0.5872
        X1: -1.0275
        X2: -0.5114
        X3: -0.4466
        X4:  0.0223
        X5: -0.0394
    """

    # Extract financial data
    at  = data.get("Total Assets", 1)
    lt  = data.get("Total Liabilities", 0)
    act = data.get("Current Assets", 0)
    lct = data.get("Current Liabilities", 0)
    ni  = data.get("Net Income", 0)

    # Calculate the 5 input variables
    X1 = math.log(max(at, 0.001))          # log(Total Assets) - firm size
    X2 = safe_divide(lt, at)               # Leverage
    X3 = safe_divide(act - lct, at)        # Working Capital / Assets
    X4 = safe_divide(lct, max(act, 0.001)) # Short-term risk
    X5 = safe_divide(ni, at)               # ROA

    # ── Coefficients from WRDS Compustat training (2010-2024) ────────────
    intercept = -0.5872
    coef_X1   = -1.0275
    coef_X2   = -0.5114
    coef_X3   = -0.4466
    coef_X4   =  0.0223
    coef_X5   = -0.0394

    # Calculate linear combination
    linear = (intercept
              + coef_X1 * X1
              + coef_X2 * X2
              + coef_X3 * X3
              + coef_X4 * X4
              + coef_X5 * X5)

    # Apply sigmoid function to get probability
    probability = 1 / (1 + math.exp(-linear))

    return {
        "probability": round(probability, 4),
        "probability_pct": round(probability * 100, 2),
        "X1": round(X1, 4),
        "X2": round(X2, 4),
        "X3": round(X3, 4),
        "X4": round(X4, 4),
        "X5": round(X5, 4)
    }


def calculate_overall_risk_score(z_result: dict,
                                  g_result: dict,
                                  m_result: dict,
                                  logit_result: dict) -> dict:
    """
    Combines Z-Score, G-Score, M-Score and logit probability into
    a single overall risk score from 1 (low risk) to 10 (high risk).

    Weighting:
        - Altman Z"-Score:     30%
        - Grover G-Score:      25%
        - Logit Probability:   30%
        - Beneish M-Score:     15% (penalty if manipulator flagged)
    """

    # ── 1. Convert Z-Score zone to partial score (0-10) ──────────────────
    z_zone = z_result.get("zone", "Grey Zone")
    if z_zone == "Safe":
        z_partial = 2.0
    elif z_zone == "Grey Zone":
        z_partial = 5.5
    else:  # Distress
        z_partial = 9.0

    # ── 2. Convert G-Score zone to partial score (0-10) ──────────────────
    g_zone = g_result.get("zone", "Caution")
    if g_zone == "Healthy":
        g_partial = 2.0
    elif g_zone == "Caution":
        g_partial = 5.5
    else:  # Distress
        g_partial = 9.0

    # ── 3. Convert logit probability to partial score (0-10) ─────────────
    prob = logit_result.get("probability", 0.5)
    logit_partial = prob * 10  # Direct mapping: 0% → 0, 100% → 10

    # ── 4. M-Score penalty ────────────────────────────────────────────────
    m_flag = m_result.get("flag", "")
    if "Manipulator" in m_flag and "Non" not in m_flag:
        m_penalty = 2.0   # Add 2 points to risk if manipulation suspected
    else:
        m_penalty = 0.0

    # ── 5. Weighted combination ───────────────────────────────────────────
    raw_score = (0.30 * z_partial +
                 0.25 * g_partial +
                 0.30 * logit_partial +
                 0.15 * m_penalty * 10)  # Normalize penalty to 0-10 scale

    # Clamp final score between 1 and 10
    final_score = max(1.0, min(10.0, raw_score))

    # ── 6. Determine final zone and recommendation ────────────────────────
    if final_score <= 3.5:
        zone  = "Low Risk"
        color = "green"
        recommendation = "Company appears financially healthy. Standard due diligence recommended."
        discount = "0-5%"
    elif final_score <= 6.5:
        zone  = "Medium Risk"
        color = "orange"
        recommendation = "Some financial concerns detected. Enhanced due diligence recommended."
        discount = "10-20%"
    else:
        zone  = "High Risk"
        color = "red"
        recommendation = "Significant distress signals detected. Proceed with caution."
        discount = "25-40%"

    return {
        "score":          round(final_score, 2),
        "zone":           zone,
        "color":          color,
        "recommendation": recommendation,
        "price_discount": discount,
        "components": {
            "z_partial":     round(z_partial, 2),
            "g_partial":     round(g_partial, 2),
            "logit_partial": round(logit_partial, 2),
            "m_penalty":     m_penalty
        }
    }