# =============================================================================
# synergy_analyzer.py
# Merger synergy scorecard for M&A analysis
# Evaluates strategic fit between two companies
# BA870/AC820 - Boston University
# Team: Sutikshna Tiwari, Kelvin Nlebemchukwu, Vicente Llinares Llata
# =============================================================================

from distress_models import calculate_z_score, calculate_g_score
from manipulation_detector import calculate_m_score
from probability_models import calculate_logit_probability


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divides two numbers, returning 0 if denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_key_ratios(data: dict) -> dict:
    """
    Calculates key financial ratios for a company.
    Used for comparison in the synergy scorecard.
    """
    at  = data.get("Total Assets", 1)
    lt  = data.get("Total Liabilities", 0)
    ni  = data.get("Net Income", 0)
    sale = data.get("Sales", 0)
    gp  = data.get("Gross Profit", 0)
    act = data.get("Current Assets", 0)
    lct = data.get("Current Liabilities", 0)

    return {
        "ROA":            round(safe_divide(ni, at), 4),         # Return on Assets
        "Leverage":       round(safe_divide(lt, at), 4),         # Debt ratio
        "Gross Margin":   round(safe_divide(gp, sale), 4),       # Gross margin
        "Current Ratio":  round(safe_divide(act, lct), 4),       # Liquidity
        "Asset Size":     at                                       # Total assets (size)
    }


def calculate_synergy_score(data_a: dict,
                             data_b: dict,
                             prev_a: dict = None,
                             prev_b: dict = None) -> dict:
    """
    Calculates a synergy score (0-100) between two companies.

    Higher score = more synergy potential.

    Scoring components:
        1. Financial health of both companies   (30 points max)
        2. Ratio complementarity                (30 points max)
        3. Size compatibility                   (20 points max)
        4. Penalties for distress/manipulation  (-20 points max)

    Parameters:
        data_a: current year financials for Company A
        data_b: current year financials for Company B
        prev_a: previous year financials for Company A (for M-Score)
        prev_b: previous year financials for Company B (for M-Score)
    """

    # Use empty dict as fallback if previous year not provided
    if prev_a is None:
        prev_a = data_a
    if prev_b is None:
        prev_b = data_b

    # ── 1. Run distress models on both companies ──────────────────────────
    z_a = calculate_z_score(data_a)
    z_b = calculate_z_score(data_b)
    g_a = calculate_g_score(data_a)
    g_b = calculate_g_score(data_b)
    m_a = calculate_m_score(data_a, prev_a)
    m_b = calculate_m_score(data_b, prev_b)
    logit_a = calculate_logit_probability(data_a)
    logit_b = calculate_logit_probability(data_b)

    # ── 2. Financial health score (30 points) ─────────────────────────────
    # Award points based on distress zones
    def zone_points(z_result, g_result):
        points = 0
        if z_result["zone"] == "Safe":
            points += 8
        elif z_result["zone"] == "Grey Zone":
            points += 4
        if g_result["zone"] == "Healthy":
            points += 7
        elif g_result["zone"] == "Caution":
            points += 3
        return points

    health_score = zone_points(z_a, g_a) + zone_points(z_b, g_b)
    health_score = min(health_score, 30)  # Cap at 30

    # ── 3. Ratio complementarity score (30 points) ────────────────────────
    ratios_a = calculate_key_ratios(data_a)
    ratios_b = calculate_key_ratios(data_b)

    complementarity = 0

    # ROA: reward if one has better profitability (knowledge transfer potential)
    roa_diff = abs(ratios_a["ROA"] - ratios_b["ROA"])
    if roa_diff < 0.05:
        complementarity += 8   # Similar profitability - stable merger
    elif roa_diff < 0.15:
        complementarity += 10  # Some difference - one can improve the other
    else:
        complementarity += 5   # Large gap - integration risk

    # Leverage: reward if combined leverage is manageable
    avg_leverage = (ratios_a["Leverage"] + ratios_b["Leverage"]) / 2
    if avg_leverage < 0.4:
        complementarity += 10  # Low combined debt
    elif avg_leverage < 0.6:
        complementarity += 6
    else:
        complementarity += 2   # High combined debt - risky

    # Gross margin: reward similarity (easier integration)
    margin_diff = abs(ratios_a["Gross Margin"] - ratios_b["Gross Margin"])
    if margin_diff < 0.05:
        complementarity += 10
    elif margin_diff < 0.15:
        complementarity += 6
    else:
        complementarity += 3

    complementarity = min(complementarity, 30)  # Cap at 30

    # ── 4. Size compatibility score (20 points) ───────────────────────────
    size_a = ratios_a["Asset Size"]
    size_b = ratios_b["Asset Size"]

    if size_a == 0 or size_b == 0:
        size_score = 5
    else:
        size_ratio = min(size_a, size_b) / max(size_a, size_b)
        if size_ratio > 0.5:
            size_score = 20   # Similar size - balanced merger
        elif size_ratio > 0.2:
            size_score = 12   # Moderate size gap
        else:
            size_score = 5    # Large size gap - acquisition dynamic

    # ── 5. Penalties for distress and manipulation ────────────────────────
    penalty = 0

    # Penalty for distress zones
    if z_a["zone"] == "Distress":
        penalty += 5
    if z_b["zone"] == "Distress":
        penalty += 5

    # Penalty for manipulation flags
    if "Manipulator" in m_a["flag"] and "Non" not in m_a["flag"]:
        penalty += 5
    if "Manipulator" in m_b["flag"] and "Non" not in m_b["flag"]:
        penalty += 5

    # ── 6. Calculate final synergy score ─────────────────────────────────
    raw_score = health_score + complementarity + size_score - penalty
    final_score = max(0, min(100, raw_score))  # Clamp between 0 and 100

    # ── 7. Determine synergy category ────────────────────────────────────
    if final_score >= 65:
        category = "High Synergy"
        color    = "green"
        interpretation = (
            "Strong strategic fit detected. Both companies show complementary "
            "financial profiles with manageable risk. Merger appears attractive."
        )
    elif final_score >= 35:
        category = "Low Synergy"
        color    = "orange"
        interpretation = (
            "Some synergy potential exists but significant challenges identified. "
            "Careful due diligence and integration planning required."
        )
    else:
        category = "No Synergy"
        color    = "red"
        interpretation = (
            "Limited strategic fit. High distress levels or incompatible financial "
            "profiles make this merger high risk. Not recommended without restructuring."
        )

    return {
        "score":          final_score,
        "category":       category,
        "color":          color,
        "interpretation": interpretation,
        "components": {
            "health_score":    health_score,
            "complementarity": complementarity,
            "size_score":      size_score,
            "penalty":         penalty
        },
        "company_a": {
            "z_score": z_a,
            "g_score": g_a,
            "m_score": m_a,
            "logit":   logit_a,
            "ratios":  ratios_a
        },
        "company_b": {
            "z_score": z_b,
            "g_score": g_b,
            "m_score": m_b,
            "logit":   logit_b,
            "ratios":  ratios_b
        }
    }