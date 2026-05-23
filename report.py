"""
ScoutAI - Scouting Report Generator  (Feature 15: LLM-powered)
==============================================================
Two modes:
  • LLM mode   — `build_scout_prompt()` feeds a compact player profile to the
                  local LLM (ollama/phi-3/llama3).  The server streams tokens
                  back as they arrive, giving a live "writing" effect.
                  Sections: [SCOUTING REPORT (LLM)] → [VALUE] → [VERDICT]

  • Template mode (fallback) — `generate_player_report()` returns the full
                  5-section template as before if no LLM is reachable.

Both modes emit the same SSE envelope so the frontend is unchanged.
"""

import logging

logger = logging.getLogger("scoutai.report")


def _age_curve_narrative(
    age: int, pos: str, peak_age: int, arc_phase: str, mv: float, fv: float
) -> str:
    """
    Return a single sentence describing where the player sits on their
    position-specific career arc, and what that means for valuation.
    Returns an empty string if there is nothing meaningful to say.
    """
    if mv <= 0:
        return ""

    appreciation = (fv / mv - 1) * 100 if mv > 0 else 0

    # Position-specific peak descriptions
    _peak_desc = {
        "FW": "Forwards peak at 24–27, driven by pace and explosiveness.",
        "MF": "Midfielders peak at 25–29, sustained by technique and reading of the game.",
        "DF": "Defenders peak at 26–30; positional intelligence and leadership compensate for physical fade.",
        "GK": "Goalkeepers peak at 28–33 and hold value longer than any other position.",
    }
    peak_desc = _peak_desc.get(pos, "Players at this position peak in their mid-to-late 20s.")

    years_to_peak = peak_age - age

    if arc_phase == "pre-peak":
        if years_to_peak >= 5:
            return (
                f"{peak_desc} At {age}, {abs(years_to_peak)} years from their prime — "
                f"the model projects a peak of {fv:.0f}M, a {appreciation:.0f}% "
                f"appreciation from today. The market consistently underprices players "
                f"this far from their ceiling."
            )
        elif years_to_peak >= 2:
            return (
                f"{peak_desc} At {age}, approaching their prime in {years_to_peak} years. "
                f"Projected peak value {fv:.0f}M — {appreciation:.0f}% upside. "
                f"Buy before the market prices in the remaining growth."
            )
        else:
            return (
                f"{peak_desc} At {age}, on the cusp of peak years. "
                f"The model sees {appreciation:.0f}% residual upside to {fv:.0f}M "
                f"before the plateau is reached."
            )

    elif arc_phase == "peak":
        if pos == "GK":
            return (
                f"Goalkeepers hold a 28–33 prime plateau — the longest of any position. "
                f"At {age}, squarely in peak years. No value decay expected for the "
                f"next {peak_age + 3 - age} years; future value mirrors current at {fv:.0f}M."
            )
        else:
            return (
                f"{peak_desc} At {age}, operating in the prime window right now. "
                f"Output and value are at or near their ceiling — "
                f"what you see is what you get. Future value holds at {fv:.0f}M."
            )

    else:  # post-peak
        years_past = age - peak_age
        decline_pct = (1 - fv / mv) * 100 if mv > 0 else 0
        if decline_pct <= 5:
            return (
                f"{peak_desc} At {age}, just past peak but decline is still modest. "
                f"The model projects minimal depreciation to {fv:.0f}M. "
                f"Short-term value remains strong; monitor over the next 12 months."
            )
        elif decline_pct <= 20:
            return (
                f"{peak_desc} At {age}, {years_past} year(s) past prime. "
                f"The model sees {decline_pct:.0f}% depreciation to {fv:.0f}M. "
                f"Factor the resale discount into the transfer fee and contract length."
            )
        else:
            return (
                f"{peak_desc} At {age}, {years_past} years past peak — "
                f"the model projects {decline_pct:.0f}% depreciation to {fv:.0f}M. "
                f"Buy only for immediate impact; future resale value is negligible."
            )


def generate_player_report(player: dict) -> list[dict]:
    """
    Return a list of {"section": str, "text": str} dicts.
    Each section is streamed separately to the frontend.
    """
    pos  = player.get("position", "MF")
    name = player.get("player", "Unknown")
    team = player.get("team",   "Unknown")
    age  = int(player.get("age", 0))

    mv   = float(player.get("market_value",    0))
    pv   = float(player.get("predicted_value", 0))
    fv          = float(player.get("future_value",    0))
    gap         = float(player.get("value_gap",       0))
    psi         = float(player.get("psi",             0))
    cyr         = int(player.get("contract_years",    2))
    cexp        = int(player.get("contract_expires",  2026))
    mv_low      = float(player.get("mv_low",  mv * 0.70))
    mv_high     = float(player.get("mv_high", mv * 1.30))
    mv_uncert   = float(player.get("mv_uncertainty", mv * 0.30))
    peak_age    = int(player.get("peak_age",  27))
    arc_phase   = str(player.get("arc_phase", "peak"))
    inj_risk    = int(player.get("injury_risk",       30))
    inj_label   = str(player.get("injury_risk_label", "Moderate"))
    inj_age_sc  = int(player.get("injury_risk_age",   20))
    inj_load_sc = int(player.get("injury_risk_load",  50))

    g90    = float(player.get("goals_per90",   0))
    xg90   = float(player.get("xg_per90",      0))
    a90    = float(player.get("assists_per90",  0))
    xag90  = float(player.get("xag_per90",     0))
    pacc   = float(player.get("pass_acc",       0))
    aerial = float(player.get("aerial_pct",    0))
    takeon = float(player.get("take_on_pct",   0))
    mins   = int(player.get("minutes",         0))

    # Stats always available from soccerdata (multi-league data)
    sh90      = float(player.get("sh_per90",     0))
    sot90     = float(player.get("sot_per90",    0))
    g_per_sh  = float(player.get("g_per_shot",   0))   # 0-1 fraction
    g_per_sot = float(player.get("g_per_sot",    0))   # 0-1 fraction
    tkl90     = float(player.get("tackles_per90",0))
    inter     = float(player.get("interceptions",0))
    fouls_w   = float(player.get("fouls_drawn",  0))
    gk_sv     = float(player.get("gk_savepct",   0))   # 0-100 %
    gk_cs     = float(player.get("gk_cspct",     0))   # 0-100 %
    gk_ga90   = float(player.get("gk_ga90",      0))

    sections: list[dict] = []

    # ------------------------------------------------------------------
    # OVERVIEW
    # ------------------------------------------------------------------
    age_txt = {
        range(0,  21): f"At {age}, one of the younger players at this level - significant development headroom remains.",
        range(21, 25): f"At {age}, entering the high-growth phase of a career - output should continue to rise.",
        range(25, 29): f"At {age}, operating in their prime years and likely performing at or near their ceiling.",
        range(29, 33): f"At {age}, a seasoned professional. Experience is an asset; contract length must be managed carefully.",
        range(33, 50): f"At {age}, in the late phase of a career. Short-term value only; future resale minimal.",
    }
    age_sentence = next((v for k, v in age_txt.items() if age in k), "")

    minutes_sentence = ""
    if mins > 2500:
        minutes_sentence = f"A first-choice starter with {mins} minutes this season - durability is not a concern."
    elif mins > 1500:
        minutes_sentence = f"Regular contributor with {mins} minutes - likely part of a rotation."
    else:
        minutes_sentence = f"Limited to {mins} minutes - either injury, form, or squad depth is a factor worth investigating."

    overview = (
        f"{name} is a {age}-year-old {_pos_label(pos)} currently at {team}. "
        f"{age_sentence} "
        f"{minutes_sentence} "
        f"Market value currently stands at {mv:.0f}M euros."
    )
    sections.append({"section": "OVERVIEW", "text": overview})

    # ------------------------------------------------------------------
    # STRENGTHS
    # ------------------------------------------------------------------
    strengths: list[str] = []

    if pos == "FW":
        if g90 > 0 and xg90 > 0:
            ratio = g90 / xg90
            if ratio > 1.15:
                strengths.append(
                    f"Elite finisher - converts {g90:.2f} goals per 90 against an xG of only {xg90:.2f}, "
                    f"a conversion ratio of {ratio:.2f}x. Consistently beats the expected model."
                )
            elif g90 > 0.4:
                strengths.append(
                    f"Reliable scorer with {g90:.2f} goals per 90 - a constant threat in the final third."
                )
        elif g90 > 0.4:
            strengths.append(
                f"Prolific finisher with {g90:.2f} goals per 90 - a consistent attacking return."
            )
        if g_per_sh > 0.12:
            strengths.append(
                f"Exceptional shot efficiency: converts {g_per_sh*100:.1f}% of attempts - "
                f"gets maximum return from the chances created."
            )
        elif sh90 > 3.5:
            strengths.append(
                f"High-volume attacker with {sh90:.1f} shots per 90 - always dangerous and hard to manage."
            )
        if xag90 > 0.18:
            strengths.append(
                f"Not just a goal threat - creates {xag90:.2f} xAG per 90, functioning as a link "
                f"player and second assist creator."
            )
        elif a90 > 0.2:
            strengths.append(
                f"Creative forward with {a90:.2f} assists per 90 - contributes beyond just goals."
            )
        if takeon > 52:
            strengths.append(
                f"Dangerous in 1v1 situations, completing {takeon:.1f}% of dribble attempts."
            )

    elif pos == "MF":
        if a90 > 0.2 or xag90 > 0.18:
            detail = f"with {xag90:.2f} xAG" if xag90 > 0 else f"({a90:.2f} per 90)"
            strengths.append(
                f"Genuine creative output {detail} - consistently puts teammates into high-value positions."
            )
        if g90 > 0.15:
            strengths.append(
                f"Contributes goals from midfield at {g90:.2f} per 90 - a double threat "
                f"that defences must account for."
            )
        if sh90 > 2.0 and g_per_sh > 0.08:
            strengths.append(
                f"Dangerous from distance: {sh90:.1f} shots per 90 converting at {g_per_sh*100:.1f}% - "
                f"a threat beyond just through-balls."
            )
        if tkl90 > 2.5:
            strengths.append(
                f"Defensively industrious with {tkl90:.2f} tackles per 90 - "
                f"wins the ball back consistently and protects the backline."
            )
        if pacc > 86:
            strengths.append(
                f"Technically excellent in possession at {pacc:.1f}% pass completion. "
                f"Rarely loses the ball under pressure."
            )
        elif pacc > 80:
            strengths.append(
                f"Reliable in possession at {pacc:.1f}% pass accuracy - keeps the team ticking."
            )

    elif pos == "DF":
        if tkl90 > 2.0:
            strengths.append(
                f"Combative and hard to beat, winning {tkl90:.2f} tackles per 90 - "
                f"a physical presence who makes life difficult for opposition attackers."
            )
        if inter > 40:
            strengths.append(
                f"Reads the game exceptionally well with {inter:.0f} interceptions this season - "
                f"cuts out danger before it develops."
            )
        if aerial > 62:
            strengths.append(
                f"Dominant in the air, winning {aerial:.1f}% of aerial duels. "
                f"An asset at set pieces and against physical strikers."
            )
        if pacc > 82:
            strengths.append(
                f"Comfortable on the ball at {pacc:.1f}% pass accuracy - "
                f"contributes to build-up rather than just clearing lines."
            )
        if tkl90 > 1.5 and inter > 25:
            strengths.append(
                f"A well-rounded defensive profile combining active ball-winning with positional awareness."
            )

    elif pos == "GK":
        if gk_sv > 70:
            strengths.append(
                f"Strong shot-stopper with a {gk_sv:.1f}% save rate - "
                f"consistently denies opponents and keeps the team in games."
            )
        if gk_cs > 35:
            strengths.append(
                f"Excellent organisational presence: contributes to a {gk_cs:.1f}% clean sheet rate - "
                f"a genuine defensive platform for the team."
            )
        if gk_ga90 > 0 and gk_ga90 < 1.0:
            strengths.append(
                f"Concedes just {gk_ga90:.2f} goals per 90 - elite-level defensive reliability."
            )
        if pacc > 75:
            strengths.append(
                f"Excellent distribution at {pacc:.1f}% pass accuracy. "
                f"A genuine ball-playing goalkeeper who extends attacking patterns from the back."
            )
        if aerial > 70:
            strengths.append(
                f"Commands the box assertively, claiming {aerial:.1f}% of aerial contests. "
                f"Reduces crosses becoming chances."
            )

    if psi > 0.72:
        strengths.append(
            f"Overall PSI of {psi:.2f} places them comfortably in the top tier of {_pos_label(pos)}s "
            f"in this dataset - consistently elite across the key performance dimensions."
        )
    elif psi > 0.52:
        strengths.append(
            f"PSI of {psi:.2f} reflects a solid above-average performer for their position."
        )

    if not strengths:
        strengths.append(
            f"Shows a balanced profile without a standout weakness. "
            f"Consistent contributor across the measured dimensions."
        )

    sections.append({"section": "STRENGTHS", "text": " ".join(strengths)})

    # ------------------------------------------------------------------
    # CONCERNS
    # ------------------------------------------------------------------
    concerns: list[str] = []

    if pos == "FW":
        if xg90 > 0.3 and g90 < xg90 * 0.78:
            concerns.append(
                f"Underperforming their xG: scoring {g90:.2f} per 90 against an expected {xg90:.2f}. "
                f"Either a finishing confidence issue or the positions being taken are lower quality."
            )
        elif g90 < 0.2 and sh90 > 2.0 and g_per_sh < 0.08:
            concerns.append(
                f"High shot volume ({sh90:.1f}/90) but low conversion ({g_per_sh*100:.1f}%). "
                f"Getting into positions but failing to finish - needs work in front of goal."
            )
        if g90 < 0.15 and (xg90 < 0.2 or xg90 == 0):
            concerns.append(
                f"Low attacking output ({g90:.2f} G/90). "
                f"Would need significant improvement to justify a starting role at a top club."
            )

    elif pos == "MF":
        if pacc > 0 and pacc < 75:
            concerns.append(
                f"Pass completion of {pacc:.1f}% is below the level expected from a central midfielder. "
                f"May struggle retaining possession under high-press opposition."
            )
        if tkl90 < 1.0 and g90 < 0.1 and a90 < 0.1:
            concerns.append(
                f"Limited contribution both offensively and defensively - "
                f"must offer more to justify a place in a competitive squad."
            )

    elif pos == "DF":
        if aerial > 0 and aerial < 45:
            concerns.append(
                f"Aerial win rate of {aerial:.1f}% is below average for a centre-back. "
                f"Could be exposed by physical strikers."
            )
        if pacc > 0 and pacc < 70:
            concerns.append(
                f"Pass completion of {pacc:.1f}% suggests difficulty playing out from the back - "
                f"a concern for any possession-based system."
            )
        if tkl90 < 1.0 and inter < 20:
            concerns.append(
                f"Quiet defensive metrics ({tkl90:.2f} tackles/90, {inter:.0f} interceptions this season). "
                f"The defensive contribution does not stand out in the data."
            )

    if age > 30:
        concerns.append(
            f"At {age}, the model projects a declining value trajectory. "
            f"Any contract beyond two years carries meaningful risk of overpaying in the back half."
        )

    if gap < -10:
        concerns.append(
            f"The model values them at {pv:.0f}M against a {mv:.0f}M market price. "
            f"That {abs(gap):.0f}M premium above model valuation is difficult to justify from the data alone."
        )

    if mins < 1200:
        concerns.append(
            f"Only {mins} minutes played this season. The sample size is limited - "
            f"the performance data should be treated with caution until a larger sample confirms it."
        )

    # ── Injury risk flag (Feature 12) ─────────────────────────────────────
    # Append a usage-pattern risk sentence for Elevated+ players
    if inj_label in ("Elevated", "High", "Very High"):
        age_driver  = inj_age_sc >= 55   # age is the primary driver
        load_driver = inj_load_sc >= 65  # high minutes load
        if inj_label == "Very High":
            concerns.append(
                f"Usage-pattern injury risk flag: {inj_risk}/100 ({inj_label}). "
                f"The combination of age ({age}) and season load ({mins} min) "
                f"sits at the extreme end of what top-flight bodies can sustain. "
                f"A medical due-diligence review is essential before committing."
            )
        elif inj_label == "High":
            driver_txt = (
                "age-related recovery decline is the primary driver" if age_driver
                else f"a heavy {mins}-minute season load is the primary driver"
            )
            concerns.append(
                f"Usage-pattern injury risk flag: {inj_risk}/100 ({inj_label}). "
                f"At this risk level, {driver_txt}. "
                f"Factor in medical screening and realistic rotation expectations."
            )
        else:  # Elevated
            driver_txt = (
                "post-peak age profile" if age_driver
                else f"{mins} minutes of season load" if load_driver
                else "combined age and workload profile"
            )
            concerns.append(
                f"Elevated usage-pattern risk ({inj_risk}/100) driven by {driver_txt}. "
                f"Not a red flag, but worth monitoring in a long contract negotiation."
            )

    if not concerns:
        concerns.append(
            "No material concerns flagged by the model. The data profile is clean and consistent."
        )

    sections.append({"section": "CONCERNS", "text": " ".join(concerns)})

    # ------------------------------------------------------------------
    # VALUE ASSESSMENT
    # ------------------------------------------------------------------
    # Confidence interval phrasing — only show if interval is meaningful (>2M width)
    if mv_high - mv_low > 2:
        ci_str = f" (model range: {mv_low:.0f}M–{mv_high:.0f}M, 80% confidence interval)"
    else:
        ci_str = ""

    val_parts = [
        f"Current market value: {mv:.0f}M euros{ci_str}. "
        f"Projected future value (age-adjusted): {fv:.0f}M euros."
    ]

    if gap > 15:
        val_parts.append(
            f"The model identifies a significant undervaluation of {gap:.0f}M. "
            f"Either the market has not yet priced in recent performance improvements, "
            f"or the selling club is motivated. Either way, this represents strong value."
        )
    elif gap > 5:
        val_parts.append(
            f"Modest upside of {gap:.0f}M above current market price. "
            f"Not a screaming buy but reasonably priced relative to output."
        )
    elif gap > -5:
        val_parts.append(
            f"Fairly priced - current market aligns closely with model valuation. "
            f"No significant discount or premium at the asking price."
        )
    elif gap < -15:
        val_parts.append(
            f"Significantly overpriced by {abs(gap):.0f}M relative to model valuation. "
            f"The market premium may reflect reputation or commercial value not captured in on-pitch stats."
        )
    else:
        val_parts.append(
            f"Slightly overpriced at {abs(gap):.0f}M above model valuation - "
            f"consider negotiating before committing."
        )

    # Position-specific age curve narrative
    _arc_text = _age_curve_narrative(age, pos, peak_age, arc_phase, mv, fv)
    if _arc_text:
        val_parts.append(_arc_text)

    # Contract context - this changes the effective acquisition cost materially
    if cyr == 0:
        val_parts.append(
            f"Contract already expired - {name} is currently a free agent. "
            f"No transfer fee is required. Any acquisition costs are limited to agent fees "
            f"and signing-on bonuses. This is a zero-cost opportunity for any club willing "
            f"to move quickly on wages."
        )
    elif cyr <= 1:
        val_parts.append(
            f"Contract situation is the defining factor here: only {cyr} year remaining "
            f"(expires summer {cexp}). The listed {mv:.0f}M market price is theoretical - "
            f"clubs typically accept 35-55% of market value for players entering their final year "
            f"rather than lose them on a free. If negotiations stall, a pre-contract can be "
            f"agreed from January. Effective acquisition cost may be well below {mv:.0f}M."
        )
    elif cyr == 2:
        val_parts.append(
            f"Contract runs until {cexp} - two years remaining. The selling club still has "
            f"negotiating leverage but the clock is ticking. A 10-20% discount on the "
            f"listed {mv:.0f}M is realistic if you move before the final year begins."
        )
    elif cyr >= 4:
        val_parts.append(
            f"Long contract ({cyr} years, expires {cexp}) means the selling club holds full "
            f"leverage. Expect to pay at or above the listed market value with limited room "
            f"to negotiate. Factor this into the budget plan."
        )

    sections.append({"section": "VALUE ASSESSMENT", "text": " ".join(val_parts)})

    # ------------------------------------------------------------------
    # VERDICT
    # Contract situation can upgrade a MONITOR to a BUY, or turn a
    # NEGOTIATE into a BUY if the expiry creates leverage.
    # ------------------------------------------------------------------

    # Effective gap: expiring contract boosts value signal because true
    # acquisition cost is below listed market value
    contract_boost = 0
    if cyr == 0:
        contract_boost = mv * 0.90   # effectively free - full market value is a boost
    elif cyr <= 1:
        contract_boost = mv * 0.45   # 45% discount implicit in expiry
    elif cyr == 2:
        contract_boost = mv * 0.15   # 15% discount in 2-year situations

    effective_gap = gap + contract_boost

    if effective_gap > 12 and psi > 0.55:
        verdict_tag  = "STRONG BUY"
        if cyr == 0:
            verdict_text = (
                f"{name} is a free agent with a PSI of {psi:.2f}. "
                f"No transfer fee required - only wages and agent fees. "
                f"This is an elite player available at zero acquisition cost. Sign immediately."
            )
        elif cyr <= 1:
            verdict_text = (
                f"{name} is a strong performer ({psi:.2f} PSI) entering the final year of their "
                f"contract. The {mv:.0f}M listed price is negotiable - clubs rarely hold firm "
                f"on expiring assets. Move now to secure a deal well below market rate. "
                f"If talks stall, a pre-contract in January is available at zero cost."
            )
        else:
            verdict_text = (
                f"{name} offers a rare combination of undervaluation and strong performance. "
                f"At {mv:.0f}M with {gap:.0f}M of model upside and a PSI of {psi:.2f}, "
                f"this is the kind of signing that defines a transfer window. Move quickly."
            )
    elif effective_gap > 0 and psi > 0.45:
        verdict_tag  = "BUY"
        contract_note = (
            f" The expiring contract gives you additional negotiating leverage."
            if cyr <= 1 else
            f" The two-year contract window means a deal is achievable at a discount." if cyr == 2 else ""
        )
        verdict_text = (
            f"Solid acquisition at current price. {name} delivers consistent output "
            f"with a positive value profile. Low-risk, reasonable-reward profile.{contract_note}"
        )
    elif gap < -12 and psi < 0.5 and cyr > 2:
        verdict_tag  = "AVOID"
        verdict_text = (
            f"Overpriced and underperforming. At {mv:.0f}M with the model showing "
            f"a {abs(gap):.0f}M premium, the risk-reward does not stack up. "
            f"Redirect the budget to better-value alternatives."
        )
    elif effective_gap < -5:
        verdict_tag  = "NEGOTIATE"
        if cyr <= 1:
            reduction = min(abs(gap) + int(mv * 0.30), int(mv * 0.55))
            verdict_text = (
                f"The performance numbers are mixed but the contract situation creates leverage. "
                f"With only {cyr} year remaining, push for a {reduction:.0f}M reduction from "
                f"the listed {mv:.0f}M - or wait until January for a pre-contract on better terms."
            )
        else:
            verdict_text = (
                f"There is interest here but not at asking price. "
                f"Push for a {min(abs(gap), 15):.0f}M reduction or walk away. "
                f"The player's output does not justify a full {mv:.0f}M outlay."
            )
    else:
        verdict_tag  = "MONITOR"
        contract_note = (
            " Check back in January when a pre-contract becomes available." if cyr <= 1 else ""
        )
        verdict_text = (
            f"{name} is a competent player at a fair price. "
            f"Not a priority target but worth keeping on the watchlist if priorities shift "
            f"or if the price drops in the final days of the window.{contract_note}"
        )

    sections.append({
        "section": f"VERDICT: {verdict_tag}",
        "text":    verdict_text,
        "verdict": verdict_tag,
    })

    return sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_label(pos: str) -> str:
    return {"GK": "goalkeeper", "DF": "defender",
            "MF": "midfielder", "FW": "forward"}.get(pos, "player")


# ---------------------------------------------------------------------------
# Feature 15 — LLM prompt builder
# ---------------------------------------------------------------------------

def build_scout_prompt(player: dict) -> str:
    """
    Build a compact, information-dense prompt for the local LLM.

    The prompt feeds only the facts the LLM needs — style, key stats,
    value signal, arc, risk — and asks for a 3-paragraph ~150-word report.
    Keeping the prompt under ~350 tokens leaves plenty of room for output
    within Phi-3-mini / llama3's context windows.
    """
    pos   = player.get("position", "MF")
    name  = player.get("player",   "Unknown")
    age   = int(player.get("age", 0))
    team  = player.get("team",  "Unknown")
    league = player.get("league", "")
    mins  = int(player.get("minutes", 0))
    psi   = float(player.get("psi", 0))
    mv    = float(player.get("market_value", 0))
    fv    = float(player.get("future_value", 0))
    gap   = fv - mv
    mv_low  = float(player.get("mv_low",  mv * 0.70))
    mv_high = float(player.get("mv_high", mv * 1.30))
    cyr   = int(player.get("contract_years", 2))
    cexp  = int(player.get("contract_expires", 2026))
    arc   = str(player.get("arc_phase", "peak"))
    peak_age = int(player.get("peak_age", 27))
    inj_label = str(player.get("injury_risk_label", "Moderate"))
    inj_score = int(player.get("injury_risk", 30))
    style = str(player.get("style_label", ""))
    nat   = str(player.get("nationality", ""))

    # Position-specific key stat line
    g90  = float(player.get("goals_per90", 0))
    a90  = float(player.get("assists_per90", 0))
    sh90 = float(player.get("sh_per90", 0))
    gps  = float(player.get("g_per_shot", 0))
    tkl  = float(player.get("tackles_tkl", 0))
    ints = float(player.get("int", 0))
    gk_sv = float(player.get("gk_savepct", 0))
    gk_cs = float(player.get("gk_cspct", 0))

    if pos == "GK":
        stats_line = f"Save% {gk_sv:.1f}%, CS% {gk_cs:.1f}%"
    elif pos == "DF":
        stats_line = (
            f"Tackles {tkl:.0f}, Interceptions {ints:.0f}, "
            f"Goals/90 {g90:.2f}, Assists/90 {a90:.2f}"
        )
    elif pos == "MF":
        stats_line = (
            f"Goals/90 {g90:.2f}, Assists/90 {a90:.2f}, "
            f"Shots/90 {sh90:.1f}, Tackles {tkl:.0f}, Int {ints:.0f}"
        )
    else:  # FW
        stats_line = (
            f"Goals/90 {g90:.2f}, Assists/90 {a90:.2f}, "
            f"Shots/90 {sh90:.1f}, Finishing {gps*100:.1f}%"
        )

    # Arc context
    if arc == "pre-peak":
        yrs = peak_age - age
        arc_ctx = f"pre-peak, {yrs} year(s) from prime — upside buy"
    elif arc == "post-peak":
        yrs = age - peak_age
        arc_ctx = f"post-peak, {yrs} year(s) past prime — immediate impact only"
    else:
        arc_ctx = "in prime — peak output now"

    # Contract context
    if cyr == 0:
        contract_ctx = "FREE AGENT — no transfer fee"
    elif cyr == 1:
        contract_ctx = f"1 year left (expires {cexp}) — pre-contract available Jan"
    else:
        contract_ctx = f"{cyr} years remaining until {cexp}"

    # Value signal
    if gap > 15:
        value_signal = f"UNDERVALUED by {gap:.0f}M — strong buy signal"
    elif gap > 5:
        value_signal = f"modest upside {gap:.0f}M above market"
    elif gap > -5:
        value_signal = "fairly priced"
    elif gap < -15:
        value_signal = f"OVERVALUED by {abs(gap):.0f}M — market premium unjustified"
    else:
        value_signal = f"slightly overpriced by {abs(gap):.0f}M"

    # PSI tier
    if psi >= 0.80:
        psi_tier = "elite"
    elif psi >= 0.60:
        psi_tier = "strong"
    elif psi >= 0.40:
        psi_tier = "average"
    else:
        psi_tier = "below average"

    prompt = f"""You are an expert football scout writing a concise scouting report. Write plain prose, no bullet points, no headers.

PLAYER DATA:
Name: {name} | Age: {age} | Position: {_pos_label(pos)} | Club: {team} ({league})
Style archetype: {style}
PSI (performance index): {psi:.2f} — {psi_tier} for position
Key stats: {stats_line}
Minutes this season: {mins}
Career arc: {arc_ctx}
Market value: {mv:.0f}M | Model range: {mv_low:.0f}–{mv_high:.0f}M | Future value: {fv:.0f}M
Value signal: {value_signal}
Contract: {contract_ctx}
Injury risk flag: {inj_label} ({inj_score}/100){f' | Nationality: {nat}' if nat else ''}

Write a scouting report in exactly 3 paragraphs, approximately 150 words total:
Paragraph 1 — Technical profile: what kind of player is this, their strengths and style.
Paragraph 2 — Value case: is this a good deal and why, factoring in contract and age trajectory.
Paragraph 3 — Risks and fit: what concerns exist, what type of team suits them.

Use direct scout language. No filler. Start writing immediately."""

    return prompt


# ---------------------------------------------------------------------------
# Feature 15 — Structured sections (VALUE + VERDICT only, no narrative prose)
# Used in LLM mode so the template still provides the reliable numbers.
# ---------------------------------------------------------------------------

def extract_structured_sections(player: dict) -> list[dict]:
    """
    Return only the VALUE ASSESSMENT and VERDICT sections from the template.
    Called in LLM mode after the narrative has already been streamed.
    """
    all_sections = generate_player_report(player)
    return [s for s in all_sections if s["section"] in ("VALUE ASSESSMENT",) or
            s.get("verdict")]


# ---------------------------------------------------------------------------
# Feature 19 — "Why this player?" explainability
# ---------------------------------------------------------------------------

def build_explain_prompt(player: dict, query: str, filters: dict) -> str:
    """
    Build a tight LLM prompt that explains *why* this specific player was
    recommended for this specific query.

    The LLM must name actual numbers and connect them to what the scout asked.
    Output: 2–3 short sentences. No headers, no bullets, no padding.
    """
    pos   = player.get("position", "MF")
    name  = player.get("player", "Unknown")
    age   = int(player.get("age", 0))
    team  = player.get("team", "Unknown")
    mins  = int(player.get("minutes", 0))
    mv    = float(player.get("market_value", 0))
    fv    = float(player.get("future_value", 0))
    gap   = fv - mv
    psi   = float(player.get("psi", 0))
    cyr   = int(player.get("contract_years", 2))
    arc   = str(player.get("arc_phase", "peak"))
    style = str(player.get("style_label", ""))

    g90   = float(player.get("goals_per90",  0))
    a90   = float(player.get("assists_per90", 0))
    sh90  = float(player.get("sh_per90",     0))
    gps   = float(player.get("g_per_shot",   0))
    tkl90 = float(player.get("tackles_per90",0))
    ints  = float(player.get("interceptions",0))
    gk_sv = float(player.get("gk_savepct",   0))
    gk_cs = float(player.get("gk_cspct",     0))
    tc    = int(player.get("transfer_confidence", 0))
    tc_lbl = str(player.get("transfer_label", ""))

    if pos == "GK":
        key_stats = f"Save% {gk_sv:.1f}%, Clean sheet% {gk_cs:.1f}%"
    elif pos == "DF":
        key_stats = f"{tkl90:.1f} tackles/90, {ints:.0f} interceptions"
    elif pos == "MF":
        key_stats = (
            f"{g90:.2f} goals/90, {a90:.2f} assists/90, "
            f"{sh90:.1f} shots/90, {tkl90:.1f} tackles/90"
        )
    else:  # FW
        key_stats = (
            f"{g90:.2f} goals/90, {a90:.2f} assists/90, "
            f"{sh90:.1f} shots/90, {gps*100:.1f}% shot conversion"
        )

    # Build filter context string so LLM knows what was asked
    filter_parts = []
    if filters.get("position"):
        filter_parts.append(f"position: {filters['position']}")
    if filters.get("max_price"):
        filter_parts.append(f"budget ≤{filters['max_price']:.0f}M")
    if filters.get("max_age"):
        filter_parts.append(f"age ≤{filters['max_age']}")
    if filters.get("min_age"):
        filter_parts.append(f"age ≥{filters['min_age']}")
    if filters.get("nationality"):
        filter_parts.append(f"nationality: {filters['nationality']}")
    filter_str = ", ".join(filter_parts) if filter_parts else "no specific filters"

    value_line = (
        f"Undervalued by {gap:.0f}M (model: {fv:.0f}M vs market: {mv:.0f}M)" if gap > 5
        else f"Overvalued by {abs(gap):.0f}M" if gap < -5
        else f"Fairly priced at {mv:.0f}M"
    )
    contract_line = (
        "free agent" if cyr == 0
        else f"{cyr} year{'s' if cyr != 1 else ''} left"
    )
    arc_line = arc.replace("-", " ")
    tc_line = f"{tc}/100 ({tc_lbl})" if tc_lbl else f"{tc}/100"

    prompt = f"""You are an expert football scout. Explain in 2-3 sentences exactly why {name} matches this search.
Be specific: name the actual numbers that matter. No generic praise.

SCOUT'S QUERY: "{query}"
APPLIED FILTERS: {filter_str}

PLAYER: {name}, {age}y {_pos_label(pos)} at {team}
Key stats: {key_stats}
Minutes: {mins} | PSI: {psi:.2f} | Style: {style}
Value: {value_line} | Contract: {contract_line} | Arc: {arc_line}
Transfer confidence: {tc_line}

In 2-3 sentences, explain why {name} specifically fits this search. Start with the strongest matching stat or trait, then mention any value or availability angle. Use concrete numbers. No filler words."""

    return prompt


def generate_explain_bullets(player: dict, query: str, filters: dict) -> list[str]:
    """
    Template fallback when no LLM is available.
    Returns 3-4 concise bullet strings explaining the match.
    """
    pos   = player.get("position", "MF")
    name  = player.get("player", "Unknown")
    age   = int(player.get("age", 0))
    mins  = int(player.get("minutes", 0))
    mv    = float(player.get("market_value", 0))
    fv    = float(player.get("future_value", 0))
    gap   = fv - mv
    psi   = float(player.get("psi", 0))
    cyr   = int(player.get("contract_years", 2))
    arc   = str(player.get("arc_phase", "peak"))
    g90   = float(player.get("goals_per90",  0))
    a90   = float(player.get("assists_per90", 0))
    sh90  = float(player.get("sh_per90",     0))
    gps   = float(player.get("g_per_shot",   0))
    tkl90 = float(player.get("tackles_per90",0))
    ints  = float(player.get("interceptions",0))
    gk_sv = float(player.get("gk_savepct",   0))
    gk_cs = float(player.get("gk_cspct",     0))
    tc    = int(player.get("transfer_confidence", 0))
    tc_lbl = str(player.get("transfer_label", ""))

    bullets: list[str] = []

    # ── Position filter match ─────────────────────────────────────────────
    fp = filters.get("position")
    if fp:
        bullets.append(
            f"Plays {_pos_label(pos)} — matches your position filter."
        )

    # ── Strongest performance stat ────────────────────────────────────────
    if pos == "GK":
        if gk_sv >= 70:
            bullets.append(f"{gk_sv:.1f}% save rate — elite shot-stopper for position.")
        if gk_cs >= 35:
            bullets.append(f"{gk_cs:.1f}% clean sheet rate — solid defensive record.")
    elif pos == "FW":
        if g90 > 0.3:
            bullets.append(f"{g90:.2f} goals per 90 — prolific finisher.")
        if gps > 0.12:
            bullets.append(f"{gps*100:.1f}% shot conversion — efficient in front of goal.")
        elif sh90 > 3.0:
            bullets.append(f"{sh90:.1f} shots per 90 — constant threat on target.")
        if a90 > 0.2:
            bullets.append(f"{a90:.2f} assists per 90 — creative as well as clinical.")
    elif pos == "MF":
        if g90 + a90 > 0.4:
            bullets.append(f"{g90:.2f}+{a90:.2f} goals+assists per 90 — productive in both phases.")
        if tkl90 > 2.5:
            bullets.append(f"{tkl90:.1f} tackles per 90 — combative and hard-working.")
        if ints > 40:
            bullets.append(f"{ints:.0f} interceptions this season — excellent positional read.")
    elif pos == "DF":
        if tkl90 > 2.0:
            bullets.append(f"{tkl90:.1f} tackles per 90 — strong defensive presence.")
        if ints > 50:
            bullets.append(f"{ints:.0f} interceptions — dominant in blocking passing lanes.")

    # ── PSI performance level ─────────────────────────────────────────────
    if psi >= 0.75:
        bullets.append(f"PSI {psi:.2f} — elite performer for position this season.")
    elif psi >= 0.55:
        bullets.append(f"PSI {psi:.2f} — above-average performer for position.")

    # ── Value angle ───────────────────────────────────────────────────────
    if gap > 10:
        bullets.append(
            f"Model values them at {fv:.0f}M vs {mv:.0f}M market — "
            f"{gap:.0f}M undervaluation is a significant buy signal."
        )
    elif gap > 4:
        bullets.append(f"Slight undervaluation ({gap:.0f}M above model) — good value at market price.")

    # ── Budget filter match ───────────────────────────────────────────────
    max_p = filters.get("max_price")
    if max_p and mv <= max_p:
        bullets.append(f"{mv:.0f}M market value — within your {max_p:.0f}M budget.")

    # ── Age / arc ─────────────────────────────────────────────────────────
    max_a = filters.get("max_age")
    if max_a:
        bullets.append(f"Aged {age} — meets your age requirement (≤{max_a}).")
    elif arc == "pre-peak":
        bullets.append(f"At {age}, still {abs(int(player.get('peak_age', 27)) - age)} years from peak — upside buy.")
    elif arc == "peak":
        bullets.append(f"At {age}, currently in peak years — ready-made impact.")

    # ── Transfer availability ─────────────────────────────────────────────
    if tc >= 65:
        bullets.append(
            f"Transfer confidence {tc}/100 ({tc_lbl}) — "
            f"{'free agent' if cyr == 0 else f'{cyr} year contract'} makes a deal very achievable."
        )
    elif tc >= 45:
        bullets.append(
            f"Transfer confidence {tc}/100 — {cyr}-year contract creates a negotiable window."
        )

    return bullets[:5]  # cap at 5 bullets
