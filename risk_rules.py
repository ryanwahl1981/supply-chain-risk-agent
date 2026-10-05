"""
Supply Chain Risk Rules Engine (no AI)

Decides, by rule, for every part:
  - risk level (HIGH / MEDIUM / LOW)
  - supplier risk tier
  - what to order (spike PO or follow-on PO), how many units, and by when

HOW TO READ THIS FILE
- SETTINGS (below) are YOUR business rules. Change a number, rerun, see the result.
- assess() applies the rules to every part.
- The helper functions at the bottom are also used by agent.py and evaluate.py.

Run:  python risk_rules.py     (writes risk_assessment.csv and supplier_tiers.csv)
"""
import math
import os
import re
from datetime import date, timedelta

import pandas as pd

# ------------------------------------------------------------------
# SETTINGS (edit these)
# ------------------------------------------------------------------
AS_OF = date(2026, 10, 2)        # fixed "today" so results are reproducible
REORDER_WINDOW_DAYS = 14        # follow-on PO due within this many days before arrival = MEDIUM
LATE_PO_GRACE_DAYS = 7          # a PO later than this is not counted as incoming
LATE_PO_ESCALATION_DAYS = 14    # a PO this late on a short item raises the supplier tier one step
SPIKE_BUFFER_WEEKS = 1.5        # extra weeks of demand added to every spike PO
BUFFER_DAYS = {"LOW": 0, "MEDIUM": 7, "HIGH": 14}   # follow-on PO released this many days before landing
FIN_HIGH, FIN_MEDIUM = 45, 60   # financial health score cutoffs
OTD_TARGET = 90                 # on-time delivery % target
OTD_DECLINE_LEVEL, OTD_DECLINE_POINTS = 85, 8      # decline trigger: last 3 months under 85%, down 8+ points
STEP_UP = {"LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "HIGH"}


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def read_data(folder="."):
    names = ["suppliers", "parts", "open_pos", "supplier_otd", "risk_alerts"]
    data = {n: pd.read_csv(os.path.join(folder, f"{n}.csv"), dtype=str, keep_default_na=False)
            for n in names}
    numeric = {"suppliers": ["lead_time_days", "financial_health_score"],
               "parts": ["unit_cost", "weekly_demand", "on_hand", "safety_stock"],
               "open_pos": ["qty"], "supplier_otd": ["otd_pct"], "risk_alerts": ["delay_days_est"]}
    for table, cols in numeric.items():
        for c in cols:
            data[table][c] = pd.to_numeric(data[table][c])
    return data


# ------------------------------------------------------------------
# SUPPLIER TIER (supplier-wide evidence only)
# ------------------------------------------------------------------
def supplier_base_tier(sid, s, data):
    alerts, otd = data["risk_alerts"], data["supplier_otd"]
    active = alerts[(alerts.status == "ACTIVE") & (alerts.scope_value == s.country)]
    o = otd[otd.supplier_id == sid].sort_values("month").otd_pct.tolist()
    first3, last3 = sum(o[:3]) / 3, sum(o[-3:]) / 3
    high, medium = [], []
    for a in active.itertuples():
        text = f"active {a.severity} alert {a.alert_id} ({a.category}) in {s.country}"
        if a.severity == "HIGH":
            high.append(text)
        elif a.severity == "MEDIUM":
            medium.append(text)
    if s.financial_health_score < FIN_HIGH:
        high.append(f"financial health score {s.financial_health_score} is below {FIN_HIGH}")
    elif s.financial_health_score < FIN_MEDIUM:
        medium.append(f"financial health score {s.financial_health_score} is below {FIN_MEDIUM}")
    if last3 < OTD_DECLINE_LEVEL and first3 - last3 >= OTD_DECLINE_POINTS:
        high.append(f"on-time delivery fell from {first3:.0f}% to {last3:.0f}% over six months")
    elif last3 < OTD_TARGET:
        medium.append(f"last-3-month on-time delivery {last3:.1f}% is below the {OTD_TARGET}% target")
    tier = "HIGH" if high else "MEDIUM" if medium else "LOW"
    return tier, high + medium


# ------------------------------------------------------------------
# ASSESS EVERY PART
# ------------------------------------------------------------------
def assess(data):
    suppliers, parts, pos, alerts = data["suppliers"], data["parts"], data["open_pos"], data["risk_alerts"]
    sup = suppliers.set_index("supplier_id")

    # Pass 1: inventory position of every part (projected stock vs open PO timing)
    info = {}
    for p in parts.itertuples():
        s = sup.loc[p.supplier_id]
        active = alerts[(alerts.status == "ACTIVE") & (alerts.scope_value == s.country)]
        delay = int(active.delay_days_est.max()) if len(active) else 0
        d = p.weekly_demand / 7                                  # daily demand
        rop = p.safety_stock + d * s.lead_time_days             # reorder point
        usable_q, first_days, late_days, late_po = 0, None, 0, ""
        for po in pos[(pos.item_id == p.item_id) & (pos.status == "OPEN")].itertuples():
            days = (date.fromisoformat(po.due_date) - AS_OF).days
            if days < 0 and -days > late_days:
                late_days, late_po = -days, po.po_id
            if days < -LATE_PO_GRACE_DAYS:
                continue                                          # too late: not counted as incoming
            usable_q += po.qty
            first_days = days if first_days is None else min(first_days, days)
        A = None if first_days is None else max(first_days, 0) + delay   # expected arrival, in days from today
        if usable_q == 0:
            p_before = p_after = None
            t_r = (p.on_hand - rop) / d                          # days until a PO must be released
            base = "HIGH" if t_r <= 0 else "MEDIUM" if t_r <= REORDER_WINDOW_DAYS else "LOW"
        else:
            p_before = p.on_hand - d * A                         # stock when the PO lands (negative = backlog)
            p_after = p_before + usable_q
            t_r = (p.on_hand + usable_q - rop) / d
            base = ("HIGH" if p_before < 0 or t_r < A - REORDER_WINDOW_DAYS
                    else "MEDIUM" if t_r <= A else "LOW")
        info[p.item_id] = dict(p=p, s=s, d=d, rop=rop, q=usable_q, A=A, delay=delay, first_days=first_days,
                               p_before=p_before, p_after=p_after, t_r=t_r, base=base,
                               late_days=late_days, late_po=late_po)

    # Pass 2: supplier tier, plus attention for a long-late PO on an item that is short
    tier_rows, tiers = [], {}
    for sid, s in sup.iterrows():
        tier, reasons = supplier_base_tier(sid, s, data)
        for iid, v in info.items():
            if (v["p"].supplier_id == sid and v["late_days"] >= LATE_PO_ESCALATION_DAYS
                    and v["base"] == "HIGH"):
                tier = STEP_UP[tier]
                reasons.append(f"raised one step: PO {v['late_po']} for {iid} is {v['late_days']} "
                               f"days late and that item is short")
                break
        tiers[sid] = tier
        tier_rows.append({"supplier_id": sid, "name": s["name"], "country": s.country, "tier": tier,
                          "reasons": "; ".join(reasons) if reasons else "no risk triggers"})

    # Pass 3: final level, action, quantity and date
    rows = []
    for iid, v in info.items():
        p, s, d, A = v["p"], v["s"], v["d"], v["A"]
        tier = tiers[p.supplier_id]
        level = "HIGH" if v["base"] == "MEDIUM" and tier == "HIGH" else v["base"]
        action, qty, release_by = "NONE", 0, ""
        if v["base"] == "HIGH":
            # spike PO: backlog + safety stock + lead-time demand + buffer, minus what is already on order
            action = "SPIKE_PO"
            need = v["rop"] + d * (A or 0) - p.on_hand - v["q"]
            qty = math.ceil(max(0, need) + SPIKE_BUFFER_WEEKS * p.weekly_demand)
            driver = ("NO_USABLE_PO_BELOW_REORDER_POINT" if v["q"] == 0
                      else "STOCKOUT_BEFORE_PO_LANDS" if v["p_before"] < 0
                      else "OPEN_PO_TOO_SMALL_FOLLOW_ON_PO_OVERDUE")
        elif v["base"] == "MEDIUM":
            # follow-on PO: one lead time of demand, released earlier for riskier suppliers
            action = "FOLLOW_ON_PO"
            qty = round(d * s.lead_time_days)
            deadline = min(v["t_r"], (A - BUFFER_DAYS[tier]) if A is not None else v["t_r"])
            release_by = "NOW" if deadline <= 0 else (AS_OF + timedelta(days=math.floor(deadline))).isoformat()
            driver = ("NO_PO_REORDER_POINT_WITHIN_WINDOW" if v["q"] == 0 else "FOLLOW_ON_PO_DUE_WITHIN_WINDOW")
        else:
            driver = "AMPLE_COVER"
        shortfall = max(0, round(-v["p_before"])) if v["p_before"] is not None else 0
        # deadline fields exist only for follow-on POs; days_until_release_by matches release_by exactly
        raw_deadline = round(v["t_r"], 1) if action == "FOLLOW_ON_PO" else ""
        days_until = ("" if action != "FOLLOW_ON_PO"
                      else 0 if release_by == "NOW" else (date.fromisoformat(release_by) - AS_OF).days)
        rows.append({
            "item_id": iid, "description": p.description, "supplier_id": p.supplier_id,
            "supplier_tier": tier, "risk_level": level, "base_level": v["base"],
            "escalated_by_supplier_tier": v["base"] == "MEDIUM" and tier == "HIGH",
            "driver": driver, "action_type": action, "recommended_qty": qty, "release_by": release_by,
            "weeks_of_stock": round(p.on_hand / p.weekly_demand, 1), "on_hand": p.on_hand,
            "weekly_demand": p.weekly_demand, "safety_stock": p.safety_stock,
            "lead_time_days": s.lead_time_days, "reorder_point": round(v["rop"]),
            "open_po_qty": v["q"],
            "po_due_date": "" if v["first_days"] is None else (AS_OF + timedelta(days=v["first_days"])).isoformat(),
            "alert_delay_days": v["delay"],
            "expected_arrival": "" if A is None else (AS_OF + timedelta(days=A)).isoformat(),
            "stock_before_po_lands": "" if v["p_before"] is None else round(v["p_before"]),
            "stock_after_po_lands": "" if v["p_after"] is None else round(v["p_after"]),
            "days_until_release_by": days_until, "raw_release_deadline_days": raw_deadline,
            "shortfall_units": shortfall, "shortfall_value": round(shortfall * p.unit_cost, 2),
            "late_po_id": v["late_po"], "late_po_days": v["late_days"], "alt_supplier_id": p.alt_supplier_id,
        })
    return pd.DataFrame(rows), pd.DataFrame(tier_rows)


# ------------------------------------------------------------------
# HELPERS shared with agent.py and evaluate.py
# ------------------------------------------------------------------
def rule_recommendation(row):
    """The order recommendation, in plain words, built by CODE from the rule results."""
    qty = int(row["recommended_qty"])
    if row["action_type"] == "SPIKE_PO":
        return (f"Raise a spike PO of {qty:,} units now, on top of the open PO (covers backlog, safety "
                f"stock, one lead time of demand and {SPIKE_BUFFER_WEEKS} weeks of buffer).")
    if row["action_type"] == "FOLLOW_ON_PO":
        when = "now" if row["release_by"] == "NOW" else f"by {row['release_by']}"
        return f"Release a follow-on PO of {qty:,} units {when} (one lead time of demand)."
    return "No order action needed."


CHECK_KEYS = ["qty_ok", "date_ok", "timing_ok", "late_ok", "tier_ok", "finance_ok", "wording_ok", "counts_ok", "ids_ok"]


def tier_patterns(reasons, own_item=""):
    """Words/IDs that show a brief has explained why a supplier has its tier."""
    pats = []
    for r in reasons.split("; "):
        pats += [re.escape(x) for x in re.findall(r"(?:ALT|PO|ITM)-\d+", r) if x != own_item]
        if "days late" in r:
            pats.append(r"\blate\b|overdue")
        pats += re.findall(r"\(([A-Z_]+)\)", r)                      # alert category, e.g. TYPHOON
        if "financial" in r:
            pats.append("financ")
        if "on-time delivery" in r:
            pats += ["on-time", "delivery"]
    return pats


def required_facts(row, data, tiers):
    """Facts a brief MUST mention. Decided by CODE, so Claude cannot skip them."""
    facts = []
    if int(row["late_po_days"]) > LATE_PO_GRACE_DAYS:
        facts.append(f"Late PO: {row['late_po_id']} is {row['late_po_days']} days late and is not "
                     f"counted as incoming.")
    t = tiers.set_index("supplier_id").loc[row["supplier_id"]]
    if t.tier != "LOW":
        facts.append(f"Supplier risk tier {t.tier}. Why: {t.reasons}. Explain why the supplier has this tier.")
    score = data["suppliers"].set_index("supplier_id").loc[row["supplier_id"]].financial_health_score
    if score < FIN_HIGH:
        alt = ("no alternate supplier is on file" if not row["alt_supplier_id"]
               else f"alternate {row['alt_supplier_id']} is on file")
        facts.append(f"Weak supplier finances (financial health score {score}; {alt}). Suggest confirming "
                     f"payment terms or asking for partial shipments before committing a large order.")
    return facts


def text_checks(row, text, data, tiers):
    """Checks on a brief's wording against the rule results. Returns pass/fail flags."""
    numbers = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text) if n.replace(",", "").isdigit()}
    qty_ok = row["action_type"] == "NONE" or int(row["recommended_qty"]) in numbers

    date_ok, timing_ok = True, True
    if row["action_type"] == "FOLLOW_ON_PO":
        if row["release_by"] == "NOW":
            date_ok = bool(re.search(r"\bnow\b|immediate|today|asap", text, re.I))
        else:
            d = date.fromisoformat(row["release_by"])
            patterns = [re.escape(d.isoformat()), rf"{d.strftime('%b')}[a-z]*\.? {d.day}\b", rf"\b{d.month}/{d.day}\b"]
            date_ok = any(re.search(pt, text) for pt in patterns)
            # a future release date must not be described as "due now"
            timing_ok = not re.search(r"due now|immediately|right now|asap|"
                                      r"(release|raise|place|submit)\w*( [\w-]+){0,3} now\b", text, re.I)

    if row["action_type"] == "SPIKE_PO":
        timing_ok = not re.search(r"follow-on[^.]{0,40}(window|deadline|closes|due)", text, re.I)

    late_ok = True
    if int(row["late_po_days"]) > LATE_PO_GRACE_DAYS:
        late_ok = row["late_po_id"] in text or (int(row["late_po_days"]) in numbers
                                                and bool(re.search(r"late|overdue", text, re.I)))

    t = tiers.set_index("supplier_id").loc[row["supplier_id"]]
    tier_ok = t.tier == "LOW" or any(re.search(p, text, re.I) for p in tier_patterns(t.reasons, row["item_id"]))

    sup = data["suppliers"].set_index("supplier_id").loc[row["supplier_id"]]
    finance_ok = sup.financial_health_score >= FIN_HIGH or bool(re.search(r"financ", text, re.I))

    alerts = data["risk_alerts"]
    allowed = {row["supplier_id"]}
    if row["alt_supplier_id"]:
        allowed.add(row["alt_supplier_id"])
    allowed |= set(alerts[(alerts.status == "ACTIVE") & (alerts.scope_value == sup.country)].alert_id)
    parts = data["parts"]
    allowed |= set(parts[parts.supplier_id == row["supplier_id"]].item_id)       # this item and its supplier's items
    cited = set(re.findall(r"(?:SUP|ALT|ITM)-\d{3}", text))
    bad = sorted(cited - allowed)
    wording_ok = not re.search(r"\bcritical(ly)?\b", text, re.I)  # unsupported urgency word
    counts_ok = not re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+other\s+(items|parts)\b",
                              text, re.I)                           # counts of other items: list IDs instead
    return {"qty_ok": qty_ok, "date_ok": date_ok, "timing_ok": timing_ok, "late_ok": late_ok,
            "tier_ok": tier_ok, "finance_ok": finance_ok, "wording_ok": wording_ok, "counts_ok": counts_ok,
            "ids_ok": not bad, "unsupported_ids": ", ".join(bad)}


def fix_instructions(row, checks, data, tiers):
    """Plain-English fixes for a brief that failed checks (sent back to Claude for one revision)."""
    t = tiers.set_index("supplier_id").loc[row["supplier_id"]]
    fixes = []
    if not checks["qty_ok"]:
        fixes.append(f"State the exact recommended quantity: {int(row['recommended_qty']):,} units.")
    if not checks["date_ok"]:
        fixes.append(f"State the release date exactly: {row['release_by']} (write it as a date, or say now if NOW).")
    if not checks["timing_ok"]:
        if row["action_type"] == "SPIKE_PO":
            fixes.append("This is a SPIKE_PO item: say the spike PO should be raised now, and do not mention "
                         "any follow-on PO window or deadline.")
        else:
            fixes.append(f"The release date is {row['release_by']}, so do not say 'due now' or 'immediately'. "
                         f"Say 'release by {row['release_by']}'.")
    if not checks["late_ok"]:
        fixes.append(f"Mention that {row['late_po_id']} is {row['late_po_days']} days late and not counted as incoming.")
    if not checks["tier_ok"]:
        fixes.append(f"Explain why the supplier is tier {t.tier}: {t.reasons}.")
    if not checks["finance_ok"]:
        fixes.append("Flag the supplier's weak financial health and suggest confirming payment terms or partial shipments.")
    if not checks["wording_ok"]:
        fixes.append("Remove the word 'critical' (or 'critically'). Show urgency with dates and numbers instead.")
    if not checks["counts_ok"]:
        fixes.append("Do not state a count of other items. List their item IDs from get_supplier_exposure instead.")
    if not checks["ids_ok"]:
        fixes.append(f"Remove these IDs, which are not supported by the data for this item: {checks['unsupported_ids']}.")
    return fixes


if __name__ == "__main__":
    data = read_data(".")
    assessment, tiers = assess(data)
    assessment.to_csv("risk_assessment.csv", index=False)
    tiers.to_csv("supplier_tiers.csv", index=False)
    print(f"Assessed {len(assessment)} parts.\n")
    print(assessment["risk_level"].value_counts().to_string())
    print("\nSupplier tiers:")
    print(tiers[["supplier_id", "name", "tier"]].to_string(index=False))
    print("\nSaved risk_assessment.csv and supplier_tiers.csv")
