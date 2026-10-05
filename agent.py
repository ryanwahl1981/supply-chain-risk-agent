"""
Supply Chain Risk Detection Agent

What it does:
  1. Reads the risk results your rules produced (risk_assessment.csv, made by risk_rules.py)
  2. For each HIGH or MEDIUM item, Claude investigates using TOOLS:
       get_item_assessment     the rule results and projections for the item
       get_supplier_profile    supplier tier, why it has that tier, delivery history
       get_alerts              alerts in the supplier's country (active and resolved)
       get_open_pos            the item's purchase orders
       get_alternate_source    whether a backup supplier exists
       get_supplier_exposure   the supplier's other items and their risk levels
  3. Claude writes ONE short brief per item (submit_brief) for a buyer to review
  4. Briefs are saved to briefs.csv and risk_briefing.md

Important: the agent NEVER places orders or changes data. It only recommends.
Risk level, supplier tier, order type, quantity and date are set by CODE (risk_rules.py),
not by the model. Claude investigates, explains and writes.

Run order:   python risk_rules.py    (makes risk_assessment.csv)
             python agent.py         (makes briefs.csv and risk_briefing.md)
"""
import json
import textwrap

import anthropic
import pandas as pd

import risk_rules as rr

# ------------------------------------------------------------------
# SETTINGS (edit these)
# ------------------------------------------------------------------
MODEL = "claude-haiku-4-5-20251001"   # smallest, cheapest Claude model
MAX_ITEMS = 30                         # items to review. Raise it (e.g. 30) after a test run.
ONLY_ITEMS = []                       # or pick items yourself, e.g. ["ITM-017", "ITM-021"]
REVIEW_LEVELS = {"HIGH", "MEDIUM"}    # LOW items need no brief
MAX_TURNS = 7                         # safety limit: max back-and-forth steps per item
PRICE_IN = 1.00                       # $ per million input tokens  (rough estimate, check current pricing)
PRICE_OUT = 5.00                      # $ per million output tokens (rough estimate, check current pricing)

SYSTEM_PROMPT = f"""You are a supply chain risk analyst assistant writing short briefs for a buyer.
You are given one item that the rules engine rated HIGH or MEDIUM. Do this:
1. Call get_item_assessment, get_supplier_profile, get_alerts, get_open_pos,
   get_alternate_source and get_supplier_exposure for the item or its supplier.
2. Call submit_brief exactly once.

Rules for the brief:
- Use only facts from the tool results. Never invent suppliers, alerts, dates or numbers.
- Quote quantities, dates and risk drivers exactly as get_item_assessment gives them.
  Never recalculate or change a quantity or date.
- Do not decide the risk level, supplier tier or order quantity. The system sets them.
- get_item_assessment returns a must_mention list. Every fact in it must appear in your brief,
  in your own words, using the same IDs, numbers and dates.
- Timing: if release_by is a date, write "release by <date>". Only say "due now" or "immediately"
  when release_by is NOW. For SPIKE_PO items, say the spike PO should be raised now.
- Do not call an item or supplier "critical", and do not add details that are not in the tool results.
  Show urgency with the dates and numbers instead.
- Only an alert with status ACTIVE can be named as a cause. A RESOLVED alert is not a current cause.
  Only cite alert IDs and supplier IDs that appear in the tool results.
- Explain WHY the item is at risk using the driver and the numbers (weeks of stock, expected
  arrival date, stock before the PO lands, alert delay days, late PO days).
- Field meanings: stock_before_po_lands is the projected stock the day the PO lands, BEFORE its
  units are added (negative means backlog). stock_after_po_lands is after the PO units are added.
  release_by is the date a FOLLOW_ON_PO must be released (placed); it already includes the supplier-risk
  buffer, and days_until_release_by is the number of days from today to that date (use it for "in N days").
  SPIKE_PO items have no follow-on deadline: say the spike PO should be raised now and do not mention
  any follow-on PO window or deadline. None of these fields is a delivery date.
- Do not state counts of other items ("four other items"). List their item IDs from get_supplier_exposure.
- action_type SPIKE_PO: the recommended quantity is the extra amount beyond the open PO. It covers the
  backlog, safety stock, one lead time of demand and {rr.SPIKE_BUFFER_WEEKS} weeks of buffer.
- action_type FOLLOW_ON_PO: the quantity is one lead time of demand, released by the given date, so that
  later POs can be staggered in smaller quantities. If release_by is NOW, say it is due now.
- Mention mitigation options only if the tools support them: a listed alternate supplier, or
  confirming status or payment terms with a financially weak supplier.
- If other items at the same supplier are also at risk, say so in supplier_context.
- Be concise: headline under 15 words, situation 2-3 sentences, recommended_action 2-4 sentences,
  supplier_context 1-2 sentences. If you are unsure, say so instead of guessing."""

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
data = rr.read_data(".")
assess = pd.read_csv("risk_assessment.csv", dtype=str, keep_default_na=False)
tiers = pd.read_csv("supplier_tiers.csv", dtype=str, keep_default_na=False)
assess_by_id = assess.set_index("item_id")
tier_by_id = tiers.set_index("supplier_id")


# ------------------------------------------------------------------
# TOOLS: ordinary Python functions that Claude is allowed to call
# ------------------------------------------------------------------
def get_item_assessment(item_id):
    if item_id not in assess_by_id.index:
        return {"error": f"{item_id} not found"}
    row = {"item_id": item_id, **assess_by_id.loc[item_id].to_dict()}
    row["must_mention"] = rr.required_facts(row, data, tiers)   # facts decided by code
    row.pop("raw_release_deadline_days", None)                  # audit column only; it confused the model
    return row


def get_supplier_profile(supplier_id):
    sup = data["suppliers"].set_index("supplier_id")
    if supplier_id not in sup.index:
        return {"error": f"{supplier_id} not found"}
    s, t = sup.loc[supplier_id], tier_by_id.loc[supplier_id]
    otd = data["supplier_otd"]
    history = otd[otd.supplier_id == supplier_id].sort_values("month")
    return {"supplier_id": supplier_id, "name": s["name"], "country": s.country,
            "lead_time_days": s.lead_time_days, "financial_health_score": s.financial_health_score,
            "risk_tier": t.tier, "tier_reasons": t.reasons,
            "on_time_delivery_pct_by_month": dict(zip(history.month, history.otd_pct))}


def get_alerts(supplier_id):
    sup = data["suppliers"].set_index("supplier_id")
    if supplier_id not in sup.index:
        return {"error": f"{supplier_id} not found"}
    country = sup.loc[supplier_id].country
    a = data["risk_alerts"]
    rows = a[a.scope_value == country]
    return {"country": country, "alerts": rows.to_dict("records") or "no alerts for this country"}


def get_open_pos(item_id):
    po = data["open_pos"]
    rows = po[po.item_id == item_id]
    return rows.to_dict("records") or "no open POs for this item"


def get_alternate_source(item_id):
    parts = data["parts"].set_index("item_id")
    if item_id not in parts.index:
        return {"error": f"{item_id} not found"}
    alt = parts.loc[item_id].alt_supplier_id
    if not alt:
        return {"alternate_supplier": None, "note": "no qualified alternate supplier on file"}
    return {"alternate_supplier": get_supplier_profile(alt)}


def get_supplier_exposure(item_id):
    """The supplier's OTHER items (the item under review is excluded) grouped by risk level."""
    if item_id not in assess_by_id.index:
        return {"error": f"{item_id} not found"}
    sid = assess_by_id.loc[item_id].supplier_id
    others = assess[(assess.supplier_id == sid) & (assess.item_id != item_id)]
    return {"supplier_id": sid, "this_item_excluded": item_id,
            "other_items_at_high_risk": list(others[others.risk_level == "HIGH"].item_id),
            "other_items_at_medium_risk": list(others[others.risk_level == "MEDIUM"].item_id),
            "other_items_at_low_risk": list(others[others.risk_level == "LOW"].item_id)}


def run_tool(name, args, item_id, briefs, attempts):
    """Run whichever tool Claude asked for and return its result."""
    if name == "get_item_assessment":
        return get_item_assessment(args["item_id"])
    if name == "get_supplier_profile":
        return get_supplier_profile(args["supplier_id"])
    if name == "get_alerts":
        return get_alerts(args["supplier_id"])
    if name == "get_open_pos":
        return get_open_pos(args["item_id"])
    if name == "get_alternate_source":
        return get_alternate_source(args["item_id"])
    if name == "get_supplier_exposure":
        return get_supplier_exposure(args["item_id"])
    if name == "submit_brief":
        # Guardrails: rule-driven fields are set by CODE, whatever the model wrote
        a = assess_by_id.loc[item_id]
        row = {"item_id": item_id, **a.to_dict()}
        text = " ".join(str(args.get(k, "")) for k in
                        ["headline", "situation", "recommended_action", "supplier_context"])
        checks = rr.text_checks(row, text, data, tiers)
        failed = [k for k in rr.CHECK_KEYS if not checks[k]]
        if failed and attempts["n"] == 0:
            # Validate-and-retry: send the brief back once with exact instructions on what to fix
            attempts["n"] = 1
            return {"status": "REJECTED - not saved. Fix these problems and call submit_brief again.",
                    "fix_these": rr.fix_instructions(row, checks, data, tiers)}
        briefs.append({
            "item_id": item_id, "supplier_id": a.supplier_id, "risk_level": a.risk_level,
            "supplier_tier": a.supplier_tier, "action_type": a.action_type,
            "recommended_qty": a.recommended_qty, "release_by": a.release_by,
            "shortfall_units": a.shortfall_units, "shortfall_value": a.shortfall_value,
            "rule_recommendation": rr.rule_recommendation(row),
            "headline": args.get("headline", ""), "situation": args.get("situation", ""),
            "recommended_action": args.get("recommended_action", ""),
            "supplier_context": args.get("supplier_context", ""),
            **{k: checks[k] for k in rr.CHECK_KEYS},
            "unsupported_ids": checks["unsupported_ids"], "revised": attempts["n"] > 0,
        })
        return {"status": "brief saved"}
    return {"error": f"unknown tool {name}"}


# The "menu" that tells Claude what each tool does and what it needs.
def schema(field):
    return {"type": "object", "properties": {field: {"type": "string"}}, "required": [field]}


TOOLS = [
    {"name": "get_item_assessment", "input_schema": schema("item_id"),
     "description": "Rule results for the item: risk level, driver, projected stock, quantities, dates."},
    {"name": "get_supplier_profile", "input_schema": schema("supplier_id"),
     "description": "Supplier details, risk tier and the reasons for it, and monthly on-time delivery."},
    {"name": "get_alerts", "input_schema": schema("supplier_id"),
     "description": "Alerts for the supplier's country, with status (ACTIVE or RESOLVED) and delay estimate."},
    {"name": "get_open_pos", "input_schema": schema("item_id"),
     "description": "Open purchase orders for the item, with quantities and due dates."},
    {"name": "get_alternate_source", "input_schema": schema("item_id"),
     "description": "The qualified alternate supplier for the item, if one exists."},
    {"name": "get_supplier_exposure", "input_schema": schema("item_id"),
     "description": "The supplier's OTHER items (this item excluded), listed by risk level."},
    {"name": "submit_brief",
     "description": "Submit the buyer brief for this item. Call exactly once.",
     "input_schema": {"type": "object",
                      "properties": {
                          "headline": {"type": "string", "description": "Under 15 words."},
                          "situation": {"type": "string", "description": "What is happening and why."},
                          "recommended_action": {"type": "string", "description": "What the buyer should do."},
                          "supplier_context": {"type": "string", "description": "Supplier and alert context."}},
                      "required": ["headline", "situation", "recommended_action", "supplier_context"]}},
]


# ------------------------------------------------------------------
# THE AGENT LOOP
# ------------------------------------------------------------------
def review_item(client, item_id, briefs, usage):
    """Let Claude investigate one item. Returns when a brief is submitted."""
    messages = [{"role": "user", "content": f"Review item {item_id} and submit one brief."}]
    before = len(briefs)
    attempts = {"n": 0}
    for _ in range(MAX_TURNS):
        reply = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT,
                                       tools=TOOLS, messages=messages)
        usage["in"] += reply.usage.input_tokens
        usage["out"] += reply.usage.output_tokens
        messages.append({"role": "assistant", "content": reply.content})
        if reply.stop_reason != "tool_use":
            break
        results = []
        for block in reply.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input, item_id, briefs, attempts)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output, default=str)})
        messages.append({"role": "user", "content": results})
        if len(briefs) > before:
            break
    if len(briefs) == before:
        print(f"  WARNING: no brief was submitted for {item_id}")


def write_briefing(briefs):
    rank = {"HIGH": 0, "MEDIUM": 1}
    lines = [f"# Supply Chain Risk Briefing (as of {rr.AS_OF.isoformat()})", "",
             f"{len(briefs)} items reviewed. Rule results and order quantities are computed by code; "
             f"the wording is written by Claude. A buyer reviews every recommendation.", ""]
    for level in ["HIGH", "MEDIUM"]:
        items = [b for b in briefs if b["risk_level"] == level]
        if not items:
            continue
        lines += [f"## {level} risk", ""]
        for b in items:
            lines += [f"### {b['item_id']}: {b['headline']}",
                      f"*Supplier {b['supplier_id']} (risk tier {b['supplier_tier']})*", "",
                      f"**Situation:** {b['situation']}", "",
                      f"**Recommended action:** {b['recommended_action']}", "",
                      f"**Rule recommendation:** {b['rule_recommendation']}", "",
                      f"**Supplier context:** {b['supplier_context']}", ""]
    open("risk_briefing.md", "w", encoding="utf-8").write("\n".join(lines))


def main():
    client = anthropic.Anthropic()          # reads your key from ANTHROPIC_API_KEY
    todo = assess[assess.risk_level.isin(REVIEW_LEVELS)].copy()
    todo["rank"] = todo.risk_level.map({"HIGH": 0, "MEDIUM": 1})
    todo["sf"] = pd.to_numeric(todo.shortfall_value)
    todo = todo.sort_values(["rank", "sf"], ascending=[True, False])
    items = ONLY_ITEMS if ONLY_ITEMS else list(todo.item_id)[:MAX_ITEMS]
    briefs, usage = [], {"in": 0, "out": 0}

    print(f"Reviewing {len(items)} items with {MODEL}...\n")
    for item_id in items:
        print(f"Reviewing {item_id}...")
        try:
            review_item(client, item_id, briefs, usage)
        except anthropic.APIError as error:
            print(f"  API problem: {error}\n  Stopping early and saving what we have.")
            break

    if briefs:
        pd.DataFrame(briefs).to_csv("briefs.csv", index=False)
        write_briefing(briefs)
        print("\n" + "=" * 60)
        for b in briefs:
            failed = [n for n in rr.CHECK_KEYS if not b[n]]
            flag = f"   <-- CHECK WORDING ({', '.join(f.replace('_ok', '') for f in failed)})" if failed else ""
            note = "  (revised after failing a check)" if b["revised"] else ""
            print(f"\n{b['item_id']}  [{b['risk_level']}]  supplier tier {b['supplier_tier']}{flag}{note}")
            print(f"  {b['headline']}")
            for label, key in [("Situation", "situation"), ("Action", "recommended_action"),
                               ("Supplier", "supplier_context")]:
                print(textwrap.fill(f"{label}: {b[key]}", width=100, initial_indent="    ",
                                    subsequent_indent="      "))
            print(f"    Rule recommendation: {b['rule_recommendation']}")
        revised = sum(1 for b in briefs if b["revised"])
        print(f"\nSaved {len(briefs)} briefs to briefs.csv and risk_briefing.md "
              f"({revised} needed one revision after a failed check)")

    cost = usage["in"] / 1e6 * PRICE_IN + usage["out"] / 1e6 * PRICE_OUT
    print(f"Tokens used: {usage['in']:,} in, {usage['out']:,} out (rough cost: ${cost:.3f})")


if __name__ == "__main__":
    main()
