"""
Evaluation: how well did the rules engine and the agent do compared with the answer key?

Part A (the rules engine, no AI):
  1. Do the risk levels match the answer key?

Part B (the agent's briefs, from briefs.csv):
  2. Coverage      - how many of the HIGH/MEDIUM items got a brief?
  3. Levels        - do the briefs carry the right risk level?
  4. Quantity      - does the wording mention the rule-computed order quantity?
  5. Date          - does the wording mention the rule-computed release date?
  6. Grounding     - does it cite only supplier and alert IDs that are valid for that item?
                     (for example, never a RESOLVED alert as a cause)
"""
import os

import pandas as pd

key = pd.read_csv("answer_key.csv", dtype=str, keep_default_na=False).set_index("item_id")
assess = pd.read_csv("risk_assessment.csv", dtype=str, keep_default_na=False).set_index("item_id")

# ---------------- Part A ----------------
wrong = [i for i in key.index if assess.loc[i, "risk_level"] != key.loc[i, "expected_risk_level"]]
print("PART A: rules engine vs answer key")
print(f"1. Risk levels:   {len(key) - len(wrong)}/{len(key)} match")
for i in wrong:
    print(f"   {i}: rules said {assess.loc[i, 'risk_level']}, key says {key.loc[i, 'expected_risk_level']}")

# ---------------- Part B ----------------
if not os.path.exists("briefs.csv"):
    print("\nPART B: no briefs.csv yet. Run agent.py first.")
    raise SystemExit

briefs = pd.read_csv("briefs.csv", dtype=str, keep_default_na=False).drop_duplicates("item_id")
needed = key[key.expected_risk_level.isin(["HIGH", "MEDIUM"])]
covered = [i for i in needed.index if i in set(briefs.item_id)]
b = briefs.set_index("item_id")

print(f"\nPART B: agent briefs ({len(briefs)} written)")
print(f"2. Coverage:      {len(covered)}/{len(needed)} HIGH/MEDIUM items have a brief")
if len(covered) < len(needed):
    print("   (the agent may have been limited by MAX_ITEMS in agent.py)")

lvl_wrong = [i for i in b.index if b.loc[i, "risk_level"] != key.loc[i, "expected_risk_level"]]
print(f"3. Levels:        {len(b) - len(lvl_wrong)}/{len(b)} briefs carry the right level")
for i in lvl_wrong:
    print(f"   {i}: brief says {b.loc[i, 'risk_level']}, key says {key.loc[i, 'expected_risk_level']}")

CHECKS = [("qty_ok", "Quantity"), ("date_ok", "Date"), ("timing_ok", "Timing"), ("late_ok", "Late PO"),
          ("tier_ok", "Tier"), ("finance_ok", "Finances"), ("wording_ok", "Wording"), ("counts_ok", "Counts"), ("ids_ok", "Grounding")]
for n, (col, label) in enumerate([c for c in CHECKS if c[0] in b.columns], start=4):
    bad = [i for i in b.index if b.loc[i, col] != "True"]
    print(f"{n}. {label + ':':<14}{len(b) - len(bad)}/{len(b)} briefs pass")
    for i in bad:
        extra = f" (unsupported: {b.loc[i, 'unsupported_ids']})" if col == "ids_ok" else ""
        print(f"   {i}{extra}")

if "revised" in b.columns:
    n_rev = int((b["revised"] == "True").sum())
    print(f"\nInfo: {n_rev}/{len(b)} briefs needed one revision after failing a check (the scores above are final results)")
