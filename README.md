# Supply Chain Risk Detection Agent

An AI agent that reviews inventory, open purchase orders, supplier health and disruption alerts, then writes a short brief for a buyer on each part at risk of a shortage.

Built as a learning project by a supply chain and demand planning professional adding AI agent skills. It was built with AI assistance (Claude), and the business rules in it are mine. All data in this repository is **synthetic**. No employer or customer data is used, and the alert feed is simulated.

## What it does

1. **Rules engine (`risk_rules.py`)** projects each part's stock against its open PO, rates every part HIGH, MEDIUM or LOW, rates every supplier's risk tier, and computes the order to place, the quantity and the date. No AI is involved.
2. **Agent (`agent.py`)** takes each HIGH or MEDIUM part and lets Claude investigate it with six tools: the rule results, the supplier profile and tier reasons, country alerts, open POs, the alternate supplier, and the supplier's other at-risk parts. Claude then writes one brief (`briefs.csv`, `risk_briefing.md`).
3. **Evaluation (`evaluate.py`)** scores the rules and the briefs against a known answer key.

The agent never places orders or changes data. A buyer reviews every recommendation.

## The rules

**Item risk level** (from projected inventory and PO timing)

| Level | When |
|---|---|
| HIGH | Stock runs out before the open PO lands, or the open PO is too small (a follow-on PO would already have been due more than 14 days before it lands), or there is no usable PO and stock is below the reorder point |
| MEDIUM | The PO lands in time, but a follow-on PO must be released within the 14 days before arrival |
| LOW | No new PO is needed until after the current one lands |

- A PO's arrival is its due date plus the estimated delay from any active alert in the supplier's country.
- A PO more than 7 days overdue is not counted as incoming.
- Reorder point = safety stock + one lead time of demand.

**Supplier risk tier** (from the worst single trigger)

| Tier | Triggers |
|---|---|
| HIGH | Active HIGH alert in its country, financial health score under 45, or on-time delivery down 8+ points to under 85% |
| MEDIUM | Active MEDIUM alert, financial score under 60, or last-3-month on-time delivery under 90% |
| LOW | None of the above |

A PO 14 or more days late on an item that is short raises its supplier's tier one step. A HIGH-risk supplier raises a MEDIUM item to HIGH.

**What to order**

- **Spike PO** (HIGH items): the units still missing to cover the backlog, safety stock and one lead time of demand, plus 1.5 weeks of buffer, minus what is already on order.
- **Follow-on PO** (MEDIUM items): one lead time of demand, released at least 0, 7 or 14 days before the open PO lands for a LOW, MEDIUM or HIGH-risk supplier.

## Design choices

- **Rules decide what can be written down.** Risk levels, supplier tiers, quantities and dates come from code. Claude cannot change them.
- **Claude investigates and writes.** It chooses which tools to call, weighs the evidence, and writes the situation, action and supplier context.
- **Code decides what must be said.** For each part, code builds a must-mention list (a late PO, the reason for an elevated supplier tier, weak supplier finances), and Claude must include it.
- **Validate and retry.** Each brief is checked for the correct quantity and date, no contradictory timing wording, the required facts, no counts of other items, no unsupported words, and no invented or inapplicable supplier, alert or item IDs. A brief that fails is sent back once with exact fixes.
- **Human in the loop.** Output is a recommendation, never an order.

## Results (development score, synthetic data)

Test data: 12 suppliers, 53 parts, 47 open POs, 7 alerts, with 23 parts rated HIGH or MEDIUM by hand in advance. Decoys include a resolved alert, an alert at a supplier with ample stock, a healthy supplier with a just-in-time PO, and a late PO on a healthy supplier.

| Measure | Result |
|---|---|
| Rules engine risk levels match the answer key | 53/53 |
| HIGH and MEDIUM parts that received a brief | 23 / 23 |
| Briefs carrying the correct risk level | 23 / 23 |
| Briefs passing each wording check (quantity, date, timing, late PO, tier explained, finance caution, wording, counts, valid IDs) | 23 / 23 on every check |
| Briefs revised after failing a check | 12 of 23 |
| Cost of one full run (Claude Haiku 4.5) | about $0.33 |

**How to read this:** the data and answer key were written by me, and the rules were refined over several rounds, so this is a development score and not a guarantee for real data. Because levels, quantities and dates are computed by code, those measures test the pipeline more than the model. The wording checks confirm that the right facts appear and the IDs are valid. They do not confirm that every sentence is sound or well written. Twelve of the 23 first drafts failed at least one check and were revised once, so the validate-and-retry step matters; all 23 final briefs pass.

A sample of real output from one run is in [example_output/risk_briefing.md](example_output/risk_briefing.md). Claude's wording differs on every run.

## Development notes

- **The first version of the rules treated low stock with no risk signal as no risk.** A planner's review showed that stock running out before the open PO lands is a risk by itself. The rules were rebuilt around projected inventory, PO timing and alert delays.
- **The answer key is written by hand from planned scenarios.** The data generator checks that the coded rules reproduce both the hand-written levels and the hand-written supplier tiers.
- **Reading early briefs by hand found problems the automated checks missed:** missing explanations of a late PO and a supplier tier, "due now" next to a future release date, an unsupported "critical," a miscounted list of other at-risk items, and misread field names. The fixes were a code-built must-mention list, new checks, validate-and-retry, clearer field names, and a tool that lists other at-risk items by ID.

## Known limitations

- Synthetic data, one product family, and a simulated alert feed. Not tested on real data.
- One open PO per part. No minimum order quantities, lot sizes, lead-time variability, or demand forecast error.
- Tier cutoffs, the 14-day window, the 1.5-week buffer and the alert delays are my judgment and would need tuning for a real business.
- The wording checks are keyword-based. A buyer should still read each brief.
- Claude's wording varies from run to run.

## How to run

Requires Python 3.11 or newer and an [Anthropic API key](https://console.anthropic.com).

```
python -m pip install -r requirements.txt
```

Set your API key as an environment variable (never put it in a file):

```
# Windows (then open a NEW terminal)
setx ANTHROPIC_API_KEY "your-key-here"

# Mac / Linux
export ANTHROPIC_API_KEY="your-key-here"
```

Then run in order:

```
python risk_rules.py     # writes risk_assessment.csv and supplier_tiers.csv
python agent.py          # writes briefs.csv and risk_briefing.md (set MAX_ITEMS or ONLY_ITEMS near the top)
python evaluate.py       # scores the rules and the briefs against answer_key.csv
```

## Files

| File | Purpose |
|---|---|
| `risk_rules.py` | Rules engine and shared checks |
| `agent.py` | Claude agent with tools, guardrails and retry |
| `evaluate.py` | Scoring against the answer key |
| `parts.csv`, `suppliers.csv`, `open_pos.csv`, `supplier_otd.csv`, `risk_alerts.csv` | Synthetic input data |
| `answer_key.csv` | Hand-written expected risk levels, used only for scoring |

## Author

Ryan Wahl, [LinkedIn](https://www.linkedin.com/in/ryanmwahl)
