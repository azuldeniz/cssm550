# R3 case cards

Three cases. Teams of 2 to 3 draw one. **Order of work:** (1) look at real inputs, outputs and tokens, (2) predict the bill for the full job, (3) run a sample and compare, (4) pull levers.

Notebook: `R3_budget_cases.ipynb`. Set `CASE = "<name>"` in the first cell.

---

## Card 1. Municipal  (`CASE = "municipal"`, task B1)

**From:** the Deputy Mayor's office (fictional).
> "Our complaint line gets 30,000 citizen messages a year, in Turkish, full of typos and slang. We have exactly **$3** for the whole classification job. Sort them into six categories so the right department sees them."

| | |
|---|---|
| Items | 30,000 |
| Budget | **$3.00** (about $0.0001 per item) |
| Quality target | Highest macro-F1 you can afford |
| Watch out for | Sarcasm, two-topic complaints, very short messages |
| Surprise to look for | The default model may already be **over budget**, and a shorter prompt can make a reasoning model cost *more* |

---

## Card 2. Survey  (`CASE = "survey"`, task B4)

**From:** a social psychologist running a trust survey (fictional).
> "20,000 open-ended answers to 'Why do you trust or distrust the local government?' Most are easy ('I don't know'); some are genuinely hard. I have **$2**. Use **one model** and make it as accurate as you can."

| | |
|---|---|
| Items | 20,000 |
| Budget | **$2.00** (about $0.0001 per item) |
| Quality target | Highest macro-F1 with a single model |
| Watch out for | Hedged, double-barrelled answers; the strong model is over budget |

---

## Card 3. Cascade  (`CASE = "cascade"`, task B4)

**From:** the same social psychologist, same survey, same **$2**.
> "Many answers are easy. Use a cheap model for everything and the strong model **only when the cheap one is unsure**."

| | |
|---|---|
| Items | 20,000 |
| Budget | **$2.00** |
| Method | `arena.cascade(...)`, vary the confidence threshold (50, 70, 90) |
| What a good answer shows | A cost-quality curve: how much quality each extra percent of escalation buys, compared with the single-model result from the Survey card |

---

### Estimation worksheet (all cases)

1. Maximum affordable cost per item = budget ÷ items: ______
2. From Step 1 (3 real items): tokens in ______, tokens out ______, cost per item ______
3. Model you plan to use and price per million tokens (in / out): ______
4. Predicted full-job cost: ______  → within budget? yes / no
5. After the sample run: actual projected cost ______  Error: ______ %
6. Lever that changed the most: ______
