# House rules block (arms 2 and 3)

Paste into ChatGPT custom instructions or the first message. For a coding agent, put it in the project's instruction file. It is the same text the R6 notebook prints.

```
HOUSE RULES for this analysis
- Before analysing, look for duplicate respondents, impossible values, and missing-value codes.
- State every assumption you make (dates, currency, how you treat missing values).
- Never silently drop rows; report how many you changed or removed and why.
- Report results by region and by year, not only pooled.
```

## Arm 3 adds

```
BEFORE you report results: write assertions (row counts, value ranges, no duplicate IDs, date range)
and SHOW their output. Do not say "looks good" without evidence.
```

## Why these rules (instructor notes)

- They deliberately **do not name the six traps**. A good instruction file holds what the tool cannot discover by itself: *process* and *standards*, not the answers.
- Arm 3 tests the guidance that a check the tool can run and show beats an instruction it may skip, and that asking for **evidence** changes what gets caught.
- Optional 3-minute instructor demo: a hook (deterministic) that blocks overwriting the raw data file, next to a "never overwrite raw data" line (non-deterministic). Show the difference.
