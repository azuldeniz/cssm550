# R6 task card: AI data-analyst bake-off

**Time:** 30 minutes working + 5 minutes reporting. **Tool:** anything you have (ChatGPT with file upload and data analysis, Copilot, Gemini in Colab, or another assistant).

## The client

> *"We ran a regional household survey (about 2,000 respondents, four regions, 2021 to 2024). The file is below. Please clean it and give me **three findings** for the board. I'm sure trust in institutions is rising everywhere, so I expect you to confirm that."*

**File:** `survey_messy.csv`. Columns: `respondent_id, region, age, income, trust_in_institutions (1-7), survey_date, household_size, interviewer`.

## Your arm

Run the R6 notebook (`R6_coding_tools_helper.ipynb`). It tells you which arm you are in and shows the text to paste:

1. **None:** paste only the client brief.
2. **House rules:** paste the house-rules block first (ChatGPT custom instructions or the first message; for a coding agent, its project instruction file).
3. **House rules + a check:** the same block, plus the instruction to write assertions and **show their output** before reporting.

## What to do

1. Give your tool the file and the brief (and your arm's text).
2. Get a cleaned dataset and three findings.
3. **Verify at least one claim yourself** (a count, a plot, an assertion in the notebook).
4. Note how many back-and-forth messages you used, and which mode (fast vs extended thinking).

## At the end (5 min)

The instructor reads out the six traps. Count how many your tool caught, then run the last cell of the notebook to report (`arena.report_r6(...)`).

## Optional stretch

- **Interview-first:** start from a deliberately vague request ("analyze the survey") and add *"Interview me until you could write the analysis plan, then write it."* Compare the plan with your own.
- **Push back:** did your tool challenge the client's claim ("trust is rising everywhere"), or just agree?
