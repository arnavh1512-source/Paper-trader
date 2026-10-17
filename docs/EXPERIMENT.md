# Pre-registration: does the Claude decider add signal beyond momentum on NSE?

Written before any `claude:in` live data exists. Nothing below may be edited
once the first Claude live cycle is journalled, except to fill in the blanks in
section 0. Changes after that point go in an appended "Deviations" section with
a date and a reason, and the original text stays.

This is a measurement of a research instrument on a paper book. Nothing in it is
investment advice, a recommendation, or a forecast of returns.

## 0. Frozen facts (fill in at the first Claude live cycle, then never again)

| Item | Value | Where it comes from |
|---|---|---|
| Claude run id | ____ | `SELECT id FROM runs WHERE kind='live' AND strategy='claude'` |
| Momentum run id | ____ | same, `strategy='momentum'` |
| Day 0 (first Claude live cycle, IST date) | ____ | `MIN(ts)` of that run's `cycles` with `market_open=1` |
| Git commit | ____ | `git rev-parse HEAD` |
| `CLAUDE_MODEL`, `CLAUDE_TEMPERATURE` | ____ | workflow env (default `claude-sonnet-5`, `0.0`) |
| `NEWS_ENABLED`, `MAX_API_CALLS`, `SEGMENT` | ____ | workflow env / repo variables |
| Momentum `min_score` | 0.002 | `strategies/momentum.py` |

Any change to a row of this table mid-run ends the arm. The data up to the change
is analysed as planned. The data after it is a new experiment.

## 1. Question and hypotheses

The question is associational and comparative: is the Claude decider better than
the momentum rule at ranking the entry candidates it was shown? It is not a
causal claim about P&L.

Population: every **entry decision** in the Claude live run, meaning a decision on
a symbol that was not held at the time, with action `buy` or `hold`. Error HOLDs
are included (see 6.6).

For each such row *i* there are three quantities:

- `s_claude(i)`: a signed conviction. It is `+confidence` for a buy and
  `-confidence` for a hold. A hold at 0, which includes `source='error'`, scores 0
  and means "indifferent".
- `s_mom(i)`: the momentum rule's own score, recomputed from that row's
  `indicators_json`. The formula is `(sma_fast - sma_slow)/sma_slow + ret_5`,
  minus `(rsi-70)/100` when rsi > 70, exactly as in `MomentumStrategy._score`.
- `y(i)`: the excess forward return at h = 4 bars, `forward_return - benchmark_return`
  (NIFTYBEES), taken from `outcomes`.

Let rho_C = Spearman(`s_claude`, y) and rho_M = Spearman(`s_mom`, y), both on the
**same rows**. Define Delta = rho_C - rho_M.

- **H0:** Delta <= 0. On Claude's own candidates, the decider ranks forward excess
  returns no better than the momentum formula does.
- **H1:** Delta > 0.

Both hypotheses can be answered from the journal alone: `decisions`, `cycles`,
`outcomes`.

> ponytail: the signed score exists because `calibrate()` puts buy, sell and hold
> confidence on one axis. On that axis a confident hold, which means "I expect
> nothing", ranks like a confident buy. Signing the score is the smallest change
> that makes the ranking mean one thing.

## 2. Primary metric and horizon

**Primary metric: Delta, as defined above. Primary horizon: h = 4 bars (about one hour).**

Why Delta and not the existing `calibrate` rho:

- Pairing on identical rows removes selection. Each live arm decides only on its
  own picks, so comparing two arms' calibration means comparing different symbols
  on different ticks.
- Momentum's journalled confidence takes three values (4, 7, 8). Its score is
  continuous and is what it actually ranks by.

Why h = 4:

- It equals the tick spacing, so consecutive decisions on the same symbol use
  adjacent forward windows that do not overlap.
- The book trades `intraday` and is squared off before the close. A 26-bar window
  crosses the overnight gap, which the strategy can never hold.
- It is one of the existing `DEFAULT_HORIZONS` and it resolves soonest.

h = 26 is reported as secondary. It is the `calibrate` default and it tests
whether any edge persists past the session.

Excess return rather than raw return: an hourly NIFTY move is shared by every
symbol on a tick. Left in, it would turn a ranking test into a market-timing test
and inflate the correlation within a day.

### Secondary metrics (exploratory: no decision is taken on any of them)

| # | Metric | Source |
|---|---|---|
| S1 | Delta at h = 26 and h = 78 | same rows |
| S2 | Partial Spearman of `s_claude` and y given `s_mom` (incremental information) | same rows |
| S3 | Picker quality: per common tick, the mean y of Claude's picks minus the mean y of momentum's picks | both runs, paired by tick |
| S4 | Existing `calibrate` output per arm at 4, 26 and 78 bars, with bucket table and `_verdict` | CLI |
| S5 | Exit calibration: decisions on held symbols, sell vs hold, against the following y | Claude run |
| S6 | `report` metrics per arm: return, Sharpe, max drawdown, excess over NIFTYBEES, net of modelled costs | CLI |
| S7 | Primary analysis excluding `source='error'` rows; excluding windows that cross the close; split by first and second half of the sample | same rows |

S6 is descriptive only. Sixty days of P&L is one draw from one regime.

## 3. Unit of analysis and pairing

The unit is one entry-decision row: (Claude run, cycle, symbol).

Pairing happens **within the row**. The momentum score is rebuilt from the
indicators Claude was shown, so every row carries both arms' signals and the same
outcome. The primary analysis therefore never joins the two runs, and a tick the
momentum job missed cannot unbalance it.

Non-overlap rules:

1. **Primary.** None needed: the pairing is within the row.
2. **S3 (picker comparison).** A tick is common only when both runs have a
   `market_open=1` cycle within 15 minutes of each other. Ticks without a partner
   are dropped and counted.
3. **Duplicate cycles** (a manual re-run inside the same hour) are deduplicated
   per run by keeping the first cycle in any 15-minute window.
4. **"Not held"** is approximated as follows: the symbol is in that cycle's
   `picks_json` and has no open `position_risk` row at `ts`. `cycle.py` computes
   `is_held` but does not journal it (gap G2).

**Reconstruction check, run before the primary analysis.** On the momentum run's
own journal, rebuild `s_mom` from `indicators_json`, re-derive the flat-branch
action and confidence, and compare them with what was journalled. Any mismatch on
a flat-symbol row means the reconstruction is wrong, and the primary result is
reported as invalid until it is fixed.

## 4. Dependence: overlap, autocorrelation, clustering

Rows are not independent:

- Symbols on one tick share a market shock, and excess return removes only part
  of it.
- The same symbol recurs across ticks in a day.
- A late GitHub cron can bunch two ticks closer than an hour, which makes their
  windows overlap.
- At h = 26, windows overlap across five or six ticks.

**Method: a day-block bootstrap.** Resample IST trading days with replacement,
keeping every row of a drawn day together. Recompute rho_C, rho_M and Delta on the
same resampled rows. Use B = 10,000 replicates, a fixed seed of 20261017, and take
the 2.5th and 97.5th percentiles as the 95% CI. Report the same CI for rho_C on
its own.

Why this method:

- With h = 4 and intraday square-off, almost every dependence path lives inside
  one session. Days are close to independent blocks.
- It handles overlap, cross-sectional correlation and same-symbol repetition
  without modelling any of them.
- It needs only `random` plus the existing `analytics.calibration.spearman`.
- It respects that Delta is a difference of two correlations that share an
  outcome, which closed-form Fisher-z tests do not.

> ponytail: there is no two-way (day x symbol) clustering. Persistence of the same
> symbol across days is covered by a sensitivity run that resamples symbols
> instead of days (S7). If the two CIs disagree in sign, the result is labelled
> fragile.

## 5. Sample size, power and stopping

Assumptions, stated so they can be checked afterwards:

- About 5 ticks per day and about 3 entry rows per tick, giving m of about 15 rows
  per day.
- Within-day correlation of the rank residuals: ICC = 0.05. The design effect is
  1 + (m - 1)ICC, about 1.7.
- Correlation between `s_claude` and `s_mom` on the same rows: r between 0.3 and
  0.7.
- The standard error of a difference of dependent correlations near zero is about
  sqrt(DEFF x 2(1 - r) / n).
- Power 80% at two-sided alpha = 0.05, so the minimum detectable effect (MDE) is
  2.8 x SE.

| Trading days | Rows (n) | MDE at r = 0.7 | r = 0.5 | r = 0.3 |
|---|---|---|---|---|
| 40 | 600 | 0.12 | 0.15 | 0.18 |
| **60** | **900** | **0.09** | **0.12** | **0.14** |
| 90 | 1350 | 0.08 | 0.10 | 0.12 |

**The smallest effect worth acting on is set at Delta = 0.10.** Be clear about what
this means. A rank correlation of 0.05 is already strong in equity work. This
design can only detect a **large** edge, and a real but modest edge (Delta around
0.03 to 0.05) will very probably come out *inconclusive*. Detecting Delta = 0.05
would take about 5,000 rows, roughly 350 trading days. That is not a reasonable
paper run.

**Stopping rule, fixed now:**

1. Stop at the end of **trading day 60** after Day 0. A trading day is an IST date
   on which the Claude run has at least one `market_open=1` cycle.
2. If fewer than 600 primary rows have **resolved** at that point, extend once, to
   trading day 90. Row count is operational, not a result, so checking it is not
   peeking.
3. **Hard calendar cap: 140 calendar days after Day 0**, whatever has happened by
   then. If the cap is reached with fewer than 300 resolved rows, the verdict is
   *inconclusive (underpowered)* without looking further.
4. Analyse only after the last decision's h = 4 window has elapsed.

## 6. Threats to validity

1. **Universe survivorship.** The universe is fixed in advance and the test runs
   forward in time, so no survivors are chosen with hindsight. It is still a
   liquid large-cap list, and the result says nothing about other universes. A
   suspended or delisted symbol leaves `forward_return` unresolved, and the
   `calibrate` join drops it silently. Count unresolved rows per horizon (gap G7).
   If they exceed 2% of rows, report it.
2. **Cost model.** y is gross of costs: this measures signal, not money. Spreads
   are modelled rather than observed (`quote.modelled`), paper fills are not real
   fills, and statutory charges are modelled. Only S6 is net of costs, and only
   under that model.
3. **LLM nondeterminism and `llm_cache`.** Temperature 0 is not determinism. The
   result describes this model, prompt and temperature as a stochastic policy, and
   a replay would make different calls.
   - The cache key is sha256(model, system, prompt), and every live prompt carries
     its timestamp and live prices. A live cache hit can only come from a
     re-executed identical tick.
   - Report `llm_cache` hit counts. Dedupe per 3.3.
   - Report the number of distinct `prompt_sha` templates. A prompt edit mid-run
     is a section-0 change.
4. **News and training-data leakage.** Headlines are fetched at decision time, so
   a live run has no news lookahead. `news_json` stores what was shown.
   `NEWS_ENABLED` is frozen in section 0.
   - The larger leakage risk belongs to backtests. The model may have memorised
     price history from before its training cutoff, so **no Claude backtest is
     evidence for this question**.
   - The live window runs after the cutoff, and that is the reason this experiment
     is live.
5. **`CLAUDE_MODEL` pin.** The model string is not in `runs.config_json` (gap G8).
   Record it in section 0, and confirm `llm_cache.model` holds one value
   throughout. If the name is an alias rather than a dated snapshot, a silent
   upstream swap is possible and cannot be detected from the journal. Say so in
   the write-up.
6. **API budget cutoffs.** `MAX_API_CALLS` is checked per process, so per tick.
   - A decider failure becomes HOLD with confidence 0 and `source='error'`. These
     rows **stay in the primary analysis** with `s_claude = 0`, as a skip that
     counts.
   - Error rows are not random. The cutoff hits the end of the review list, which
     holds picks after held symbols. S7 re-runs without them.
   - A **picker** failure abstains the whole cycle (`strategy_note='model
     unavailable'`), and that cycle produces no rows at all. Count these cycles.
     If they exceed 10% of cycles, the verdict carries a "degraded" label.
7. **Market regime.** Sixty trading days is one regime. Momentum's edge depends on
   how trendy the period is, so Delta can change sign in a different market.
   Report the NIFTYBEES return and the half-split (S7) as context, never as a
   test.
8. **Code drift.** Any commit that touches `strategies/`, `llm/prompts.py`,
   `risk/` or `engine/cycle.py` during the run is a section-0 change.
   Documentation and tests are exempt.
9. **The gate does not shape the outcome.** The `MIN_CONFIDENCE` = 7 gate decides
   which buys execute. The analysis uses decisions, never executions, so the gate
   cannot bias it.

## 7. No-peeking rule

Weekly checks are **operational only**. Allowed:

- cycles per day and `market_open` cycles
- abstain rate, plus the "model unavailable" count
- error-HOLD count, API calls, cache hits
- backlog of unresolved outcomes
- whether the journal artifact is archived
- the model string and the `prompt_sha` count
- row count against the 600 threshold

Not allowed until stopping:

- `calibrate`, `report`
- the calibration and performance panels of `dashboard`
- any query that touches `outcomes.forward_return`, the equity curve, or P&L

Since the stopping rule is mechanical, a peek cannot change N. The real damage a
peek does is a mid-run change to a prompt, a threshold or the universe, and any
such change ends the arm (section 0). The dashboard has no blind mode (gap G5), so
do not open the dashboard artifact the workflow uploads.

## 8. Decision rule (written in advance)

All CIs are the day-block 95% intervals from section 4. The practical threshold is
delta* = 0.10.

| Verdict | Condition |
|---|---|
| **Claude adds signal** | lower CI bound of Delta > 0 **and** lower CI bound of rho_C > 0 |
| **No detectable edge** | upper CI bound of Delta < delta* (an edge as large as the design targets is ruled out). If the upper bound is also < 0, add "control ranks better". |
| **Inconclusive** | anything else, or a failed reconstruction check (3), or under 300 resolved rows at the cap |

The rho_C condition stops a weak control from passing for signal. Claude being
merely "less wrong" than a momentum rule that is itself inverted does not count as
signal.

Labels added to any verdict, none of which flips it:

- "degraded": picker failures above 10%, or error rows above 10% of primary rows
- "fragile": the symbol-block CI disagrees in sign with the day-block CI
- "unresolved-heavy": more than 2% of rows unresolved

None of the S-metrics can promote or demote the verdict.

### The `momentum_plus` arm

A third, keyless arm runs on the same ticks on its own paper account. It is the
momentum rule with fixed filters stacked on top (breadth, opening window, 20-bar
and volume confirmation, a cost floor, volatility-scaled ranking), all stated in
`strategies/momentum_plus.py` before any data was looked at.

- It is **outside the primary analysis**. The verdict above is Claude against
  `momentum`, and nothing about `momentum_plus` can promote or demote it.
- Every comparison involving it is exploratory, reported with S6-style metrics
  and labelled as such. Three arms give three pairwise comparisons; none of them
  is corrected for multiplicity, so none of them is a finding.
- `strategies/momentum.py` stays byte-for-byte unchanged for the length of the
  experiment. Improvements to the rule go into `momentum_plus.py` or a new
  file, never into the control (G4 depends on it).
- Its filters are frozen on the same terms as section 0. Tuning one after
  looking at the journal ends the arm, and the data after the change is a new
  experiment.

## 9. What this cannot tell you

- Whether running the Claude strategy makes or loses money, in paper or for real.
  y is gross, and S6 is one short sample under a modelled cost book.
- Whether the result holds in another regime, another universe, another market,
  or after `CLAUDE_MODEL` changes.
- Whether the picker adds value. The primary analysis conditions on Claude's own
  picks, and S3 is exploratory.
- Why Claude ranks as it does. `reason` text describes the decision; it is not
  evidence of the mechanism.
- Anything that should be read as a recommendation, a prediction of returns, or
  investment advice. It is none of these.

## 10. How to compute it

Work on a **copy** of the archived journal (the `journal-in` workflow artifact),
never on the cached live file. `calibrate` writes to `outcomes`.

```bash
J=analysis/journal-final.sqlite3     # copy of the archived artifact
py -3.11 -m claude_trader --market in --journal $J doctor

# resolves outcomes at 4, 26, 78 bars for each run (writes to the copy), then S4
py -3.11 -m claude_trader --market in --journal $J calibrate --run <claude_id>   --horizon 4
py -3.11 -m claude_trader --market in --journal $J calibrate --run <momentum_id> --horizon 4
py -3.11 -m claude_trader --market in --journal $J calibrate --run <claude_id>   --horizon 26

# S6, descriptive only
py -3.11 -m claude_trader --market in --journal $J report --run <claude_id>   --horizon 4
py -3.11 -m claude_trader --market in --journal $J report --run <momentum_id> --horizon 4
```

After `calibrate` has filled `outcomes`, the primary analysis is read-only
(`sqlite3.connect("file:...?mode=ro", uri=True)`). It joins `decisions`, `cycles`
and `outcomes` (horizon_bars = 4) for the Claude run, applies section 3, builds
`s_claude`, `s_mom` and y, and runs the section 4 bootstrap using
`analytics.calibration.spearman`. No CLI command does this yet (G1).

### Gap list (no code was written; each item is a future change)

| # | Gap | Effect here |
|---|---|---|
| G1 | No script computes Delta, S2, S3 or the bootstrap. It needs one read-only stdlib script that reuses `spearman`. | The primary metric cannot be computed from the CLI |
| G2 | `decisions` has no `held_at_decision` column, although `cycle.py` computes `is_held`. Add it as an additive column per invariant 15. | "Not held" has to be approximated (3.4) |
| G3 | The `calibrate` CLI does not expose `actions=`, uses raw rather than excess return, and pools buy, sell and hold confidence on one axis. | S4 is not comparable with the primary metric |
| G4 | Momentum's continuous score is not journalled, only its three-level confidence. | Reconstruction depends on `momentum.py` staying unchanged |
| G5 | The dashboard has no blind or operational-only mode, and the workflow publishes it every run. | No-peeking depends on discipline |
| G6 | A picker failure is distinguishable from a deliberate abstention only through the `strategy_note` text. | The degraded-rate count relies on a string match |
| G7 | Unresolved outcomes are dropped by the join with no count reported. | Missingness has to be counted by hand |
| G8 | `runs.config_json` omits `CLAUDE_MODEL`, temperature, `NEWS_ENABLED`, `MAX_API_CALLS` and the commit. | Section 0 has to be filled in by hand |
