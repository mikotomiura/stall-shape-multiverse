# English summary of the conditions

The conditions of this study were written in Japanese and committed to the author's version control before each
computation ran. They are records in the author's repository, not third-party registrations. This file summarises each
of them in English, in the order of the computations given in Supplement S1 of the manuscript. Where this summary and
a Japanese file differ, the Japanese file is the record. Each file is identified by its SHA-256.

Status labels follow S1: [P] primary, quantity and criterion committed before any result; [A] added after the primary
results, quantity and criterion committed before that quantity was seen; [E] exploratory; [D] described after seeing
the results.

Common to all conditions: one seed (20261107, file `SEED`); the data and intermediate files stay outside the
repository; only aggregate values are written to `results/` (no per-writer values, times or identifiers; CS1 by
agreement with its authors); no data, intermediate file or result is sent to an external language model; every
interval is a 95% percentile interval from 2,000 resamples of writers.

| Order | File | SHA-256 |
|---|---|---|
| 1 | `conditions.md` | `daa11c3a21ab79dfad259cbb3ac43facaa6ab47e971b33a0f8a2ab3b76e41a3b` |
| 2 | `conditions-length90.md` | `c1fd53daa345d074665f813652557415209e8f1ceb30549bdcc790baf11bfd25` |
| 3 | `conditions-round1.md` | `27b067c6b45f09b611ca6a196c8085efd69cfd2024142a2c69ecee5354153116` |
| 4 | `conditions-round2.md` | `9de34ac966e69f0bee98c357a80b4a1179f435ed70e652244d7537df7212ab2f` |
| 5 | `conditions-round3.md` | `16557faaa685b61891e15e7469c15988d31029c8ab4aaafd931dbb5e773a59e9` |
| 6 | `conditions-round4.md` | `7fa4af13a1b2128a7d7a7201621f2cae9c5f172e4736fcf32aa6f2407d1d8d7d` |
| 7 | `conditions-round5.md` | `40194a9a6cebd4e3627cac11c896bab2e662e29f84f66709ffbcef91e391fc5e` |
| 8 | `conditions-round7.md` | `42374d34765d39316cc57d5b4d98b89173a9e2c35b7e7b388674e528de5c6c8c` |
| 9 | `conditions-round8.md` | `350e9d68b53d6105454a70d918d451df5707c1b77e04a4984a49acb1112fa975` |
| 10 | (no condition; recount of existing intervals, `scripts/round9_classes.py`) | |
| 11 | `conditions-round9.md` | `29f91f09e497c18a037de4bf78829bc3cefd697d4be94ca34a0aef84c02032bd` |

The file names do not follow the order numbers: `round5` is the sixth added condition, and there is no `round6`.

## 1. Primary conditions [P] (`conditions.md`)

- Seen before: only that the Python copy of the detector reproduced the onsets (s, t0) of the reference
  implementation in 60 units of each dataset (the number of agreements, not rates or lengths), and values published
  in earlier work on these datasets.
- Data and units: KLiCKe writers with at least 10 minutes of observation (one essay each); CoAuthor seen writers
  (sessions within writers; the 18 held-out writers not opened); CS1 2019 seen writers and the 146 writers opened in
  the companion study; CS1 2021 seen writers (13 held-out writers not opened). CS1 units are assignments, divided into
  work blocks (30-minute gaps) and days (gaps of 6 hours or more within an assignment). Checks that the assignment
  identifiers and the holdout lists match, stopping on a mismatch.
- Exposure is the length of the observation intervals; a stall is assigned to a unit by its onset t0; units with less
  than 5 minutes of exposure are dropped from count analyses (their number reported). Stall length is from t0 to the
  next character press, on a log scale; unresumed stalls are excluded and counted.
- Baseline definition (B0): window W = 30 s; condition A = character presses minus deletion presses in the window < 1;
  θ = the nearest-rank 0.99 quantile of the unit's inter-keystroke intervals so far (from 100 intervals); the default
  observation intervals.
- Decomposition of variation: a Poisson log-normal mixed model of counts (writer, assignment, day, block, and a crossed
  assignment effect; variational Bayes) and a linear mixed model of log length (REML). A recovery check on simulated
  data with known components (within ±30% of the SD) decides which levels may be used in the decision rule.
- Overdispersion: the Pearson dispersion index φ = Σ(n − ê)²/ê/(k − p), with ê from (a) one overall rate or (b) each
  writer's own rate (levels below the writer only).
- Multiverse (90 definitions): W 15, 30, 60 s; θ a writer quantile 0.95, 0.99, 0.995 or a fixed 10 or 30 s; condition
  A net or characters only; the default observation intervals or intervals split at key-press gaps of 5 or 30 minutes.
  For each: rates, the rank order of the four datasets, φ, the share of writers with no stall, agreement in time with
  B0, median length. The decomposition of variation is run for the 9 combinations of W and quantile only. An
  estimator-level comparison (time shares of a stuck state in rows close to the last key press) is also fixed.
- Checks before the computation: onsets agree with the reference implementation in every unit; positive controls on
  synthetic streams; mutants of the condition, the quantile and the segmentation must fail the checks.
- Criteria ("moves"): M1 the rank order differs from B0 for at least 25% of the definitions; M2 the conclusion that φ
  is at least 1.5 reverses for at least 25% of the definitions in a primary cell; M3 the estimator-level ratio halves
  when rows are aligned; M4 low agreement in time together with similar rates. Within-writer criterion O: φ (b) of at
  least 1.5 with an interval above 1 in at least two of the five cells. A rule combining M, O and a level criterion V
  decided whether the study stands alone; it was a decision aid for the author, not a claim.
- Stated limits: a stall onset is not a judgement that the writer was stuck; no definition can be called correct;
  differences between datasets are not attributed to a cause.

## 2. Length decomposition for all 90 definitions [A] (`conditions-length90.md`)

- Seen before: the length decomposition for the 9 definitions of order 1.
- The same length model for all 90 definitions (CS1 2019, CS1 2021, CoAuthor), from the stored stalls; the 9 earlier
  values must be reproduced to three decimals.
- Criterion ("stays"): in each dataset the stall-to-stall share is larger than every other share in all 90
  definitions; a definition whose fit did not converge counts as not meeting it. The writer share is reported, not
  judged.

## 3. Intervals and the score association [A] (`conditions-round1.md`)

- Seen before: the point values of the 90 definitions and the number of reversed ratio point values per pair. Of the
  KLiCKe score file only its columns, the number of identifiers, the four prompts and the score frequencies.
- (a) Intervals for the four rates, the six ratios and φ in the five primary cells. Counted over the 89 non-baseline
  definitions: A1 a ratio whose interval lies on the other side of 1 from the baseline (pairs whose baseline interval
  contains 1 are not counted); A2 φ with an interval above 1; A3 φ whose interval lies on the other side of 1.5.
  Criterion: a pair reverses with the definition if A1 reaches 25%; the two CS1 editions are reported first. This
  criterion was set after the reversed point values had been seen and before the intervals.
- (b) The Spearman correlation between a KLiCKe writer's stall rate and the holistic score, with intervals, and the
  median within-prompt correlation. Criteria: the direction changes if at least one definition has an interval on the
  other side of zero from the baseline; presence moves if 25–75% of the intervals exclude zero; otherwise the
  association is robust.
- Checks: the baseline φ interval agrees with the earlier one within 0.05; positive controls for the correlation.

## 4. Partial association given the key-press rate [E] (`conditions-round2.md`)

- The partial rank correlation of score and stall rate given the writer's key presses per observed hour, and the
  correlation of stall rate with key-press rate, with readings set in advance. The covariate was later withdrawn
  because its denominator contains the stall time.

## 5. Partial association given typing speed [E] (`conditions-round3.md`)

- Typing speed: the median of a writer's inter-keystroke intervals under 2 s. The partial rank correlation of score and
  stall rate given speed, with intervals, and readings set in advance. Exploratory, because the covariate was chosen
  after seeing results.

## 6. Six ways of measuring speed [E] (`conditions-round4.md`)

- Speed as the median of intervals under 1, 2 or 3 s, the mean under 2 s, the median of character-to-character
  intervals under 2 s, and the median of intervals of 10–750 ms; a within-prompt version for one of them. Reading: if
  all six give negative intervals for all 89 definitions, the speed-adjusted association is negative throughout;
  otherwise the counts per measure are reported.

## 7. Pause family, re-estimated θ and length adjustment [A] (`conditions-round5.md`)

- Seen before: all earlier results. The form of the study (a diagnostic of the spread of definitions with intervals)
  is not changed by any result.
- (a) Pause family: a pause is an interval between consecutive key presses in an observation interval longer than a
  threshold. 36 definitions: any key or characters only; fixed 2, 5, 10 s or a writer quantile 0.95, 0.99, 0.995; three
  segmentations. The same quantities as for stalls. Reading: the direction changes if at least one interval is above
  zero and one below; the 25% criterion for ratios as for stalls; the pause family is the same diagnostic on a second
  detector, not a check of the stall results.
- (b) Two-stage intervals that redraw the writer-quantile θ within each unit (200 inner draws), for the 54 quantile
  definitions. Reading set in advance: these intervals replace the earlier ones regardless of which is narrower.
- (c) Partial correlations of score and rate given essay length (characters), observation length, or both, for all
  stall and pause definitions. Reading: a uniform sign if only one side has intervals excluding zero; if the score
  correlates with essay length at 0.5 or more, the limitations state that the score reflects length strongly.
- Checks: positive controls for the pause counting; exposures equal those of the stall detector; without redrawing,
  counts equal the stored counts exactly.

## 8. Restriction to the time after the 100th interval [A] (`conditions-round7.md`)

- A writer-quantile θ is undefined before a unit's 100th interval, so no stall can start there, while a fixed θ can.
  All 90 definitions are restricted to the time after the 100th interval (exposure and counts).
- Reading: the primary values stay those of the full time; the restriction is a sensitivity analysis with one sentence
  in the main text. If fewer than 9 of the CS1 reversals by a fixed θ remain, the reading that they come mainly from a
  fixed θ is withdrawn. Any count that crosses the 25% criterion is noted at its conclusion.
- Checks: without the restriction, counts equal the stored counts.

## 9. Causal two-stage intervals [A] (`conditions-round8.md`)

- Seen before: the intervals of order 7 (b) and how often point values fell outside them.
- The inner draw puts Poisson(1) weights on a unit's intervals and recomputes θ at each time from the earlier
  intervals only (weighted nearest-rank quantile), keeping the estimator causal.
- New counting rule: a reversal needs both the point value and the interval on the other side; cells whose baseline
  interval contains the boundary are not counted.
- Reading: if at most 5% of point values fall outside their intervals for the score, the ratios and φ, these intervals
  replace the one-stage ones; otherwise the one-stage intervals stay primary and both two-stage versions go to the
  supplement. (27% of φ values fell outside, so the one-stage intervals stayed primary.)
- Checks: with all weights 1, counts equal the stored counts in every unit; θ at time s does not depend on weights
  after s.

## 10. Reading by classes of interval [D] (no condition)

- No new computation. Existing intervals were sorted into below −0.1, within ±0.1, above +0.1 or imprecise, and the
  reading of groups and the grid of default and precedented levels were defined after seeing the results.

## 11. The first W seconds of each observation interval [A] (`conditions-round9.md`)

- Seen before: all earlier results, including those of order 8; not the share of the first W seconds or the results
  without them.
- Condition A is evaluated only from W after the start of an observation interval, so no definition can detect a
  stall in the first W seconds. Output 1: the share of exposure in those seconds by dataset, segmentation and W.
  Output 2: each interval [a, b] is cut to [max(a + 60 s, w), b] (w the time of the 100th interval; empty intervals
  dropped) for all 90 definitions, which are detected again and counted when t0 lies in the cut interval.
- Reading: as for order 8; in addition, if no definition keeps a positive score interval, the abstract and results
  state that the change of direction depends on the undetectable time.
- Checks: without the cut, counts and exposures equal the stored values in every unit; positive controls and mutants.
