# Pre-registration R<n>: <title>

Commit this file **before** the device run and tag the commit `prereg-R<n>`. Do not edit it afterwards; if a rule
has to change, add `R<n>-amendment-1.md` and a new tag.

- **Date / author:**
- **Campaign / model / file sha256:**
- **Agent models** (full IDs used this round):
- **Build under test** (app build, kit, llama.cpp `BASE`, patch series):
- **Labels:** every number in this file is *estimated* until a result block says *device* or *host*.

## 1. What is tested

One lever or one package, and its knobs (environment variables, defaults).

## 2. Budget accounting

| item | PPL | % over stock |
|---|---|---|
| stock served file | | 0 |
| config before this round | | |
| config under test (full wikitext-2, ctx 512, batch 512, all chunks, host) | | |
| budget (stock × 1.05) | | |
| hard ceiling (F16 × 1.20) | | |

Identity checks required (if the lever is declared bit-identical): per-chunk PPL equal at `-ub 1`, `-ub 32`,
`-b 512`; greedy text identical on the kit prompts.

## 3. Prediction

Quantity, unit, clock, band and central value. What the instrument can resolve (expected σ, number of blocks),
so a verdict is possible at all.

## 4. Pass rule and kill rule

- **Pass:**
- **Falsified / kill:**
- **Unresolved** (what happens if σ is too large):

## 5. Claim rule (only for a goal claim)

"The median of <N> plain runs of the official benchmark is at least <X> tok/s", with the cool-down gate and the
number of runs fixed here.

## 6. Hygiene counts

Exact counts that must appear in every B block (for example barriers, nodes, op counts), or the run is void.

## 7. What a miss means

Written now: which of the next steps is taken on pass, on kill, and on unresolved.
