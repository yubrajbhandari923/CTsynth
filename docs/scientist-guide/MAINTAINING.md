# Maintaining this Scientist Guide

Everything is in `docs/scientist-guide/` (this folder). `*.qmd` are the chapters; `provenance/*.json` link claims and figures
to Research OS records; `_output/` and `_generated/` are rebuilt and git-ignored. Commands use `research` = the project's
`.research/bin/research` (or `python -m research_cli --root .`).

## Look first
    research guide status          # pending syntheses, per-chapter freshness (fresh | stale | invalid), uncovered work
    research guide validate        # missing, stale, contradicted, superseded, inconsistent evidence (read-only)
    research guide coverage        # experiments and findings no chapter cites yet
    research guide review status   # per-chapter: unreviewed | reviewed with concerns | reviewed, awaiting approval | approved | review outdated

A chapter turns **stale** when a record it cites changes (finding status or supersession, a review, a metric, an artifact, a
synthesis, a decision). Only the affected chapters are stale; nothing is regenerated automatically.

## After a new experiment or finding
1. `research guide coverage` names what is not yet in the book. Decide the home: an existing thematic chapter, or a new one
   (`research guide chapter add investigations/<name>.qmd --title "..."`). Do not add one chapter per experiment.
2. Hand the update to an agent so the required `scientific-briefing` skill is enforced and hash-recorded:
       research guide refresh --run          # prepares/starts the managed guide agent (blocked if the skill is missing)
       research dispatch analyst --require-skill scientific-briefing --objective "update investigations/x.qmd for EXP-0NN" --print
   The agent edits only stale/uncovered chapters, registers claims, then you (or it) run `research guide section pin <chapter>`.
3. Register what the prose says: `research guide claim add "<statement>" --kind observation --section <chapter> --source F-0NN
   --source RUN-0NNN#metric`, and mark it in the text as `[sentence]{#clm-00NN .claim}`. Retire a claim you delete:
   `research guide claim retire CLM-00NN`.
4. Update the storyline (`storyline.qmd`) and briefing (`briefing.qmd`) when understanding changed; a superseded or contradicted
   finding is told as history, never as a current conclusion.

## Figures
Script reads recorded values (never reruns a simulation) and writes `figures/<name>.svg|png`; register with
`research guide figure add figures/<name>.svg --fig-kind recreated --section <chapter> --label fig-<name> --title "..." --source RUN-0NNN#metric --script docs/scientist-guide/scripts/<script>.py`.
After deliberately regenerating a figure: `research guide figure refresh FIG-00N --notes "why"`.
Kinds: archived (project's own figure, unchanged) | recreated | derived | conceptual (never evidence for an observation).

## Human edits and ownership
`research guide section set <chapter> --owner human` makes a chapter yours: agents must not edit it and put proposed changes in
`_proposals/<chapter>.md`; `guide status` lists open proposals. Agent-owned chapters may be rewritten by agents. Routine refreshes
never regenerate the book and never overwrite a chapter; `guide quarto init` and `chapter add` never overwrite files.

## Build and read
    research guide quarto doctor
    research guide quarto build --edition both        # HTML for the scientific and researcher editions
    research guide quarto build --edition scientific --pdf
    research guide quarto preview --edition researcher
Outputs: `_output/scientific/index.html`, `_output/researcher/index.html`, `_output/scientific/*.pdf`. Rendering never runs code.
The scientific edition refuses to build over provenance errors (a retracted claim shown as current); the build also checks that no
internal identifier leaked into the scientific pages.

## Review
    research guide review request [chapter]          # prepares an independent reviewer (verifier role, skill enforced)
    research guide review record <chapter> --verdict sound|concerns|unsound --reviewer "<who>" --rating evidence_fidelity=ok ...
    research guide review approve <chapter> --by "<you>"    # human approval: yours alone; outdated when text or evidence changes
A model review is not approval. Any change to a chapter or to evidence it cites marks earlier reviews OUTDATED.
