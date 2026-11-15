# LLM interaction log

Curated record of the sessions in which an LLM worked on the Part B reproduction. Raw transcripts
are kept privately by the group. Each entry gives the group's prompts verbatim, what the LLM did,
and the decisions taken.

## Session 1: planning and bootstrap (2026-11-15)

- **Tool and model:** Claude Code, model Claude Fable 5.2 (`claude-fable-5-2`).
- **Where:** the original multi-branch clone, on branch `partB-LLM`. No code was written.
- **Sub-agents:** none. The LLM declined to spawn planning agents because a cold agent could have
  read the `original_code` branch.

### Prompts (verbatim)

1. > We are studying the paper in @docs/asymmetrics_duos.pdf and need to implement their results on our own. It is crucial that we do not use the authors code to directly affect our code, so make sure to reference each and every source that you use to get to your answer. You can find our project proposal in @docs/project-proposal.md and the feedback we received on said proposal in @docs/G18 - DD2610 - HT25 - Project Proposal Feedback Form.pdf. The gist is that we can maybe simplify our experiments in terms of computational need, so we should keep that in mind. We should build with the software packages and libraries specified in the proposal. Ask lots of questions until there are no ambiguities left and everything is clear.
2. > write the full plan into docs/plans/LLM_RECREATION_PLAN.md

   (Given instead of approving the plan in Claude Code's plan-approval dialog.)
3. > do step 0

### What the LLM read

The paper (all 30 pages), the project proposal, the TA's feedback form, and the course assignment,
grading and proposal instructions. From git it saw branch names and commit titles only. Full list in
`docs/SOURCES.md`.

### Clarifying questions and answers

The LLM asked 19 multiple-choice questions in five rounds. "(rec.)" marks the option the LLM had
recommended.

| # | Question | Answer chosen |
|---|---|---|
| 1 | Which datasets should the reproduction cover? | ImageNet + ImageNet V2, then Caltech 256 (rec.) |
| 2 | What may the LLM consult besides the paper? | Free-text answer: "Paper only, and with no access to other branches in this git repo as well" |
| 3 | How should isolation be enforced? | Clean single-branch clone (rec.) |
| 4 | Where will experiments run? | RTX 3080, run by the group (rec.) |
| 5 | With paper only, how may library facts be looked up? | Offline introspection of installed packages (rec.) |
| 6 | Which base models? | All 4 on ImageNet, ConvNeXt-Base on Caltech (rec.) |
| 7 | How to pick sidekicks, given the paper lists only families? | All family members on ImageNet, about 8 on Caltech (rec.) |
| 8 | Which costlier extra results? (multi-select, no recommendation) | Duos vs deep ensembles (Tables 1-2) only |
| 9 | How much hyperparameter search per Caltech model? | Tune per family, reuse. The LLM had recommended a 4-point LR grid per model. |
| 10 | How should the code find the datasets? | Hugging Face for ImageNet V2, local folders for the rest (rec.) |
| 11 | How to handle details the paper leaves unspecified? | Flagged defaults plus sensitivity analysis (rec.) |
| 12 | What to save per model? | Logits only (rec.) |
| 13 | How to keep the interaction log? | Curated log in the repo, raw transcripts private (rec.) |
| 14 | How many fine-tuning seeds? | 1 for the search, 3 for the final setting (rec.) |
| 15 | How should commits work? | The LLM commits milestones locally, the group pushes (rec.) |
| 16 | What does the RTX 3080 machine run? (no recommendation) | Windows, native |
| 17 | Which sidekicks on Caltech 256? | The models the paper names, plus ResNet-152 (rec.) |
| 18 | How to pick the 10 ResNet-50 checkpoints? | Top 10 by validation accuracy (rec.) |
| 19 | How to choose the unnamed timm base checkpoints? | Match the paper's base accuracy (rec.) |

Fifteen of the 19 answers followed the LLM's recommendation, one was free text, one went against the
recommendation, and two had no recommendation.

### Output

- `docs/plans/LLM_RECREATION_PLAN.md`: scope, decisions, paper spec, gap list, package layout, milestones.
- `CLAUDE.md` and `.claude/settings.json`: the isolation rules as a standing prompt and as tool deny rules.
- `docs/SOURCES.md`, `docs/paper-gaps.md`, this log.
- 24 unspecified details (G1-G24) and 9 internal inconsistencies (I1-I9) identified in the paper.

### Isolation notes

- The authors' code was present on disk in this clone during the session, on branch `original_code`.
  The LLM did not open, list or diff it; it learned of it from a commit title.
- No web access was used.
- The LLM filled several planning estimates from background knowledge rather than the paper
  (library metadata, runtime estimates, metric conventions). These are listed as `[PRIOR]` in
  `docs/SOURCES.md` and are to be verified offline before use.
- All code is to be written in a fresh single-branch clone that does not contain `original_code`.
