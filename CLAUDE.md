# Clean-room reproduction of "Asymmetric Duos"

Branch `partB-LLM` is a **clean-room** reimplementation: an LLM writes the code from the paper alone, and the course grades how well that isolation holds. Read `docs/plans/LLM_RECREATION_PLAN.md` before writing code; it holds the scope, the group's decisions, the paper spec and the milestones.

## Clean-room sources

Take every method decision from `docs/asymmetrics_duos.pdf`, main text and appendix. Take library facts (names, signatures, weight metadata) from the installed packages themselves: docstrings, `inspect`, `weights.meta`, `timm.list_models()`. Where the paper is silent, choose a default, record it as a gap, and continue.

Hard limits; a single breach invalidates the reproduction:

- The authors' code is off limits: the `original_code` branch, every other branch of this repository, and the GitHub repository linked in the paper's abstract.
- Web content is off limits: no browsing, fetching or searching. The network is for tooling only (package installs, pretrained weights, the dataset the group names).
- Work in one session without sub-agents, so every lookup lands in one log.

When a task seems to need a forbidden source, stop and ask the group.

## Provenance

Each non-trivial line of method code carries one tag in a comment:

- `[P §3.1]`, `[P App. A.2]`, `[P Fig. 3]`, `[P Tab. 1]`: stated in the paper.
- `[LIB pkg==version: object]`: read offline from an installed package.
- `[GAP Gnn]`: the paper is silent; default documented in `docs/paper-gaps.md`.
- `[PRIOR]`: the LLM's background knowledge; always paired with a `[GAP]` id.
- `[USER date]`: decided by the group.
- `[DEV]`: deviation forced by compute.

A change is done when every new decision has a tag, every new lookup has a row in `docs/SOURCES.md`, and every new gap has a row in `docs/paper-gaps.md`.

## Session log

Before a session ends, append an entry to `docs/llm-log.md`: the group's prompts verbatim, what was done, and the decisions taken. Committed files carry placeholder paths such as `<repo>` and no email addresses.

## Git

Commit locally at each milestone. The group pushes and pulls; `git fetch`, `pull`, `clone` and `remote` are denied in `.claude/settings.json` so no other branch can arrive in this clone.
