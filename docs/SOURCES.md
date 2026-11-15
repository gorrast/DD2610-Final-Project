# Source log

Every source consulted for the Part B reproduction, in the order it was first used. A row is added in
the same change that uses the source. Tags are defined in `CLAUDE.md`.

## Method source

| ID | Date | Source | Used for | Tag |
|---|---|---|---|---|
| S1 | 2026-11-15 | `docs/asymmetrics_duos.pdf`, pages 1-30 (main text, references, appendices A-G, checklist) | The whole method, the experiment spec, the reference numbers and the gap list | `[P ...]` |

## Requirement sources (scope only, no method content)

| ID | Date | Source | Used for |
|---|---|---|---|
| S2 | 2026-11-15 | `docs/project-proposal.md` | Datasets, software packages, compute budget, success criteria |
| S3 | 2026-11-15 | Proposal feedback form from the TA | "Doable for one or two datasets", "show the trend in one configuration", smaller-dataset fallback |
| S4 | 2026-11-15 | Course assignment text, grading guidelines and proposal instructions (kept outside the repository) | The Part B isolation requirement and the best-practice criteria |
| S5 | 2026-11-15 | The group's answers to 19 clarifying questions (listed in `docs/llm-log.md`) | Every `[USER 2026-11-15]` decision |

## Environment and tooling

| ID | Date | Source | Used for |
|---|---|---|---|
| S6 | 2026-11-15 | Git metadata of the original clone: branch names, commit titles, and file names on `partB-LLM` | Learning that the authors' code sits on `original_code` (commit title "copy code from authors"). No file on that branch was opened, listed or diffed. |
| S7 | 2026-11-15 | Claude Code's bundled `update-config` skill text | Permission-rule syntax for `.claude/settings.json` |
| S8 | 2026-11-15 | The group's `writing-for-agents` skill text | Structure and wording of `CLAUDE.md` |

## LLM background knowledge (`[PRIOR]`)

None of this comes from the paper. Each item stays unverified until the listed check replaces it with
a `[LIB]` row or a measurement.

| ID | Recalled item | Where it was used | Check that replaces it |
|---|---|---|---|
| PR1 | Per-model GFLOPS, parameter counts and accuracies in torchvision's weight metadata | Estimating the FLOPs balance of candidate sidekicks; the observation that the paper's §3.1 and Table 1 numbers look like torchvision's published values (G8, G10) | Read `weights.meta` offline |
| PR2 | timm checkpoint naming and the existence of several CLIP ViT-B/16 and EfficientNetV2-L variants | G9 | `timm.list_pretrained()` offline |
| PR3 | AdamW applies weight decay decoupled from the gradient, scaled by the learning rate | I8, and fixing WD instead of searching it (G17) | `torch.optim.AdamW` docstring |
| PR4 | PyTorch ships a FLOP counter under `torch.utils.flop_counter` | G10 | Import it and cross-check against `weights.meta` |
| PR5 | ImageNet class indices follow sorted WordNet ids in torchvision and timm | Label mapping in the ImageNet loader | Measured accuracy must match `weights.meta` |
| PR6 | ImageNet V2 has three variants; matched-frequency costs roughly ten accuracy points | G14 | Measured ConvNeXt-B accuracy against `[P Fig. 14]` |
| PR7 | Conventional definitions of ECE (15 bins), AURC, selective accuracy and Brier score | G3-G6 | None possible from the paper; sensitivity script reports the alternatives |
| PR8 | Common fine-tuning conventions: warm-up length, RandAugment defaults, random-resized-crop and flip, feature caching for linear probing | G16, G18, G19, G21 | Library defaults via `[LIB]`; the rest remain flagged assumptions |
| PR9 | `git clone --no-local --single-branch` transfers only the requested branch's objects | Handoff instructions in the plan | `git cat-file -e 54717af` must fail in the new clone |
| PR10 | Training and inference throughput of an RTX 3080 | Runtime estimates in the plan and in the clarifying questions | The benchmark at the start of each GPU job |
| PR11 | Common on-disk layouts of ImageNet validation data and Caltech 256, and that Kaggle and Hugging Face host copies | Options offered in a clarifying question; loader design | The layout the group actually has |
| PR12 | Softmax cross-entropy is convex in the logits, hence in (T_large, T_small) | G1 | Sensitivity script: different initialisations reach the same optimum |

## Offline library lookups (`[LIB]`)

None yet. One row per lookup:

| ID | Date | Package and version | Object inspected | What was read | Used in |
|---|---|---|---|---|---|

## Not consulted

No web page, no search result, no paper from the reference list, and no file from any branch other
than `partB-LLM`. The authors' repository linked in the paper's abstract was not opened.
