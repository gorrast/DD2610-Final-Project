# Plan: LLM reproduction of "Asymmetric Duos" (Part B, branch `partB-LLM`)

## Context

DD2610 final project, Group 18. Part B of the grading criteria requires reproducing the paper's main
experiments **with an LLM that has access only to the paper and its appendix**, with isolation
enforced by prompt and by jailing, and then discussing what the paper left unspecified. The repo
branch `partB-LLM` contains only `README.md`, `.gitignore` and `docs/asymmetrics_duos.pdf`; there is
no existing code to reuse. The authors' code lives on branch `original_code` and must never be read.

Outcome: a PyTorch package that (1) saves logits for pretrained/fine-tuned models, (2) builds Duos and
all metrics from saved logits on CPU, (3) reproduces the paper's core figures and Tables 1-2 at reduced
compute, and (4) produces a categorised list of paper gaps with their measured effect.

## Decisions taken with the user (2026-11-15)

| Topic | Decision |
|---|---|
| Datasets | ImageNet + ImageNet V2 (inference only) first, then Caltech 256 (fine-tuning). No iWildCam. |
| Sources | Paper only. No web, no other git branches. Library facts via **offline introspection** of installed packages. |
| Jail | Fresh single-branch clone of `partB-LLM`; all code is written in a new session there. |
| Hardware | The LLM writes code + CPU smoke tests on the laptop; the group runs GPU jobs on an **RTX 3080 (10 GB), native Windows**. |
| Base models | All 4 (ConvNeXt-B, SwinV2-B, EfficientNetV2-L, CLIP ViT-B/16) on ImageNet; ConvNeXt-B only on Caltech. |
| Sidekicks | ImageNet: every torchvision model in the App. A.1 families with FLOPs <= base. Caltech: MNASNet0.75, ShuffleNetV2 x2.0, EfficientNet-B2, RegNetX-1.6GF, ResNet-34/50/101 (all named in the paper) + ResNet-152. |
| Extras | Tables 1-2 (Duos vs deep ensembles). No soups (Fig. 7), no throughput (Fig. 8). |
| HP search | Tune per family, reuse for siblings (ResNet-50's setting reused for ResNet-34/101/152). |
| Seeds | 1 seed for the search, 3 seeds for each model's final setting. |
| RN50 pool | All distinct plain ResNet-50 ImageNet-1k checkpoints in torchvision + timm; keep top 10 by accuracy on our validation split. |
| timm bases | Enumerate candidates offline, keep the one whose ImageNet accuracy is closest to the paper's App. G base line. |
| Gaps | Flagged default + config switch per gap; CPU sensitivity script for post-processing gaps. |
| Saved outputs | Logits + labels only (float32). |
| Data access | ImageNet val and Caltech 256 from local folders; ImageNet V2 via HF `datasets` with a Hub ID the group supplies. |
| LLM log | Curated `docs/llm-log.md` in the repo; raw transcripts stay private. |
| Commits | The LLM makes local milestone commits on `partB-LLM`; the group reviews and pushes. |
| Dependencies | Only the proposal's packages: torch, torchvision, timm, datasets, scikit-learn, numpy, pandas, matplotlib, seaborn (+ pytest for tests; config via stdlib `tomllib`/argparse). |

## Isolation and source protocol

The living copy of these rules is `CLAUDE.md`; this section records them as planned.

Every non-trivial line of method code carries one of these tags, and every lookup is appended to
`docs/SOURCES.md`:

- `[P §3.1]`, `[P App. A.2]`, `[P Fig. 3]`, `[P Tab. 1]`: stated in the paper.
- `[LIB pkg==version: object]`: read offline from an installed package (docstring, signature, `weights.meta`).
- `[GAP Gnn]`: not in the paper; default chosen, documented in `docs/paper-gaps.md`.
- `[PRIOR]`: the LLM's built-in background knowledge (never the paper); always paired with a GAP id.
- `[USER 2026-11-15]`: decided by the group in the table above.
- `[DEV]`: deviation forced by compute (e.g. mixed precision, gradient accumulation).

Network use is limited to tooling: package installs, pretrained-weight downloads by torchvision/timm,
and the HF dataset the group names. The LLM never reads web content, never fetches the GitHub URL in
the paper's abstract, and never spawns sub-agents.

## Step 0: bootstrap in the original clone, on `partB-LLM` (then stop)

1. `CLAUDE.md`: the isolation rules and tag convention above, plus "read `docs/plans/LLM_RECREATION_PLAN.md` first".
2. `.claude/settings.json`: deny WebFetch, WebSearch, `curl`, `wget`, `gh`, and `git fetch/pull/clone/remote/ls-remote` (exact rule syntax via the `update-config` skill).
3. `docs/plans/LLM_RECREATION_PLAN.md` (this plan), `docs/SOURCES.md` (seeded with everything consulted so far, including the recalled-from-memory items at the end of this plan), `docs/paper-gaps.md` (tables below), `docs/llm-log.md` (session 1: the prompt verbatim and the decisions).
4. Extend `.gitignore` (`artifacts/`, `.venv/`, caches, `.claude/settings.local.json`).
5. One local commit containing only these files. The untracked proposal and feedback PDF are left untouched.

**Handoff, run by the group:**

```bash
git clone --no-local --single-branch --no-tags --branch partB-LLM \
    <path-to-original-clone>  <new-dir>
cd <new-dir> && git remote set-url origin <project GitHub URL>
git branch -a && git log --all --oneline   # must show only partB-LLM, no "copy code from authors"
git cat-file -e 54717af                    # must fail: the authors' commit is absent from the object store
```

`--no-local` matters: a plain local clone copies the whole object store, including the authors' code.
The `cat-file` check is the one that proves it, because unreferenced objects never show up in `git log`.
Then start a new Claude Code session in `<new-dir>` and ask it to follow `docs/plans/LLM_RECREATION_PLAN.md`.

## What the paper specifies (implementation spec)

- **Duo** `[P §3.1]`: `f_duo = T_large * f_large + T_small * f_small` on logits; prediction = argmax; uncertainty = `1 - max softmax` `[P §2 Eq. 1]`. Both temperatures minimise NLL on the validation set.
- **Unweighted Duo** `[P §3.1]`: `T_large = T_small = 0.5`. **UQ-only Duo**: prediction from `f_large`, uncertainty = `1 - softmax(f_duo)[y_hat_large]`.
- **FLOPs balance** `[P §3.2]` = FLOPs(small) / FLOPs(large).
- **Single models** `[P §4]`: temperature scaling by L-BFGS on validation NLL.
- **Data** `[P App. A.3]`: ImageNet: 5% of the official validation set is our validation split, the rest is test; ImageNet V2: whole set is OOD test (temperatures come from ImageNet validation); Caltech 256: 30,607 images, 257 classes, random 15% val / 15% test.
- **Training** `[P §4, App. A.2]`: LP (head only) 8 epochs, then FT (all weights) 16 epochs; cross-entropy, AdamW, linear warm-up + cosine; LR in [1e-6, 3e-4], WD in [1e-8, 1e-5]; batch 128; RandAugment in FT only; best LP checkpoint initialises FT; best FT model by validation is kept.
- **Metrics** `[P §5, App. C, App. G]`: accuracy, macro-F1, NLL, Brier, ECE, AUROC of correctness prediction, AURC, SAC@98; entropy as alternative uncertainty.
- **Deep ensembles** `[P §2.1, §5.6, App. A.5, App. E]`: logit average, pool-then-calibrate, 32 random subsets of size m from 10 ResNet-50 checkpoints; Duos pair one ResNet-50 with MNASNet0.75 / ShuffleNetV2 x2.0 / EfficientNet-B2.
- **Reference numbers for sanity checks**: ConvNeXt-B 89M params / 15 GFLOPs / 84.1%, RegNetX-1.6GF 9M / 1.6 GFLOPs / 79.7% `[P §3.1]`; ResNet-50 4.1 GFLOPs and balances 5% / 14% / 27% `[P Tab. 1-2]`; all rows of Tables 1-2; base-model lines read off App. G figures (approximate, +/-0.2): ImageNet accuracy ConvNeXt-B 84.1, SwinV2-B 84.0, EfficientNetV2-L 85.9, ViT-B 86.2; ImageNet V2 accuracy ConvNeXt-B 73.6.

## Paper gaps (seed for `docs/paper-gaps.md`)

"Sens." = alternative evaluated by the CPU sensitivity script.

| ID | Missing detail | Default | Sens. |
|---|---|---|---|
| G1 | Optimiser, init, constraints for (T_large, T_small) `[P §3.1]` | torch L-BFGS, float64, init (1, 1), unconstrained; objective is convex | inits, Adam |
| G2 | Unweighted Duo: raw or individually calibrated logits | raw logits, literal reading of T = 0.5 | calibrated inputs |
| G3 | ECE bin count and scheme | 15 equal-width bins on max probability `[PRIOR]` | 10 / 20 |
| G4 | Brier normalisation (reported magnitudes imply mean over classes) | mean over samples and classes | sum over classes |
| G5 | AURC estimator and tie handling (only cited) | mean selective risk over all n coverages `[PRIOR]` | tie variants |
| G6 | SAC@98 exact rule; value when the target is never reached | largest coverage with selective accuracy >= 0.98, else 0 | strict variant |
| G7 | Exact sidekick list (App. A.1 gives families only) | all torchvision members of those families, FLOPs <= base | n/a |
| G8 | Which pretrained weights per architecture | identify offline by matching §3.1 accuracies to `weights.meta` | other version (optional GPU) |
| G9 | Exact timm checkpoints and resolution for ViT-B and EfficientNetV2-L | closest to the paper's App. G base accuracy `[USER]` | runner-up |
| G10 | FLOPs tool and input resolution | torchvision `weights.meta` value; PyTorch's built-in counter for timm, cross-checked on torchvision models | n/a |
| G11 | Whether sidekicks larger than the base are used | no (figures stop at balance 1) | n/a |
| G12 | Eval preprocessing | each checkpoint's own default transform | n/a |
| G13 | ImageNet val/test split: random or stratified, seed | uniform random, seed 0, indices saved | 10 seeds, stratified, 1% / 10% |
| G14 | ImageNet V2 variant | matched-frequency (only one consistent with ~73.6% `[P Fig. 14]` `[PRIOR]`) | n/a |
| G15 | Caltech split seed, stratification; train = remaining 70% | uniform random, seed 0 | n/a |
| G16 | LP learning rate, schedule, augmentation; "best LP checkpoint" criterion | LP on cached eval-transform features, small LR grid, best validation accuracy `[DEV]` | n/a |
| G17 | Number of HP trials and sampling | 4 log-spaced LRs over the paper's range, WD fixed mid-range, per family `[USER]` | LR curve reported |
| G18 | Warm-up length, final LR, step granularity | 1 epoch warm-up, cosine to 0, per-iteration | n/a |
| G19 | RandAugment strength, other augmentation, input resolution | torchvision defaults, random-resized-crop + flip, checkpoint's resolution | n/a |
| G20 | "Validation score" for model selection | validation accuracy, ties by NLL | n/a |
| G21 | AdamW betas/eps, precision, clipping, new-head init | PyTorch defaults; mixed precision and accumulation to batch 128 if 10 GB is exceeded `[DEV]` | n/a |
| G22 | Seeds / number of runs (scatter plots are single runs) | 3 seeds for final Caltech settings `[USER]`; bootstrap over test images | yes |
| G23 | Which 10 ResNet-50 checkpoints; what the +/- in Tables 1-2 is over | top 10 by validation accuracy `[USER]`; std over subsets (DE) and over bases (AD) | all checkpoints |
| G24 | Which pairs make up AD[m, m+1) in Table 2 | all (large, small) pairs in the pool with total FLOPs in range | n/a |

Inconsistencies found in the paper (for the presentation):

- I1: §4 / App. A.1 say EfficientNetV2-L is ImageNet-21k and CLIP ViT-B is LAION-2B; App. G headings swap them.
- I2: §4 promises "a complete list" of sidekicks in App. A; A.1 lists families only. ViT appears as a family in Fig. 2 and Fig. 8 but not in A.1; Swin is cited as [62], the ShuffleNet paper.
- I3: Fig. 7 axis says SAC@99; caption and text say SAC@98.
- I4: Figures state NLL x10 and Brier x1e5; Tables 1-3 imply NLL x100 and Brier x1e4, unstated.
- I5: App. A.3 calls the ImageNet validation set the "official test split".
- I6: App. B text gives R^2 = 0.795 for Duos, Fig. 8 shows 0.826; the Fig. 8 caption says ConvNeXt-Large, its annotation says ConvNeXt-Base.
- I7: The same DE(m=2) and DE(m=5) setups report slightly different numbers in Table 1 and Table 2; App. E says the DE setup is "identical to Table 2" (itself).
- I8: The WD range [1e-8, 1e-5] is negligible under AdamW's decoupled decay `[PRIOR, verify in docstring]`, so that search dimension cannot matter.

## Package layout (new clone)

```
pyproject.toml, README.md (PowerShell run instructions), configs/default.toml (one key per GAP id)
duos/
  data/{imagenet,imagenet_v2,caltech256,splits}.py   local-folder + HF loaders, saved split indices
  models/{registry,flops}.py                         family enumeration, weights, transforms, FLOPs
  inference.py        model x dataset -> artifacts/logits/<dataset>/<model>.npy + JSON sidecar
  finetune.py         cached-feature LP, then FT; family-level LR grid; seeds; resumable
  calibration.py      single-model temperature scaling (L-BFGS)
  duo.py              weighted / unweighted / UQ-only Duos, logit ensembles
  metrics.py          sklearn for accuracy, F1, AUROC; numpy for NLL, Brier, ECE, AURC, SAC, entropy
  analysis/{duos,ensembles,sensitivity,plots,report}.py
  cli.py              python -m duos.cli {infer,finetune,analyze,ensembles,sensitivity,plots,check}
tests/                pytest: metrics vs sklearn and hand-computed cases, Duo fit, smoke pipeline
artifacts/            (gitignored) logits, checkpoints, cached features
results/              (committed) CSV tables and figures
```

Windows constraints: Python entry points only (no shell scripts), `pathlib`, picklable top-level
dataset/transform classes, `if __name__ == "__main__"` guards, `persistent_workers`.

## Milestones (new session), one local commit each

- **M1 Core maths (CPU):** `metrics`, `calibration`, `duo`, unit tests on synthetic logits (Duo fit recovers a known optimum, T_small -> 0 for a noise sidekick, sklearn cross-checks).
- **M2 Inference stage:** registry, FLOPs, loaders, `infer` CLI, CPU smoke test on a handful of images with two tiny models. Includes `check`: measured accuracy vs `weights.meta` and FLOPs vs the paper's quoted numbers.
- **M3 Analysis:** results table for every (base, sidekick, variant, dataset); analogues of Fig. 2, 3-6, 9, 10 and the App. G metrics; Tables 1-2; sensitivity script; paired bootstrap for Duo-minus-base.
- **M4 Caltech 256:** split, cached-feature LP, FT trainer, family HP grid, 3 seeds, then the same inference and analysis path; same-architecture ConvNeXt-B ensemble from the 3 seeds as an extra baseline.
- **M5 Write-up:** `docs/paper-gaps.md` filled with measured effects, `docs/reproduction-report.md` checking the paper's claims against our numbers and its reference values.

**GPU jobs for the group (resumable; each starts with a short benchmark that prints a projected runtime):**

- Job A after M2: ImageNet + V2 inference for ~50 sidekicks, 4 bases plus timm candidates, and the ResNet-50 pool. Needs the ImageNet val path, the V2 Hub ID and roughly 40 GB of free disk. Unmeasured estimate: 3-6 hours.
- Job B after M4: Caltech 256, 24 search runs + 3 sibling runs + 18 extra-seed runs. Needs the `256_ObjectCategories` path. Unmeasured estimate: 11-12 hours.
- Analysis is CPU-only and runs wherever the logits are; `results/` is committed from there.

## Verification

1. `pytest` passes on the laptop (CPU): metric implementations agree with scikit-learn where it has an equivalent and with hand-computed toy cases; the smoke pipeline runs end to end on synthetic data.
2. `python -m duos.cli check` on the 3080 after Job A: every torchvision model's full-validation accuracy is within ~0.1 pt of its `weights.meta` value (validates labels and preprocessing); FLOPs reproduce 15 / 1.6 / 4.1 GFLOPs and balances 5 / 14 / 27% `[P §3.1, Tab. 1-2]`.
3. Base-model metrics land near the paper's App. G base lines; differences are logged, not tuned away.
4. Claim checklist in the report: accuracy >= base once balance >~ 0.1 while Unweighted drops at low balance `[P §5.1]`; AUROC, AURC and SAC@98 improve at all balances with UQ-only comparable `[P §5.2-5.4]`; T_large/T_small > 1 at low balance and -> 1 as it grows `[P App. D]`; Duos approach DE(m=5) at a fraction of the FLOPs `[P §5.6]`.
5. Isolation audit: `docs/SOURCES.md` lists every lookup; in the clean clone `git cat-file -e 54717af` fails and `git log --all` never shows the authors' commit.

## Recalled from memory, to be verified offline before use

torchvision GFLOPS values used for the size estimates above, timm checkpoint naming, the AdamW decay
formula, `torch.utils.flop_counter`, ImageNet class-index ordering, and the `git clone --no-local`
behaviour. None of these come from the paper; each is logged as `[PRIOR]` until confirmed by `[LIB]`.

## Sources consulted to write this plan

- `docs/asymmetrics_duos.pdf`, all 30 pages (the only method source).
- `docs/project-proposal.md`, the proposal feedback form, and the course assignment / grading /
  proposal instructions under `_personal/docs/` (requirements only, no method content).
- Git metadata of this clone: branch names and commit titles only. No file on `original_code` was
  opened, listed or diffed.
- The group's answers to 19 clarifying questions on 2026-11-15 (decision table above).
- LLM background knowledge for the items listed in the previous section, flagged `[PRIOR]`.
