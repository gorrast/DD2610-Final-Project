# Paper gaps and inconsistencies

Details the paper leaves unspecified or states inconsistently, found while reproducing it from the
text alone. Each gap has a default (what the code does), the basis for that default, and the
alternative the CPU sensitivity script evaluates. "Measured effect" is filled in once logits exist.

Tags are defined in `CLAUDE.md`; `[P ...]` locations refer to `docs/asymmetrics_duos.pdf`.

## Method

| ID | Missing detail | Default | Basis | Sensitivity | Measured effect |
|---|---|---|---|---|---|
| G1 | Optimiser, initial values and constraints for (T_large, T_small); the paper says "any standard optimizer" `[P §3.1]` | torch L-BFGS in float64, init (1, 1), unconstrained | L-BFGS is what the paper uses for single-model temperature scaling `[P §4]`; the objective is convex so the optimum should not depend on these `[PRIOR]` | Other initialisations; Adam | not yet measured |
| G2 | Whether the Unweighted Duo averages raw logits or individually temperature-scaled logits `[P §3.1]` | Raw logits | Literal reading of "T_large = T_small = 0.5" in the Duo formula; matches the logit average of `[P §2.1]` | Calibrated inputs | not yet measured |

## Metrics

| ID | Missing detail | Default | Basis | Sensitivity | Measured effect |
|---|---|---|---|---|---|
| G3 | ECE bin count and binning scheme `[P §5.2, App. G]` | 15 equal-width bins on the maximum probability | `[PRIOR]` | 10 and 20 bins | not yet measured |
| G4 | Brier score normalisation `[P App. G, Tab. 1]` | Mean over samples and classes | The reported magnitudes (about 20-24 after the stated x1e5 scaling on ImageNet, `[P Fig. 11, 14]`) only fit a mean over the 1000 classes `[PRIOR]` | Sum over classes | not yet measured |
| G5 | AURC estimator and tie handling; only a citation is given `[P §5.3]` | Mean selective risk over all n coverage levels, samples ordered by confidence | `[PRIOR]` | Tie-handling variants | not yet measured |
| G6 | Exact SAC@98 rule and its value when 98% is never reached `[P §5.4]` | Largest coverage whose selective accuracy is at least 0.98; 0 if none | Wording of `[P §5.4]`; `[P Fig. 6]` shows values at 0 on ImageNet V2 | Requiring the accuracy at every lower coverage too | not yet measured |

## Models

| ID | Missing detail | Default | Basis | Sensitivity | Measured effect |
|---|---|---|---|---|---|
| G7 | The exact sidekick list; `[P §4]` promises one in App. A, `[P App. A.1]` names families only | Every torchvision member of the named families with FLOPs no larger than the base | `[USER 2026-11-15]` | n/a | n/a |
| G8 | Which pretrained weight version per architecture `[P App. A.1]` | Identified offline by matching the accuracies quoted in `[P §3.1]` (84.1%, 79.7%) to `weights.meta` | `[P §3.1]` plus `[LIB]` once read | The other weight version (optional, needs GPU) | not yet measured |
| G9 | Exact timm checkpoints and input resolution for CLIP ViT-B/16 and EfficientNetV2-L `[P §4, App. A.1]` | The candidate whose ImageNet accuracy is closest to the base line in `[P Fig. 11, 13]` | `[USER 2026-11-15]` | Runner-up candidate | not yet measured |
| G10 | FLOPs counting tool and input resolution `[P §3.2]` | torchvision `weights.meta` value; PyTorch's built-in counter for timm models, cross-checked on torchvision models | The numbers in `[P §3.1, Tab. 1-2]` are compared against `weights.meta` `[PRIOR]` until verified | n/a | n/a |
| G11 | Whether sidekicks larger than the base model are used | Excluded | The FLOPs-balance axis ends at 1 in `[P Fig. 3-6]` | n/a | n/a |
| G12 | Evaluation preprocessing (resize, crop, interpolation, normalisation) | Each checkpoint's own default transform | `[PRIOR]`, then `[LIB]` | n/a | n/a |

## Data

| ID | Missing detail | Default | Basis | Sensitivity | Measured effect |
|---|---|---|---|---|---|
| G13 | ImageNet validation/test split: random or stratified, and the seed `[P App. A.3]` | Uniform random 5% / 95%, seed 0, indices saved | `[P App. A.3]` gives only the fraction | 10 seeds; stratified; 1% and 10% validation | not yet measured |
| G14 | Which ImageNet V2 variant `[P App. A.3]` | Matched-frequency | Only variant consistent with about 73.6% for ConvNeXt-B in `[P Fig. 14]` `[PRIOR]` | n/a | n/a |
| G15 | Caltech 256 split seed and stratification; that training uses the remaining 70% is implied `[P App. A.3]` | Uniform random 70 / 15 / 15, seed 0 | `[P App. A.3]` | n/a | n/a |

## Training

| ID | Missing detail | Default | Basis | Sensitivity | Measured effect |
|---|---|---|---|---|---|
| G16 | Linear-probe learning rate, schedule and augmentation; what "best LP checkpoint" is judged by `[P App. A.2]` | Probe trained on cached evaluation-transform features, small LR grid, best validation accuracy | RandAugment is stated for the FT phase only `[P App. A.2]`; caching is a compute saving `[DEV]` | n/a | n/a |
| G17 | Number of hyperparameter trials and how they are sampled `[P App. A.2]` | 4 log-spaced learning rates across the paper's range, weight decay fixed mid-range, searched once per architecture family | `[USER 2026-11-15]`, `[DEV]` | Validation accuracy per learning rate is reported | not yet measured |
| G18 | Warm-up length, final learning rate, per-step or per-epoch schedule `[P §4, App. A.2]` | 1 epoch linear warm-up, cosine to 0, stepped per iteration | `[PRIOR]` | n/a | n/a |
| G19 | RandAugment strength, other augmentations, input resolution `[P App. A.2]` | torchvision's RandAugment defaults, random-resized-crop and horizontal flip, the checkpoint's own resolution | `[PRIOR]`, then `[LIB]` | n/a | n/a |
| G20 | What "validation score" selects the model `[P App. A.2]` | Validation accuracy, ties broken by validation NLL | `[PRIOR]` | n/a | n/a |
| G21 | AdamW betas and epsilon, numeric precision, gradient clipping, initialisation of the new 257-class head `[P App. A.2]` | PyTorch defaults, no clipping; mixed precision, and gradient accumulation to an effective batch of 128 if 10 GB is exceeded | `[LIB]` defaults, `[DEV]` | n/a | n/a |
| G22 | Number of runs and seeds; the scatter plots are single runs `[P checklist item 7]` | 3 seeds for each final Caltech setting; paired bootstrap over test images | `[USER 2026-11-15]` | Spread across seeds and bootstrap intervals | not yet measured |

## Deep-ensemble comparison

| ID | Missing detail | Default | Basis | Sensitivity | Measured effect |
|---|---|---|---|---|---|
| G23 | Which 10 ResNet-50 checkpoints, and what the +/- in Tables 1-2 is taken over `[P App. A.5, Tab. 1-2]` | The 10 distinct plain ResNet-50 ImageNet-1k checkpoints with the best accuracy on our validation split; std over subsets for ensembles and over base checkpoints for Duos | `[USER 2026-11-15]`; the std reading is an assumption | All available checkpoints | not yet measured |
| G24 | Which model pairs make up AD[m, m+1) in Table 2 `[P App. E]` | Every (large, small) pair in the pool whose total FLOPs fall in the range | Assumption | n/a | n/a |

## Inconsistencies in the paper

| ID | Location | Inconsistency |
|---|---|---|
| I1 | `[P §4, App. A.1]` vs `[P App. G]` | The text says EfficientNetV2-L is pretrained on ImageNet-21k and CLIP ViT-B on LAION-2B; the App. G headings swap the two. |
| I2 | `[P §4]` vs `[P App. A.1, Fig. 2, Fig. 8]` | §4 promises "a complete list" of sidekicks in App. A, which names families only. ViT appears as a family in Fig. 2 and Fig. 8 but not in A.1. Swin is cited as [62], the ShuffleNet paper. |
| I3 | `[P Fig. 7]` | The axis label reads SAC@99; the caption and text say SAC@98. |
| I4 | `[P App. G footers]` vs `[P Tab. 1-3]` | Figures state NLL x10 and Brier x1e5. Table values imply NLL x100 and Brier x1e4, and the tables do not state their scaling. |
| I5 | `[P App. A.3]` | The ImageNet validation set is called the "official test split". |
| I6 | `[P App. B]` vs `[P Fig. 8]` | The text gives R^2 = 0.795 for Duos, the figure shows 0.826. The caption says ConvNeXt-Large-based Duos, the annotation says ConvNeXt-Base. |
| I7 | `[P Tab. 1]` vs `[P Tab. 2, App. E]` | DE(m=2) and DE(m=5) report slightly different numbers in the two tables for what is described as the same setup. App. E says the ensemble setup is "identical to Table 2", the table it sits under. |
| I8 | `[P App. A.2]` | The weight-decay range [1e-8, 1e-5] is negligible under AdamW's decoupled decay, so that search dimension cannot matter `[PRIOR, verify in docstring]`. |
| I9 | `[P §3.1]` | The example sidekick is named "RegNeXt-1.6GF"; the citation [46] and the quoted size point to RegNetX-1.6GF `[PRIOR]`. |
