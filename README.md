# DD2610 Final Project · Asymmetric Duos

This project studies [Asymmetric Duos: Sidekicks Improve Uncertainty](docs/asymmetrics_duos.pdf) by Tim G. Zhou, Evan Shelhamer, and Geoff Pleiss. The method pairs a large image classifier with a cheaper sidekick. Two weights fitted on validation logits determine how much each classifier contributes. The paper asks whether the pair improves predictions and helps identify errors at a modest increase in inference compute.

## Branches

| Branch | What belongs there |
| --- | --- |
| `main` | The authors’ released code, the paper, and this guide. The import is commit `54717af`. |
| `partA` | Our scoped replication using the released implementation. Keep experiment configs, run instructions, and results documentation here. |
| `partB-LLM` | The paper-only implementation. The paper is the sole reference for this part. Work is deferred. |

Use `git switch partA` for Part A work. Return to `main` to browse the reference implementation and project overview.

## Part A · Scoped replication

We target Figures 3–6 and use Appendix G Figures 11 and 14 to compare our two base models on Caltech-256. We keep one setting for each purpose in the original study.

| Dataset | Purpose | Protocol |
| --- | --- | --- |
| ImageNet-1K | Large-scale evaluation | Use pretrained classifiers. Split the 50,000 official validation images into 2,500 for calibration and 47,500 for testing. |
| ImageNetV2 | Distribution shift | Evaluate the same ImageNet classifiers and temperatures on MatchedFrequency, our chosen variant. |
| Caltech-256 | Transfer to a new task | Use a 70/15/15 train/validation/test split. Train the classifier head for 8 epochs, then fine-tune the whole model for 16 epochs. |

Our base models are TorchVision **ConvNeXt-Base** with `IMAGENET1K_V1` weights and **CLIP ViT-B/16** with the `timm` checkpoint `vit_base_patch16_clip_224.laion2b_ft_in12k_in1k`. Pair each base with TorchVision **ResNet-18, ResNet-50, and ResNet-101** to sweep three inference compute budgets within one sidekick family.

Compare the temperature-scaled base, the equal-weight Duo, the learned-weight Duo, and the UQ-only ablation. UQ-only keeps the base model’s predicted class and reads the weighted Duo’s confidence for that class. Report top-1 accuracy, correctness prediction AUROC, area under the risk–coverage curve (AURC), and coverage at 98% accuracy (SAC@98). Plot scores against sidekick FLOPs divided by base FLOPs.

Start with a Caltech-256 subset in Colab to estimate training time. Use the full split for the results above. For each run, record its paper reference, Git commit, dataset split and seed, checkpoint IDs, run configuration, GPU time, results location, and any deviations from the paper or reference code.

The paper also reports greedy model soup, deep ensembles, throughput, entropy-based uncertainty, learned weight ratios, Bayesian baselines, and full results across all models and benchmarks. Those studies provide context for the four core tests.

## Navigate the authors’ code

| Location | What to find |
| --- | --- |
| [Paper](docs/asymmetrics_duos.pdf) | §3 defines the method. Figures 3–6 report the main experiments. Appendices A–G provide protocols and additional results. |
| [`load_data/`](load_data/) and [`load_models/`](load_models/) | Dataset splits, preprocessing, pretrained model loading, and Duo wrappers. |
| [`train/`](train/) | Linear probing, fine-tuning, and hyperparameter search, plus model soup and IVON baselines. |
| [`evaluate/`](evaluate/) | Export predictions, fit single-model and Duo temperatures, and compute metrics. |
| [`utils/`](utils/) and [`plot_notebooks/`](plot_notebooks/) | Metric implementations and the authors’ plotting notebooks. |

The released code refers to dataset YAML files, model and pairing CSVs, and checkpoints that the public repository does not include. Part A will supply its own experiment configuration and document any changes needed to run the reference code. The [upstream repository](https://github.com/timgzhou/asymmetric-duos) has the authors’ original README and citation.
