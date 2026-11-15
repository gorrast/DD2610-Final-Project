# Project Proposal: Reproducing Asymmetric Duos

## Group 18:

Erik Mallmin: erikmall@kth.se  
Nina Kasimova: ninakas@kth.se  
Hugo Wigh: hwigh@kth.se

## Topic, paper and target grade

* **Topic:** Probabilistic Deep Networks  
* **Target grade:** A/B  
* **Working title:** Reproduction of Asymmetric Duos, extended with an adaptive weighting scheme

## Project Plan 

### Experiments 

We will reproduce the main results of the paper by combining large pre-trained models with smaller “sidekicks”, for example, ConvNeXt-Base copying the paper setup. For the sidekicks, we use smaller architectures from torchvision and timm. We compare three versions of each Duo: weighted, unweighted, and UQ-only.  
To evaluate, we compute accuracy for ImageNet, ImageNet V2, and Caltech 256\.

In order to ensure a fair re-implementation of the research, we are going to need a proper LLM isolation. We will be tracking the sources used, specifying in the prompt to only use the paper text and logging the interactions with the LLM.

**Fine-tuning**   
The models we use were trained on ImageNet’s classes, so they can be used on ImageNet and ImageNet V2 without any training. Caltech 256 has its own 257 classes, so each model needs to be fine tuned. We follow the paper’s approach. First we train only the fc layer (linear probing), then fine tune the whole network (LP-FT approach). We use the AdamW optimiser with a warm-up and cosine learning rate schedule, and calibrate each model with temperature changes on the val set.Depending on available compute, we try only a few learning rates and reuse the best settings across similar models. If we have enough GPU time, we will tune the learning rate and weight decay for every model. After fine-tuning, we save each model’s logits once, so all Duo experiments can be run from the saved results without GPU again.

**Extension**  
In the original paper, the large model and the sidekick are combined using fixed weights. Following the suggestion from the authors we will make the weighting adaptive (Dynamic Duos).

### Datasets 

ImageNet-1k  
ImageNet V2  
Caltech 256  
iWildCam

### Additional Methods 

### Why this matches A / B

We will reproduce the results using an LLM based only on the paper’s content and making sure it does not fetch the original code from GitHub and then compare the results. This will satisfy the B requirement. Then we will also write an extension of the paper by introducing adaptive weighing for an A grade. 

## Compute 

### Computational Resources 

T4 GPU \- 5h/week with Google Colab  
Tesla P100 or T4 x2 \- 30h/week with Kaggle  
RTX 3080 GPU \- unlimited use

### Software Packages 

Training: PyTorch, torchvision, timm  
Data: HuggingFace datasets for ImageNet V2  
Metrics: scikit-learn   
Analysis and tracking: numpy, pandas, matplotlib/seaborn

## Measuring Success

The project would be considered a success if each stage can provide a clear, quantified answer. A negative result for the extension would also be considered a success, if it is well supported. 

- **Reproduction**: we confirm the results of the paper by presenting results that support the conclusions drawn by the authors.   
- **LLM reproduction**: a working implementation, plus a categorised list of missing or inconsistent details with their measured effect on results.   
- **Extension (Dynamic Duos)**: An adaptive weighting scheme should either improve NLL, AURC or AUROC over static Duos with statistical significance across random seeds, or we explain when and why they do not. 

## Skills we are hoping to gain

- Fine-tuning pre-trained models from model zoos   
- Measuring and reducing a model’s uncertainty in an efficient way   
- Effectively using an LLM to reproduce the paper accurately and receive the correct results   
- Learning the best ML practices (ablations, fair baseline, data splits)

