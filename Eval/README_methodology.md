# Bug Bite Classifier — Methodology & Evaluation

## Overview

This project builds an 8-class image classifier to identify the likely cause of a bug bite from a photograph, using transfer learning on EfficientNet-B0 with Monte Carlo Dropout for uncertainty quantification. The task is framed around risk-stratified triage rather than diagnosis: the goal is to support a "self-care vs. seek medical advice" decision, not to replace clinical judgement.

**Classes:** `ants`, `bed_bugs`, `chiggers`, `fleas`, `mosquitos`, `no_bites`, `spiders`, `ticks`

## Dataset

The base dataset (`eceunal/bug-bite-images-hf` on Hugging Face) contains 1,055 labelled images pre-split into train (896), validation (106), and test (53). Class balance is close to even across all splits (roughly 10–14% per class).

An augmented derivative of this dataset (`bug-bite-images-aug_v3`, ~8,300 images) was also explored. Early experiments pooled this augmented data with the clean data to build a larger 70/15/15 split. This approach was ultimately abandoned in favour of a cleaner methodology (below), because pooling pre-generated augmented images risked **train/validation/test leakage**: augmented variants of the same source photo could end up split across different sets, since augmentation lineage is not traceable in the source dataset. Where a diluted/pooled test set was used in early experiments, this is noted explicitly as a limitation of that specific result.

## Final Methodology

The final, reported model uses the following pipeline, chosen specifically to guarantee no leakage:

1. **Data is split before any augmentation is generated.** Only the original 896/106/53 raw photos are used, in their original split.
2. **Augmentation is applied live, on-the-fly, during training only**, via `torchvision.transforms`. Every epoch generates a fresh, randomly-augmented version of each training image; nothing is pre-baked or saved to disk. Validation and test images are never augmented.
3. **Augmentation pipeline** (train only): horizontal/vertical flip, rotation, affine transform (translate/scale/shear), colour jitter, random resized crop, autocontrast, sharpness adjustment, Gaussian blur, and random erasing (Cutout-style occlusion).

### Model & Training

- **Backbone:** EfficientNet-B0 (ImageNet-pretrained), chosen over larger architectures (e.g. ResNet50/101) given the small dataset size (~900 training images) and consequent overfitting risk with higher-capacity models.
- **Two-phase training:**
  - *Phase 1:* backbone fully frozen; only the classification head is trained.
  - *Phase 2:* the final 3 feature blocks are unfrozen and fine-tuned at a lower learning rate.
- **Uncertainty quantification:** MC Dropout (p=0.3) retained active at inference; 20–30 stochastic forward passes per prediction are averaged to produce a mean prediction and an uncertainty estimate.
- **Early stopping** on validation loss (patience = 7) in both phases; the checkpoint with the best validation accuracy is retained, not the final epoch.
- **Hyperparameter search:** a constrained grid search over dropout (0.3/0.4/0.5), fine-tune learning rate (1e-5/3e-5/1e-4), and number of unfrozen blocks (1 vs 3) was run using short, capped-epoch training runs to rank configurations. The best-ranked configuration (dropout=0.3, fine-tune LR=1e-4, last 3 blocks unfrozen) was then retrained to full convergence to produce the final reported model.

### Model Variants Compared

| Variant | Training data | Notes |
|---|---|---|
| Clean-only | 896 raw images, moderate augmentation | Baseline |
| Pooled/Augmented | ~6,800 pooled images (raw + pre-augmented) | Early approach; some leakage risk in its evaluation |
| Aggressive-Aug | 896 raw images, aggressive live augmentation, deeper fine-tuning | Improves on clean-only baseline |
| **Tuned (final)** | 896 raw images, aggressive live augmentation, tuned hyperparameters | Best result; reported below |

## Final Model Results

Evaluated on the clean, held-out, leakage-free 53-image test set.

**Test accuracy: 75.47%**

### Uncertainty Quantification (MC Dropout, 30 samples)

| Metric | Correct predictions | Incorrect predictions |
|---|---|---|
| Mean predictive entropy | 0.6832 | 0.8610 |

The model is measurably less certain when it is wrong than when it is right, supporting its use in a confidence-tiered triage setting (e.g. flagging low-confidence predictions for human review rather than acting on them directly).

### Calibration

**Expected Calibration Error (ECE): 0.0988**

The reliability diagram shows the model is reasonably well-calibrated in the mid-to-high confidence range, with a tendency toward mild overconfidence in some bins. An ECE below 0.1 is generally considered acceptable for a model of this scale and dataset size.

### Discriminative Power (ROC-AUC, one-vs-rest)

| Class | AUC |
|---|---|
| mosquitos | 1.000 |
| ants | 0.993 |
| spiders | 0.987 |
| bed_bugs | 0.982 |
| fleas | 0.980 |
| no_bites | 0.944 |
| ticks | 0.848 |
| chiggers | 0.801 |

**Mean AUC: 0.9421**

Chiggers and ticks show the weakest discriminative power, consistent with the confusion matrix below, where chiggers are most often confused with bed_bugs and no_bites, and ticks are most often confused with spiders and no_bites.

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| ants | 1.00 | 0.90 | 0.95 | 10 |
| bed_bugs | 0.67 | 1.00 | 0.80 | 6 |
| chiggers | 1.00 | 0.43 | 0.60 | 7 |
| fleas | 0.67 | 1.00 | 0.80 | 2 |
| mosquitos | 1.00 | 1.00 | 1.00 | 4 |
| no_bites | 0.58 | 0.88 | 0.70 | 8 |
| spiders | 0.86 | 0.67 | 0.75 | 9 |
| ticks | 0.50 | 0.43 | 0.46 | 7 |
| **Accuracy** | | | **0.75** | **53** |
| **Macro avg** | 0.78 | 0.79 | 0.76 | 53 |
| **Weighted avg** | 0.80 | 0.75 | 0.75 | 53 |

## Limitations

- **Test set size.** The clean, leakage-free test set contains only 53 images (2–10 per class). This is a deliberate trade-off: a larger pooled test set was available but carried leakage risk, so a smaller, guaranteed-clean set was prioritised. Per-class metrics, particularly for classes with single-digit support (e.g. fleas, mosquitos), should be treated as indicative rather than statistically robust. Overall accuracy is comparatively more stable at this sample size than the per-class breakdown.
- **External validation.** Multiple external, out-of-distribution datasets were investigated (Wikimedia Commons, Openverse, Kaggle, Roboflow) to test generalisation beyond the source dataset's visual characteristics. Each had a specific, documented limitation: insufficient volume for bite-specific classes, licensing/access restrictions, a non-matching class taxonomy, or the presence of synthetic augmentation artefacts (e.g. Cutout occlusion) within the "test" images themselves, making a fair comparison impossible with the available data. Where an external comparison was attempted regardless, accuracy dropped substantially (e.g. one available external set produced ~34% accuracy), indicating the model's true real-world generalisation is likely materially lower than its in-distribution test accuracy. This is reported as an open limitation rather than resolved.
- **Dataset skin-tone diversity.** As with dermatology/skin-condition datasets generally, the source dataset was not audited for skin-tone representativeness. This is a known, unaddressed limitation given project scope and time constraints, and would be a priority for any future extension of this work.

## Future Work

- K-fold cross-validation across the full dataset to obtain a statistically robust performance estimate given the small test set size.
- A properly sourced, class-matched, artefact-free external validation set to reliably quantify out-of-distribution generalisation.
- Skin-tone-stratified evaluation, contingent on either sourcing more diverse data or auditing the existing dataset.
