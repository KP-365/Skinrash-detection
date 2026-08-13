# Bite Classification - A Comparative Study of Transfer Learning Strategies for Insect Bite Classification

This repository contains the full implementation, trained models, and evaluation code for our 8-class insect bite image classifier, built using transfer learning on EfficientNet-B0 with Monte Carlo Dropout for uncertainty quantification.

**Classes:** ants, bed_bugs, chiggers, fleas, mosquitos, no_bites, spiders, ticks

---

## Where to find things

### Training notebooks

| File | What it contains |
|---|---|
| [`BugBites_NoAug.ipynb`](./BugBites_NoAug.ipynb) | Clean-only baseline model - trained on the original 896 raw images only, moderate augmentation. See Section 3.3 of the report ("Clean-only" strategy). |
| [`BugBitesAug.ipynb`](./BugBitesAug.ipynb) | Pooled/Augmented model training - includes the dataset merging and leakage safe 70/15/15 split (Section 3.1 and 5.1 of the report). |
| [`BugBitesAug2.ipynb`](./BugBitesAug2.ipynb) | Aggressive-Aug and Tuned (final) model training the split before augmenting methodology (live, on the fly augmentation only, Section 3.2) and the two phase training loop. This notebook produces `best_model_aggaug_final.pt` and, using the winning config from the grid search, `best_model_tuned_final.pt`. |
| [`bruteforcelol.ipynb`](./bruteforcelol.ipynb) | Hyperparameter grid search (Section 3.4 of the report) tests all combinations of dropout probability, fine-tune learning rate, and number of unfrozen blocks using short, capped epoch runs, ranked by validation accuracy. The winning configuration (dropout=0.3, fine-tune LR=1e-4, last 3 blocks unfrozen) was then retrained to full convergence in `BugBitesAug2.ipynb` to produce the final tuned model. |

### Evaluation

| File | What it contains |
|---|---|
| [`Eval/Eval.ipynb`](./Eval/Eval.ipynb) | Full evaluation suite for the final tuned model: test accuracy, MC Dropout uncertainty and predictive entropy, Expected Calibration Error and reliability diagram, per-class ROC-AUC, confusion matrix, and the precision/recall/F1 classification report. Corresponds to Section 5 (Experimentation and Evaluation) of the report. |
| [`GetTestImagesnEval.ipynb`](./GetTestImagesnEval.ipynb) | Loads and inspects the held-out clean test set (53 images) used for all final reported results, and the external validation attempts discussed in the report's limitations. |


### Trained models

All checkpoints are in [`models/trained_models/`](./models/trained_models/):

| Checkpoint | Strategy | Test accuracy |
|---|---|---|
| `best_model_clean_only_final.pt` | Clean-only baseline | 54.7% |
| `best_model_aggaug_final.pt` | Aggressive-Aug | 67.9% |
| `best_model_final.pt` | Pooled/Augmented (see note below) | 73.0% |
| `best_model_tuned_final.pt` | **Tuned (final) - reported model** | **75.5%** |

> **Note on `best_model_final.pt`:** this model's reported accuracy was measured on an expanded test set that included images drawn from the same pool used for training, meaning it carries a structural risk of train/test leakage not present in the other three checkpoints. See Section 5.1 of the report for the full explanation. `best_model_tuned_final.pt` is the model referenced throughout the Results and Analysis sections of the report.

### Report and figures

The full IEEE-format report (PDF) covers methodology, all four evaluation figures (confusion matrix, ROC curves, entropy histogram, reliability diagram), and the complete analysis. All figures referenced in the report are generated directly by `Eval/Eval.ipynb` and can be reproduced by running that notebook end to end.

---

## Reproducing the results

1. Open `BugBites_NoAug.ipynb` or `BugBitesAug.ipynb` in Google Colab
2. Run all cells in order - datasets are pulled automatically from Hugging Face (`eceunal/bug-bite-images-hf` and `eceunal/bug-bite-images-aug_v3`)
3. Trained checkpoints save automatically; best checkpoints (by validation accuracy) are retained
4. Open `Eval/Eval.ipynb`, point it at the checkpoint you want to evaluate, and run all cells to reproduce every reported metric and figure

## Dataset sources

- Base dataset: [`eceunal/bug-bite-images-hf`](https://huggingface.co/datasets/eceunal/bug-bite-images-hf) (1,055 images)
- Augmented dataset: [`eceunal/bug-bite-images-aug_v3`](https://huggingface.co/datasets/eceunal/bug-bite-images-aug_v3) (~8,300 images)
