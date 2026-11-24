# Bone Age Prediction from Hand X-rays (RSNA)

Deep learning model that estimates skeletal bone age (in months) from a paediatric hand/wrist radiograph and the child's sex.

> **Research project only. Not clinically validated. Not for diagnosis.**

## Results (held-out test set, n = 1,262)

| Metric | Value |
|---|---|
| Baseline MAE (always predict the mean age) | 33.6 months |
| **Test MAE** | **6.33 months** |
| **Test MAE with test-time augmentation (TTA)** | **6.21 months** |
| Predictions within 12 months of the true age | 88% |
| MAE, girls / boys (TTA) | 6.46 / 5.99 months |

Error by true bone-age bracket (TTA):

| Bracket (months) | n | MAE (months) |
|---|---|---|
| 0-36 | 38 | 8.34 |
| 36-72 | 123 | 6.32 |
| 72-108 | 259 | 7.17 |
| 108-144 | 362 | 6.98 |
| 144-180 | 412 | 4.85 |
| 180-240 | 68 | 5.28 |

![Predicted vs actual](pred_vs_actual.png)

## Data
- RSNA Pediatric Bone Age dataset (Kaggle: `kmader/rsna-bone-age`), 12,611 labelled radiographs with bone age in months and sex.
- Random 80/10/10 train/validation/test split (10,088 / 1,261 / 1,262), stratified by sex and 2-year age bin. Seed 42.

## Method
- **Preprocessing:** pad to square, resize, grayscale replicated to 3 channels, ImageNet normalisation.
- **Augmentation (train only):** random affine (±10° rotation, 5% shift, 0.9-1.1 scale) and brightness/contrast jitter.
- **Model:** ImageNet-pretrained ResNet-50 backbone; the child's sex is embedded (Linear 1→32) and concatenated with the 2048-d image features, followed by an MLP head (256 units, dropout 0.3) that outputs age.
- **Target and loss:** age standardised with training-set mean/std; L1 loss (directly optimises MAE).
- **Training:** AdamW, one-cycle LR schedule, mixed precision. Stage 1 at 256px (12 epochs, lr 3e-4), then fine-tune at 512px (8 epochs, lr 1e-4). Best checkpoint by validation MAE.
- **Inference:** TTA averaging 6 views (original, horizontal flip, ±5° rotation, ±10px shift).

Validation MAE was 6.86 months at 256px and 6.44 months at 512px.

## Limitations
- Single dataset with no external validation; performance on other scanners, populations or hospitals is unknown.
- Labels are radiologist assessments, which carry their own inter-observer variability.
- Highest error in infants (very few examples) and in the 72-144 month range, where carpal bones appear and change quickly.
- Girls score slightly worse than boys (about 0.5 months).
- Not clinically validated; not a medical device.

## Run
```bash
pip install torch torchvision gradio pillow
# put stage2_512.pt next to app.py and set AGE_MEAN / AGE_STD in app.py
python app.py
```

`AGE_MEAN` and `AGE_STD` are the training-split mean and standard deviation of bone age, printed in the training notebook.

## Files
- `bone_age_notebook.ipynb`: full training and evaluation pipeline
- `app.py`: Gradio prediction interface
- `pred_vs_actual.png`: predicted-vs-actual and error plots

## Acknowledgements
Dataset: RSNA Pediatric Bone Age Machine Learning Challenge (Halabi et al., 2019).
