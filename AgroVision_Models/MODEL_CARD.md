# AgroVision AI — Model Card

## 1. Model Overview

AgroVision AI uses crop-specific deep learning models for leaf disease classification.

The current model package contains four independently trained EfficientNet-B0 models:

- Apple_EfficientNetB0
- Corn_EfficientNetB0
- Tomato_EfficientNetB0
- Potato_EfficientNetB0

Each model is specialized for one crop and predicts only the disease classes
defined for that crop.

The application does not use a crop-classification model.
The user selects the crop, and the application routes the image to the corresponding
crop-specific disease classifier.

## 2. Model Architecture

Base architecture: EfficientNet-B0
Pretrained weights: ImageNet
Input resolution: 224 x 224
Input channels: 3 (RGB)
Pooling: Global Average Pooling
Dropout: 0.30
Output: Crop-specific softmax classification head

The models were trained using transfer learning followed by selective fine-tuning.
Batch Normalization layers were kept frozen during fine-tuning.

## 3. Training Configuration

Random seed: 42
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Primary training metric: Sparse Categorical Accuracy
Batch size: 32
Image size: 224 x 224

Training augmentation:
- Horizontal flip
- Rotation: 0.08
- Zoom: 0.10
- Contrast: 0.10

Augmentation was applied only during training.
Validation and test inference used deterministic preprocessing.

## 4. Dataset Integrity Procedure

Before model training, the datasets were processed through an integrity pipeline.

The pipeline included:

- Corrupted-image detection
- Exact duplicate detection using SHA-256
- Perceptual-hash near-duplicate detection
- Cross-label duplicate inspection
- Manual verification of near-duplicate candidates
- Removal of confirmed duplicate or conflicting samples
- Fixed train/validation/test split

The locked test sets were not used for model selection.

The final evaluation therefore represents performance on the predefined
held-out test sets rather than training accuracy.

## 5. Apple Model

### Model

File: Apple_EfficientNetB0.keras

Number of classes: 6

Class mapping:

0 -> Apple___Apple_scab
1 -> Apple___Black_rot
2 -> Apple___Cedar_apple_rust
3 -> Apple___Mosaic_virus
4 -> Apple___Rust
5 -> Apple___healthy

### Fine-tuning

The Apple model used staged transfer learning.
The final stage selectively unfroze the deeper EfficientNet layers while
keeping Batch Normalization layers frozen.

Final fine-tuning learning rate: 2e-6

### Locked Test Results

Test samples: 559
Accuracy: 92.67%
Balanced Accuracy: 87.24%
Macro Precision: 83.04%
Macro Recall: 87.24%
Macro F1: 84.36%
Weighted F1: 93.12%

### Apple Evaluation Notes

The Apple model shows strong performance on the healthy class and the larger
disease classes.

The smaller Mosaic virus and Rust classes have lower F1 scores, indicating
that class imbalance and visual similarity remain important limitations.

## 6. Corn Model

### Model

File: Corn_EfficientNetB0.keras

Number of classes: 4

Class mapping:

0 -> Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
1 -> Corn_(maize)___Common_rust_
2 -> Corn_(maize)___Northern_Leaf_Blight
3 -> Corn_(maize)___healthy

### Dataset Scope

The original dataset contained a Smut class.
The class was excluded from the final training set after visual inspection
showed that the images did not match the intended leaf-disease classification target.

### Fine-tuning

The Corn model used staged transfer learning followed by selective fine-tuning.
The deeper EfficientNet layers were unfrozen while Batch Normalization layers
remained frozen.

Final fine-tuning learning rate: 1e-5

### Locked Test Results

Test samples: 639
Accuracy: 96.09%
Balanced Accuracy: 94.82%
Macro Precision: 95.48%
Macro Recall: 94.82%
Macro F1: 95.12%
Weighted F1: 96.07%

### Corn Evaluation Notes

The final test performance is strong across the four retained classes.
However, the model is a closed-set classifier and can still produce a known
disease prediction for an image outside the supported distribution.

A separate non-target/OOD evaluation is therefore required before deployment.

## 7. Tomato Model

### Model

File: Tomato_EfficientNetB0.keras

Number of classes: 10

Class mapping:

0 -> Tomato_Bacterial_spot
1 -> Tomato_Early_blight
2 -> Tomato_Late_blight
3 -> Tomato_Leaf_Mold
4 -> Tomato_Septoria_leaf_spot
5 -> Tomato_Spider_mites_Two_spotted_spider_mite
6 -> Tomato__Target_Spot
7 -> Tomato__Tomato_YellowLeaf__Curl_Virus
8 -> Tomato__Tomato_mosaic_virus
9 -> Tomato_healthy

### Fine-tuning

The Tomato model used staged transfer learning followed by selective fine-tuning.
Batch Normalization layers remained frozen during fine-tuning.

Final recovery/fine-tuning learning rate: 1e-5

### Locked Test Results

Test samples: 2527
Accuracy: 94.90%
Balanced Accuracy: 94.32%
Macro Precision: 93.81%
Macro Recall: 94.32%
Macro F1: 94.04%
Weighted F1: 94.90%

### Tomato Evaluation Notes

The model performs strongly across the majority of disease classes.
The Early Blight and Mosaic Virus classes have comparatively lower F1 scores,
indicating remaining visual ambiguity and class imbalance.

## 8. Potato Model

### Model

File: Potato_EfficientNetB0.keras

Number of classes: 3

Class mapping:

0 -> Potato___Early_blight
1 -> Potato___Late_blight
2 -> Potato___healthy

### Fine-tuning

The Potato model used staged transfer learning followed by selective fine-tuning.
The deeper EfficientNet layers were unfrozen while Batch Normalization layers
remained frozen.

Final fine-tuning learning rate: 1e-5

### Locked Test Results

Test samples: 358
Accuracy: 96.65%
Balanced Accuracy: 97.61%
Macro Precision: 96.42%
Macro Recall: 97.61%
Macro F1: 97.00%
Weighted F1: 96.65%

### Potato Evaluation Notes

The Potato model performs strongly across all three retained classes.
The main observed confusion is between Early Blight and Late Blight.
The healthy class achieved full recall on the locked test set, although its
support is substantially smaller than the two disease classes.

## 9. Inference Pipeline

The intended inference pipeline is:

1. User uploads image.
2. User selects crop.
3. Validate image.
4. Perform image-quality checks including blur detection.
5. Convert image to RGB.
6. Resize to 224 x 224.
7. Convert to float32.
8. Route the image to the selected crop model.
9. Generate disease probabilities.
10. Apply confidence/uncertainty/OOD assessment.
11. Return prediction or abstain.

The crop selection happens before model inference.
There is no crop-classification stage.

## 10. Closed-Set Classification Limitation

These models are trained as closed-set classifiers.

A softmax classifier assigns probability values to the classes it knows.
Consequently, an unrelated image may still receive a high probability for one
of the supported classes.

For example, an image containing a non-target object can potentially be forced
into one of the known disease classes.

Therefore, softmax confidence alone should not be interpreted as a guarantee
that the diagnosis is valid.

## 11. OOD and Abstention Requirement

For production deployment, AgroVision AI should include an additional
confidence/OOD decision layer.

The system should be capable of returning:

Prediction accepted
OR
Prediction rejected / uncertain

The threshold should be determined experimentally using validation data and
a deliberately collected non-target/OOD dataset.

An arbitrary threshold such as 70% should not be treated as a scientifically
validated rejection criterion.

## 12. Generalization Limitations

Dataset performance does not guarantee equivalent performance in real-world
field conditions.

Potential distribution shifts include:

- Different lighting conditions
- Different camera devices
- Leaf orientation
- Background variation
- Occlusion
- Multiple leaves in one image
- Different growth stages
- Geographic variation
- Disease severity variation
- Images outside the supported crop/disease taxonomy

External field evaluation is therefore required before making high-stakes
agricultural recommendations.

## 13. Supported Scope

This package currently contains models for:

- Apple — 6 classes
- Corn/Maize — 4 classes
- Tomato — 10 classes
- Potato — 3 classes

Total: 4 independently trained crop-specific models covering 23 disease/health
classes.

The broader AgroVision taxonomy also includes Grape and other planned crops,
but those models are not included in this package yet.

## 14. Intended Use

The models are intended for research and project-level crop leaf disease
classification within AgroVision AI.

The predictions should be treated as AI-assisted outputs rather than definitive
agricultural diagnoses.

## 15. Model Artifacts

The package contains:

- Trained Keras model files
- Class mapping JSON files
- Evaluation summary JSON files
- README documentation
- This model card

The class mapping files must be used when converting model output indices
into disease labels.

## 16. Reproducibility

The training pipeline used a fixed seed of 42 and deterministic dataset manifests
for train, validation, and test splits.

The exact training environment, dataset snapshot, preprocessing pipeline,
and training configuration should be preserved alongside future model versions
to allow reproducibility and meaningful comparison.

## 17. Version

AgroVision AI Disease Classification Models
Model Card Version: 1.1
Model Package Version: 1.1
