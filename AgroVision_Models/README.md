# AgroVision AI — Disease Classification Models

## 1. Overview

This package contains the finalized crop-specific disease classification models for AgroVision AI.

The application uses separate disease classification models for each supported crop.
The user selects the crop from a predefined list. The selected crop determines which
crop-specific EfficientNet-B0 model is used for disease classification.

The models perform disease classification only.
There is NO crop classification model in this package.

Supported models:
- Apple — EfficientNet-B0
- Corn/Maize — EfficientNet-B0
- Tomato — EfficientNet-B0
- Potato — EfficientNet-B0

## 2. Application Inference Pipeline

The intended application pipeline is:

1. User uploads a leaf image.
2. User selects the crop from the predefined crop list.
3. The uploaded image goes through image quality and validity checks.
4. Image preprocessing is performed.
5. The application selects the corresponding crop-specific EfficientNet-B0 model.
6. The model predicts disease probabilities.
7. A confidence/uncertainty or OOD decision is applied.
8. The application returns a disease result or abstains when the prediction is unreliable.

Conceptually:

User upload
-> Crop selection
-> Image quality and validity checks
-> RGB conversion
-> Resize to 224 x 224
-> Convert to float32
-> Select crop-specific model
-> Disease classification
-> Confidence / OOD assessment
-> Result or abstention

There is no automatic crop classification step.
The crop selected by the user determines the model used.

## 3. Image Quality and Preprocessing

Quality checks are performed before model inference.

The application should check the uploaded image for:
- Image validity
- Corruption
- Excessive blur
- Other quality conditions defined by the application

After passing the quality checks:
- Convert image to RGB.
- Resize to 224 x 224 pixels.
- Convert to float32.
- Keep pixel values in the 0-255 range.

The Keras EfficientNet implementation handles its internal preprocessing.
Do not apply an additional external ImageNet normalization step unless it has
been explicitly validated against the trained models.

Random training augmentations must NOT be applied during normal inference.

## 4. Model Routing

The crop selection determines the model:

Apple -> Apple_EfficientNetB0.keras
Corn/Maize -> Corn_EfficientNetB0.keras
Tomato -> Tomato_EfficientNetB0.keras
Potato -> Potato_EfficientNetB0.keras

An image selected as Apple must be sent to the Apple model.
An image selected as Corn/Maize must be sent to the Corn model.
An image selected as Tomato must be sent to the Tomato model.
An image selected as Potato must be sent to the Potato model.

## 5. Apple Model

File: models/Apple_EfficientNetB0.keras

Classes:
0 -> Apple___Apple_scab
1 -> Apple___Black_rot
2 -> Apple___Cedar_apple_rust
3 -> Apple___Mosaic_virus
4 -> Apple___Rust
5 -> Apple___healthy

Locked test performance:
- Accuracy: 92.67%
- Balanced Accuracy: 87.24%
- Macro Precision: 83.04%
- Macro Recall: 87.24%
- Macro F1: 84.36%
- Weighted F1: 93.12%
- Test samples: 559

## 6. Corn Model

File: models/Corn_EfficientNetB0.keras

Classes:
0 -> Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
1 -> Corn_(maize)___Common_rust_
2 -> Corn_(maize)___Northern_Leaf_Blight
3 -> Corn_(maize)___healthy

Locked test performance:
- Accuracy: 96.09%
- Balanced Accuracy: 94.82%
- Macro Precision: 95.48%
- Macro Recall: 94.82%
- Macro F1: 95.12%
- Weighted F1: 96.07%
- Test samples: 639

The original Corn dataset contained a Smut class.
It was excluded from the final model because inspected images did not
represent the intended leaf-disease classification target.

## 7. Tomato Model

File: models/Tomato_EfficientNetB0.keras

Classes:
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

Locked test performance:
- Accuracy: 94.90%
- Balanced Accuracy: 94.32%
- Macro Precision: 93.81%
- Macro Recall: 94.32%
- Macro F1: 94.04%
- Weighted F1: 94.90%
- Test samples: 2527

## 8. Potato Model

File: models/Potato_EfficientNetB0.keras

Classes:
0 -> Potato___Early_blight
1 -> Potato___Late_blight
2 -> Potato___healthy

Locked test performance:
- Accuracy: 96.65%
- Balanced Accuracy: 97.61%
- Macro Precision: 96.42%
- Macro Recall: 97.61%
- Macro F1: 97.00%
- Weighted F1: 96.65%
- Test samples: 358

## 9. Model Architecture

All four models use:
- EfficientNet-B0
- ImageNet pretrained weights
- 224 x 224 RGB input
- Global Average Pooling
- Dropout
- Crop-specific softmax classification head

The models were trained using staged transfer learning and fine-tuning.
Batch Normalization layers were kept frozen during fine-tuning.

## 10. Dataset and Training Integrity

The datasets underwent integrity and duplicate checks before training.

Checks included:
- Corrupted-image detection
- SHA-256 exact duplicate detection
- Perceptual-hash near-duplicate detection
- Cross-label duplicate inspection
- Manual verification of near-duplicate candidates
- Frozen train/validation/test splits

The locked test set was kept separate from model selection.

Training seed: 42

Training augmentation included:
- Horizontal flip
- Random rotation: 0.08
- Random zoom: 0.10
- Random contrast: 0.10

These random augmentations are training-time operations and are not part
of normal inference.

## 11. OOD and Abstention

A standard softmax classifier always produces probabilities for its known
classes, even when an input image is outside the training distribution.

Therefore, the highest softmax probability must not automatically be treated
as a reliable diagnosis.

The production application should include a confidence and/or OOD decision layer.

Conceptually:

Disease probabilities
-> Confidence/OOD assessment
-> Reliable result OR abstention

The abstention threshold should be selected using validation data and deliberately
collected non-target/OOD examples.

Do not use an arbitrary confidence threshold without validation.

## 12. Example Inference

Example using the Apple model:

```python
import numpy as np
from PIL import Image
from tensorflow import keras

model = keras.models.load_model(
    "models/Apple_EfficientNetB0.keras"
)

image = Image.open("leaf.jpg").convert("RGB")
image = image.resize((224, 224))

x = np.asarray(image, dtype=np.float32)
x = np.expand_dims(x, axis=0)

probabilities = model.predict(x, verbose=0)[0]
predicted_class = int(np.argmax(probabilities))
confidence = float(np.max(probabilities))

print("Class:", predicted_class)
print("Confidence:", confidence)
```

Use the corresponding class mapping JSON to convert the class index
into the disease name.

## 13. Package Contents

AgroVision_Models/
├── README.md
├── MODEL_CARD.md
├── models/
│   ├── Apple_EfficientNetB0.keras
│   ├── Corn_EfficientNetB0.keras
│   ├── Tomato_EfficientNetB0.keras
│   └── Potato_EfficientNetB0.keras
├── class_mappings/
│   ├── apple_class_mapping.json
│   ├── corn_class_mapping.json
│   ├── tomato_class_mapping.json
│   └── potato_class_mapping.json
└── evaluation/
    ├── apple_metrics.json
    ├── corn_metrics.json
    ├── tomato_metrics.json
    └── potato_metrics.json

## 14. Intended Use

These models are intended for the AgroVision AI research/project
application for crop leaf disease classification.

The predictions should be presented as AI predictions with appropriate
confidence and uncertainty handling.

## 15. Version

AgroVision AI — EfficientNet-B0 Disease Models
Version: 1.1

Current trained model scope:
- Apple — 6 classes
- Corn/Maize — 4 classes
- Tomato — 10 classes
- Potato — 3 classes

Total: 4 crop-specific models and 23 disease/health classes.

Grape and other crops in the broader AgroVision taxonomy are not included
in this model package yet.
