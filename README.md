# DIT5411-ChongKaHo

THEi **DIT5411 Machine Learning** assignment: Traditional Chinese handwritten character recognition with TensorFlow.

| Item | Value |
| --- | --- |
| Student | Chong Ka Ho |
| Classes | **13,065** |
| Framework | TensorFlow / Keras |
| Input size | 64×64 grayscale (RGB for transfer learning) |
| Train split | first **40** samples / character |
| Test split | remaining samples / character |
| Best model | **Model_3_MobileNetV2** |
| Best test accuracy | **0.0578 (5.78%)** |

## Dataset

Full Traditional Chinese handwriting dataset (not the 4,803 common-word subset):

- https://github.com/chenkenanalytic/handwritting_data_all  
- Project description / license: https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset  

Total **13,065** characters, **684,677** images (about 50 samples per character).  
Zip member names are Big5-encoded. This project maps folder IDs `1..13065` to labels `0..13064`.

## Method

### 1. Train / test split

For each character, images are sorted by the numeric suffix in the filename (`字_0.png`, `字_1.png`, …):

- **Training originals:** first 40 images  
- **Testing:** all remaining images  

### 2. Augmented image creation

Because only ~40 originals exist per class, training uses **online TensorFlow augmentation**:

- `RandomRotation` (rotation)  
- `RandomZoom` (scaling)  
- `RandomTranslation` (shear-like geometric distortion)  

Across multiple epochs with live augmentation, each character receives **≥ 200** augmented training views (40 originals × many epochs with random transforms).

Reference idea: https://github.com/darshitajain/Introduction-to-OpenCV

### 3. Models compared

All classification heads end with **Dense(13065, softmax)**.

| Model | Architecture (summary) | Test Acc |
| --- | --- | ---: |
| **Model_1_MLP** | Flatten → Dense(1024) → Dense(512) → Softmax(13065) | 0.0001 |
| **Model_2_Scratch_CNN** | 3× Conv+MaxPool (32/64/128) → Dense(512) → Softmax(13065) | 0.0001 |
| **Model_3_MobileNetV2** | ImageNet MobileNetV2 (frozen) → GAP → Dropout → Softmax(13065) | **0.0578** |

Training details: Adam optimizer, `sparse_categorical_crossentropy`, EarlyStopping on `val_loss` where applicable.

### 4. Diagnostic check

A from-scratch CNN trained on only **100 classes** reached **test accuracy = 0.483**, which confirms that the data loading / labeling pipeline is correct.  
The very low from-scratch accuracy on the full **13,065**-way task is due to task difficulty and limited GPU time, not a broken dataset.

## How to run

Main submission notebook:

- [`notebooks/DIT5411_ChongKaHo_Submission.ipynb`](notebooks/DIT5411_ChongKaHo_Submission.ipynb)

Also available in Downloads as `DIT5411_ChongKaHo_Submission.ipynb`.

1. Open the notebook in Google Colab.  
2. `Runtime → Change runtime type → GPU`.  
3. Run parts in order.  
4. Part 6A writes the recorded results used for this report (no long re-training required).  
5. Part 6B is optional if you want to reproduce MobileNet training.

Recorded metrics are also stored in [`outputs/results.json`](outputs/results.json).

## Results summary

```text
Model_1_MLP:            Test = 0.0001
Model_2_Scratch_CNN:    Test = 0.0001
Model_3_MobileNetV2:    Test = 0.0578   ← Best
```

**Best model:** Model_3_MobileNetV2 (Test Accuracy = **0.0578**)

## Project layout

```text
DIT5411-ChongKaHo/
├── notebooks/DIT5411_ChongKaHo_Submission.ipynb
├── outputs/results.json
├── src/                    # helper modules used during development
├── scripts/
├── requirements.txt
└── README.md
```

## References

1. AI.FREE Team — Traditional Chinese Handwriting Dataset  
   https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset  
2. Full 13,065-character data mirror  
   https://github.com/chenkenanalytic/handwritting_data_all  
3. OpenCV augmentation reference (assignment link)  
   https://github.com/darshitajain/Introduction-to-OpenCV  
4. Handwritten Digit Recognition using Keras and TensorFlow (README format reference)  
   https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow  
5. TensorFlow / Keras documentation  
   https://www.tensorflow.org/

## License note

Dataset license: **CC BY-NC-SA 4.0** (non-commercial).  
This coursework project is for academic use only.
