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
| Best model | **Model_3_MobileNetV2 (fine-tuned)** |
| Best test accuracy | **0.4902 (49.02%)** |

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
| Model_1_MLP | Flatten → Dense(1024) → Dense(512) → Softmax(13065) | 0.0001 |
| Model_2_Scratch_CNN | 3× Conv+MaxPool (32/64/128) → Dense(512) → Softmax(13065) | 0.0001 |
| Model_3_MobileNetV2 (frozen) | ImageNet MobileNetV2 (frozen) → GAP → Dropout → Softmax(13065) | 0.0578 |
| Model_3_MobileNetV2 (phase 1) | Frozen backbone, head-only training | 0.0315 |
| **Model_3_MobileNetV2 (fine-tuned)** | Unfreeze last ~30 layers + small LR | **0.4902** |

Training details: Adam optimizer, `sparse_categorical_crossentropy`, EarlyStopping / ReduceLROnPlateau on `val_loss`.

**Fine-tune strategy**

1. **Phase 1:** freeze MobileNetV2 backbone, train classification head  
2. **Phase 2:** unfreeze the last ~30 backbone layers and continue with a lower learning rate (`1e-4`)

### 4. Diagnostic check

A from-scratch CNN trained on only **100 classes** reached **test accuracy = 0.483**, which confirms that the data loading / labeling pipeline is correct.  
Baselines on the full **13,065**-way task were weak without transfer learning + fine-tuning.

## How to run

Recommended all-in-one notebook (full pipeline + fine-tune):

- [`notebooks/DIT5411_Full_Plus_FineTune.ipynb`](notebooks/DIT5411_Full_Plus_FineTune.ipynb)

Also available:

- [`notebooks/DIT5411_ChongKaHo_Submission.ipynb`](notebooks/DIT5411_ChongKaHo_Submission.ipynb) — baseline report notebook  
- `DIT5411_Full_Plus_FineTune.ipynb` in Downloads  

1. Open the notebook in Google Colab.  
2. `Runtime → Change runtime type → GPU`.  
3. Run Parts 1–7 for setup / recorded baselines.  
4. Run Part 8 to reproduce fine-tuning (needs GPU; longer runtime).  

Recorded metrics:

- [`outputs/results.json`](outputs/results.json)  
- [`outputs/results_finetune.json`](outputs/results_finetune.json)

## Results summary

```text
Model_1_MLP:                         Test = 0.0001
Model_2_Scratch_CNN:                 Test = 0.0001
Model_3_MobileNetV2 (frozen):        Test = 0.0578
Model_3_MobileNetV2 (phase 1):       Test = 0.0315
Model_3_MobileNetV2 (fine-tuned):    Test = 0.4902   ← Best
```

**Best model:** Model_3_MobileNetV2 fine-tuned (Test Accuracy = **0.4902**)

## Project layout

```text
DIT5411-ChongKaHo/
├── notebooks/
│   ├── DIT5411_Full_Plus_FineTune.ipynb
│   └── DIT5411_ChongKaHo_Submission.ipynb
├── outputs/
│   ├── results.json
│   └── results_finetune.json
├── src/
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
