<div align="center">

<h1>Human Activity Recognition</h1>
<h3>Hybrid TCN-Transformer Architecture on MHEALTH Wearable Sensor Data</h3>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Architecture-TCN%20%2B%20Transformer-7B2FBE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Digital%20Health-0A9396?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-2D6A4F?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Dataset-MHEALTH%20Benchmark-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Activities-12%20Classes-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Sensors-Accelerometer%20%7C%20Gyroscope%20%7C%20Magnetometer-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Inference-Deployment%20Ready-red?style=flat-square"/>
</p>

<br/>

> **A production-grade deep learning system for multiclass human activity recognition from multimodal wearable sensor streams, built on a custom hybrid architecture combining Temporal Convolutional Networks and Transformer self-attention — designed with deployment-realistic constraints from the ground up.**

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why This Problem Matters](#why-this-problem-matters)
- [Technical Highlights](#technical-highlights)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Dataset](#dataset)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Training Configuration](#training-configuration)
- [Evaluation Framework](#evaluation-framework)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Deployment-Ready Inference](#deployment-ready-inference)
- [Technology Stack](#technology-stack)
- [Engineering Principles](#engineering-principles)
- [Potential Extensions](#potential-extensions)
- [Author](#author)

---

## Project Overview

This project builds a fully production-grade **Human Activity Recognition (HAR)** system that classifies 12 distinct physical activities from raw multimodal inertial sensor data using a novel hybrid deep learning architecture. The model fuses the local temporal feature extraction power of **Temporal Convolutional Networks (TCN)** with the global sequence modeling capacity of **Transformer self-attention** into a unified two-stage classifier — purpose-built for the unique challenges of wearable sensor time-series.

The system is trained and evaluated on the **MHEALTH (Mobile Health) benchmark dataset**, which provides synchronized accelerometer, gyroscope, magnetometer, and ECG readings from body-worn sensors across multiple placement sites. Every component of the pipeline — preprocessing, segmentation, normalization, model design, training strategy, and artifact serialization — reflects real deployment constraints, making this a complete solution ready for integration into wearable health platforms, rehabilitation systems, or real-time activity monitoring pipelines.

---

## Why This Problem Matters

Automatic recognition of human physical activities from wearable sensors sits at the intersection of several high-impact application domains.

**Digital Health and Remote Patient Monitoring** — Continuous, objective activity data is increasingly central to chronic disease management, post-surgical rehabilitation tracking, fall detection in elderly populations, and mental health correlates of physical behavior. Replacing manual patient reporting with automated sensor-based recognition dramatically improves data quality and clinical utility.

**Sports Science and Athletic Performance** — Granular activity classification enables real-time biomechanical feedback, fatigue monitoring, injury risk detection, and training load quantification for athletes — applications where model latency and label granularity directly translate to competitive and medical value.

**Ambient Assisted Living** — Recognizing activities of daily living from unobtrusive wearables supports aging-in-place systems that can trigger interventions when anomalous or absent activity patterns indicate health events.

**Industrial Safety and Ergonomics** — Activity-aware systems deployed in workplace wearables can detect unsafe postures, repetitive strain patterns, or proximity events in manufacturing, construction, and logistics environments.

Despite the breadth of these applications, HAR from raw inertial data remains technically demanding: sensor signals are noisy, activities produce overlapping frequency signatures, recording durations create severe class imbalance, and inference must often run under tight latency and memory budgets. This project addresses each of these challenges with targeted architectural and training decisions.

---

## Technical Highlights

**Architecture**
- Custom two-stage hybrid model: TCN backbone for causal local feature extraction, Transformer encoder head for global sequence attention
- Dilated causal convolutions with exponentially increasing dilation rates for multi-scale receptive field without future information leakage
- Sinusoidal positional encoding preserving temporal order prior to multi-head self-attention
- Residual connections with pointwise projection across TCN blocks for stable gradient flow through deep stacks
- Global average pooling over the attended sequence for permutation-invariant activity embedding

**Data Engineering**
- Sliding window segmentation converting continuous sensor streams into fixed-length classification inputs
- Per-channel `StandardScaler` normalization fitted exclusively on the training partition, applied consistently at inference
- Null-class removal and null-row exclusion for clean label semantics
- Stratified train / validation / test split preserving class proportions across all partitions

**Training Robustness**
- Class-weighted sparse categorical cross-entropy addressing MHEALTH's uneven activity distribution
- Early stopping on validation accuracy with full state restoration of the best epoch
- Dual `ModelCheckpoint` strategy persisting both best-epoch and final-epoch model states
- Fixed global random seed propagated across all stochastic components for fully reproducible training

**Deployment Readiness**
- All preprocessing artifacts serialized: `feature_scaler.pkl`, `label_to_idx.pkl`, `idx_to_label.pkl`
- Models saved in native Keras format for direct loading without re-instantiation
- Inference wrapper requires only three artifact loads — no training code dependency at prediction time
- Test predictions exported to `test_predictions.csv` for auditable downstream review

---

## Architecture Deep Dive

### Conceptual Data Flow

```
Raw Sensor Stream  (continuous, multi-channel, multi-placement)
        |
        v
+-----------------------------------------------+
|           PREPROCESSING STAGE                 |
|  - Null-class and null-row removal            |
|  - Per-channel StandardScaler fit             |
|  - Sliding window segmentation                |
|  - Stratified split                           |
+-------------------+---------------------------+
                    |
                    v   Shape: (batch, window_len, num_channels)
+---------------------------------------------------------------+
|               STAGE 1 -- TCN BACKBONE                        |
|                                                               |
|  Block 1: Conv1D(dilation=1) -> WeightNorm -> ReLU -> Drop   |
|       + Residual projection (if channels differ)              |
|  Block 2: Conv1D(dilation=2) -> WeightNorm -> ReLU -> Drop   |
|       + Residual                                              |
|  Block 3: Conv1D(dilation=4) -> WeightNorm -> ReLU -> Drop   |
|       + Residual                                              |
|  Block N: Conv1D(dilation=2^N) -> ...                        |
|                                                               |
|  Receptive field grows exponentially -- captures both         |
|  fine-grained motion transients and coarse activity rhythms   |
+------------------------+--------------------------------------+
                         |
                         v   Shape: (batch, window_len, tcn_filters)
             + Sinusoidal Positional Encoding
                         |
                         v
+---------------------------------------------------------------+
|           STAGE 2 -- TRANSFORMER ENCODER HEAD                |
|                                                               |
|  Multi-Head Self-Attention                                    |
|  (every timestep attends to all others simultaneously)        |
|       + Add & LayerNorm                                       |
|                                                               |
|  Position-Wise Feed-Forward Network                           |
|  (Dense -> ReLU -> Dense per timestep)                        |
|       + Add & LayerNorm                                       |
|                                                               |
|  Global Average Pooling                                       |
|  (collapses sequence -> fixed activity embedding)             |
+------------------------+--------------------------------------+
                         |
                         v   Shape: (batch, embedding_dim)
+-------------------------------------+
|      CLASSIFICATION HEAD            |
|  Dense -> Dropout -> Dense(Softmax) |
+-------------------------------------+
                         |
                         v
               Predicted Activity Label
```

### Why TCN First, Transformer Second?

This ordering is a deliberate inductive bias decision. Raw sensor signals contain rich local structure — impact transients, oscillation cycles, directional impulses — that benefits from the local receptive field of convolutions before any global attention is computed. Applying Transformer attention directly to raw sensor channels would force the model to learn local structure purely from pairwise position interactions, which is sample-inefficient.

By running the TCN first, the Transformer receives a sequence of already-meaningful local feature vectors rather than raw sensor readings. The attention mechanism then performs its intended function: discovering which temporal regions within the window are most relevant to the activity being classified — without needing to re-derive local motion primitives from scratch.

### Why Not a Pure Transformer?

Vanilla Transformers are quadratic in sequence length and lack inductive bias for local temporal structure. For sensor-length windows (typically 128–512 timesteps at 50 Hz), a pure Transformer must learn locality from data alone, requiring substantially more training examples to converge. The TCN backbone provides this locality inductive bias efficiently, making the hybrid more sample-efficient on the relatively compact MHEALTH dataset.

### Why Not a Pure TCN?

TCNs are inherently local — each output depends only on a fixed receptive field of past inputs. While stacking dilated layers extends this receptive field exponentially, a TCN has no native mechanism for a position to directly reference a distant position without passing through all intermediate layers. For activities with long-range temporal dependencies (e.g., the full arc of a jumping cycle), direct position-to-position attention provides a shortcut that pure TCNs cannot replicate efficiently.

---

## Dataset

### MHEALTH (Mobile Health) Benchmark

The **MHEALTH dataset** was collected from 10 volunteers performing 12 distinct physical activities while wearing three inertial measurement units simultaneously — on the chest, right wrist, and left ankle. Sensor readings were recorded at 50 Hz, providing high temporal resolution for activity dynamics. The dataset is fully included in this repository as `mhealth_raw_data.zip` and requires no external download or account.

### Sensor Configuration

| Sensor Type | Placement Site | Channels | Physical Unit |
|---|---|---|---|
| Accelerometer | Chest | X, Y, Z | m/s² |
| ECG | Chest | Lead I, Lead II | mV |
| Accelerometer | Left ankle | X, Y, Z | m/s² |
| Gyroscope | Left ankle | X, Y, Z | deg/s |
| Magnetometer | Left ankle | X, Y, Z | local |
| Accelerometer | Right wrist | X, Y, Z | m/s² |
| Gyroscope | Right wrist | X, Y, Z | deg/s |
| Magnetometer | Right wrist | X, Y, Z | local |

**Total sensor channels: 23** (across all placement sites and modalities)

### Activity Classes

| Label | Activity | Biomechanical Complexity |
|---|---|---|
| 1 | Standing still | Low — static posture |
| 2 | Sitting and relaxing | Low — static posture |
| 3 | Lying down | Low — static posture |
| 4 | Walking | Medium — periodic gait |
| 5 | Climbing stairs | Medium — periodic, asymmetric gait |
| 6 | Waist bends forward | Medium — single-axis trunk flexion |
| 7 | Frontal elevation of arms | Medium — upper limb gesture |
| 8 | Knees bending (crouching) | High — full-body compound movement |
| 9 | Cycling | High — bilateral periodic lower-limb motion |
| 10 | Jogging | High — high-impact periodic full-body |
| 11 | Running | High — high-impact, higher cadence |
| 12 | Jump front and back | High — ballistic, impact-heavy |

The null activity class (label 0, representing unlabeled or transition periods) is excluded before training to ensure clean label semantics. Low-variance static posture classes and high-variance dynamic classes coexist in the label space, creating a challenging multi-scale classification problem.

### Dataset Characteristics

| Property | Value |
|---|---|
| Sampling Rate | 50 Hz across all modalities |
| Subjects | 10 volunteers |
| Activity Classes | 12 (after null-class exclusion) |
| Total Channels | 23 sensor features |
| Recording Environment | Controlled laboratory with defined activity protocols |
| Availability | Fully self-contained — no external download required |

---

## End-to-End Pipeline

```
Step 1   Extract mhealth_raw_data.zip and load raw sensor DataFrame

Step 2   Data Cleaning
         |- Remove null-class rows (label == 0)
         +- Drop rows with any missing sensor readings

Step 3   Feature Normalization
         |- Fit StandardScaler on training channels only
         |- Transform train, validation, and test partitions
         +- Serialize scaler to feature_scaler.pkl

Step 4   Sliding Window Segmentation
         |- Apply fixed-length window with configurable stride
         +- Produce tensors of shape (N_windows, window_len, num_channels)

Step 5   Label Encoding and Splitting
         |- Map activity strings to integer indices
         |- Serialize label_to_idx.pkl and idx_to_label.pkl
         +- Stratified train / validation / test split

Step 6   Class Weight Computation
         +- Compute inverse-frequency weights from training labels

Step 7   Model Construction
         |- Build TCN backbone (N blocks, exponential dilation)
         |- Add sinusoidal positional encoding
         |- Attach Transformer encoder head (multi-head attention + FFN)
         +- Attach dense classification head (softmax output)

Step 8   Training
         |- Optimizer: Adam with tuned learning rate
         |- Loss: class-weighted sparse categorical cross-entropy
         |- Callbacks: EarlyStopping + ModelCheckpoint (best_model.keras)
         +- Save final state to final_transformer_model.keras

Step 9   Evaluation
         |- Classification report (precision, recall, F1 per class)
         |- Macro F1, weighted F1, overall accuracy
         +- Confusion matrix heatmap

Step 10  Export
         +- test_predictions.csv (predicted label + true label per window)
```

---

## Machine Learning Methodology

### Sliding Window Segmentation

Wearable sensor data is inherently a continuous stream — there are no natural sample boundaries in the raw signal. Sliding window segmentation converts this stream into a supervised learning problem by treating each fixed-length temporal window as a single training example.

The window length controls the amount of temporal context visible to the model per inference step. A 50 Hz sensor stream with a 128-sample window provides approximately 2.6 seconds of motion context, sufficient to capture multiple gait cycles, jump phases, or upper-limb gesture arcs. Configurable stride controls the density of the sliding window and the degree of temporal overlap between consecutive samples, trading training set size against data redundancy.

### Dilated Causal Convolution

Standard 1D convolutions on sensor sequences require causal padding to prevent any output timestep from depending on future inputs — a requirement that is both theoretically necessary for causal systems and practically important for real-time inference, which cannot observe future sensor readings.

Dilated convolutions extend the receptive field by inserting gaps between filter taps. With dilation rate `d`, each convolution tap is spaced `d` steps apart, allowing the network to cover a span of `(kernel_size - 1) * d + 1` timesteps per block with a fixed number of parameters. Stacking blocks with dilation rates 1, 2, 4, 8 produces a receptive field that grows exponentially in the number of blocks, enabling the model to capture both short-duration motion transients (small dilation) and longer-duration activity rhythms (large dilation) without increasing model depth quadratically.

Residual connections across each TCN block enable gradient flow through deep stacks and allow the network to optionally bypass any given block's transformation — functioning as an identity shortcut when the block's learned features are not beneficial for a given activity.

### Sinusoidal Positional Encoding

Transformer self-attention is inherently permutation-equivariant — it produces the same output regardless of the order of input positions unless position information is explicitly injected. For time-series data, temporal order is semantically critical: a rising acceleration followed by a falling one implies a different activity phase than the reverse.

Sinusoidal positional encodings add fixed, frequency-based vectors to the TCN output at each timestep before attention. Sine and cosine functions of varying frequencies are chosen so that each position receives a unique encoding, and so that the model can learn to attend to relative positions through linear combinations of these encodings — without introducing additional learned parameters.

### Multi-Head Self-Attention

The Transformer encoder applies self-attention over the full TCN-encoded sequence. Attention allows the model to directly relate any two timesteps in the window — for example, associating the initial impact signature at the beginning of a jump with the landing deceleration at the end, without requiring information to pass through all intermediate timesteps.

Multi-head attention runs H independent attention operations in parallel over projected subspaces of the input. Each head is free to specialize in attending to different relational patterns: one head might track peak acceleration events, another might track periodicity phase offsets. The outputs from all heads are concatenated and projected back to the original dimension.

### Class Imbalance Handling

The MHEALTH dataset does not have equal recording time across all activities. Static postures such as sitting and lying down tend to have longer total duration than dynamic activities such as jumping. When converted to windows, this produces a training label distribution that is substantially skewed toward static classes.

Without correction, a model trained with uniform loss weighting would maximize expected accuracy by over-predicting high-frequency classes at the expense of minority-class recall — a failure mode that is particularly problematic when the minority classes (high-intensity dynamic activities) are often the most clinically or practically interesting.

Class weights inversely proportional to class frequency are computed from the training partition and applied to the cross-entropy loss at every training step. This reweights the gradient contribution of each sample so that minority-class examples have amplified influence during optimization, pushing the model toward balanced performance across all 12 activity types.

---

## Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Input Shape | `(window_len, 23)` | Full 23-channel sensor vector per timestep |
| Window Length | Configurable (e.g., 128 samples) | ~2.6 seconds at 50 Hz — covers full gait cycles |
| TCN Blocks | Configurable stack depth | Exponential receptive field growth per block |
| TCN Dilation Rates | 1, 2, 4, 8, … (doubling) | Covers multi-scale temporal structure |
| Attention Heads | 4 (default) | Parallel relational pattern discovery |
| Feed-Forward Dimension | 128 (default) | Position-wise transformation capacity |
| Dropout | Applied in TCN blocks and attention sublayers | Regularization against sensor noise overfitting |
| Optimizer | Adam | Adaptive per-parameter learning rates |
| Loss Function | Sparse Categorical Crossentropy | Native integer label support; class-weighted |
| Class Weighting | Inverse-frequency weights | Minority-class gradient amplification |
| Early Stopping | Validation accuracy, configurable patience | Best-epoch restoration on plateau |
| Checkpoint (Best) | `best_model.keras` | Persists globally best validation epoch |
| Checkpoint (Final) | `final_transformer_model.keras` | Persists terminal training state |
| Normalization | `StandardScaler` per channel | Unit-variance, zero-mean sensor features |
| Label Encoding | Integer indices with pkl mappings | Serialized for exact inference-time decoding |
| Random Seed | Fixed globally | Fully reproducible across runs |

---

## Evaluation Framework

The model is evaluated on a held-out test partition that is never touched during training or hyperparameter decisions. Evaluation spans multiple complementary metrics to deliver a complete performance picture.

| Metric | Description | Diagnostic Value |
|---|---|---|
| Overall Accuracy | Fraction of correctly classified windows | Aggregate performance baseline |
| Macro F1 Score | Unweighted average F1 across all 12 classes | Primary metric — treats all activities equally regardless of support |
| Weighted F1 Score | F1 weighted by class sample count | Reflects performance on the natural class distribution |
| Per-Class Precision | True positives / predicted positives per class | Identifies classes with high false-positive rates |
| Per-Class Recall | True positives / actual positives per class | Identifies classes with high false-negative rates |
| Per-Class F1 | Harmonic mean of precision and recall per class | Activity-level diagnostic for targeted improvement |
| Confusion Matrix | Normalized heatmap with raw counts | Reveals systematic inter-class confusion patterns |
| Classification Report | Full sklearn report across all 12 labels | Comprehensive per-activity performance summary |

Macro F1 is the primary model selection and reporting metric because it treats all activity classes with equal weight — ensuring that strong performance on static posture classes (which have high sample counts) cannot mask poor performance on dynamic activity classes (which have lower sample counts but equal practical importance).

---

## Repository Structure

```
human-activity-recognition-transformer/
|
+-- har-mhealth-hybrid-tcn-transformer.ipynb   # Complete end-to-end pipeline notebook
|
+-- mhealth_raw_data.zip                        # MHEALTH benchmark dataset (self-contained)
|
+-- best_model.keras                            # Best validation checkpoint
+-- final_transformer_model.keras              # Final model after training completion
|
+-- feature_scaler.pkl                          # Fitted StandardScaler for all 23 channels
+-- label_to_idx.pkl                            # Activity name to integer index mapping
+-- idx_to_label.pkl                            # Integer index to activity name mapping
|
+-- test.csv                                    # Preprocessed held-out test partition
+-- test_predictions.csv                        # Exported predictions (predicted + true label)
|
+-- README.md                                   # Project documentation
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/sushantkothari/human-activity-recognition-transformer.git
cd human-activity-recognition-transformer
```

### 2. Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

Python 3.10 or higher is recommended. TensorFlow 2.x with GPU support is optional but substantially reduces training time on the full dataset.

### 3. Run the Notebook

Open `har-mhealth-hybrid-tcn-transformer.ipynb` in Google Colab or Jupyter Notebook and run all cells sequentially. The pipeline is fully self-contained:

- `mhealth_raw_data.zip` is extracted automatically — no manual data download required
- All preprocessing artifacts are generated and saved during execution
- All evaluation outputs render inline within the notebook
- Final predictions are exported to `test_predictions.csv` in the working directory

The entire pipeline is deterministic given the fixed global random seed. Re-running the notebook from scratch will reproduce identical preprocessing splits, model initialization, training trajectory, and evaluation results.

---

## Deployment-Ready Inference

All preprocessing and decoding artifacts are serialized alongside the trained model, enabling clean inference on new sensor data with no dependency on training code.

```python
import pickle
import numpy as np
from tensorflow import keras

# Load artifacts
model = keras.models.load_model("final_transformer_model.keras")

with open("feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("idx_to_label.pkl", "rb") as f:
    idx_to_label = pickle.load(f)

# Preprocess a raw sensor window
# raw_window: numpy array of shape (window_length, 23)
# Columns must match the original 23-channel sensor ordering from training

normalized_window = scaler.transform(raw_window)        # (window_length, 23)
batch = normalized_window[np.newaxis, ...]              # (1, window_length, 23)

# Predict
logits = model.predict(batch)                           # (1, num_classes)
predicted_index = np.argmax(logits, axis=-1)[0]
predicted_activity = idx_to_label[predicted_index]
confidence = float(np.max(logits))

print(f"Predicted Activity : {predicted_activity}")
print(f"Confidence         : {confidence:.4f}")
```

**Artifact Dependency Summary**

| Artifact | Required For | Purpose |
|---|---|---|
| `final_transformer_model.keras` | Inference | Forward pass and logit computation |
| `feature_scaler.pkl` | Preprocessing | Exact replication of training-time normalization |
| `idx_to_label.pkl` | Post-processing | Decoding integer prediction to activity name |
| `label_to_idx.pkl` | Optional | Re-encoding labels when evaluating against ground truth |

---

## Technology Stack

| Library | Role |
|---|---|
| Python 3.10+ | Core runtime |
| TensorFlow / Keras 2.x | Model construction, training, checkpointing, and serialization |
| Scikit-learn | Preprocessing, evaluation metrics, class weight computation |
| NumPy | Array operations, sliding window construction, seed management |
| Pandas | Data loading, cleaning, label encoding, and CSV export |
| Matplotlib | Training curves and confusion matrix visualization |
| Seaborn | Enhanced heatmap styling for confusion matrix analysis |
| Google Colab | GPU-accelerated notebook execution environment |

---

## Engineering Principles

**Causal Temporal Modeling** — Dilated causal convolutions enforce strict temporal causality throughout the TCN backbone. No output timestep accesses future sensor readings, making the model valid for both offline batch processing and real-time streaming inference without architectural modification.

**Complementary Representational Stages** — The TCN-first, Transformer-second ordering encodes a deliberate inductive bias: local motion primitives are extracted first by the convolution stack, then global temporal relationships across those primitives are modeled by attention. This separation of concerns improves sample efficiency and training stability compared to applying attention directly to raw sensor channels.

**Leak-Free Normalization** — The `StandardScaler` is fitted exclusively on the training partition. Validation and test channel statistics never influence the scaler parameters, preventing the subtle data leakage that occurs when normalization is fitted on the full dataset prior to splitting.

**Serialized Preprocessing Contracts** — Every transformation applied during training — normalization statistics, label index mappings — is serialized as a standalone artifact. Inference code loads these artifacts directly, guaranteeing bit-identical preprocessing between training and deployment without any coupling to training code.

**Imbalance-Aware Optimization** — Class-weighted loss ensures that gradient updates during training treat all 12 activity classes with proportional importance, regardless of how many windows each class contributes to the training set. This produces a model that generalizes across the full activity label space rather than optimizing primarily for the majority class.

**Dual Checkpoint Strategy** — Training persists both the best-validation-epoch model and the final model. The best checkpoint is optimal for deployment, maximizing generalization. The final model captures any additional convergence that may occur after the best epoch — useful for analysis and comparison.

**Reproducibility by Design** — A fixed global random seed is propagated across Python, NumPy, and TensorFlow's internal random state at process initialization. Every stochastic operation — weight initialization, data shuffling, dropout masking — is seeded identically, ensuring that any researcher can reproduce exact training results from this repository without additional configuration.

**Modular Architecture Construction** — The TCN blocks, positional encoding, Transformer encoder, and classification head are implemented as composable, independently testable components. This modularity enables straightforward substitution of any stage — for example, replacing the TCN backbone with a state-space model or swapping the classification head for a regression output — without restructuring the surrounding pipeline.

---

## Potential Extensions

**Architecture Upgrades**
- Replace the Transformer encoder with a Mamba or S4 state-space model for sub-quadratic sequence modeling on longer recording windows
- Integrate Conformer blocks (convolution and attention within a single residual unit) to more tightly couple local and global temporal modeling
- Explore multi-scale TCN branches with different dilation schedules merged before attention for richer feature diversity

**Data and Generalization**
- Subject-independent cross-validation — train on N-1 subjects, evaluate on the held-out subject — to rigorously assess cross-person generalization for clinical deployment
- Sensor stream augmentation: time warping, jitter injection, channel masking, and magnitude scaling to improve robustness to sensor placement variability and hardware differences
- Multi-dataset training combining MHEALTH, UCI HAR, PAMAP2, and OPPORTUNITY datasets for a more generalizable cross-benchmark model

**Advanced Training Strategies**
- Contrastive pre-training on unlabeled sensor streams using SimCLR or BYOL before supervised fine-tuning on labeled activity windows
- Knowledge distillation from the full TCN-Transformer into a compact TCN-only student model for edge deployment
- Federated learning across multiple subject devices without centralizing raw sensor data — relevant for privacy-preserving health applications

**Interpretability and Validation**
- Attention weight visualization overlaid on raw sensor signals to identify which temporal regions drive activity predictions
- SHAP-based feature attribution at the sensor-channel level to determine which modalities contribute most to each activity class
- Grad-CAM adaptation for 1D convolutional layers to localize discriminative timestep regions within each sensor window

**Deployment Targets**
- TensorFlow Lite conversion with post-training integer quantization for microcontroller deployment on resource-constrained wearable hardware
- ONNX export for cross-framework and cross-platform inference compatibility
- CoreML conversion for native iOS integration in Apple Watch or iPhone health applications
- FastAPI REST endpoint wrapping the inference pipeline for real-time activity stream classification via HTTP
- MQTT-based streaming inference client for integration with IoT sensor gateways

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Sushant Kothari**

[GitHub](https://github.com/sushantkothari)

---

<div align="center">
If this project was useful or informative, consider starring the repository on GitHub.
</div>
