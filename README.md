# Video Captioning Using Deep Learning | ResNet-50 + BiLSTM

A deep learning-based video captioning system that generates natural-language descriptions of video content by combining **CNN-based visual feature extraction with recurrent neural network-based language generation**. The model uses a pretrained ResNet-50 encoder to extract spatial representations from sampled video frames and a bidirectional LSTM decoder to predict caption tokens conditioned on the extracted visual features.

The system is implemented in **PyTorch** and trained on a subset of the **MSR-VTT (Microsoft Research Video-to-Text) dataset**, supporting an end-to-end pipeline for video preprocessing, visual representation learning, vocabulary construction, sequence modeling, model optimization, and autoregressive caption generation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technical Workflow](#technical-workflow)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Dataset Configuration](#dataset-configuration)
- [Training](#training)
- [Inference](#inference)
- [Implementation Details](#implementation-details)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [Repository Structure](#repository-structure)
- [References](#references)

---

## Overview

Understanding video content requires interpreting visual information across multiple frames and translating that information into coherent natural-language descriptions.

This project implements a video-to-text pipeline that learns a mapping between sampled video frames and their corresponding textual descriptions.

The model consists of two principal components:

1. **Visual Encoder:** A pretrained ResNet-50 convolutional neural network extracts 2,048-dimensional feature vectors from individual video frames.
2. **Language Decoder:** A bidirectional Long Short-Term Memory (BiLSTM) network processes caption token embeddings alongside aggregated visual features to predict the next token in a caption sequence.

Video frames are sampled at approximately uniform intervals, resized to the dimensions expected by ResNet-50, and normalized using ImageNet statistics. During training, a caption associated with each video is selected, tokenized, and converted into integer indices using a vocabulary constructed from the training captions.

The decoder is optimized using cross-entropy loss, with padding tokens excluded from the loss calculation. At inference time, the trained model generates captions autoregressively until an end-of-sequence token is produced or the maximum generation length is reached.

### Project Objectives

- Extract meaningful visual representations from video frames using transfer learning.
- Combine frame-level visual information with a recurrent language-generation architecture.
- Train a neural sequence model to predict caption tokens from video representations.
- Implement an autoregressive inference pipeline for generating video descriptions.
- Develop a modular deep learning workflow using PyTorch and OpenCV.

---

## Key Features

- **CNN-based visual encoding:** Uses pretrained ResNet-50 with its final classification layer removed.
- **Transfer learning:** Reuses ImageNet-pretrained visual representations while freezing the encoder parameters.
- **Video frame sampling:** Extracts a fixed number of frames at approximately uniform intervals.
- **Feature aggregation:** Computes the mean of frame-level feature vectors to obtain a global video representation.
- **Sequence modeling:** Uses a bidirectional LSTM decoder conditioned on visual features and caption token embeddings.
- **Custom vocabulary:** Builds a word-level vocabulary from training captions, filtering out words with fewer than five occurrences.
- **Variable-length caption handling:** Uses special tokens for sequence boundaries, unknown words, and padding.
- **Masked sequence loss:** Ignores padding tokens during cross-entropy loss computation.
- **Autoregressive inference:** Predicts caption tokens iteratively using temperature-scaled sampling.
- **GPU acceleration:** Automatically selects CUDA when available and otherwise uses the CPU.
- **Model checkpointing:** Saves and reloads trained model parameters through PyTorch state dictionaries.

---

## System Architecture

The model follows a CNN–RNN architecture.

```text
                    Input Video (.mp4)
                           |
                           v
                  Video Frame Sampling
                   (4 Frames per Video)
                           |
                           v
                 RGB Conversion & Resize
                     224 x 224
                           |
                           v
                  ImageNet Normalization
                           |
                           v
                 Pretrained ResNet-50
                 (Classification Head
                      Removed)
                           |
                           v
               Frame-Level Feature Vectors
                    2048 Dimensions
                           |
                           v
                 Temporal Mean Pooling
                           |
                           v
                Global Video Representation
                    2048 Dimensions
                           |
                           +---------------------+
                           |                     |
                           v                     v
                  Caption Token IDs       Global Video Features
                           |                     |
                           v                     |
                    Token Embedding              |
                       256-D                    |
                           |                     |
                           +----------+----------+
                                      |
                                      v
                            Feature Concatenation
                              256 + 2048
                                      |
                                      v
                              BiLSTM Decoder
                               Hidden Size 512
                                      |
                                      v
                             Fully Connected Layer
                               1024 -> Vocabulary
                                      |
                                      v
                              Softmax / Sampling
                                      |
                                      v
                             Generated Caption
```

**Important architectural detail:** Although the encoder extracts features from multiple frames, the current implementation averages these features before passing them to the decoder. Consequently, the decoder receives a global video representation rather than an explicit sequence of temporally ordered frame features.

---

## Technical Workflow

### 1. Dataset Loading and Caption Mapping

The project loads video metadata and caption annotations from `train_val_videodatainfo.json`.

The JSON annotations are grouped by `video_id`, allowing each video to be associated with multiple reference captions.

Videos marked with the `train` split are selected to construct the training caption mapping.

This provides the correspondence between video samples and their natural-language supervision.

### 2. Video Preprocessing

OpenCV is used to load videos and extract frames.

The preprocessing pipeline performs the following operations:

- Reads the total number of frames in the video.
- Calculates an approximate sampling interval.
- Extracts frames at regular intervals.
- Converts frames from BGR to RGB.
- Resizes frames to `224 × 224` pixels.
- Converts frames into PyTorch tensors.
- Normalizes RGB channels using ImageNet mean and standard deviation.

The dataset implementation samples up to four frames per video to reduce computational overhead during training.

The normalization parameters are:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These statistics correspond to the normalization convention used with ImageNet-pretrained ResNet-50 weights.

### 3. Visual Feature Extraction

A pretrained ResNet-50 model is used as the visual encoder.

The final fully connected classification layer is removed, leaving the convolutional backbone and global average pooling operation.

For an input batch containing `B` videos and `N` sampled frames per video, the encoder reshapes the input from:

```text
(B, N, 3, 224, 224)
```

to:

```text
(B × N, 3, 224, 224)
```

The frames are processed independently by ResNet-50, producing 2,048-dimensional feature vectors.

The extracted features are then reshaped to:

```text
(B, N, 2048)
```

All encoder parameters have `requires_grad=False`, so the pretrained visual backbone remains frozen during training.

### 4. Caption Vocabulary Construction

A word-level vocabulary is constructed from the training captions.

The vocabulary includes four reserved tokens:

| Token | Purpose |
|---|---|
| `<PAD>` | Pads sequences to a fixed length |
| `<SOS>` | Marks the beginning of a caption |
| `<EOS>` | Marks the end of a caption |
| `<UNK>` | Represents out-of-vocabulary words |

Captions are converted to lowercase and split into whitespace-delimited tokens.

Words appearing fewer than five times are excluded from the vocabulary. Any excluded word encountered during encoding is mapped to `<UNK>`.

Two mappings are created:

- `word2idx`: Maps tokens to integer indices.
- `idx2word`: Maps integer indices back to tokens.

These mappings are essential for converting captions into model inputs and reconstructing generated token sequences into readable text.

### 5. Caption Sequence Preparation

For each training sample, one caption is randomly selected from the captions associated with its video.

The caption is transformed into a sequence containing the start and end tokens:

```text
<SOS> a person is playing guitar <EOS>
```

Each token is mapped to its corresponding vocabulary index.

The sequence is padded or truncated to a maximum length of 20 tokens.

During training, teacher forcing is implemented by passing the ground-truth caption prefix to the decoder and comparing its predictions against the next-token targets.

For a caption tensor of shape:

```text
(B, T)
```

the model receives:

```python
captions[:, :-1]
```

and is trained against:

```python
captions[:, 1:]
```

This shifts the input and target sequences by one position, aligning each decoder prediction with the expected next token.

### 6. Visual-Linguistic Fusion

The decoder combines the caption token embeddings with the global video representation.

The embedding layer converts each caption token into a 256-dimensional vector.

The frame-level visual features are averaged across the frame dimension:

```python
video_feat = features.mean(1)
```

The resulting 2,048-dimensional vector is repeated across the caption sequence length and concatenated with the token embeddings.

The decoder therefore receives a sequence of vectors with dimensionality:

```text
256 + 2048 = 2304
```

This conditioning mechanism provides visual context at every caption decoding position.

### 7. Training and Optimization

The model is trained using the Adam optimizer and cross-entropy loss.

The current configuration uses:

| Hyperparameter | Value |
|---|---:|
| Visual backbone | ResNet-50 |
| Frame sampling | 4 frames per video |
| Input resolution | 224 × 224 |
| Visual feature dimension | 2,048 |
| Token embedding dimension | 256 |
| LSTM hidden dimension | 512 |
| LSTM directionality | Bidirectional |
| LSTM layers | 1 |
| Maximum caption length | 20 tokens |
| Batch size | 2 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Loss function | Cross-entropy |
| Training epochs | 10 |
| Training subset | 200 video-caption mappings |

Padding tokens are excluded from the loss calculation using:

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=word2idx["<PAD>"]
)
```

The training loop performs the following steps:

1. Loads a batch of video frames and caption sequences.
2. Computes visual representations using the encoder.
3. Predicts caption-token logits using the decoder.
4. Calculates the next-token prediction loss.
5. Backpropagates the loss.
6. Updates trainable model parameters.
7. Tracks batch-level and epoch-level loss.

The visual encoder is frozen, while the decoder parameters remain trainable.

### 8. Autoregressive Caption Generation

During inference, the trained model receives the sampled frames of a video.

The encoder extracts visual features, which are averaged into a global representation.

Generation starts with the `<SOS>` token. At each decoding step, the model predicts a probability distribution over the vocabulary and samples the next token using temperature-scaled softmax.

The generated token is fed back into the decoder as the next input.

Generation terminates when the model predicts `<EOS>` or reaches the maximum generation length.

The temperature parameter controls the sharpness of the sampling distribution:

- Lower temperatures concentrate probability on more likely tokens.
- Higher temperatures produce a flatter distribution and potentially more varied outputs.

The current implementation uses a temperature of `0.8` for the sample inference call.

---

## Model Architecture

### Visual Encoder: ResNet-50

**Architecture:** Residual Convolutional Neural Network

**Input:** RGB video frame of shape `(3, 224, 224)`

**Output:** Feature vector of dimension `2048`

The encoder uses pretrained ResNet-50 weights and removes the final classification layer.

Residual connections allow deeper convolutional networks to learn visual representations effectively. In this project, these pretrained representations are reused for video captioning rather than image classification.

The encoder is frozen to reduce the number of trainable parameters and computational requirements.

### Language Decoder: Bidirectional LSTM

**Architecture:** Bidirectional Long Short-Term Memory network

**Input:** Concatenated caption embeddings and global video features

**Input dimension:** `2304`

**Hidden size:** `512` per direction

**Output dimension:** `1024`

The decoder combines token-level linguistic information with visual context to predict caption tokens.

Because the LSTM is bidirectional, its output combines forward and backward recurrent representations. The concatenated output has dimensionality:

```text
512 × 2 = 1024
```

A linear projection maps the 1,024-dimensional decoder output to the vocabulary dimension:

```text
Linear(1024, vocabulary_size)
```

The resulting logits represent the model's unnormalized scores for the next caption token.

**Implementation note:** The bidirectional LSTM is trained on complete caption prefixes during teacher forcing. At inference, it is called incrementally with one token at a time while recurrent states are carried forward. This creates a mismatch between training and inference behavior, so the current decoder should be treated as an experimental implementation rather than a fully validated autoregressive language model.

### Objective Function

The model minimizes token-level cross-entropy loss.

For each position in the target caption, the predicted vocabulary distribution is compared against the ground-truth next token. Padding positions are ignored.

Conceptually, the training objective is:

\[
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P(y_t \mid y_{<t}, V)
\]

where:

- \(V\) represents the visual features extracted from the video.
- \(y_t\) is the ground-truth token at position \(t\).
- \(y_{<t}\) represents the preceding ground-truth tokens.
- \(T\) is the target sequence length.

The expression describes the intended next-token prediction objective; the actual decoder's bidirectional configuration introduces the architectural limitation discussed above.

---

## Dataset

This project uses the **MSR-VTT (Microsoft Research Video-to-Text) dataset**, a benchmark for video understanding and natural-language video description.

The dataset provides video clips paired with human-written captions, making it suitable for supervised video captioning experiments.

The implementation expects:

- Video files in MP4 format.
- A JSON annotation file containing video metadata and caption sentences.
- Video identifiers that can be matched to the corresponding filenames.

The code references a Kaggle dataset downloaded through `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("vishnutheepb/msrvtt")
print(path)
```

The notebook subsequently references:

```text
/kaggle/input/msrvtt/TrainValVideo
```

**Dataset configuration note:** The downloaded path and the hard-coded `/kaggle/input/` path are not necessarily the same. Verify the actual directory structure after downloading the dataset and update the notebook paths accordingly.

The current experiment uses 200 video-caption mappings rather than the complete training dataset. This subset is intended to reduce training cost and support initial experimentation; it is not sufficient by itself to establish benchmark-level captioning performance.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Visual Feature Extraction | Torchvision, ResNet-50 |
| Sequence Modeling | PyTorch LSTM |
| Image Preprocessing | Torchvision Transforms, Pillow |
| Data Processing | JSON, Python Collections |
| Dataset Acquisition | KaggleHub |
| Training Utilities | PyTorch DataLoader, tqdm |
| Model Optimization | Adam |
| Loss Function | Cross-Entropy Loss |
| Execution Environment | Google Colab / Kaggle Notebook |
| Hardware Acceleration | CUDA-enabled GPU when available |

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Replace the placeholders with your repository URL and directory name.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision opencv-python Pillow pandas tqdm kagglehub
```

For GPU acceleration, install a PyTorch build compatible with your CUDA environment by following the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

The dataset must also be downloaded and configured before running the training pipeline.

---

## Dataset Configuration

Ensure that the repository or notebook environment can access the following files:

```text
dataset/
├── train_val_videodatainfo.json
└── TrainValVideo/
    ├── video_id_1.mp4
    ├── video_id_2.mp4
    └── ...
```

The filenames above are illustrative. Actual filenames must match the `video_id` values stored in the annotation file.

Update the dataset path in the notebook:

```python
video_folder = "/path/to/TrainValVideo"
```

Load the annotation file:

```python
with open(
    "/path/to/train_val_videodatainfo.json",
    "r"
) as f:
    data = json.load(f)
```

The JSON file is expected to contain `videos` and `sentences` entries. Video metadata must include `video_id` and `split`, while each caption entry must contain `video_id` and `caption`.

Ensure that the selected training videos exist in the video directory and that their filenames correspond to the identifiers in the annotations.

---

## Training

After configuring the dataset, construct the model and optimizer:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = VideoCaptioningModel(
    embed_size=256,
    hidden_size=512,
    vocab_size=len(word2idx)
).to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=word2idx["<PAD>"]
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)
```

Create the dataset and data loader:

```python
dataset = MSRVTTDataset(
    video_folder=video_folder,
    video_captions=sliced_captions,
    word2idx=word2idx,
    transform=transform,
    max_len=20
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

Execute the training loop for the configured number of epochs.

The notebook reports average training loss for each epoch. These values can help assess optimization behavior, but they do not independently establish caption quality or generalization performance.

**Reproducibility:** The current notebook does not explicitly set random seeds for Python, NumPy, or PyTorch. As a result, training subsets, selected reference captions, and model initialization can vary across runs.

---

## Inference

After training, save the model parameters:

```python
torch.save(
    model.state_dict(),
    "video_caption_model.pth"
)
```

Reload the model parameters:

```python
model.load_state_dict(
    torch.load(
        "video_caption_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()
```

The checkpoint contains the model's learned parameters. It does **not** contain the vocabulary mappings, model configuration, or dataset metadata, so these must be preserved separately for reliable inference.

Generate a caption from a sample video:

```python
frames, _ = dataset[0]

caption = generate_caption(
    model,
    frames,
    word2idx,
    idx2word,
    max_len=20,
    device=device,
    temperature=0.8
)

print("Generated Caption:", caption)
```

The example uses an existing dataset sample. To caption a new video, first extract and preprocess its frames using the same preprocessing pipeline, then pass the resulting frame tensor to `generate_caption()`.

**Checkpoint compatibility:** The vocabulary used during inference must exactly match the vocabulary used during training. The same applies to model dimensions and preprocessing settings.

---

## Implementation Details

### Tensor Dimensions

The following table summarizes the main tensor transformations implemented by the model.

| Stage | Tensor Shape |
|---|---|
| Input video batch | `(B, N, 3, 224, 224)` |
| Flattened frame batch | `(B × N, 3, 224, 224)` |
| ResNet-50 output | `(B × N, 2048, 1, 1)` |
| Frame feature sequence | `(B, N, 2048)` |
| Mean-pooled video features | `(B, 2048)` |
| Caption token indices | `(B, T)` |
| Token embeddings | `(B, T, 256)` |
| Repeated video features | `(B, T, 2048)` |
| Fused decoder input | `(B, T, 2304)` |
| Bidirectional LSTM output | `(B, T, 1024)` |
| Vocabulary logits | `(B, T, vocabulary_size)` |

Here:

- `B` = batch size.
- `N` = number of sampled frames.
- `T` = caption sequence length.
- `vocabulary_size` = number of tokens in the constructed vocabulary.

### Training and Inference Considerations

- The ResNet-50 encoder is frozen, reducing the number of parameters updated during training.
- Video features are recomputed when each sample is loaded; the current pipeline does not cache extracted features.
- Random caption selection provides different reference captions for a video across dataset accesses.
- Mean pooling reduces a sequence of frame features to a single global representation, sacrificing explicit temporal ordering.
- Temperature-based multinomial sampling introduces stochasticity into generated captions.
- The vocabulary is constructed from the selected training split, reducing direct exposure to validation or test captions during vocabulary creation.

---

## Current Limitations

The following limitations are relevant when interpreting the implementation and its results.

### 1. Limited Training Data

The experiment uses only 200 video-caption mappings and runs for 10 epochs. This is a small experimental subset and should not be interpreted as a fully trained MSR-VTT captioning model.

### 2. Loss of Explicit Temporal Information

The model averages frame-level features before decoding. This captures global visual content but does not explicitly model motion, event ordering, or changes between frames.

### 3. Bidirectional Decoder During Autoregressive Generation

The decoder uses a bidirectional LSTM, but inference generates one token at a time. The backward recurrent direction is therefore not provided with the complete caption prefix it encountered during training.

A unidirectional LSTM decoder would be a more appropriate choice for conventional autoregressive generation.

### 4. Fixed-Length Captions

Caption sequences are limited to 20 tokens and are truncated when longer. This restricts the length and detail of generated descriptions.

### 5. Potential Empty-Frame Failures

If a video cannot be opened or contains no readable frames, the dataset implementation may fail when attempting to stack an empty frame list. Robust production code should validate video loading and handle corrupted or missing files.

### 6. Vocabulary and Checkpoint Management

The model checkpoint does not preserve the vocabulary mappings. Without the original `word2idx` and `idx2word` mappings, the saved model cannot reliably convert predictions back into the original token sequence.

### 7. Limited Evaluation

The current code prints training loss but does not implement a dedicated validation or test pipeline. It also does not calculate established captioning metrics such as BLEU, METEOR, ROUGE-L, or CIDEr.

Consequently, the implementation demonstrates the modeling pipeline, but its captioning quality and generalization have not been quantitatively established by the code shown here.

---

## Future Improvements

The following improvements would strengthen the model's architecture, reliability, and evaluation.

- **Temporal modeling:** Replace global mean pooling with temporal attention, an LSTM over frame features, or a transformer-based video encoder.
- **Autoregressive decoder correction:** Use a unidirectional LSTM for next-token generation or adopt a decoder architecture explicitly designed for causal language modeling.
- **Attention mechanism:** Implement visual attention so the decoder can focus on relevant frame representations at different decoding steps.
- **Larger-scale training:** Train on a larger portion of MSR-VTT and use a dedicated validation split for hyperparameter selection.
- **Captioning evaluation:** Implement BLEU, METEOR, ROUGE-L, and CIDEr using an appropriate evaluation pipeline.
- **Decoding improvements:** Compare multinomial sampling against greedy decoding and beam search.
- **Feature caching:** Precompute and store frozen ResNet-50 features to reduce repeated visual inference during training.
- **Reproducibility:** Set random seeds and save training configuration, vocabulary mappings, and checkpoint metadata.
- **Robust preprocessing:** Validate frame extraction, handle unreadable videos, and enforce consistent frame counts.
- **Model optimization:** Evaluate mixed-precision training and tune batch size based on available GPU memory.
- **Improved visual representations:** Compare the current CNN-based encoder against pretrained video models or vision-language encoders.

---

## Repository Structure

A suggested repository structure is:

```text
video-captioning/
│
├── Video_Captioning.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   └── video_caption_model.pth
│
└── assets/
    └── architecture.png
```

The structure is illustrative. Include `models/video_caption_model.pth` only if you intend to distribute the trained checkpoint. Large model weights and the MSR-VTT videos should generally be stored outside Git or distributed through an appropriate external hosting mechanism.

For a more maintainable implementation, the notebook can later be refactored into separate modules for dataset handling, model definitions, training, inference, and evaluation.

---

## References

1. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [Research paper](https://arxiv.org/abs/1512.03385)

2. **Hochreiter, S., & Schmidhuber, J.** (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780. [Research paper](https://doi.org/10.1162/neco.1997.9.8.1735)

3. **Xu, J., et al.** (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* International Conference on Machine Learning (ICML). [Research paper](https://arxiv.org/abs/1502.03044)

4. **MSR-VTT Dataset:** Xu, J., et al. (2016). *MSR-VTT: A Large Video Description Dataset for Bridging Video and Language.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [Research paper](https://arxiv.org/abs/1603.07678)

5. **PyTorch Documentation:** [Official documentation](https://pytorch.org/docs/stable/index.html)

6. **Torchvision Documentation:** [Official documentation](https://pytorch.org/vision/stable/index.html)

---

## Project Summary

This project implements an experimental deep learning pipeline for generating natural-language descriptions from video content. By combining pretrained ResNet-50 visual representations with a recurrent sequence decoder, it explores the integration of computer vision and natural-language generation within a supervised video captioning framework.

The implementation covers video preprocessing, transfer learning, visual feature aggregation, vocabulary construction, sequence modeling, teacher-forced training, and autoregressive token generation.

**Primary technical focus:** Computer Vision, Deep Learning, Transfer Learning, CNN Feature Extraction, LSTM-Based Sequence Modeling, Natural Language Generation, and Video Understanding.
