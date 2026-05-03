# AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/MTUCI/AASIST3)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-red.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

This repository contains the original implementation of **AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization for the ASVspoof 2024 Challenge**.

## Paper

**AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization for the ASVspoof 2024 Challenge**

*This is the original implementation of the paper. The model weights provided here are NOT the same weights used in the paper results.*

## Overview

AASIST3 is an enhanced version of the AASIST (Anti-spoofing with Adaptive Softmax and Instance-wise Temperature) architecture that incorporates **Kolmogorov-Arnold Networks (KAN)** for improved speech deepfake detection. The model leverages:

- **Self-Supervised Learning (SSL) Features**: Uses Wav2Vec2 encoder for robust audio representation
- **KAN Linear Layers**: Kolmogorov-Arnold Networks for enhanced feature transformation
- **Graph Attention Networks (GAT)**: For spatial and temporal feature modeling
- **Multi-branch Inference**: Multiple inference branches for robust decision making

## Architecture

The AASIST3 model consists of several key components:

1. **Wav2Vec2 Encoder**: Extracts SSL features from raw audio
2. **KAN Bridge**: Transforms SSL features using Kolmogorov-Arnold Networks
3. **Residual Encoder**: Processes features through multiple residual blocks
4. **Graph Attention Networks**: 
   - GAT-S: Spatial attention mechanism
   - GAT-T: Temporal attention mechanism
5. **Multi-branch Inference**: Four parallel inference branches with master tokens
6. **KAN Output Layer**: Final classification using KAN linear layers

### Key Innovations

- **KAN Integration**: Replaces traditional linear layers with KAN linear layers for better feature approximation
- **Enhanced Regularization**: Additional dropout and regularization techniques
- **Multi-dataset Training**: Trained on multiple ASVspoof datasets for robustness

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mtuciru/AASIST3.git
cd AASIST3
pip install -r requirements.txt
```

### Loading the Model

```python
from model import aasist3

# Load the model from Hugging Face Hub
model = aasist3.from_pretrained("MTUCI/AASIST3")
model.eval()
```

### Basic Usage

```python
import torch
import torchaudio

# Load and preprocess audio
audio, sr = torchaudio.load("audio_file.wav")
# Ensure audio is 16kHz and mono
if sr != 16000:
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)

# Prepare input (model expects ~4 seconds of audio at 16kHz)
# Pad or truncate to 64600 samples
if audio.shape[1] < 64600:
    audio = torch.nn.functional.pad(audio, (0, 64600 - audio.shape[1]))
else:
    audio = audio[:, :64600]

# Run inference
with torch.no_grad():
    output = model(audio)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    
    # prediction: 0 = bonafide, 1 = spoof
    print(f"Prediction: {'Bonafide' if prediction.item() == 0 else 'Spoof'}")
    print(f"Confidence: {probabilities.max().item():.3f}")
```

## Training Details

### Datasets Used

The model was trained on a combination of multiple datasets:

- **ASVspoof 2019 LA** (Logical Access)
- **ASVspoof 2024 (ASVspoof5)** 
- **MLAAD** (Multi-Language Audio Anti-Spoofing Dataset)
- **M-AILABS** (Multi-Language Audio Dataset)

### Training Configuration

- **Epochs**: 20
- **Batch Size**: 12 (training), 24 (validation)
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss
- **Gradient Accumulation Steps**: 2

### Hardware

- **GPUs**: 2xA100 40GB
- **Framework**: PyTorch with Accelerate for distributed training

##  Advanced Usage

### Custom Training

```bash
# Train the model
bash train.sh
```

### Validation

```bash
# Run validation on test sets
bash validate.sh
```

### Model Configuration

The model can be configured through the `configs/train.yaml` file:

```yaml
# Key parameters
num_epochs: 20
train_batch_size: 12
val_batch_size: 24
learning_rate: 1e-4
gradient_accumulation_steps: 2
```


## 🤝 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{borodin24_asvspoof,
  title     = {AASIST3: KAN-enhanced AASIST speech deepfake detection using SSL features and additional regularization for the ASVspoof 2024 Challenge},
  author    = {Kirill Borodin and Vasiliy Kudryavtsev and Dmitrii Korzh and Alexey Efimenko and Grach Mkrtchian and Mikhail Gorodnichev and Oleg Y. Rogov},
  year      = {2024},
  booktitle = {The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof 2024)},
  pages     = {48--55},
  doi       = {10.21437/ASVspoof.2024-8},
}
```

##  License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0) - see the [LICENSE](LICENSE) file for details.

This license allows you to:
- **Share**: Copy and redistribute the material in any medium or format
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made

But does NOT allow:
- **Commercial use**: You may not use the material for commercial purposes
- **Derivatives**: You may not distribute modified versions of the material

For more information, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/


**Disclaimer**: This is a research implementation. The model weights provided are for demonstration purposes and may not match the exact performance reported in the paper.
