# WeedZSL: Multi-Modal Framework for Open-Set Weed Detection

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.7%2B-red) ![MIT License](https://img.shields.io/badge/license-MIT-green) [![Hugging Face Model](https://img.shields.io/badge/huggingface-emanfj%2FWeedZSLmodel-orange)](https://huggingface.co/emanfj/WeedZSLmodel)


**A novel generalized zero-shot learning pipeline for weed detection using multimodal embeddings and semantic alignment**

This repository implements the methodology described in our paper: *"A Multi-Modal Framework for Open-Set Weed Detection Using Generalized Zero-Shot Learning"*. WeedZSL implements zero-shot weed detection using text–image embeddings and class mapping. Given a top-view field image and the primary crop label, the pipeline segments individual plants, generates candidate classes, and assigns pseudo-labels via multimodal encoders or LLMs.



---

## Overview

WeedZSL addresses the limitations of supervised detectors by reframing weed detection as an open-set recognition problem using generalized zero-shot learning (GZSL).

### Key Features

- **ExG-based Segmentation**: Vegetation extraction using Excess Green Index thresholding
- **Lightweight CNN Classification**: Pre-trained models (MobileNet, ResNet18, ShuffleNet variants, SqueezeNet)
- **Multimodal Zero-Shot Mapping**: Semantic alignment between visual and textual representations
- **Unseen Species Support**: Generalization to new weed species without retraining

---

## Repository Structure

```
WeedZSL/
├── classification_models/          # CNN architectures for pseudo-label generation
│   ├── MobileNet.py               # MobileNet V2 implementation
│   ├── ResNet18.py                # ResNet-18 architecture
│   ├── ShuffleNet.py              # Standard ShuffleNet V2
│   ├── ShuffleNet_SE.py           # ShuffleNet + Squeeze-and-Excitation
│   ├── ShuffleNet_SEPCONV.py      # ShuffleNet + Separable Convolutions
│   ├── ShuffleNet_SEPCONV_SE.py   # ShuffleNet + SE + SepConv (best variant)
│   └── SqueezeNet.py              # SqueezeNet architecture
├── embedding_models/              # Multimodal encoders and LLM interfaces
│   ├── embedding_clip.py          # CLIP encoder interface
│   ├── embeddings_gemini.py       # Gemini 2.5 Flash API wrapper
│   ├── embeddings_imagebind.py    # ImageBind encoder
│   ├── embeddings_llama.py        # Llama 3.2-11B Vision interface
│   ├── embeddings_openclip.py     # OpenCLIP variants
│   └── embeddings_siglip.py       # SigLIP encoder
├── segmentation.py                # ExG-based vegetation segmentation
├── classification.py              # Pseudo-label generation pipeline
├── class_mapping.py               # Zero-shot semantic alignment
├── infer.py                       # End-to-end inference pipeline
├── exg_testing.py                 # Segmentation evaluation utilities
├── requirements.txt               # Python dependencies
└── README.md                      
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- 24GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SyArsRa/WeedZSL.git
   cd WeedZSL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Download pre-trained weights**
   ```bash
   # create checkpoint folder
   mkdir -p classification_models/checkpoints

   # download the ShuffleNet+SepConv+SE checkpoint
   wget https://huggingface.co/emanfj/WeedZSLmodel/resolve/main/shufflenet_sep_conv_squeeze_excitation.pt \
        -O classification_models/checkpoints/shufflenet_sep_conv_squeeze_excitation.pt

   # browse the full model repo:
   # https://huggingface.co/emanfj/WeedZSLmodel

---

## Usage

### 1. Vegetation Segmentation

Extract vegetation masks from field images using ExG thresholding:

```bash
python segmentation.py 
```

**Parameters:**
- `--exg_threshold`: ExG threshold value (default: 30, optimal range: 20-40)
- `--min_area`: Minimum component area in pixels (default: 15)

### 2. Pseudo-Label Classification

Generate candidate class labels for segmented vegetation:

```bash
python classification.py 
```

**Available Models:**
- `MobileNet` - Lightweight, mobile-optimized
- `ResNet18` - Standard residual architecture
- `ShuffleNet` - Efficient channel shuffling
- `ShuffleNet_SEPCONV_SE` - **Best performing variant**
- `SqueezeNet` - Ultra-lightweight architecture

### 3. Zero-Shot Semantic Mapping

Align pseudo-labels with crop semantics using multimodal encoders:

```bash
python class_mapping.py 
```

**Available Embedders:**
- `CLIP` - Contrastive Language-Image Pre-training
- `OpenCLIP` - Open-source CLIP variants
- `ImageBind` - **Best performing multimodal encoder**
- `SigLip` - Sigmoid loss pre-training
- `Gemini` - **Best overall performance** (requires API key)
- `Llama` - Llama 3.2-11B Vision model

### 4. End-to-End Inference

Run the complete pipeline with a single command:

```bash
python infer.py 
```
---

## Configuration

### Environment Variables

For API-based embedders, set your credentials:

```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

### Model Hyperparameters

Key parameters can be adjusted in the respective scripts:

- **ExG threshold**: Controls vegetation sensitivity (20-40 optimal)
- **Similarity threshold**: Semantic matching strictness (0.02-0.10 range)
- **Minimum area**: Filters out noise components (10-50 pixels)


## Datasets

This project utilizes several open-source agricultural datasets:

1. **CropAndWeed** [[Steininger et al., 2023]](https://github.com/cropandweed/cropandweed-dataset)
   - 80,508 instances across 83 classes (after filtering)
   - Used for training classification models

2. **PhenoBench** [[Weyler et al., 2024]](https://www.phenobench.org/)
   - Top-view plant images with segmentation masks
   - Used for ExG threshold optimization

3. **Plant Phenotyping** [[Minervini et al., 2016]](https://www.plant-phenotyping.org/datasets)
   - Tobacco and Arabidopsis (Thale Cress) images
   - Used for generalization testing

4. **Custom Field Data**
   - Wheat: 224 instances (DJI Mavic Pro, 1m height)
   - Chili & Tomato: 337 instances (greenhouse conditions)

---


## Contributing

We welcome contributions! To get started:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## Citation

If you use WeedZSL in your research, please contact us for citation reference.


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [CropAndWeed Dataset](https://github.com/cropandweed/cropandweed-dataset) contributors
- [PhenoBench](https://www.phenobench.org/) team
