# WeedZSL: Multi-Modal Framework for Open-Set Weed Detection

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.7%2B-red) ![MIT License](https://img.shields.io/badge/license-MIT-green)

**A generalized zero-shot learning pipeline for weed detection using multimodal embeddings and semantic alignment**

WeedZSL reframes weed detection as an open-set recognition problem using generalized zero-shot learning (GZSL). The pipeline performs vegetation segmentation, pseudo-label classification using CNNs, and zero-shot semantic mapping using multimodal embeddings. It supports unseen species recognition and enables inference using various foundation models.

---

## Overview

WeedZSL is designed to support the detection, classification, and segmentation of weeds in agricultural environments — even when weed species were not seen during training.

### Key Features

- **Weed Classification**: Train and evaluate CNNs to classify weed vs. crop species
- **Weed Segmentation**: Identify vegetation pixels using ExG thresholding
- **Embedding Model Evaluation**: Compare multimodal models for visual-semantic alignment
- **Zero-Shot Detection**: Use large vision-language models for unseen weed species
- **Inference Pipeline**: End-to-end inference over new images

---

## Repository Structure

```
WeedZSL/
├── classification_models/          # CNN architectures for pseudo-label generation
│   ├── MobileNet.py
│   ├── ResNet18.py
│   ├── ShuffleNet.py
│   ├── ShuffleNet_SE.py
│   ├── ShuffleNet_SEPCONV.py
│   ├── ShuffleNet_SEPCONV_SE.py
│   └── SqueezeNet.py
├── embedding_models/               # Multimodal encoders and LLM interfaces
│   ├── embedding_clip.py
│   ├── embeddings_gemini.py
│   ├── embeddings_imagebind.py
│   ├── embeddings_llama.py
│   ├── embeddings_openclip.py
│   └── embeddings_siglip.py
├── datasets/                       # Dataset folder for classification/segmentation
├── segmentation.py                 # ExG-based vegetation segmentation
├── classification.py               # Pseudo-label generation pipeline
├── class_mapping.py                # Zero-shot semantic alignment
├── evaluate_embedding_models.py    # Multimodal model performance comparison
├── infer.py                        # End-to-end inference pipeline
├── exg_testing.py                  # Segmentation evaluation
├── train.ipynb                     # Jupyter notebook for training classifiers
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- 24GB+ RAM

---

## Data Preparation

Ensure your dataset is structured as follows:

- **Classification**:  
  ```
  datasets/classifier_data/
  ├── class_1/
  ├── class_2/
  └── ...
  ```

**Available Models:**
- `CLIP` - Contrastive Language-Image Pre-training
- `OpenCLIP` - Open-source CLIP variants
- `ImageBind` - **Best performing multimodal encoder**
- `SigLip` - Sigmoid loss pre-training
- `Gemini` - **Best overall performance** (requires API key)
- `Llama` - Llama 3.2-11B Vision Instruct model

- **End-to-End Inference**

Run the complete pipeline with a single command:

```bash
python infer.py 
```

### Pipeline Hyperparameters

Key parameters can be adjusted in the respective scripts:

- **ExG threshold**: Controls vegetation sensitivity (20-40 optimal)
- **Similarity threshold**: Semantic matching strictness (0.00-0.10 range)


- **Segmentation**:  
  Prepare RGB images and corresponding binary masks.

---

## Usage

## Multimodal Model Setup

### LLaMA-3 Vision

- Requires a Hugging Face token and access approval  
- Visit: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

### Gemini 2.5 Flash

- Free API key available  
- Visit: https://aistudio.google.com/

### ImageBind

- Setup instructions available at: https://github.com/facebookresearch/ImageBind

## .env requires the following vars
GOOGLE_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
---

## External Datasets

This project uses both public and custom datasets:

1. **CropAndWeed**  
   - 83-class dataset for classifier training  
   - [GitHub](https://github.com/cropandweed/cropandweed-dataset)

2. **PhenoBench**  
   - Used for segmentation threshold optimization  
   - [Website](https://www.phenobench.org/)

3. **Plant Phenotyping Datasets**  
   - Arabidopsis, Tobacco  
   - [Website](https://www.plant-phenotyping.org/datasets)

4. **Custom Field Data**  
   - Wheat, chili, tomato images under real-world conditions

---

## Citation

To cite this project, please contact the authors. Citation details will be released post-publication.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- Contributors of CropAndWeed, PhenoBench, and other open-source datasets  
- Developers of CLIP, ImageBind, SigLIP, and open multimodal tools  
