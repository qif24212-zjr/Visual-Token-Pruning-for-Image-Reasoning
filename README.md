# Task-Driven Visual Token Pruning for Image Reasoning 🚀
Our project builds a robust multimodal predictor upon the LLaVA architecture by incorporating semantic representations, YOLO-based object cues, and mask-based visual tokens. The pipeline consists of:

1. **Data Processing:** Extracting visual features, generating YOLO bounding boxes, and creating semantic masks.
2. **Token Generation:** Mapping questions to concepts and building token labels for multimodal alignment.
3. **Training:** Training a lightweight predictor to align visual tokens (including global/mask/semantic tokens) with the LLaVA language model.
4. **Evaluation:** Evaluating the model on multiple multimodal benchmarks (e.g., ScienceQA, VQAv2, MME) and benchmarking inference speed.

The framework is designed for efficient multimodal reasoning, utilizing both visual and textual cues.
# 1. Installation ⚙️

First, clone the repository:
```bash
git clone https://github.com/qif24212-zjr/Visual-Token-Pruning-for-Image-Reasoning.git
cd Visual-Token-Pruning-for-Image-Reasoning
```
Then create and activate the Conda environment:
```bash
conda create -n multimodal-predictor python=3.10 -y
conda activate multimodal-predictor
```
Install PyTorch with CUDA 12.1 support (or adjust based on your CUDA version):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Install the remaining dependencies:
```bash
pip install -r requirements.txt
```
Note: A compatible NVIDIA driver is required to run PyTorch with CUDA support. You can verify the installation with:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```
## Potential Issues
1.Missing packages: If you encounter `ModuleNotFoundError`, make sure all required dependencies are installed in the active environment.

2.CUDA compatibility: Ensure that your PyTorch installation is compatible with your CUDA environment.

3.Configuration paths: Check that the dataset paths and pretrained checkpoint paths (e.g., `predictor.pth`) are correctly set before running experiments.
# 2. Repository Structure 📁
```text
Visual-Token-Pruning-for-Image-Reasoning/
├── data_processing/                  # Dataset building and feature extraction
│   ├── build_predictor_dataset.py
│   ├── build_token_labels.py
│   ├── build_training_data.py
│   ├── build_training_data_scienceqa.py
│   ├── extract_features.py
│   ├── make_features.py
│   ├── make_label.py
│   ├── make_token_labels.py
│   └── question_to_concepts.py
├── models/                           # Model architectures and encoders
│   ├── llava_arch.py
│   ├── mask_encoder.py
│   ├── question_encoder.py
│   ├── semantic_encoder.py
│   └── semantic_predictor.py
├── train/                            # Training scripts
│   ├── train_predictor.py
│   ├── train_predictor_dataset.py
│   └── train_predictor_real.py
│   ├── eval_mme_fast.py
│   ├── eval_predictor.py
│   ├── eval_scienceqa_fast.py
│   ├── eval_vqav2_fast.py
│   ├── test_predictor.py
│   ├── test_predictor_mask.py
│   ├── test_scienceqa_one.py
│   ├── test_semantic_pipeline.py
│   └── benchmark_speed.py
├── pipeline/                         # One-click execution scripts
│   ├── run_full_pipeline.py
│   └── run_llava_semantic.py
│   ├── bbox_to_mask.py
│   ├── get_bbox.py
│   ├── global_mask.py
│   ├── global_semantic.py
│   ├── yolo_cues.py
│   ├── yolo_mask.py
│   └── visualize_tokens.py
└── README.md
```
Notes:

- `data_processing/` contains scripts for building training datasets, generating labels, and extracting features.
- `models/`  contains the core components, including LLaVA architecture, semantic predictor, and various encoders.
- `train/` contain corresponding training and evaluation scripts for various benchmarks.
- `pipeline/` provides one-click execution scripts.
# 3. Datasets 📦
The following datasets are used in our experiments. Please download them and update the paths in the configuration or script files accordingly.
### VQAv2 (Visual Question Answering v2)
- **Official website:**  http://visualqa.org/ 
### ScienceQA
- **Official repository:** https://scienceqa.github.io.
### MME (Multimodal Evaluation Benchmark)
- **Official repository:** https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation.
- # 4. Support 🛠️

If you meet issues, please open a GitHub issue with:
- Operating system and CUDA version
- Full error log
- Script and command-line arguments

If you need further assistance, please contact **Jiaren Zou** at [jiarenzou@acm.org](mailto:jiarenzou@acm.org).
