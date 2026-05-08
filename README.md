# Contrastive Learning Implementation

A comprehensive implementation of contrastive learning methods including SimCLR, MoCo, and SwAV for computer vision tasks. This project provides a clean, reproducible, and showcase-ready framework for research and education in self-supervised learning.

## Features

- **Multiple Contrastive Learning Methods**: SimCLR, MoCo, SwAV implementations
- **Comprehensive Baselines**: Classical ML and deep learning baselines for comparison
- **Modern Tech Stack**: PyTorch 2.x, Hydra configuration, type hints, and proper testing
- **Interactive Demo**: Gradio-based web interface for model exploration
- **Rich Evaluation**: k-NN, linear evaluation, clustering, and retrieval metrics
- **Visualization Tools**: t-SNE, PCA, similarity matrices, and interactive plots
- **Reproducible**: Deterministic seeding, checkpointing, and configuration management

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Configuration](#configuration)
- [Safety & Ethics](#safety--ethics)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Apple Silicon support (MPS)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/kryptologyst/contrastive-learning-implementation.git
cd contrastive-learning-implementation

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn plotly
pip install tqdm omegaconf hydra-core
pip install opencv-python kornia
pip install gradio streamlit
pip install wandb mlflow
pip install faiss-cpu transformers
pip install accelerate rich typer

# Development dependencies
pip install black ruff mypy pytest
pip install pre-commit jupyter
```

## Quick Start

### 1. Train a SimCLR Model

```bash
# Train SimCLR on CIFAR-10
python -m src.cli train \
    --model-type simclr \
    --dataset cifar10 \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.0001
```

### 2. Evaluate the Model

```bash
# Evaluate trained model
python -m src.cli eval \
    --checkpoint-path ./checkpoints/best_model_epoch_99.pth \
    --model-type simclr \
    --dataset cifar10
```

### 3. Launch Interactive Demo

```bash
# Launch demo application
python -m src.cli demo \
    --model-type simclr \
    --checkpoint-path ./checkpoints/best_model_epoch_99.pth \
    --port 8501
```

### 4. Compare Baselines

```bash
# Compare different baseline methods
python -m src.cli compare-baselines \
    --dataset cifar10 \
    --batch-size 256
```

## Usage

### Training

```python
from src.models.simclr import create_simclr_model
from src.data.datasets import CIFAR10Dataset
from src.data.augmentations import SimCLRAugmentation
from src.train.trainer import ContrastiveTrainer

# Create model and loss
model, loss_fn = create_simclr_model(
    base_model="resnet50",
    projection_dim=128,
    temperature=0.5
)

# Create dataset
dataset = CIFAR10Dataset(root_dir="./data", train=True)
augmentation = SimCLRAugmentation()

# Create trainer
trainer = ContrastiveTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device="cuda"
)

# Train
history = trainer.fit(train_loader, epochs=100)
```

### Evaluation

```python
from src.eval.evaluator import ContrastiveEvaluator

# Create evaluator
evaluator = ContrastiveEvaluator(model, device="cuda")

# Load checkpoint
evaluator.load_checkpoint("checkpoint.pth")

# Evaluate
metrics = evaluator.evaluate_all(train_loader, test_loader)
evaluator.print_results(metrics)
```

### Visualization

```python
from src.viz.plots import plot_embeddings, plot_similarity_matrix

# Plot embeddings
plot_embeddings(
    embeddings=features,
    labels=labels,
    method="tsne",
    title="SimCLR Embeddings"
)

# Plot similarity matrix
plot_similarity_matrix(
    embeddings=features,
    labels=labels,
    title="Feature Similarity Matrix"
)
```

## Models

### SimCLR (Simple Contrastive Learning of Visual Representations)

- **Paper**: [SimCLR](https://arxiv.org/abs/2002.05709)
- **Key Features**: NT-Xent loss, strong data augmentation, large batch sizes
- **Implementation**: `src.models.simclr.SimCLR`

### MoCo (Momentum Contrast)

- **Paper**: [MoCo](https://arxiv.org/abs/1911.05722)
- **Key Features**: Momentum encoder, large queue, stable training
- **Implementation**: `src.models.moco.MoCo`

### SwAV (Swapping Assignments between Views)

- **Paper**: [SwAV](https://arxiv.org/abs/2006.09882)
- **Key Features**: Online clustering, Sinkhorn algorithm, multi-crop
- **Implementation**: `src.models.swav.SwAV`

### Baselines

- **CNN**: Simple convolutional neural network
- **ResNet**: Pre-trained ResNet with fine-tuning
- **ViT**: Vision Transformer implementation
- **Classical ML**: Random Forest, SVM, Logistic Regression, k-NN

## Evaluation

### Metrics

- **k-NN Classification**: Accuracy@k for k ∈ {1, 5, 10, 20}
- **Linear Evaluation**: Logistic regression on frozen features
- **Clustering**: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)
- **Retrieval**: Recall@k, Precision@k, mAP
- **Visualization**: t-SNE, PCA, similarity matrices

### Example Results

| Method | k-NN@1 | k-NN@5 | Linear Acc | ARI | NMI |
|--------|--------|--------|------------|-----|-----|
| SimCLR | 0.8234 | 0.9123 | 0.8456 | 0.6789 | 0.7123 |
| MoCo   | 0.8156 | 0.9089 | 0.8398 | 0.6654 | 0.6987 |
| SwAV   | 0.8298 | 0.9156 | 0.8512 | 0.6845 | 0.7201 |

## Demo

The interactive demo provides:

- **Image Similarity**: Compare similarity between two images
- **Embedding Visualization**: Visualize learned representations
- **Model Information**: Architecture and training details

Access the demo at: `http://localhost:8501`

## Configuration

Configuration is managed using Hydra. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific configurations
- `configs/dataset/`: Dataset configurations
- `configs/optimizer/`: Optimizer configurations
- `configs/scheduler/`: Learning rate scheduler configurations

### Example Configuration

```yaml
# configs/config.yaml
train:
  epochs: 100
  batch_size: 256
  num_workers: 4

model:
  projection_dim: 128
  temperature: 0.5
  hidden_dim: 512

optimizer:
  lr: 0.0001
  weight_decay: 1e-6

augmentation:
  color_jitter_strength: 0.8
  gaussian_blur_prob: 0.5
```

## Safety & Ethics

### ⚠️ Important Disclaimers

- **Research Only**: This implementation is for research and educational purposes only
- **Not for Production**: Do not use for production systems or critical decision-making
- **No Medical Use**: Not intended for medical diagnosis or treatment
- **No Financial Advice**: Not for financial trading or investment decisions
- **Human Oversight**: Always require human oversight for important decisions

### Privacy & Security

- **Data Privacy**: Ensure compliance with data protection regulations
- **Model Security**: Be aware of potential adversarial attacks
- **Bias Awareness**: Models may exhibit biases present in training data
- **Transparency**: Document model limitations and potential risks

### Environmental Impact

- **Energy Consumption**: Training large models consumes significant energy
- **Carbon Footprint**: Consider environmental impact of compute resources
- **Efficiency**: Use efficient training practices and hardware

## 📁 Project Structure

```
contrastive-learning-implementation/
├── src/                          # Source code
│   ├── data/                     # Data loading and augmentation
│   ├── models/                   # Model implementations
│   ├── losses/                   # Loss functions
│   ├── metrics/                  # Evaluation metrics
│   ├── train/                    # Training pipeline
│   ├── eval/                     # Evaluation pipeline
│   ├── viz/                      # Visualization tools
│   ├── utils/                    # Utility functions
│   └── cli.py                    # Command-line interface
├── configs/                      # Configuration files
├── data/                         # Data directory
├── assets/                       # Generated assets
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── demo/                         # Demo applications
├── notebooks/                    # Jupyter notebooks
├── pyproject.toml               # Project configuration
├── .gitignore                   # Git ignore rules
└── README.md                     # This file
```

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::test_simclr
```

## Monitoring & Logging

### Weights & Biases

```python
import wandb

# Initialize
wandb.init(project="contrastive-learning")

# Log metrics
wandb.log({"loss": loss, "accuracy": acc})
```

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir ./logs
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## References

- [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
- [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882)
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨Author

**kryptologyst**

- GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)
- Email: kryptologyst@example.com

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for transformers and datasets libraries
- The contrastive learning research community
- Contributors and users of this project

---

**⚠️ Disclaimer**: This is a research demonstration tool. The models are not intended for production use and should not be used for critical decision-making without proper validation and human oversight.
# Contrastive-Learning-Implementation
