"""Interactive demo application for contrastive learning."""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.simclr import create_simclr_model
from ..models.moco import create_moco_model
from ..models.swav import create_swav_model
from ..data.augmentations import SimCLRAugmentation, MoCoAugmentation, SwAVAugmentation
from ..utils.device import get_device
from .plots import plot_embeddings, plot_similarity_matrix


class ContrastiveDemo:
    """Interactive demo for contrastive learning."""
    
    def __init__(
        self,
        model_type: str = "simclr",
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize demo.
        
        Args:
            model_type: Type of model to use
            checkpoint_path: Path to model checkpoint
            device: Device to use
        """
        self.model_type = model_type
        self.device = device or get_device()
        
        # Load model
        if model_type.lower() == "simclr":
            self.model, _ = create_simclr_model()
            self.augmentation = SimCLRAugmentation()
        elif model_type.lower() == "moco":
            self.model, _ = create_moco_model()
            self.augmentation = MoCoAugmentation()
        elif model_type.lower() == "swav":
            self.model, _ = create_swav_model()
            self.augmentation = SwAVAugmentation()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def process_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process image for contrastive learning.
        
        Args:
            image: Input PIL image
            
        Returns:
            Tuple of (view1, view2)
        """
        # Apply augmentation
        view1, view2 = self.augmentation(image)
        
        # Add batch dimension
        view1 = view1.unsqueeze(0).to(self.device)
        view2 = view2.unsqueeze(0).to(self.device)
        
        return view1, view2
    
    def get_embeddings(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings for an image.
        
        Args:
            image: Input PIL image
            
        Returns:
            Tuple of (embeddings1, embeddings2)
        """
        # Process image
        view1, view2 = self.process_image(image)
        
        # Get embeddings
        with torch.no_grad():
            if hasattr(self.model, "get_embeddings"):
                embeddings1 = self.model.get_embeddings(view1)
                embeddings2 = self.model.get_embeddings(view2)
            else:
                embeddings1 = self.model(view1)
                embeddings2 = self.model(view2)
        
        return embeddings1, embeddings2
    
    def compute_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Compute similarity between two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Similarity score
        """
        # Get embeddings
        embeddings1, _ = self.get_embeddings(image1)
        embeddings2, _ = self.get_embeddings(image2)
        
        # Compute similarity
        similarity = F.cosine_similarity(embeddings1, embeddings2, dim=-1)
        
        return similarity.item()
    
    def visualize_embeddings(self, images: list, labels: list) -> str:
        """
        Visualize embeddings of multiple images.
        
        Args:
            images: List of PIL images
            labels: List of labels
            
        Returns:
            Path to saved plot
        """
        # Get embeddings for all images
        all_embeddings = []
        all_labels = []
        
        for image, label in zip(images, labels):
            embeddings, _ = self.get_embeddings(image)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(label)
        
        # Concatenate embeddings
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.tensor(all_labels)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Use t-SNE for visualization
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_tensor.numpy())
        
        # Plot
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels_tensor.numpy(),
            cmap="tab10",
            alpha=0.7,
            s=100,
        )
        plt.colorbar(scatter)
        plt.title("Embedding Visualization (t-SNE)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        
        # Save plot
        plot_path = "./assets/plots/embedding_visualization.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        return plot_path
    
    def create_demo_interface(self) -> gr.Blocks:
        """
        Create Gradio demo interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Contrastive Learning Demo") as demo:
            gr.Markdown(
                """
                # Contrastive Learning Demo
                
                This demo showcases contrastive learning methods including SimCLR, MoCo, and SwAV.
                Upload images to explore how these models learn representations.
                """
            )
            
            with gr.Tab("Image Similarity"):
                gr.Markdown("## Image Similarity Comparison")
                
                with gr.Row():
                    with gr.Column():
                        image1 = gr.Image(
                            label="First Image",
                            type="pil",
                            height=200,
                        )
                        image2 = gr.Image(
                            label="Second Image",
                            type="pil",
                            height=200,
                        )
                    
                    with gr.Column():
                        similarity_score = gr.Number(
                            label="Similarity Score",
                            precision=4,
                        )
                        similarity_plot = gr.Plot(
                            label="Similarity Visualization",
                        )
                
                def compute_similarity_and_plot(img1, img2):
                    if img1 is None or img2 is None:
                        return 0.0, None
                    
                    # Compute similarity
                    similarity = self.compute_similarity(img1, img2)
                    
                    # Create similarity plot
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(["Similarity"], [similarity], color="skyblue")
                    ax.set_ylim(0, 1)
                    ax.set_title("Image Similarity Score")
                    ax.set_ylabel("Cosine Similarity")
                    
                    return similarity, fig
                
                image1.change(
                    compute_similarity_and_plot,
                    inputs=[image1, image2],
                    outputs=[similarity_score, similarity_plot],
                )
                
                image2.change(
                    compute_similarity_and_plot,
                    inputs=[image1, image2],
                    outputs=[similarity_score, similarity_plot],
                )
            
            with gr.Tab("Embedding Visualization"):
                gr.Markdown("## Embedding Visualization")
                
                with gr.Row():
                    with gr.Column():
                        images = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        labels = gr.Textbox(
                            label="Labels (comma-separated)",
                            placeholder="0, 1, 2, 3, 4",
                        )
                    
                    with gr.Column():
                        embedding_plot = gr.Image(
                            label="Embedding Visualization",
                            height=400,
                        )
                
                def visualize_embeddings(files, label_text):
                    if files is None or not label_text:
                        return None
                    
                    # Parse labels
                    try:
                        labels_list = [int(x.strip()) for x in label_text.split(",")]
                    except ValueError:
                        return None
                    
                    # Load images
                    images_list = []
                    for file in files:
                        image = Image.open(file.name)
                        images_list.append(image)
                    
                    if len(images_list) != len(labels_list):
                        return None
                    
                    # Create visualization
                    plot_path = self.visualize_embeddings(images_list, labels_list)
                    
                    return plot_path
                
                images.change(
                    visualize_embeddings,
                    inputs=[images, labels],
                    outputs=[embedding_plot],
                )
                
                labels.change(
                    visualize_embeddings,
                    inputs=[images, labels],
                    outputs=[embedding_plot],
                )
            
            with gr.Tab("Model Information"):
                gr.Markdown("## Model Information")
                
                model_info = gr.Markdown(
                    f"""
                    **Model Type:** {self.model_type.upper()}
                    
                    **Device:** {self.device}
                    
                    **Model Architecture:**
                    - Base Model: ResNet-50
                    - Projection Dimension: 128
                    - Hidden Dimension: 512
                    
                    **Augmentation Strategy:**
                    - Random Resized Crop
                    - Random Horizontal Flip
                    - Color Jittering
                    - Gaussian Blur
                    
                    **Safety Notice:**
                    This is a research demonstration tool. The model is not intended for production use
                    and should not be used for critical decision-making without proper validation.
                    """
                )
        
        return demo


def create_demo_app(
    model_type: str = "simclr",
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> ContrastiveDemo:
    """
    Create demo application.
    
    Args:
        model_type: Type of model to use
        checkpoint_path: Path to model checkpoint
        device: Device to use
        
    Returns:
        Demo application
    """
    return ContrastiveDemo(model_type, checkpoint_path, device)


def launch_demo(
    model_type: str = "simclr",
    checkpoint_path: Optional[str] = None,
    port: int = 8501,
    device: Optional[torch.device] = None,
) -> None:
    """
    Launch demo application.
    
    Args:
        model_type: Type of model to use
        checkpoint_path: Path to model checkpoint
        port: Port to run on
        device: Device to use
    """
    # Create demo
    demo = create_demo_app(model_type, checkpoint_path, device)
    
    # Create interface
    interface = demo.create_demo_interface()
    
    # Launch
    interface.launch(server_port=port)
