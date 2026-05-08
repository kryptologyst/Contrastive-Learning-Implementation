"""Test suite for contrastive learning implementation."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.simclr import SimCLR, ContrastiveLoss, create_simclr_model
from src.models.moco import MoCo, MoCoLoss, create_moco_model
from src.models.swav import SwAV, SwAVLoss, create_swav_model
from src.models.baselines import BaselineCNN, BaselineResNet, create_baseline_model
from src.losses.contrastive import NTXentLoss, InfoNCELoss, get_loss_function
from src.data.augmentations import SimCLRAugmentation, MoCoAugmentation, SwAVAugmentation
from src.utils.device import get_device, set_seed, count_parameters
from src.utils.safety import SafetyChecker, print_safety_disclaimer


class TestSimCLR:
    """Test SimCLR implementation."""
    
    def test_simclr_creation(self):
        """Test SimCLR model creation."""
        model = SimCLR(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
            hidden_dim=512,
        )
        
        assert isinstance(model, nn.Module)
        assert model.projection_dim == 128
        assert model.hidden_dim == 512
    
    def test_simclr_forward(self):
        """Test SimCLR forward pass."""
        model = SimCLR(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
        )
        
        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_simclr_embeddings(self):
        """Test SimCLR embeddings."""
        model = SimCLR(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
        )
        
        x = torch.randn(2, 3, 224, 224)
        embeddings = model.get_embeddings(x)
        
        # Check normalization
        norms = torch.norm(embeddings, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_contrastive_loss(self):
        """Test contrastive loss."""
        loss_fn = ContrastiveLoss(temperature=0.5)
        
        # Create dummy embeddings
        z_i = torch.randn(4, 128)
        z_j = torch.randn(4, 128)
        
        loss = loss_fn(z_i, z_j)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_create_simclr_model(self):
        """Test SimCLR model creation function."""
        model, loss_fn = create_simclr_model(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
        )
        
        assert isinstance(model, SimCLR)
        assert isinstance(loss_fn, ContrastiveLoss)


class TestMoCo:
    """Test MoCo implementation."""
    
    def test_moco_creation(self):
        """Test MoCo model creation."""
        model = MoCo(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
            momentum=0.999,
            queue_size=1024,
        )
        
        assert isinstance(model, nn.Module)
        assert model.momentum == 0.999
        assert model.queue_size == 1024
    
    def test_moco_forward(self):
        """Test MoCo forward pass."""
        model = MoCo(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
            queue_size=1024,
        )
        
        # Create dummy inputs
        im_q = torch.randn(2, 3, 224, 224)
        im_k = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        q, k = model(im_q, im_k)
        
        assert q.shape == (2, 128)
        assert k.shape == (2, 128)
        assert not torch.isnan(q).any()
        assert not torch.isnan(k).any()
    
    def test_moco_loss(self):
        """Test MoCo loss."""
        loss_fn = MoCoLoss(temperature=0.07)
        
        # Create dummy data
        q = torch.randn(4, 128)
        k = torch.randn(4, 128)
        queue = torch.randn(128, 1024)
        
        loss = loss_fn(q, k, queue)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestSwAV:
    """Test SwAV implementation."""
    
    def test_swav_creation(self):
        """Test SwAV model creation."""
        model = SwAV(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
            num_prototypes=1000,
        )
        
        assert isinstance(model, nn.Module)
        assert model.num_prototypes == 1000
    
    def test_swav_forward(self):
        """Test SwAV forward pass."""
        model = SwAV(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
            num_prototypes=1000,
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()
    
    def test_swav_prototypes(self):
        """Test SwAV prototypes."""
        model = SwAV(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
            num_prototypes=1000,
        )
        
        prototypes = model.get_prototypes()
        
        assert prototypes.shape == (1000, 128)
        assert not torch.isnan(prototypes).any()


class TestBaselines:
    """Test baseline models."""
    
    def test_baseline_cnn(self):
        """Test baseline CNN."""
        model = BaselineCNN(num_classes=10)
        
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
    
    def test_baseline_resnet(self):
        """Test baseline ResNet."""
        model = BaselineResNet(num_classes=10, pretrained=False)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
    
    def test_create_baseline_model(self):
        """Test baseline model creation."""
        model = create_baseline_model("cnn", num_classes=10)
        
        assert isinstance(model, BaselineCNN)
        assert model.fc2.out_features == 10


class TestLossFunctions:
    """Test loss functions."""
    
    def test_ntxent_loss(self):
        """Test NT-Xent loss."""
        loss_fn = NTXentLoss(temperature=0.5)
        
        z_i = torch.randn(4, 128)
        z_j = torch.randn(4, 128)
        
        loss = loss_fn(z_i, z_j)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_infonce_loss(self):
        """Test InfoNCE loss."""
        loss_fn = InfoNCELoss(temperature=0.07)
        
        query = torch.randn(4, 128)
        positive = torch.randn(4, 128)
        
        loss = loss_fn(query, positive)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_get_loss_function(self):
        """Test loss function creation."""
        loss_fn = get_loss_function("ntxent", temperature=0.5)
        
        assert isinstance(loss_fn, NTXentLoss)
        assert loss_fn.temperature == 0.5


class TestAugmentations:
    """Test data augmentations."""
    
    def test_simclr_augmentation(self):
        """Test SimCLR augmentation."""
        aug = SimCLRAugmentation(image_size=224)
        
        # Create dummy PIL image
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="red")
        
        view1, view2 = aug(img)
        
        assert isinstance(view1, torch.Tensor)
        assert isinstance(view2, torch.Tensor)
        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)
    
    def test_moco_augmentation(self):
        """Test MoCo augmentation."""
        aug = MoCoAugmentation(image_size=224)
        
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="blue")
        
        view1, view2 = aug(img)
        
        assert isinstance(view1, torch.Tensor)
        assert isinstance(view2, torch.Tensor)
        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("cpu")
        
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Check if seed is set
        assert True  # Seed setting doesn't return anything
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(10, 5)
        count = count_parameters(model)
        
        assert count == 55  # 10*5 + 5 bias terms


class TestSafety:
    """Test safety and compliance."""
    
    def test_safety_checker(self):
        """Test safety checker."""
        checker = SafetyChecker()
        
        model = nn.Linear(10, 5)
        config = {
            "train": {"batch_size": 256},
            "optimizer": {"lr": 0.001},
            "model": {"temperature": 0.5},
            "safety": {"disclaimer": True, "research_only": True},
        }
        
        report = checker.check_model_safety(model, config)
        
        assert isinstance(report, dict)
        assert "safe" in report
        assert "warnings" in report
        assert "errors" in report
        assert "recommendations" in report
    
    def test_safety_disclaimer(self):
        """Test safety disclaimer."""
        # This should not raise an exception
        print_safety_disclaimer()
        assert True


# Integration tests
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_simclr(self):
        """Test end-to-end SimCLR training."""
        # Create model
        model, loss_fn = create_simclr_model(
            base_model="resnet18",
            pretrained=False,
            projection_dim=128,
        )
        
        # Create dummy data
        x1 = torch.randn(4, 3, 224, 224)
        x2 = torch.randn(4, 3, 224, 224)
        
        # Forward pass
        z1 = model(x1)
        z2 = model(x2)
        
        # Compute loss
        loss = loss_fn(z1, z2)
        
        # Backward pass
        loss.backward()
        
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_model_comparison(self):
        """Test model comparison."""
        models = {
            "simclr": create_simclr_model(pretrained=False)[0],
            "moco": create_moco_model(pretrained=False)[0],
        }
        
        x = torch.randn(2, 3, 224, 224)
        
        for name, model in models.items():
            output = model(x)
            assert output.shape[0] == 2
            assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
