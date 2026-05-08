"""Loss functions module."""

from .contrastive import (
    NTXentLoss,
    InfoNCELoss,
    TripletLoss,
    SupConLoss,
    BarlowTwinsLoss,
    get_loss_function,
)

__all__ = [
    "NTXentLoss",
    "InfoNCELoss",
    "TripletLoss",
    "SupConLoss",
    "BarlowTwinsLoss",
    "get_loss_function",
]
