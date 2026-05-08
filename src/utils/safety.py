"""Safety and compliance measures for contrastive learning."""

import os
import warnings
from typing import Any, Dict, List, Optional

import torch
from rich.console import Console
from rich.panel import Panel

console = Console()


class SafetyChecker:
    """Safety checker for contrastive learning models."""
    
    def __init__(self):
        """Initialize safety checker."""
        self.warnings = []
        self.errors = []
    
    def check_model_safety(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check model safety and compliance.
        
        Args:
            model: PyTorch model
            config: Model configuration
            
        Returns:
            Safety report
        """
        report = {
            "safe": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }
        
        # Check model size
        self._check_model_size(model, report)
        
        # Check configuration
        self._check_configuration(config, report)
        
        # Check for potential issues
        self._check_potential_issues(model, report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _check_model_size(self, model: torch.nn.Module, report: Dict[str, Any]) -> None:
        """Check model size and complexity."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check parameter count
        if total_params > 100_000_000:  # 100M parameters
            report["warnings"].append(
                "Model has over 100M parameters. Consider using smaller models for efficiency."
            )
        
        if trainable_params > 50_000_000:  # 50M trainable parameters
            report["warnings"].append(
                "Model has many trainable parameters. Ensure sufficient computational resources."
            )
        
        # Check model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        if model_size_mb > 500:  # 500MB
            report["warnings"].append(
                f"Model size is {model_size_mb:.1f}MB. Consider model compression techniques."
            )
    
    def _check_configuration(self, config: Dict[str, Any], report: Dict[str, Any]) -> None:
        """Check configuration for safety issues."""
        # Check batch size
        batch_size = config.get("train", {}).get("batch_size", 32)
        if batch_size > 512:
            report["warnings"].append(
                "Large batch size detected. Ensure sufficient GPU memory."
            )
        
        # Check learning rate
        lr = config.get("optimizer", {}).get("lr", 0.001)
        if lr > 0.01:
            report["warnings"].append(
                "High learning rate detected. Monitor for training instability."
            )
        
        # Check temperature parameter
        temperature = config.get("model", {}).get("temperature", 0.5)
        if temperature < 0.01:
            report["warnings"].append(
                "Very low temperature parameter. May cause numerical instability."
            )
        
        # Check for safety flags
        safety_config = config.get("safety", {})
        if not safety_config.get("disclaimer", False):
            report["errors"].append(
                "Safety disclaimer not enabled. Enable in configuration."
            )
        
        if not safety_config.get("research_only", False):
            report["errors"].append(
                "Research-only flag not enabled. Enable in configuration."
            )
    
    def _check_potential_issues(self, model: torch.nn.Module, report: Dict[str, Any]) -> None:
        """Check for potential issues in the model."""
        # Check for NaN parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                report["errors"].append(f"NaN detected in parameter: {name}")
        
        # Check for infinite parameters
        for name, param in model.named_parameters():
            if torch.isinf(param).any():
                report["errors"].append(f"Infinite values detected in parameter: {name}")
        
        # Check for very large parameters
        for name, param in model.named_parameters():
            if param.abs().max() > 100:
                report["warnings"].append(
                    f"Large parameter values detected in: {name}"
                )
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> None:
        """Generate safety recommendations."""
        recommendations = [
            "Always validate model outputs before deployment",
            "Monitor model performance on diverse datasets",
            "Implement proper error handling and logging",
            "Use deterministic seeding for reproducibility",
            "Regularly backup model checkpoints",
            "Document model limitations and assumptions",
            "Test model robustness to adversarial inputs",
            "Consider model interpretability and explainability",
        ]
        
        report["recommendations"].extend(recommendations)
    
    def print_safety_report(self, report: Dict[str, Any]) -> None:
        """Print safety report."""
        if report["errors"]:
            console.print(Panel(
                "\n".join(report["errors"]),
                title="❌ Safety Errors",
                border_style="red"
            ))
        
        if report["warnings"]:
            console.print(Panel(
                "\n".join(report["warnings"]),
                title="⚠️ Safety Warnings",
                border_style="yellow"
            ))
        
        if report["recommendations"]:
            console.print(Panel(
                "\n".join(report["recommendations"]),
                title="💡 Recommendations",
                border_style="blue"
            ))
        
        # Overall safety status
        if report["safe"]:
            console.print("✅ Model passed safety checks")
        else:
            console.print("❌ Model failed safety checks")


class ComplianceManager:
    """Compliance manager for ethical AI practices."""
    
    def __init__(self):
        """Initialize compliance manager."""
        self.compliance_checks = []
        self.ethical_guidelines = []
    
    def check_data_compliance(
        self,
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check dataset compliance.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            Compliance report
        """
        report = {
            "compliant": True,
            "issues": [],
            "recommendations": [],
        }
        
        # Check for sensitive data
        if dataset_info.get("contains_pii", False):
            report["issues"].append("Dataset contains personally identifiable information")
            report["compliant"] = False
        
        # Check for bias
        if dataset_info.get("bias_risk", "high") == "high":
            report["issues"].append("Dataset has high bias risk")
            report["recommendations"].append("Implement bias detection and mitigation")
        
        # Check for consent
        if not dataset_info.get("consent_obtained", False):
            report["issues"].append("Consent not obtained for data use")
            report["compliant"] = False
        
        # Check for privacy
        if not dataset_info.get("privacy_protected", False):
            report["issues"].append("Privacy protection not implemented")
            report["recommendations"].append("Implement privacy-preserving techniques")
        
        return report
    
    def check_model_compliance(
        self,
        model_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check model compliance.
        
        Args:
            model_info: Model information
            
        Returns:
            Compliance report
        """
        report = {
            "compliant": True,
            "issues": [],
            "recommendations": [],
        }
        
        # Check for fairness
        if not model_info.get("fairness_tested", False):
            report["issues"].append("Model fairness not tested")
            report["recommendations"].append("Implement fairness testing")
        
        # Check for transparency
        if not model_info.get("transparent", False):
            report["issues"].append("Model not transparent")
            report["recommendations"].append("Implement explainability features")
        
        # Check for accountability
        if not model_info.get("accountable", False):
            report["issues"].append("Model not accountable")
            report["recommendations"].append("Implement accountability measures")
        
        return report
    
    def generate_compliance_report(
        self,
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            dataset_info: Dataset information
            model_info: Model information
            
        Returns:
            Comprehensive compliance report
        """
        dataset_report = self.check_data_compliance(dataset_info)
        model_report = self.check_model_compliance(model_info)
        
        overall_compliant = (
            dataset_report["compliant"] and model_report["compliant"]
        )
        
        return {
            "overall_compliant": overall_compliant,
            "dataset_compliance": dataset_report,
            "model_compliance": model_report,
            "recommendations": (
                dataset_report["recommendations"] + model_report["recommendations"]
            ),
        }


class EthicsFramework:
    """Ethics framework for contrastive learning."""
    
    def __init__(self):
        """Initialize ethics framework."""
        self.principles = [
            "Fairness and Non-discrimination",
            "Transparency and Explainability",
            "Privacy and Data Protection",
            "Accountability and Responsibility",
            "Human Autonomy and Oversight",
            "Beneficence and Non-maleficence",
        ]
    
    def assess_ethical_impact(
        self,
        use_case: str,
        stakeholders: List[str],
        potential_harms: List[str],
    ) -> Dict[str, Any]:
        """
        Assess ethical impact of model use.
        
        Args:
            use_case: Intended use case
            stakeholders: Affected stakeholders
            potential_harms: Potential harms
            
        Returns:
            Ethical impact assessment
        """
        assessment = {
            "ethical_risk": "low",
            "recommendations": [],
            "mitigation_strategies": [],
        }
        
        # Assess risk level
        if any(harm in ["discrimination", "privacy_violation", "bias"] for harm in potential_harms):
            assessment["ethical_risk"] = "high"
        elif any(harm in ["unfairness", "lack_of_transparency"] for harm in potential_harms):
            assessment["ethical_risk"] = "medium"
        
        # Generate recommendations
        if assessment["ethical_risk"] == "high":
            assessment["recommendations"].extend([
                "Conduct thorough ethical review",
                "Implement robust monitoring",
                "Establish human oversight",
                "Develop mitigation strategies",
            ])
        
        # Generate mitigation strategies
        assessment["mitigation_strategies"].extend([
            "Regular bias testing",
            "Diverse stakeholder engagement",
            "Transparent documentation",
            "Continuous monitoring",
        ])
        
        return assessment
    
    def print_ethics_guidelines(self) -> None:
        """Print ethics guidelines."""
        console.print(Panel(
            "\n".join(f"• {principle}" for principle in self.principles),
            title="🤝 Ethical AI Principles",
            border_style="green"
        ))


def create_safety_checker() -> SafetyChecker:
    """Create a new safety checker instance."""
    return SafetyChecker()


def create_compliance_manager() -> ComplianceManager:
    """Create a new compliance manager instance."""
    return ComplianceManager()


def create_ethics_framework() -> EthicsFramework:
    """Create a new ethics framework instance."""
    return EthicsFramework()


def print_safety_disclaimer() -> None:
    """Print safety disclaimer."""
    disclaimer = """
    ⚠️  SAFETY DISCLAIMER  ⚠️
    
    This is a research demonstration tool for contrastive learning.
    
    IMPORTANT NOTICES:
    • This model is NOT intended for production use
    • Do NOT use for critical decision-making without proper validation
    • Always require human oversight for important decisions
    • Be aware of potential biases and limitations
    • Ensure compliance with applicable regulations
    
    The authors and contributors are not responsible for any misuse
    or damages resulting from the use of this software.
    """
    
    console.print(Panel(
        disclaimer,
        title="🛡️ Safety Notice",
        border_style="red"
    ))
