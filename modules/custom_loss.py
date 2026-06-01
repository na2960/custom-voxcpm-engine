import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, l1_weight: float = 1e-5):
        """
        Custom loss wrapper for fine-tuning foundational speech models.
        
        Args:
            l1_weight: Scalar weight for L1 regularization of trainable parameters.
                       Helps prevent overfitting to small personal voice datasets.
        """
        super(CustomLoss, self).__init__()
        self.l1_weight = l1_weight

    def forward(self, model_output, model_parameters=None) -> torch.Tensor:
        """
        Args:
            model_output: The output object returned from calling VoxCPM2. 
                           Must contain a '.loss' attribute.
            model_parameters: Iterator of model parameters (model.named_parameters())
                              used to calculate regularization penalties.
        """
        # 1. Extract the base continuous flow-matching/diffusion loss tensor
        base_loss = model_output.loss
        
        # If no custom parameter metrics are passed, pass the base loss straight through
        if model_parameters is None or self.l1_weight == 0.0:
            return base_loss
            
        # 2. Add an L1 regularization tensor to keep LoRA weight matrices stable
        l1_reg = torch.tensor(0.0, device=base_loss.device, dtype=base_loss.dtype)
        
        for name, param in model_parameters:
            if param.requires_grad:  # Only track our active LoRA adapter weights
                l1_reg += torch.sum(torch.abs(param))
                
        # 3. Calculate total loss matrix
        total_loss = base_loss + (self.l1_weight * l1_reg)
        
        return total_loss