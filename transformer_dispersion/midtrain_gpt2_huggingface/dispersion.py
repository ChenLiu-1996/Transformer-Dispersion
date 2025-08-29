from typing import Literal
import torch
from einops import rearrange


class DispersionLoss(torch.nn.Module):
    '''
    Variants (exactly as in the table):

      InfoNCE_l2:     log E_{i,j}[exp(-D(z_i, z_j) / \tau_1)], D(z_i, z_j) = pdist(z_i, z_j, p=2)**2
      InfoNCE_cosine: log E_{i,j}[exp(-D(z_i, z_j) / \tau_2)], D(z_i, z_j) = - z_i z_j / (||z_i|| ||z_j||)
      Hinge:          E_{i,j}[max(0, margin - D(z_i, z_j))^2]
      Covariance:     \sum_{(m,n), m != n} Cov_{mn}^2, after l2-normalization

    Notes:
      - \tau_1, \tau_2 and margin are kept as internal constants for simplicity.
    '''
    def __init__(self,
                 variant: Literal["infonce_l2", "infonce_cosine", "hinge", "covariance"],
                 tau_1: float = 500,
                 tau_2: float = 0.05,
                 margin: float = 1.0):
        super().__init__()
        variant = variant.lower()
        assert variant in {"infonce_l2", "infonce_cosine", "hinge", "covariance"}
        self.variant = variant
        self.tau_1 = float(tau_1)
        self.tau_2 = float(tau_2)
        self.margin = float(margin)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        z: [B, L, F],
            where B: batch size. L: sequence length. F: feature dimension.
        '''
        if z.dim() != 3:
            raise ValueError(f'DispersionLoss only supports 3D [B, L, F]; got {tuple(z.shape)}.')

        B, L, F = z.shape

        if F < 2:
            raise ValueError(f'DispersionLoss expects F >= 2 in [B, L, F]; got {F}.')

        if self.variant == "covariance":
            # NOTE: The covariance matrix `Cov` has shape [B, L, L].
            # \sum_{(m,n), m != n} Cov_{mn}^2, after l2-normalization
            z_norm = (z - z.mean(dim=2, keepdim=True)) / z.std(dim=2, keepdim=True)
            Cov = torch.matmul(z_norm, rearrange(z_norm, 'b l f -> b f l')) / (F - 1)
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device).unsqueeze(0).repeat(B, 1, 1)
            # divide by L to make the loss scale more reasonable.
            loss_b = (Cov.pow(2) * non_diag).sum(dim=(1, 2)) / L
            return loss_b.mean()

        elif self.variant == "infonce_l2":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = torch.cdist(z, z, p=2) ** 2
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            loss_b = torch.exp(-D / self.tau_1).masked_select(non_diag)
            return torch.log(loss_b.mean())

        elif self.variant == "infonce_cosine":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = - z @ rearrange(z, 'b l f -> b f l') / (torch.linalg.norm(z, dim=2) ** 2).unsqueeze(1).repeat(1, L, 1)
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            loss_b = torch.exp(-D / self.tau_2).masked_select(non_diag)
            return torch.log(loss_b.mean())

        else:
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = - z @ rearrange(z, 'b l f -> b f l') / (torch.linalg.norm(z, dim=2) ** 2).unsqueeze(1).repeat(1, L, 1)
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            residual = torch.clamp(self.margin - D, min=0.0)
            # divide by L to make the loss scale more reasonable.
            loss_b = (residual.pow(2) * non_diag).sum(dim=(1, 2)) / L
            return loss_b.mean()


if __name__ == '__main__':
    for variant in ['covariance', 'infonce_l2', 'infonce_cosine', 'hinge']:
        print(f"Variant: {variant}")
        loss_fn = DispersionLoss(variant=variant)
        z = torch.randn(16, 1024, 768, requires_grad=True)
        loss = loss_fn(z)
        print(f"Loss: {loss.item():.3f}")
        loss.backward()
        print(f"Gradient norm: {torch.norm(z.grad).item():.6f}")
