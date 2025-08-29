from typing import Literal
import torch


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
                 tau_1: float = 100,
                 tau_2: float = 0.1,
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
        z: [N, L] tensor. N: sample size. L: sequence length.
        '''
        if z.dim() != 2 or z.size(0) < 2:
            raise ValueError(f'DispersionLoss only supports 2D tensors, but got {z.shape}.')

        N, L = z.shape

        if self.variant == "covariance":
            # NOTE: The covariance matrix `Cov` has shape [L, L].
            # \sum_{(m,n), m != n} Cov_{mn}^2, after l2-normalization
            z_norm = (z - z.mean(0)) / z.std(0)
            Cov = z_norm.T @ z_norm / (N - 1)
            non_diag_mask = ~torch.eye(L, dtype=torch.bool, device=z.device)
            # divide by L to make the loss scale more reasonable.
            return (Cov ** 2)[non_diag_mask].sum() / L

        elif self.variant == "infonce_l2":
            # NOTE: The distance matrix matrix `D` has shape [N, N].
            D = torch.cdist(z, z, p=2) ** 2
            non_diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
            return torch.log(torch.mean(torch.exp(-D[non_diag_mask] / self.tau_1)))

        elif self.variant == "infonce_cosine":
            # NOTE: The distance matrix matrix `D` has shape [N, N].
            D = - z @ z.T / (torch.linalg.norm(z, dim=1) ** 2)
            non_diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
            return torch.log(torch.mean(torch.exp(-D[non_diag_mask] / self.tau_2)))

        else:
            # NOTE: The distance matrix matrix `D` has shape [N, N].
            D = - z @ z.T / (torch.linalg.norm(z, dim=1) ** 2)
            non_diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
            margin = torch.clamp(self.margin - D[non_diag_mask], min=0.0)
            # divide by L to make the loss scale more reasonable.
            return (margin ** 2).sum() / L


if __name__ == '__main__':
    for variant in ['covariance', 'infonce_l2', 'infonce_cosine', 'hinge']:
        print(f"Variant: {variant}")
        loss_fn = DispersionLoss(variant=variant)
        z = torch.randn(128, 2048, requires_grad=True)
        loss = loss_fn(z)
        print(f"Loss: {loss.item():.3f}")
        loss.backward()
        print(f"Gradient norm: {torch.norm(z.grad).item():.6f}")
