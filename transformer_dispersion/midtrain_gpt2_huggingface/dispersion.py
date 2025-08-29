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
                 tau_1: float = 50,
                 tau_2: float = 0.5,
                 margin: float = 0.5,  # 0.5 angular cosine distance = orthogonal.
                 epsilon: float = 1e-4):
        super().__init__()
        variant = variant.lower()
        assert variant in {"infonce_l2", "infonce_cosine", "hinge", "covariance"}
        self.variant = variant
        self.tau_1 = float(tau_1)
        self.tau_2 = float(tau_2)
        self.margin = float(margin)
        self.epsilon = float(epsilon)

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
            z_centered = (z - z.mean(dim=2, keepdim=True)) / z.std(dim=2, keepdim=True)
            Cov = torch.matmul(z_centered, rearrange(z_centered, 'b l f -> b f l')) / (F - 1)
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device).unsqueeze(0).repeat(B, 1, 1)
            loss = Cov.pow(2).masked_select(non_diag).mean()
            return loss

        elif self.variant == "infonce_l2":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = torch.cdist(z, z, p=2).pow(2) / L
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            loss_b = torch.exp(-D / self.tau_1).masked_select(non_diag)
            return torch.log(loss_b.mean())

        elif self.variant == "infonce_cosine":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            z_norm = z / torch.linalg.norm(z, dim=2, keepdim=True)
            cossim = z_norm @ rearrange(z_norm, 'b l f -> b f l')
            D = torch.arccos(torch.clamp(cossim, self.epsilon, 1 - self.epsilon)) / torch.pi
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            loss = torch.log(torch.exp(-D / self.tau_2).masked_select(non_diag).mean())
            return loss

        else:
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            z_norm = z / torch.linalg.norm(z, dim=2, keepdim=True)
            cossim = z_norm @ rearrange(z_norm, 'b l f -> b f l')
            D = torch.arccos(torch.clamp(cossim, self.epsilon, 1 - self.epsilon)) / torch.pi
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            diff = torch.clamp(self.margin - D, min=0.0)
            loss = diff.pow(2).masked_select(non_diag).mean()
            return loss


if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from midtrain_gpt2 import CausalLMLoss

    import os
    import sys
    import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
    sys.path.insert(0, os.path.join(import_dir, 'prelim'))
    from utils.text_data import get_random_long_text

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # grab the first non-empty line
    text = get_random_long_text('wikipedia', min_word_count=1024, max_word_count=1500)
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=False
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.train()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    base_loss_fn = CausalLMLoss()
    base_loss = base_loss_fn(out.logits, input_ids)

    # Use all layer activations as z: [B, L, F]
    # Detach so each test is a fresh leaf tensor.
    z_base_list = [vec.detach() for vec in out.hidden_states]

    # 3) Your exact test loop, but with real hidden states
    for variant in ["covariance", "infonce_l2", "infonce_cosine", "hinge"]:
        print(f"\nVariant: {variant}")
        loss_fn = DispersionLoss(variant=variant)

        print(f"Base loss: {base_loss.item():.3f}")
        loss = 0
        for z_base in z_base_list:
            # fresh leaf w/ grads each time
            z = z_base.clone().requires_grad_(True)
            loss += loss_fn(z)
        loss /= len(z_base_list)
        print(f"Dispersion loss: {loss.item():.3f}")

        loss.backward()
        print(f"Gradient norm: {torch.norm(z.grad).item():.6f}")
