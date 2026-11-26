"""ViT Shapley for uncertainty: a small learned explainer that outputs Shapley values in one pass.

Idea (Covert, Kim & Lee, ICLR 2023, "Learning to Estimate Shapley Values with Vision Transformers"):
instead of evaluating hundreds of coalitions for every new image (KernelSHAP), train a model once
that maps an image to its Shapley values. They train it **without** ground-truth Shapley values,
with the KernelSHAP objective: Shapley values are the solution of

    minimize    E_{S ~ Shapley kernel} [ ( v(S) - v(empty) - sum_{i in S} phi_i )^2 ]
    subject to  sum_i phi_i = v(N) - v(empty)          (efficiency)

So training only needs game values v(S) of randomly drawn coalitions S, which we already compute
for the masked training heads (:func:`vit_uncertainty.features.masked_training_features`).

What we change, and why it is new:

* **Four games at once**: the explainer predicts Shapley values for P, TU, AU, and EU of a probly
  predictor. Covert et al. explain one class probability.
* **Superpixel players**: it predicts one value per patch token and sums the tokens of each SLIC
  superpixel. Covert et al. use the fixed patch grid and name superpixels as future work.
* **Light version**: the ViT stays frozen. The tokens of each superpixel are averaged into one *player
  token*, and a small transformer over the 16 player tokens predicts their values (trained in minutes on
  a CPU). The original fine-tunes a whole ViT as explainer and predicts one value per patch.

Efficiency is guaranteed by construction with the *additive efficient normalization* of Covert et
al.: whatever the network predicts, the missing amount ``(v(N) - v(empty)) - sum_i phi_i`` is spread
evenly over the players.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

QUANTITY_ORDER = ("P", "TU", "AU", "EU")


def player_tokens(tokens: torch.Tensor, token_players: torch.Tensor, n_players: int) -> torch.Tensor:
    """Average the frozen ViT tokens of each player and append the player's area share.

    Args:
        tokens: ``(B, T, D)`` frozen ViT patch tokens.
        token_players: ``(B, T)`` player id of every token.
        n_players: Number of players per image.

    Returns:
        ``(B, n_players, D + 1)``: mean token of each player, then the share of tokens it owns.
    """
    batch, _, dim = tokens.shape
    index = token_players.long()
    sums = torch.zeros(batch, n_players, dim, dtype=torch.float32).scatter_add(
        1, index[..., None].expand(-1, -1, dim), tokens.float()
    )
    counts = torch.zeros(batch, n_players).scatter_add(1, index, torch.ones_like(index, dtype=torch.float32))
    means = sums / counts.clamp_min(1)[..., None]
    return torch.cat([means, (counts / tokens.shape[1])[..., None]], dim=-1)


class ExplainerNet(nn.Module):
    """Small transformer over player tokens: one output per player and quantity.

    Every player sees all other players through attention, so its value can depend on the whole image,
    as a Shapley value does.

    Args:
        token_dim: Dimension of a player token (ViT width + 1 for the area share).
        width: Internal width.
        depth: Number of transformer layers.
        heads: Attention heads.
        n_quantities: Outputs per player (P, TU, AU, EU).
    """

    def __init__(
        self, token_dim: int = 385, width: int = 128, depth: int = 2, heads: int = 4, n_quantities: int = 4
    ) -> None:
        """Build the input projection, the transformer layers, and the output layer."""
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, width))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(width, n_quantities)

    def forward(self, players: torch.Tensor) -> torch.Tensor:
        """``(B, n_players, token_dim)`` -> ``(B, n_players, n_quantities)``."""
        return self.out(self.encoder(self.inp(players)))


def efficient(phi: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    """Additive efficient normalization (Covert et al.): spread the missing amount evenly over the players.

    Args:
        phi: ``(B, n_players, Q)`` raw estimates.
        gap: ``(B, Q)`` target sum ``v(N) - v(empty)``.

    Returns:
        ``(B, n_players, Q)`` estimates whose sum over players equals ``gap``.
    """
    return phi + (gap - phi.sum(1))[:, None, :] / phi.shape[1]


class AmortizedUncertaintyExplainer:
    """Train once, then explain P, TU, AU, and EU of a probly predictor in one forward pass.

    Args:
        n_players: Players (superpixels) per image.
        token_dim: Dimension of the frozen ViT patch tokens.
        width: Width of the explainer network (32: larger networks did not generalize better with our data).
        depth: Transformer layers of the explainer network.
        seed: Seed of the initial weights and the minibatch order.
    """

    def __init__(
        self, n_players: int = 16, *, token_dim: int = 384, width: int = 32, depth: int = 1, seed: int = 0
    ) -> None:
        """Create an untrained explainer network."""
        torch.manual_seed(seed)
        self.n_players, self.seed = n_players, seed
        self.net = ExplainerNet(token_dim + 1, width, depth)
        # Scale of each quantity (set in fit), so that P (0..1) and EU (about 0.05) count equally in the loss.
        self.scale = torch.ones(len(QUANTITY_ORDER))
        self.history: list[dict] = []

    # The training data is one dictionary per split:
    #   tokens        (I, T, D)   frozen ViT patch tokens of image i (float16 is fine)
    #   token_players (I, T)      superpixel of every token
    #   empty         (Q,)        v(empty) for the four quantities (the same for every image)
    #   full          (I, Q)      v(N) of image i
    #   rows          (M,)        image index of every sampled coalition
    #   coalitions    (M, n)      sampled coalitions (Shapley kernel sizes)
    #   values        (M, Q)      v(S) of every sampled coalition

    def _players(self, data: dict) -> torch.Tensor:
        # The tokens are frozen, so the player tokens of a dataset are computed once and kept.
        if "_players" not in data:
            data["_players"] = torch.cat(
                [
                    player_tokens(
                        data["tokens"][first : first + 256], data["token_players"][first : first + 256], self.n_players
                    )
                    for first in range(0, len(data["tokens"]), 256)
                ]
            )
        return data["_players"]

    def _loss(self, data: dict, images: torch.Tensor) -> torch.Tensor:
        gap = data["full"][images] - data["empty"]
        phi = efficient(self.net(self._players(data)[images]), gap)
        # Coalitions that belong to these images.
        select = torch.isin(data["rows"], images)
        rows = data["rows"][select]
        local = torch.searchsorted(images, rows)
        coalitions = data["coalitions"][select].float()
        predicted = data["empty"] + torch.einsum("mn,mnq->mq", coalitions, phi[local])
        error = (predicted - data["values"][select]) / self.scale
        return (error**2).mean()

    def fit(
        self,
        train: dict,
        val: dict,
        *,
        epochs: int = 40,
        batch_images: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        verbose: bool = True,
    ) -> list[dict]:
        """Train with the KernelSHAP objective; keep the epoch with the lowest validation loss.

        Returns:
            Training history (one dictionary per epoch).
        """
        self.scale = train["values"].std(0).clamp_min(1e-6)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(self.seed)
        n_images = len(train["tokens"])
        best, best_state = float("inf"), None
        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            self.net.train()
            order = torch.randperm(n_images, generator=generator)
            losses = []
            for first in range(0, n_images, batch_images):
                images = order[first : first + batch_images].sort().values
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(train, images)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            self.net.eval()
            with torch.no_grad():
                val_loss = float(
                    np.mean(
                        [
                            self._loss(val, torch.arange(first, min(first + 256, len(val["tokens"])))).item()
                            for first in range(0, len(val["tokens"]), 256)
                        ]
                    )
                )
            self.history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"Epoche {epoch:3d} | Train {np.mean(losses):.4f} | Val {val_loss:.4f} | "
                    f"{time.perf_counter() - start:.0f} s"
                )
        if best_state is None:  # only possible if every validation loss was NaN
            msg = "Training found no epoch with a finite validation loss."
            raise RuntimeError(msg)
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self.history

    @torch.no_grad()
    def explain(
        self, tokens: torch.Tensor, token_players: torch.Tensor, full: torch.Tensor, empty: torch.Tensor
    ) -> np.ndarray:
        """Shapley value estimates for new images in one forward pass.

        Args:
            tokens: ``(B, T, D)`` frozen ViT patch tokens.
            token_players: ``(B, T)`` superpixel of every token.
            full: ``(B, Q)`` v(N) of every image.
            empty: ``(Q,)`` v(empty).

        Returns:
            ``(B, n_players, Q)`` array, quantities in the order ``P, TU, AU, EU``.
        """
        self.net.eval()
        phi = efficient(self.net(player_tokens(tokens, token_players, self.n_players)), full - empty)
        return phi.double().numpy()

    def state_dict(self) -> dict:
        """Weights, scale, and history (for saving)."""
        return {
            "net": self.net.state_dict(),
            "scale": self.scale,
            "history": self.history,
            "n_players": self.n_players,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore what :meth:`state_dict` returned."""
        self.net.load_state_dict(state["net"])
        self.scale, self.history = state["scale"], state["history"]
