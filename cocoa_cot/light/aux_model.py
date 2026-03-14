"""
CoCoA-CoT Light: Auxiliary MLP model and training pipeline.

Implements Eq. 15 from the paper:
    Û^L_CoCoA-CoT = u_A(a*|c*,x) · g_b(e_c(y*|x), e_a(y*|x))

Architecture:
    Linear(2*d_model, hidden_dim) → ReLU → Dropout(p)
    → Linear(hidden_dim, hidden_dim//2) → ReLU → Dropout(p)
    → Linear(hidden_dim//2, 1) → Softplus

The AuxiliaryModel g_b takes [e_c; e_a] as input and predicts a non-negative
scalar that approximates the CoCoA-CoT uncertainty score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class AuxiliaryModel(nn.Module):
    """2-layer MLP auxiliary model for CoCoA-CoT Light.

    Takes concatenated [e_c; e_a] ∈ R^{2*d_model} and produces a non-negative
    uncertainty scalar via Softplus activation at the output.

    Args:
        d_model: Hidden size of the base language model (size of each embedding).
        hidden_dim: Hidden dimension of the MLP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 4096,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        input_dim = 2 * d_model

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # ensures non-negative output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, 2*d_model)``.

        Returns:
            Output tensor of shape ``(batch, 1)`` with non-negative values.
        """
        return self.net(x)


class CoCoACoTLight:
    """Training and inference wrapper for CoCoA-CoT Light.

    Training procedure:
    1. Generate greedy outputs + hidden states on holdout set (no labels needed)
    2. Compute full CoCoA-CoT targets via M=10 sampling (expensive; cached)
    3. Extract (e_c, e_a) from greedy hidden states
    4. Train AuxiliaryModel to predict targets via MSE loss

    Inference (single forward pass, no sampling):
        Û^L = u_A(a*|c*,x) · g_b(e_c, e_a)

    Args:
        d_model: Hidden size of the base LM.
        hidden_dim: MLP hidden dimension.
        dropout: MLP dropout.
        layer_idx: Transformer layer for hidden state extraction.
        device: Compute device.
    """

    def __init__(
        self,
        d_model: int = 4096,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        layer_idx: int = 16,
        device: str = "cuda",
    ) -> None:
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.layer_idx = layer_idx
        self.device = device
        self.aux_model: Optional[AuxiliaryModel] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        holdout_prompts: list[str],
        full_cocoa_cot: "CoCoACoT",  # type: ignore[name-defined]
        cfg: dict,
        save_path: Optional[str] = None,
    ) -> dict:
        """Train the auxiliary model on holdout data.

        Args:
            holdout_prompts: List of unlabeled prompts for training.
            full_cocoa_cot: Trained :class:`~cocoa_cot.uncertainty.CoCoACoT`
                instance used to generate supervision targets.
            cfg: Configuration dict with keys from ``configs/base.yaml``
                (``light.hidden_dim``, ``light.dropout``, ``light.lr``,
                ``light.batch_size``, ``light.epochs``).
            save_path: If provided, saves the trained model to this path.

        Returns:
            Dictionary with training history: ``{"train_loss": list, "val_loss": list}``.
        """
        from cocoa_cot.light.dual_embedding import DualEmbeddingExtractor
        from cocoa_cot.uncertainty.information import PPLEstimator

        light_cfg = cfg.get("light", {})
        hidden_dim = light_cfg.get("hidden_dim", self.hidden_dim)
        dropout = light_cfg.get("dropout", self.dropout)
        lr = light_cfg.get("lr", 3e-4)
        batch_size = light_cfg.get("batch_size", 64)
        epochs = light_cfg.get("epochs", 30)

        hf_model = full_cocoa_cot.model

        # ── Step 1: Extract features ──────────────────────────────────────────
        logger.info("Extracting dual embeddings for %d holdout prompts", len(holdout_prompts))
        extractor = DualEmbeddingExtractor(hf_model, layer_idx=self.layer_idx)
        features = extractor.extract_batch(holdout_prompts)
        e_c_list = [f[0] for f in features]
        e_a_list = [f[1] for f in features]

        # Infer d_model from actual embeddings
        actual_d_model = e_c_list[0].shape[0] if e_c_list else self.d_model
        self.d_model = actual_d_model

        # ── Step 2: Compute CoCoA-CoT targets ─────────────────────────────────
        logger.info("Computing CoCoA-CoT targets for training…")
        targets = []
        for prompt in holdout_prompts:
            result = full_cocoa_cot.estimate(prompt)
            targets.append(float(result["uncertainty"]))

        # ── Step 3: Build AuxiliaryModel ──────────────────────────────────────
        self.aux_model = AuxiliaryModel(
            d_model=actual_d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        # ── Step 4: Train ─────────────────────────────────────────────────────
        X = np.concatenate([
            np.stack(e_c_list),
            np.stack(e_a_list),
        ], axis=1).astype(np.float32)

        y = np.array(targets, dtype=np.float32)

        # Train/val split (80/20)
        n = len(X)
        val_size = max(1, n // 5)
        X_train, X_val = X[val_size:], X[:val_size]
        y_train, y_val = y[val_size:], y[:val_size]

        X_train_t = torch.tensor(X_train, device=self.device)
        y_train_t = torch.tensor(y_train, device=self.device).unsqueeze(1)
        X_val_t = torch.tensor(X_val, device=self.device)
        y_val_t = torch.tensor(y_val, device=self.device).unsqueeze(1)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.aux_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.aux_model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.aux_model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= len(X_train)

            # Validation loss
            self.aux_model.eval()
            with torch.no_grad():
                val_pred = self.aux_model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d — train MSE: %.6f, val MSE: %.6f",
                    epoch + 1, epochs, epoch_loss, val_loss,
                )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.aux_model.state_dict(),
                    "d_model": actual_d_model,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "layer_idx": self.layer_idx,
                },
                save_path,
            )
            logger.info("AuxiliaryModel saved to %s", save_path)

        return history

    def load(self, path: str) -> None:
        """Load a trained auxiliary model from disk.

        Args:
            path: Path to the saved model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        d_model = checkpoint["d_model"]
        hidden_dim = checkpoint["hidden_dim"]
        dropout = checkpoint["dropout"]
        self.layer_idx = checkpoint.get("layer_idx", self.layer_idx)
        self.d_model = d_model

        self.aux_model = AuxiliaryModel(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)
        self.aux_model.load_state_dict(checkpoint["model_state_dict"])
        self.aux_model.eval()
        logger.info("AuxiliaryModel loaded from %s", path)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, prompt: str, model: "HFModel") -> float:  # type: ignore[name-defined]
        """Predict CoCoA-CoT Light uncertainty for a single prompt.

        Single forward pass through the auxiliary model — no sampling required.

        Û^L = u_A(a*|c*,x) · g_b(e_c, e_a)

        Args:
            prompt: Input prompt string.
            model: :class:`~cocoa_cot.models.HFModel` white-box model.

        Returns:
            Light uncertainty score (scalar).
        """
        if self.aux_model is None:
            raise RuntimeError("AuxiliaryModel not trained or loaded. Call train() or load() first.")

        from cocoa_cot.light.dual_embedding import DualEmbeddingExtractor
        from cocoa_cot.uncertainty.information import PPLEstimator

        extractor = DualEmbeddingExtractor(model, layer_idx=self.layer_idx)
        e_c, e_a = extractor.extract(prompt)

        # Auxiliary model prediction
        feat = np.concatenate([e_c, e_a]).astype(np.float32)
        feat_t = torch.tensor(feat, device=self.device).unsqueeze(0)

        self.aux_model.eval()
        with torch.no_grad():
            g_b = self.aux_model(feat_t).item()

        # u_A from greedy pass
        gen_out = model.generate_greedy(prompt)
        ppl_est = PPLEstimator(cot_mode=True)
        u_a = ppl_est.estimate(gen_out)

        return float(u_a * g_b)

    # ── Batch training from pre-computed arrays ───────────────────────────────

    def train(  # type: ignore[override]
        self,
        features: "np.ndarray",
        targets: "np.ndarray",
        cfg: dict,
    ) -> tuple[list[float], list[float]]:
        """Train from pre-computed feature matrix and target array.

        This variant is used by :mod:`cocoa_cot.experiments.run_light`
        which pre-computes the ``[e_c; e_a]`` features externally and
        passes them directly.

        Args:
            features: Float32 array of shape ``(N, 2*d_model)``.
            targets: Float32 array of shape ``(N,)`` — CoCoA-CoT targets.
            cfg: Light training config (``lr``, ``epochs``, ``batch_size``, etc.).

        Returns:
            ``(train_losses, val_losses)`` — per-epoch MSE history.
        """
        import numpy as _np

        light_cfg = cfg if isinstance(cfg, dict) else {}
        hidden_dim = light_cfg.get("hidden_dim", self.hidden_dim)
        dropout = light_cfg.get("dropout", self.dropout)
        lr = light_cfg.get("lr", 3e-4)
        batch_size = light_cfg.get("batch_size", 64)
        epochs = light_cfg.get("epochs", 30)

        X = features.astype(_np.float32)
        y = targets.astype(_np.float32)

        actual_d_model = X.shape[1] // 2
        self.d_model = actual_d_model
        self.aux_model = AuxiliaryModel(
            d_model=actual_d_model, hidden_dim=hidden_dim, dropout=dropout
        ).to(self.device)

        n = len(X)
        val_size = max(1, n // 5)
        X_train, X_val = X[val_size:], X[:val_size]
        y_train, y_val = y[val_size:], y[:val_size]

        X_train_t = torch.tensor(X_train, device=self.device)
        y_train_t = torch.tensor(y_train, device=self.device).unsqueeze(1)
        X_val_t = torch.tensor(X_val, device=self.device)
        y_val_t = torch.tensor(y_val, device=self.device).unsqueeze(1)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.aux_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(epochs):
            self.aux_model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.aux_model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss /= max(len(X_train), 1)
            self.aux_model.eval()
            with torch.no_grad():
                val_loss = criterion(self.aux_model(X_val_t), y_val_t).item()

            train_losses.append(epoch_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d — train MSE: %.6f, val MSE: %.6f",
                    epoch + 1, epochs, epoch_loss, val_loss,
                )

        return train_losses, val_losses

    def predict_batch(self, features: "np.ndarray") -> "np.ndarray":
        """Batch inference from pre-computed feature matrix.

        Args:
            features: Float32 array of shape ``(N, 2*d_model)``.

        Returns:
            Float32 array of shape ``(N,)`` — predicted uncertainty scores.
        """
        import numpy as _np

        if self.aux_model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = torch.tensor(features.astype(_np.float32), device=self.device)
        self.aux_model.eval()
        with torch.no_grad():
            preds = self.aux_model(X).squeeze(1).cpu().numpy()
        return preds.astype(_np.float32)

    def save(self, path: str) -> None:
        """Save trained auxiliary model checkpoint to *path*."""
        if self.aux_model is None:
            raise RuntimeError("No model to save. Call train() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.aux_model.state_dict(),
                "d_model": self.d_model,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "layer_idx": self.layer_idx,
            },
            path,
        )
        logger.info("AuxiliaryModel saved to %s", path)
