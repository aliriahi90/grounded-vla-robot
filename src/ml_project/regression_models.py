"""Deterministic action regression controls, compatible with existing VLA evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ml_project.cross_attention_model import CrossAttentionVLA
from ml_project.model import TinyVLAConfig, VLAOutput


def masked_action_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    per_sample = (pred - target).square().mean(dim=(1, 2))
    if mask is None:
        return per_sample.mean()
    weights = mask.float().reshape(-1)
    if not bool(weights.any()):
        raise ValueError("No labeled examples in batch")
    return (per_sample * weights).sum() / weights.sum()


class CrossAttentionRegressionVLA(CrossAttentionVLA):
    """Same FiLM/cross-attention/action-query encoder, deterministic MSE head.

    Uses one linear projection per query timestep. No flow sampling noise.
    """

    def __init__(self, config: TinyVLAConfig):
        super().__init__(config, use_language_cross_attention=True, use_action_queries=True)
        self.action_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.action_dim)
        )

    def _predict(
        self,
        *,
        images,
        image_masks,
        input_ids,
        attention_mask,
        state,
        return_attentions=False,
        return_hidden_states=False,
    ):
        context, attn, hidden = self.encode(
            images=images,
            image_masks=image_masks,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )
        return self.action_head(context), context, attn, hidden

    def compute_loss(self, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
        pred, _, _, _ = self._predict(
            images=batch["images"],
            image_masks=batch["image_masks"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            state=batch["state"],
        )
        loss = masked_action_mse(pred, batch["actions"], batch.get("action_mask"))
        return {"total": loss, "action": loss.detach()}

    def forward(
        self,
        *,
        images,
        image_masks,
        input_ids,
        attention_mask,
        state,
        flow_steps=6,
        sample_seed=0,
        return_attentions=False,
        return_hidden_states=False,
    ):
        pred, context, attn, hidden = self._predict(
            images=images,
            image_masks=image_masks,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )
        return VLAOutput(
            actions=pred,
            text_logits=None,
            auxiliary={"context": context},
            attentions=attn,
            hidden_states=hidden,
        )


class StateOnlyMLP(nn.Module):
    """Small deterministic state -> [H, D] control with same model interface."""

    def __init__(self, config: TinyVLAConfig, width: int = 64):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.state_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, config.action_horizon * config.action_dim),
        )

    def parameter_count(self, *, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def _predict(self, state):
        return self.net(state).reshape(-1, self.config.action_horizon, self.config.action_dim)

    def compute_loss(self, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
        loss = masked_action_mse(
            self._predict(batch["state"]), batch["actions"], batch.get("action_mask")
        )
        return {"total": loss, "action": loss.detach()}

    def forward(
        self,
        *,
        images,
        image_masks,
        input_ids,
        attention_mask,
        state,
        flow_steps=6,
        sample_seed=0,
        return_attentions=False,
        return_hidden_states=False,
    ):
        return VLAOutput(
            actions=self._predict(state),
            text_logits=None,
            auxiliary={},
            attentions=None,
            hidden_states=None,
        )
