"""Language-conditioned FiLM experiment for TinyVLA.

Modulates *visual patch embeddings* before camera/modality embeddings and fusion.
All other baseline components, including the action head, remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ml_project.model import TinyVLA, TinyVLAConfig


class FiLMTinyVLA(TinyVLA):
    def __init__(self, config: TinyVLAConfig) -> None:
        super().__init__(config)
        h = config.hidden_dim
        self.film = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2 * h))
        # Exact identity modulation at initialization, while allowing learning.
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def encode(
        self,
        *,
        images: Mapping[str, torch.Tensor],
        image_masks: Mapping[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        missing = set(self.config.camera_names) - set(images)
        if missing:
            raise KeyError(f"Missing configured cameras: {sorted(missing)}")
        batch_size = input_ids.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + self.modality_embedding.weight[0][None, None]
        state_tokens = self.state_encoder(state).unsqueeze(1)
        state_tokens = state_tokens + self.modality_embedding.weight[1][None, None]
        text_tokens = self.token_embedding(input_ids) + self.text_position_embedding
        text_tokens = text_tokens + self.modality_embedding.weight[2][None, None]

        # Exclude padding from pooled language context. No attention-mask leakage.
        text_valid = attention_mask.bool()
        weights = text_valid.unsqueeze(-1).to(text_tokens.dtype)
        language_context = (text_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        gamma, beta = self.film(language_context).chunk(2, dim=-1)

        token_groups = [cls_tokens, state_tokens, text_tokens]
        valid_groups = [
            torch.ones(batch_size, 1, dtype=torch.bool, device=input_ids.device),
            torch.ones(batch_size, 1, dtype=torch.bool, device=input_ids.device),
            text_valid,
        ]
        for camera_name in self.config.camera_names:
            camera_index = self.camera_to_index[camera_name]
            image_tokens = self.vision_encoder(images[camera_name])
            image_tokens = image_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            image_tokens = image_tokens + self.camera_embedding.weight[camera_index][None, None]
            image_tokens = (
                image_tokens + self.modality_embedding.weight[3 + camera_index][None, None]
            )
            token_groups.append(image_tokens)
            camera_valid = image_masks[camera_name].bool().reshape(batch_size, 1)
            valid_groups.append(camera_valid.expand(-1, self.patches_per_image))

        tokens = torch.cat(token_groups, dim=1)
        key_padding_mask = ~torch.cat(valid_groups, dim=1)
        attention_outputs: dict[str, torch.Tensor] | None = {} if return_attentions else None
        hidden_outputs: dict[str, torch.Tensor] | None = {} if return_hidden_states else None
        for index, layer in enumerate(self.fusion_layers):
            tokens, weights = layer(
                tokens, key_padding_mask=key_padding_mask, return_attention=return_attentions
            )
            if attention_outputs is not None and weights is not None:
                attention_outputs[f"fusion.layer_{index}"] = weights
            if hidden_outputs is not None:
                hidden_outputs[f"fusion.layer_{index}"] = tokens
        tokens = self.final_norm(tokens)
        if hidden_outputs is not None:
            hidden_outputs["fused_tokens"] = tokens
        return tokens[:, 0], attention_outputs, hidden_outputs
