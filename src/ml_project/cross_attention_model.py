"""Language-guided camera attention and timestep-specific flow conditioning.

Compatible with TinyVLA's compute_loss/forward/evaluate_model interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ml_project.film_model import FiLMTinyVLA
from ml_project.model import FlowMatchingActionHead, TinyVLAConfig


class ActionQueryFlowHead(FlowMatchingActionHead):
    """Flow velocity conditioned on [B,H,D] timestep-specific representations."""

    def velocity(
        self, context: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        if context.ndim != 3 or context.shape[:2] != noisy_actions.shape[:2]:
            raise ValueError("Expected context [batch, action_horizon, hidden_dim]")
        tokens = self.action_projection(noisy_actions) + self.position_embedding
        tokens = tokens + self.context_projection(context)
        tokens = tokens + self.time_projection(self._time_features(time)).unsqueeze(1)
        return self.output_projection(self.decoder(tokens))


class CrossAttentionVLA(FiLMTinyVLA):
    """FiLM vision + language-guided visual attention + action-query decoder.

    `use_language_cross_attention=False` is an architectural ablation.
    `use_action_queries=False` returns the CLS context and uses original flow head.
    """

    def __init__(
        self,
        config: TinyVLAConfig,
        *,
        use_language_cross_attention: bool = True,
        use_action_queries: bool = True,
    ) -> None:
        super().__init__(config)
        h = config.hidden_dim
        self.use_language_cross_attention = use_language_cross_attention
        self.use_action_queries = use_action_queries
        self.language_to_vision = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.language_norm = nn.LayerNorm(h)
        self.visual_norm = nn.LayerNorm(h)
        self.visual_gate = nn.Parameter(torch.tensor(0.0))
        self.action_queries = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.action_cross_attention = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.action_norm = nn.LayerNorm(h)
        self.action_mlp = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h * 2), nn.GELU(), nn.Linear(h * 2, h)
        )
        self.action_gate = nn.Parameter(torch.tensor(0.0))
        if use_action_queries:
            self.action_head = ActionQueryFlowHead(config)

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
    ):
        batch_size = input_ids.shape[0]
        text_valid = attention_mask.bool()
        text = self.token_embedding(input_ids) + self.text_position_embedding
        text = text + self.modality_embedding.weight[2][None, None]
        weights = text_valid.unsqueeze(-1).to(text.dtype)
        language_context = (text * weights).sum(1) / weights.sum(1).clamp_min(1)
        gamma, beta = self.film(language_context).chunk(2, -1)

        cls = (
            self.cls_token.expand(batch_size, -1, -1)
            + self.modality_embedding.weight[0][None, None]
        )
        state_token = (
            self.state_encoder(state).unsqueeze(1) + self.modality_embedding.weight[1][None, None]
        )
        camera_tokens = []
        camera_valid = []
        for camera in self.config.camera_names:
            i = self.camera_to_index[camera]
            visual = self.vision_encoder(images[camera])
            visual = visual * (1 + gamma[:, None, :]) + beta[:, None, :]
            visual = visual + self.camera_embedding.weight[i][None, None]
            visual = visual + self.modality_embedding.weight[3 + i][None, None]
            camera_tokens.append(visual)
            camera_valid.append(
                image_masks[camera].bool().reshape(batch_size, 1).expand(-1, visual.shape[1])
            )
        visual = torch.cat(camera_tokens, 1)
        visual_valid = torch.cat(camera_valid, 1)
        attentions = {} if return_attentions else None
        hidden = {} if return_hidden_states else None

        # Language queries select task-relevant patches; preserve original visual tokens.
        if self.use_language_cross_attention:
            # For samples with no camera, avoid all-masked attention NaNs by adding
            # one zero key/value; the gated update is explicitly zeroed for them.
            has_camera = visual_valid.any(1)
            safe_valid = visual_valid.clone()
            safe_valid[~has_camera, 0] = True
            safe_visual = self.visual_norm(visual).clone()
            safe_visual[~has_camera, 0] = 0
            selected, attn = self.language_to_vision(
                self.language_norm(text),
                safe_visual,
                safe_visual,
                key_padding_mask=~safe_valid,
                need_weights=return_attentions,
                average_attn_weights=False,
            )
            selected = selected * text_valid.unsqueeze(-1) * has_camera[:, None, None]
            text = text + torch.tanh(self.visual_gate) * selected
            if attentions is not None:
                attentions["language_to_vision"] = attn
        tokens = torch.cat([cls, state_token, text, visual], 1)
        valid = torch.cat(
            [
                torch.ones(batch_size, 2, dtype=torch.bool, device=input_ids.device),
                text_valid,
                visual_valid,
            ],
            1,
        )
        for index, layer in enumerate(self.fusion_layers):
            tokens, attn = layer(
                tokens, key_padding_mask=~valid, return_attention=return_attentions
            )
            if attentions is not None and attn is not None:
                attentions[f"fusion.layer_{index}"] = attn
            if hidden is not None:
                hidden[f"fusion.layer_{index}"] = tokens
        tokens = self.final_norm(tokens)
        if hidden is not None:
            hidden["fused_tokens"] = tokens
        if not self.use_action_queries:
            return tokens[:, 0], attentions, hidden

        queries = self.action_queries.expand(batch_size, -1, -1) + tokens[:, :1]
        action_context, attn = self.action_cross_attention(
            self.action_norm(queries),
            tokens,
            tokens,
            key_padding_mask=~valid,
            need_weights=return_attentions,
            average_attn_weights=False,
        )
        queries = queries + torch.tanh(self.action_gate) * action_context
        queries = queries + self.action_mlp(queries)
        if attentions is not None:
            attentions["action_to_fusion"] = attn
        if hidden is not None:
            hidden["action_queries"] = queries
        return queries, attentions, hidden


def load_cross_attention_checkpoint(
    path: str, device: str | torch.device = "cpu"
) -> CrossAttentionVLA:
    """Load a trusted checkpoint produced by run_cross_attention.py."""
    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = TinyVLAConfig(**payload["model_config"])
    metadata = payload["metadata"]
    model = CrossAttentionVLA(
        cfg,
        use_language_cross_attention=bool(metadata["language_cross_attention"]),
        use_action_queries=bool(metadata["action_queries"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()
