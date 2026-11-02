from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass(frozen=True)
class TinyVLAConfig:
    vocab_size: int
    image_size: int = 64
    patch_size: int = 8
    state_dim: int = 7
    action_dim: int = 7
    action_horizon: int = 8
    max_text_length: int = 16
    hidden_dim: int = 64
    num_heads: int = 4
    num_fusion_layers: int = 2
    num_action_layers: int = 1
    camera_names: tuple[str, ...] = ("front", "wrist")
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not self.camera_names:
            raise ValueError("At least one camera is required")


@dataclasses.dataclass
class VLAOutput:
    actions: torch.Tensor
    text_logits: torch.Tensor | None
    auxiliary: dict[str, torch.Tensor]
    attentions: dict[str, torch.Tensor] | None
    hidden_states: dict[str, torch.Tensor] | None


class PatchEncoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.grid_size = image_size // patch_size
        self.projection = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.position_embedding = nn.Parameter(torch.randn(1, self.grid_size**2, hidden_dim) * 0.02)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(images).flatten(2).transpose(1, 2)
        return self.normalization(tokens + self.position_embedding)


class AttentionBlock(nn.Module):
    """Transformer block that can explicitly return per-head attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor,
        return_attention: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.attention_norm(tokens)
        # A distinct key/value tensor avoids a noisy boolean-mask fast-path warning
        # in PyTorch 2.0 while preserving self-attention semantics.
        key_value = normalized.clone()
        attended, weights = self.attention(
            normalized,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return tokens, weights if return_attention else None


class FlowMatchingActionHead(nn.Module):
    """Small conditional flow head for continuous action chunks."""

    def __init__(self, config: TinyVLAConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.action_projection = nn.Linear(config.action_dim, hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.action_horizon, hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config.num_action_layers)
        self.output_projection = nn.Linear(hidden_dim, config.action_dim)

    @staticmethod
    def _time_features(time: torch.Tensor) -> torch.Tensor:
        time = time.float()
        return torch.stack(
            [time, time.square(), torch.sin(math.pi * time), torch.cos(math.pi * time)],
            dim=-1,
        )

    def velocity(
        self,
        context: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.action_projection(noisy_actions)
        tokens = tokens + self.position_embedding
        tokens = tokens + self.context_projection(context).unsqueeze(1)
        tokens = tokens + self.time_projection(self._time_features(time)).unsqueeze(1)
        return self.output_projection(self.decoder(tokens))

    def loss(
        self,
        context: torch.Tensor,
        target_actions: torch.Tensor,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(target_actions)
        time = torch.rand(target_actions.shape[0], device=target_actions.device)
        interpolation = time[:, None, None]
        noisy_actions = (1.0 - interpolation) * noise + interpolation * target_actions
        target_velocity = target_actions - noise
        prediction = self.velocity(context, noisy_actions, time)
        per_example = F.mse_loss(prediction, target_velocity, reduction="none").mean((1, 2))

        if label_mask is None:
            return per_example.mean()
        weights = label_mask.float()
        if not bool(weights.any()):
            raise ValueError("The batch contains no labeled actions")
        return (per_example * weights).sum() / weights.sum()

    def sample(
        self,
        context: torch.Tensor,
        *,
        num_steps: int = 6,
        seed: int = 0,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        actions = torch.randn(
            context.shape[0],
            self.action_horizon,
            self.action_dim,
            generator=generator,
            dtype=context.dtype,
        ).to(context.device)
        step_size = 1.0 / num_steps
        for step in range(num_steps):
            time = torch.full(
                (context.shape[0],),
                step / num_steps,
                dtype=context.dtype,
                device=context.device,
            )
            actions = actions + step_size * self.velocity(context, actions, time)
        return actions


class TinyVLA(nn.Module):
    """Inspectible VLA with named cameras and a chunked continuous action head."""

    def __init__(self, config: TinyVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = PatchEncoder(config.image_size, config.patch_size, config.hidden_dim)
        self.camera_to_index = {name: index for index, name in enumerate(config.camera_names)}
        self.camera_embedding = nn.Embedding(len(config.camera_names), config.hidden_dim)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.text_position_embedding = nn.Parameter(
            torch.randn(1, config.max_text_length, config.hidden_dim) * 0.02
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        # CLS, state, text, then one modality id for each camera.
        self.modality_embedding = nn.Embedding(3 + len(config.camera_names), config.hidden_dim)
        self.fusion_layers = nn.ModuleList(
            [
                AttentionBlock(config.hidden_dim, config.num_heads, config.dropout)
                for _ in range(config.num_fusion_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.action_head = FlowMatchingActionHead(config)

    @property
    def patches_per_image(self) -> int:
        return self.vision_encoder.grid_size**2

    def token_slices(self, text_length: int | None = None) -> dict[str, slice]:
        """Return token locations for attention visualization."""

        text_length = text_length or self.config.max_text_length
        cursor = 0
        slices = {"cls": slice(cursor, cursor + 1)}
        cursor += 1
        slices["state"] = slice(cursor, cursor + 1)
        cursor += 1
        slices["text"] = slice(cursor, cursor + text_length)
        cursor += text_length
        for camera_name in self.config.camera_names:
            slices[camera_name] = slice(cursor, cursor + self.patches_per_image)
            cursor += self.patches_per_image
        return slices

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
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor] | None,
        dict[str, torch.Tensor] | None,
    ]:
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

        token_groups = [cls_tokens, state_tokens, text_tokens]
        valid_groups = [
            torch.ones(batch_size, 1, dtype=torch.bool, device=input_ids.device),
            torch.ones(batch_size, 1, dtype=torch.bool, device=input_ids.device),
            attention_mask.bool(),
        ]
        for camera_name in self.config.camera_names:
            camera_index = self.camera_to_index[camera_name]
            image_tokens = self.vision_encoder(images[camera_name])
            image_tokens = image_tokens + self.camera_embedding.weight[camera_index][None, None]
            modality = self.modality_embedding.weight[3 + camera_index][None, None]
            image_tokens = image_tokens + modality
            token_groups.append(image_tokens)
            camera_valid = image_masks[camera_name].bool().reshape(batch_size, 1)
            valid_groups.append(camera_valid.expand(-1, self.patches_per_image))

        tokens = torch.cat(token_groups, dim=1)
        valid_tokens = torch.cat(valid_groups, dim=1)
        key_padding_mask = ~valid_tokens
        attention_outputs: dict[str, torch.Tensor] | None = {} if return_attentions else None
        hidden_outputs: dict[str, torch.Tensor] | None = {} if return_hidden_states else None

        for index, layer in enumerate(self.fusion_layers):
            tokens, weights = layer(
                tokens,
                key_padding_mask=key_padding_mask,
                return_attention=return_attentions,
            )
            if attention_outputs is not None and weights is not None:
                attention_outputs[f"fusion.layer_{index}"] = weights
            if hidden_outputs is not None:
                hidden_outputs[f"fusion.layer_{index}"] = tokens

        tokens = self.final_norm(tokens)
        if hidden_outputs is not None:
            hidden_outputs["fused_tokens"] = tokens
        return tokens[:, 0], attention_outputs, hidden_outputs

    def compute_loss(self, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
        context, _, _ = self.encode(
            images=batch["images"],  # type: ignore[arg-type]
            image_masks=batch["image_masks"],  # type: ignore[arg-type]
            input_ids=batch["input_ids"],  # type: ignore[arg-type]
            attention_mask=batch["attention_mask"],  # type: ignore[arg-type]
            state=batch["state"],  # type: ignore[arg-type]
        )
        action_loss = self.action_head.loss(
            context,
            batch["actions"],  # type: ignore[arg-type]
            batch.get("action_mask"),  # type: ignore[arg-type]
        )
        return {"total": action_loss, "action": action_loss.detach()}

    def forward(
        self,
        *,
        images: Mapping[str, torch.Tensor],
        image_masks: Mapping[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        flow_steps: int = 6,
        sample_seed: int = 0,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
    ) -> VLAOutput:
        context, attentions, hidden_states = self.encode(
            images=images,
            image_masks=image_masks,
            input_ids=input_ids,
            attention_mask=attention_mask,
            state=state,
            return_attentions=return_attentions,
            return_hidden_states=return_hidden_states,
        )
        actions = self.action_head.sample(context, num_steps=flow_steps, seed=sample_seed)
        return VLAOutput(
            actions=actions,
            text_logits=None,
            auxiliary={"context": context},
            attentions=attentions,
            hidden_states=hidden_states,
        )

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
