"""Flow-matching acoustic transformer - predicts audio codes from LLM hidden states."""

import torch
import torch.nn as nn

from voxtral.config import AcousticTransformerConfig, NUM_SPECIAL_TOKENS, EMPTY_AUDIO_ID, END_AUDIO_ID
from voxtral.acoustic.attention import AcousticTransformerBlock
from voxtral.acoustic.time_embedding import TimeEmbedding


class FlowMatchingAudioTransformer(nn.Module):
    """Predicts one frame of audio codes from an LLM hidden state using
    classifier-free-guided Euler ODE integration.

    Attribute names (input_projection, time_projection, llm_projection,
    semantic_codebook_output, acoustic_codebook_output, layers, norm)
    match checkpoint keys.
    """

    def __init__(self, config: AcousticTransformerConfig | None = None):
        super().__init__()
        if config is None:
            config = AcousticTransformerConfig.voxtral_4b()

        self.semantic_codebook_size = config.semantic_codebook_size
        self.acoustic_levels = config.acoustic_levels
        self.num_acoustic_codes = config.num_acoustic_codes

        # Semantic output head: padded to nearest multiple of 128
        semantic_vocab_with_special = self.semantic_codebook_size + NUM_SPECIAL_TOKENS
        semantic_vocab_padded = 128 * ((semantic_vocab_with_special + 127) // 128)

        # Projections
        self.time_embedding = TimeEmbedding(config.model_dim)
        self.input_projection = nn.Linear(self.num_acoustic_codes, config.model_dim, bias=False)
        self.time_projection = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.llm_projection = nn.Linear(config.input_dim, config.model_dim, bias=False)

        self.semantic_codebook_output = nn.Linear(
            config.model_dim, semantic_vocab_padded, bias=config.use_biases,
        )
        self.acoustic_codebook_output = nn.Linear(
            config.model_dim, self.num_acoustic_codes, bias=False,
        )

        # Transformer layers
        self.layers_ids = list(range(config.num_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            self.layers[str(layer_id)] = AcousticTransformerBlock(
                layer_id,
                config.model_dim, config.hidden_dim,
                config.num_heads, config.num_kv_heads, config.head_dim,
                config.use_biases, config.norm_eps,
            )
        self.norm = nn.RMSNorm(config.model_dim, eps=config.norm_eps)

        # Flow-matching ODE parameters
        self._num_ode_steps = config.num_ode_steps
        self._classifier_free_guidance_scale = config.classifier_free_guidance_scale
        self._initial_noise_scale = config.initial_noise_scale

        timesteps = torch.linspace(0, 1, self._num_ode_steps)
        self.register_buffer("_timesteps", timesteps, persistent=False)
        self.register_buffer("_step_sizes", timesteps[1:] - timesteps[:-1], persistent=False)

        # Lazily computed on first forward call for the correct device
        self._cached_time_embeddings: list[torch.Tensor] | None = None

    def _predict_velocity(
        self,
        noisy_acoustic_state: torch.Tensor,
        llm_hidden_state: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field for one Euler step.

        Args:
            noisy_acoustic_state: [B, num_acoustic_codes] current ODE state.
            llm_hidden_state: [B, D] conditioning from the LLM.
            time_embedding: [B, D] sinusoidal timestep embedding.

        Returns:
            [B, num_acoustic_codes] predicted velocity.
        """
        noisy_acoustic_state = noisy_acoustic_state.to(llm_hidden_state.dtype)
        time_projected = self.time_projection(time_embedding)
        llm_projected = self.llm_projection(llm_hidden_state)

        # Stack 3 tokens: [acoustic_input, time, llm_condition]
        transformer_input = torch.cat([
            self.input_projection(noisy_acoustic_state.unsqueeze(1)),
            time_projected.unsqueeze(1),
            llm_projected.unsqueeze(1),
        ], dim=1)  # [B, 3, D]

        for layer_id in self.layers_ids:
            transformer_input = self.layers[str(layer_id)](transformer_input)

        normalised_output = self.norm(transformer_input)
        return self.acoustic_codebook_output(normalised_output[:, 0, :])

    def forward(self, llm_hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict one frame of audio codes from an LLM hidden state.

        Args:
            llm_hidden_state: [B, D] last-layer hidden state from the LLM.

        Returns:
            [B, 37] integer codes (1 semantic + 36 acoustic).
        """
        # Semantic code via greedy argmax
        semantic_logits = self.semantic_codebook_output(llm_hidden_state).float()
        semantic_logits[:, EMPTY_AUDIO_ID] = float("-inf")
        semantic_logits[:, NUM_SPECIAL_TOKENS + self.semantic_codebook_size :] = float("-inf")
        semantic_codes = semantic_logits.argmax(dim=-1, keepdim=True)

        batch_size = semantic_codes.shape[0]
        is_active_frame = semantic_codes.squeeze(1) != END_AUDIO_ID

        # Pre-compute time embeddings once (same for every frame)
        if (
            self._cached_time_embeddings is None
            or self._cached_time_embeddings[0].device != llm_hidden_state.device
        ):
            timesteps = self._timesteps.to(dtype=llm_hidden_state.dtype)
            self._cached_time_embeddings = [
                self.time_embedding(timesteps[i].view(-1, 1)).to(llm_hidden_state.dtype)
                for i in range(len(timesteps) - 1)
            ]

        # Flow-matching ODE: Euler integration with classifier-free guidance
        ode_state = (
            torch.randn(
                batch_size, self.num_acoustic_codes,
                dtype=llm_hidden_state.dtype, device=llm_hidden_state.device,
            )
            * self._initial_noise_scale
        )

        # Doubled tensors for conditional + unconditional (CFG)
        llm_conditional_unconditional = torch.cat(
            [llm_hidden_state, torch.zeros_like(llm_hidden_state)], dim=0,
        )
        step_sizes = self._step_sizes.to(dtype=llm_hidden_state.dtype)

        for step_index in range(self._num_ode_steps - 1):
            time_embedding = self._cached_time_embeddings[step_index].expand(batch_size, -1)
            time_embedding_doubled = torch.cat([time_embedding, time_embedding], dim=0)

            velocity_all = self._predict_velocity(
                torch.cat([ode_state, ode_state], dim=0),
                llm_conditional_unconditional,
                time_embedding_doubled,
            )
            velocity_conditional = velocity_all[:batch_size]
            velocity_unconditional = velocity_all[batch_size:]
            velocity_guided = (
                self._classifier_free_guidance_scale * velocity_conditional
                + (1 - self._classifier_free_guidance_scale) * velocity_unconditional
            )
            ode_state = ode_state + velocity_guided * step_sizes[step_index]

        # Quantize continuous ODE output to discrete acoustic codes
        ode_state = ode_state.clamp(-1, 1)
        scaled = ((ode_state + 1) / 2) * (self.acoustic_levels - 1)
        acoustic_codes = scaled.round().long()
        acoustic_codes[~is_active_frame] = EMPTY_AUDIO_ID
        acoustic_codes = acoustic_codes + NUM_SPECIAL_TOKENS

        return torch.cat([semantic_codes, acoustic_codes], dim=1)  # [B, 37]
