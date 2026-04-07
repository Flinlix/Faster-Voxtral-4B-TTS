"""Weight loading from the Voxtral consolidated.safetensors checkpoint."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from voxtral.config import REPO_ID

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mistral_inference.transformer import Transformer

    from voxtral.acoustic.flow_matching import FlowMatchingAudioTransformer
    from voxtral.codec.decoder import CodecDecoder
    from voxtral.embedding import AudioTokenEmbedding

# Key mapping from checkpoint names to mistral-inference Transformer state_dict names.
_LLM_KEY_PATTERNS = [
    (r"^mm_audio_embeddings\.tok_embeddings\.weight$", "tok_embeddings.weight"),
    (r"^(layers\.\d+\.attention\.w[qkvo]\.weight)$", r"\1"),
    (r"^(layers\.\d+\.attention_norm\.weight)$", r"\1"),
    (r"^(layers\.\d+\.feed_forward\.w[123]\.weight)$", r"\1"),
    (r"^(layers\.\d+\.ffn_norm\.weight)$", r"\1"),
    (r"^(norm\.weight)$", r"\1"),
]


def _remap_checkpoint_key_to_llm_key(checkpoint_key: str) -> str | None:
    """Map a checkpoint key to the corresponding mistral-inference key, or None."""
    for pattern, replacement in _LLM_KEY_PATTERNS:
        if re.match(pattern, checkpoint_key):
            return re.sub(pattern, replacement, checkpoint_key)
    return None


def load_checkpoint_weights(
    llm: Transformer,
    acoustic_transformer: FlowMatchingAudioTransformer,
    codec_decoder: CodecDecoder,
    audio_token_embedding: AudioTokenEmbedding,
    repo_id: str = REPO_ID,
) -> None:
    """Load all weights from consolidated.safetensors into the four model components.

    Args:
        llm: mistral-inference ``Transformer`` instance.
        acoustic_transformer: ``FlowMatchingAudioTransformer`` instance.
        codec_decoder: ``CodecDecoder`` instance.
        audio_token_embedding: ``AudioTokenEmbedding`` instance.
        repo_id: HuggingFace repository ID for the checkpoint.
    """
    safetensors_path = hf_hub_download(repo_id, "consolidated.safetensors")

    llm_state_dict: dict[str, torch.Tensor] = {}
    acoustic_state_dict: dict[str, torch.Tensor] = {}
    codec_state_dict: dict[str, torch.Tensor] = {}
    audio_embedding_state_dict: dict[str, torch.Tensor] = {}

    with safe_open(safetensors_path, framework="pt", device="cpu") as safetensors_file:
        for key in safetensors_file.keys():
            tensor = safetensors_file.get_tensor(key)

            if key.startswith("acoustic_transformer."):
                acoustic_state_dict[key.removeprefix("acoustic_transformer.")] = tensor

            elif key.startswith("audio_tokenizer."):
                codec_state_dict[key.removeprefix("audio_tokenizer.")] = tensor

            elif key.startswith("mm_audio_embeddings.audio_codebook_embeddings.embeddings."):
                stripped = key.removeprefix("mm_audio_embeddings.audio_codebook_embeddings.")
                audio_embedding_state_dict[stripped] = tensor

            else:
                mistral_key = _remap_checkpoint_key_to_llm_key(key)
                if mistral_key is not None:
                    llm_state_dict[mistral_key] = tensor
                    # Tied embeddings: output.weight = tok_embeddings.weight
                    if mistral_key == "tok_embeddings.weight":
                        llm_state_dict["output.weight"] = tensor

    # Load LLM (strict - all keys must match)
    llm.load_state_dict(llm_state_dict, strict=True)

    # Load acoustic transformer
    _missing, unexpected = acoustic_transformer.load_state_dict(acoustic_state_dict, strict=False)
    if _missing:
        logger.warning("Acoustic transformer missing keys: %s", _missing)
    if unexpected:
        logger.warning("Acoustic transformer unexpected keys: %s", unexpected)

    # Load audio token embedding
    _missing, unexpected = audio_token_embedding.load_state_dict(
        audio_embedding_state_dict, strict=False,
    )
    if _missing:
        logger.warning("Audio token embedding missing keys: %s", _missing)
    if unexpected:
        logger.warning("Audio token embedding unexpected keys: %s", unexpected)

    # Load codec decoder
    _missing, unexpected = codec_decoder.load_state_dict(codec_state_dict, strict=False)
    if _missing:
        logger.warning("Codec decoder missing keys: %s", _missing)
    if unexpected:
        logger.warning("Codec decoder unexpected keys: %s", unexpected)

    # Invalidate cached semantic embedding since the underlying buffers were just loaded
    codec_decoder.quantizer.semantic_codebook._embedding = None
    codec_decoder.quantizer.semantic_codebook.register_buffer("_embedding", None, persistent=False)
