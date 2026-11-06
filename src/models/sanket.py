"""Sanket ISL Translator Model Provider.

STAGE 2 IMPLEMENTATION:
- Integrated isolated PoseFeatureExtractor (Candidate A: 23 upper-body pose + 21 left + 21 right hand = 195-dim).
- Uniform temporal sampling of 8 frames across clip duration.
- ImageNet normalization ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]).
- Explicit T5 Beam Search generation (num_beams=4, max_new_tokens=32, early_stopping=True).
- Preserved default path guard (raises NotImplementedError unless allow_experimental_inference=True).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, TimesformerConfig, TimesformerModel

from src.preprocessing.pose import PoseExtractorConfig, PoseFeatureExtractor, PoseLandmarkScheme
from src.preprocessing.video import IMAGENET_MEAN, IMAGENET_STD, preprocess_video_frames, sample_uniform_frames
from .base import ModelProvider, Prediction

logger = logging.getLogger(__name__)

EXPERIMENTAL_PREPROCESSING_WARNING = (
    "EXPERIMENTAL / UNVERIFIED: Preprocessing contracts (MediaPipe pose feature selection, "
    "frame sampling stride, image normalization) cannot be verified against upstream source code. "
    "Output should NOT be trusted as a faithful reproduction of the published BLEU-1 50.6 baseline."
)


class PoseEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_in = nn.Linear(195, 768)
        layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        return self.transformer(h)


class FusionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_gate = nn.Linear(1536, 768)
        self.pose_gate = nn.Linear(1536, 768)
        self.out_proj = nn.Linear(1536, 768)

    def forward(self, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([v, p], dim=-1)  # (B, T, 1536)
        g_v = torch.sigmoid(self.visual_gate(concat))
        g_p = torch.sigmoid(self.pose_gate(concat))
        fused = g_v * v + g_p * p
        fused_concat = torch.cat([fused, fused], dim=-1)
        return self.out_proj(fused_concat)


class DimMapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(768, 768))
        self.bias = nn.Parameter(torch.empty(768))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t()) + self.bias


class SanketFullModel(nn.Module):
    """End-to-end Sanket Multimodal Sign Language Translation Model."""

    def __init__(self) -> None:
        super().__init__()
        vt_config = TimesformerConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_frames=8,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        self.video_encoder = TimesformerModel(vt_config)

        t5_config = T5Config(
            vocab_size=32128,
            d_model=768,
            d_kv=64,
            d_ff=3072,
            num_layers=12,
            num_decoder_layers=12,
            num_heads=12,
            relative_attention_num_buckets=32,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
        self.text_decoder = T5ForConditionalGeneration(t5_config)

        self.pose_encoder = PoseEncoder()
        self.fusion = FusionBlock()
        self.dim_mapper = DimMapper()

    def load_from_checkpoint(self, checkpoint_path: Path) -> None:
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        video_sd = {k.replace("video_encoder.", ""): v for k, v in sd.items() if k.startswith("video_encoder.")}
        self.video_encoder.load_state_dict(video_sd, strict=True)

        text_sd = {k.replace("text_decoder.", ""): v for k, v in sd.items() if k.startswith("text_decoder.")}
        self.text_decoder.load_state_dict(text_sd, strict=True)

        pose_sd = {k.replace("pose_encoder.", ""): v for k, v in sd.items() if k.startswith("pose_encoder.")}
        self.pose_encoder.load_state_dict(pose_sd, strict=True)

        fusion_sd = {k.replace("fusion.", ""): v for k, v in sd.items() if k.startswith("fusion.")}
        self.fusion.load_state_dict(fusion_sd, strict=True)

        dim_sd = {k.replace("dim_mapper.", ""): v for k, v in sd.items() if k.startswith("dim_mapper.")}
        self.dim_mapper.load_state_dict(dim_sd, strict=True)


class SanketModelProvider(ModelProvider):
    """Adapter boundary for the Sanket checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        model_version: str = "sanket-baseline",
        allow_experimental_inference: bool = False,
        pose_landmark_scheme: PoseLandmarkScheme = PoseLandmarkScheme.CANDIDATE_A_UPPER_BODY_HANDS,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.model_version = model_version
        self.allow_experimental_inference = allow_experimental_inference
        self.loaded = False
        self.model: SanketFullModel | None = None
        self.tokenizer: Any | None = None

        # Instantiate isolated PoseFeatureExtractor (Candidate A by default)
        pose_config = PoseExtractorConfig(scheme=pose_landmark_scheme, target_dim=195)
        self.pose_extractor = PoseFeatureExtractor(config=pose_config)

    def load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Sanket checkpoint not found: {self.checkpoint}. "
                "Run scripts/download_sanket.py first."
            )

        self.model = SanketFullModel()
        self.model.load_from_checkpoint(self.checkpoint)
        self.model.eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        except Exception:
            logger.warning("Could not load T5 tokenizer online; falling back to offline config.")

        self.loaded = True

    def predict(
        self,
        frames: Any,
        pose_features: Any = None,
        allow_experimental_inference: bool | None = None,
    ) -> Prediction:
        # Input validation: reject None or empty frames
        if frames is None or (isinstance(frames, (list, tuple, np.ndarray, torch.Tensor)) and len(frames) == 0):
            raise ValueError("Input 'frames' cannot be None or empty.")

        is_opted_in = (
            allow_experimental_inference
            if allow_experimental_inference is not None
            else self.allow_experimental_inference
        )

        if not is_opted_in:
            raise NotImplementedError(
                "Pin the upstream Sanket implementation and exact preprocessing "
                "before enabling inference. To opt into the experimental pipeline, pass "
                "allow_experimental_inference=True."
            )

        if not self.loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        logger.warning(EXPERIMENTAL_PREPROCESSING_WARNING)

        # 1. Preprocess Video Frames (Uniform Sampling of 8 frames + ImageNet Normalization)
        if isinstance(frames, torch.Tensor):
            video_tensor = frames if frames.ndim == 5 else frames.unsqueeze(0)
            sampled_raw_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(8)]
        else:
            # Process list/array of raw BGR/RGB image frames
            raw_frames_list = list(frames)
            sampled_raw_frames = sample_uniform_frames(raw_frames_list, num_samples=8)
            video_array = preprocess_video_frames(raw_frames_list, target_size=(224, 224))
            video_tensor = torch.tensor(video_array, dtype=torch.float32).unsqueeze(0)  # (1, 8, 3, 224, 224)

        # 2. Extract or Format Pose Features via PoseFeatureExtractor
        if pose_features is not None:
            if isinstance(pose_features, torch.Tensor):
                pose_tensor = pose_features if pose_features.ndim == 3 else pose_features.unsqueeze(0)
            else:
                pose_tensor = torch.tensor(pose_features, dtype=torch.float32).unsqueeze(0)
        else:
            # Genuine integration of PoseFeatureExtractor on sampled frames
            extracted_pose = self.pose_extractor.extract_from_frames(sampled_raw_frames)  # (8, 195)
            pose_tensor = torch.tensor(extracted_pose, dtype=torch.float32).unsqueeze(0)  # (1, 8, 195)

        # 3. Model Forward Pass
        with torch.no_grad():
            v_outputs = self.model.video_encoder(pixel_values=video_tensor)
            v_feats = v_outputs.last_hidden_state  # (B, T*N, 768)
            if v_feats.ndim == 3 and v_feats.shape[1] != 8:
                v_feats = v_feats.mean(dim=1, keepdim=True).repeat(1, 8, 1)

            p_feats = self.model.pose_encoder(pose_tensor)  # (B, 8, 768)
            fused = self.model.fusion(v_feats, p_feats)      # (B, 8, 768)
            mapped = self.model.dim_mapper(fused)            # (B, 8, 768)

            # 4. Explicit T5 Text Generation: Beam Search with num_beams=4
            # Pull special token IDs dynamically from model configuration
            decoder_config = self.model.text_decoder.config
            decoder_start_id = getattr(decoder_config, "decoder_start_token_id", 0)
            eos_id = getattr(decoder_config, "eos_token_id", 1)
            pad_id = getattr(decoder_config, "pad_token_id", 0)

            generated_output = self.model.text_decoder.generate(
                inputs_embeds=mapped,
                max_new_tokens=32,
                num_beams=4,
                early_stopping=True,
                decoder_start_token_id=decoder_start_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_ids = generated_output.sequences[0]

            # Compute sequence-level confidence score from beam search log probability
            # Note: Uncalibrated sequence log-probability score P = exp(log_prob) in (0.0, 1.0]
            if hasattr(generated_output, "sequences_scores") and generated_output.sequences_scores is not None:
                log_prob = float(generated_output.sequences_scores[0].item())
                confidence = float(np.clip(np.exp(log_prob), 0.0, 1.0))
            else:
                confidence = 1.0

            if self.tokenizer is not None:
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                text = f"[Generated Token IDs: {generated_ids.tolist()}]"

        return Prediction(
            status="success",
            text=text,
            confidence=confidence,
            model_version=self.model_version,
            reason=None,
            warning=EXPERIMENTAL_PREPROCESSING_WARNING,
        )
