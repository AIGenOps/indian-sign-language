"""Pilot benchmark runner for evaluating test split clips end-to-end.

Requires REAL video files on disk (data/videos/<uid>.mp4).
Synthetic or placeholder data fallback is STRICTLY PROHIBITED and raises MissingVideoDataError.
"""

import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any

from src.dataset.fetch import load_isl_translate_records
from src.dataset.exceptions import MissingVideoDataError
from src.models.sanket import SanketModelProvider
from src.preprocessing.pose import PoseFeatureExtractor
from src.preprocessing.video import preprocess_video_frames, sample_uniform_frames
from src.evaluation.harness import BenchmarkEvaluator
from src.evaluation.results import EvaluationResult, UNVERIFIED_PREPROCESSING_DISCLAIMER

DEFAULT_VIDEO_DIR = Path("data/videos")


def run_pilot(num_clips: int = 10, video_dir: Path = DEFAULT_VIDEO_DIR) -> Dict[str, Any]:
    """Run pilot evaluation on num_clips test split items.

    Args:
        num_clips: Number of test clips to evaluate.
        video_dir: Directory containing real video files (<uid>.mp4).

    Returns:
        Structured evaluation report dict.

    Raises:
        MissingVideoDataError: If real video file for any requested UID is missing.
    """
    manifest_path = Path("configs/eval_split_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError("Split manifest not found. Run Stage B first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_uids = manifest["clip_uids"]["test"][:num_clips]
    all_records = load_isl_translate_records()
    records_by_uid = {r["uid"]: r for r in all_records}

    pilot_records = [records_by_uid[uid] for uid in test_uids if uid in records_by_uid]

    print(f"Loaded {len(pilot_records)} test split records for pilot run.")

    # HARD RULE: Verify all real video files exist before starting evaluation.
    # Synthetic/placeholder fallbacks are strictly prohibited.
    video_dir = Path(video_dir)
    missing_uids = []
    for rec in pilot_records:
        uid = rec["uid"]
        v_path = video_dir / f"{uid}.mp4"
        if not v_path.exists():
            missing_uids.append((uid, str(v_path)))

    if missing_uids:
        missing_str = "\n".join([f"  - UID '{uid}' missing at {p}" for uid, p in missing_uids])
        raise MissingVideoDataError(
            f"Cannot run pilot evaluation: {len(missing_uids)} real video file(s) are missing.\n"
            f"{missing_str}\n"
            "Silent fallback to synthetic or placeholder data is strictly prohibited."
        )

    # Initialize model provider & pose extractor
    print("Initializing SanketModelProvider and PoseFeatureExtractor...")
    model_provider = SanketModelProvider("artifacts/models/model_best.pth")
    model_provider.load()
    pose_extractor = PoseFeatureExtractor()

    references: List[str] = []
    hypotheses: List[str] = []
    latencies: List[float] = []

    crashes = 0
    start_total = time.perf_counter()

    for idx, rec in enumerate(pilot_records, 1):
        uid = rec["uid"]
        target_text = rec["text"]
        video_path = video_dir / f"{uid}.mp4"

        t0 = time.perf_counter()
        try:
            # 1. Read frames from real video clip
            cap = cv2.VideoCapture(str(video_path))
            raw_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)
            cap.release()

            if not raw_frames:
                raise ValueError(f"No frames could be decoded from real video file {video_path}")

            # 2. Preprocessing & Pose extraction
            v_np = preprocess_video_frames(raw_frames)
            video_tensor = torch.tensor(v_np, dtype=torch.float32).unsqueeze(0)

            p_np = pose_extractor.extract_from_frames(raw_frames)
            p_sampled = np.array(sample_uniform_frames(p_np, num_samples=8))
            pose_tensor = torch.tensor(p_sampled, dtype=torch.float32).unsqueeze(0)

            # 3. Model inference path
            pred = model_provider.predict(
                frames=video_tensor,
                pose_features=pose_tensor,
                allow_experimental_inference=True,
            )

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies.append(elapsed_ms)

            references.append(target_text)
            hypotheses.append(pred.text)

            print(
                f"Clip [{idx}/{len(pilot_records)}] UID={uid} | Latency: {elapsed_ms:.1f}ms | "
                f"Target: '{target_text}' | Output: '{pred.text}'"
            )

        except Exception as e:
            crashes += 1
            print(f"ERROR processing clip {uid}: {e}")

    total_time_s = time.perf_counter() - start_total

    # Run benchmark metrics
    evaluator = BenchmarkEvaluator()
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    eval_result = evaluator.evaluate(
        references, hypotheses, split_name="pilot_test", average_latency_ms=mean_latency
    )

    # Calculate extrapolation for full test split (3,137 clips)
    full_test_clips = len(manifest["clip_uids"]["test"])  # 3,137
    extrapolated_seconds = (mean_latency / 1000.0) * full_test_clips
    extrapolated_hours = extrapolated_seconds / 3600.0

    report = {
        "pilot_samples": len(pilot_records),
        "successful_clips": len(hypotheses),
        "crashes": crashes,
        "mean_latency_ms_per_clip": round(mean_latency, 2),
        "total_pilot_wall_clock_s": round(total_time_s, 2),
        "full_test_split_clips": full_test_clips,
        "extrapolated_full_test_runtime_hours": round(extrapolated_hours, 2),
        "metrics": eval_result.to_dict(),
    }

    return report


if __name__ == "__main__":
    report = run_pilot(num_clips=10)
    print("\n" + "=" * 60)
    print("PILOT BENCHMARK REPORT:")
    print("=" * 60)
    print(json.dumps(report, indent=2))
