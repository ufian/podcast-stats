"""Speaker diarization using pyannote.audio."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm


@dataclass
class DiarizationSegment:
    """A segment with speaker label."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class TqdmProgressHook:
    """Progress hook for pyannote pipeline using tqdm."""

    # Steps that are called once (no progress updates)
    INSTANT_STEPS = {"speaker_counting", "clustering"}

    def __init__(self):
        self.pbar: tqdm | None = None
        self.current_step = 0
        self.current_total = 0
        self.last_step_name = ""
        self.shown_steps: set[str] = set()

    def __call__(
        self,
        step_name: str,
        step_artifact: any,
        file: dict | None = None,
        total: int | None = None,
        completed: int | None = None
    ):
        """Called by pyannote pipeline during processing."""
        # Track step changes to show current phase
        if step_name != self.last_step_name:
            # Close previous progress bar
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None

            self.last_step_name = step_name
            self.current_step = 0
            self.current_total = 0

            # For instant steps (called once without progress), show completion
            if step_name in self.INSTANT_STEPS and step_name not in self.shown_steps:
                self.shown_steps.add(step_name)
                tqdm.write(f"  Diarization ({step_name}): done")
                return

        if completed is not None:
            # Create or update progress bar
            if self.pbar is None:
                # Use total if known, otherwise show spinner-style
                self.current_total = total if total else 0
                self.pbar = tqdm(
                    total=self.current_total if self.current_total > 0 else None,
                    desc=f"Diarization ({step_name})",
                    unit="step",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" if self.current_total > 0
                        else "{desc}: {n_fmt} steps [{elapsed}]"
                )

            # Update progress
            delta = completed - self.current_step
            if delta > 0:
                self.pbar.update(delta)
                self.current_step = completed

            # Update total if it changed
            if total and total != self.current_total and self.current_total == 0:
                self.current_total = total
                self.pbar.total = total
                self.pbar.refresh()

    def close(self):
        """Close the progress bar."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class Diarizer:
    """Wrapper for pyannote speaker diarization."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str | None = None,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        # Use provided token, fall back to env var if empty/None
        self.hf_token = hf_token if hf_token else os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "Hugging Face token required for pyannote models. "
                "Set it in config.yaml under huggingface.token. "
                "You also need to accept the model terms at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
        self._pipeline = None

    def _get_device(self) -> torch.device:
        """Get the torch device based on configuration."""
        if self.device == "cpu":
            return torch.device("cpu")
        elif self.device == "mps":
            return torch.device("mps")
        elif self.device == "cuda":
            return torch.device("cuda")
        else:  # auto
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

    @property
    def pipeline(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token
            )

            device = self._get_device()
            if device.type != "cpu":
                self._pipeline.to(device)

        return self._pipeline

    def diarize(
        self,
        audio_path: str | Path,
        show_progress: bool = True,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None
    ) -> list[DiarizationSegment]:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            show_progress: Show progress bar during diarization
            num_speakers: Exact number of speakers (if known, speeds up processing)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            List of DiarizationSegment objects with speaker labels
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Build kwargs for pipeline
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        if show_progress:
            # Use progress hook for pyannote pipeline
            hook = TqdmProgressHook()
            kwargs["hook"] = hook
            diarization_output = self.pipeline(str(audio_path), **kwargs)
            hook.close()
        else:
            diarization_output = self.pipeline(str(audio_path), **kwargs)

        # pyannote 4.x returns DiarizeOutput, extract the Annotation
        if hasattr(diarization_output, 'speaker_diarization'):
            diarization = diarization_output.speaker_diarization
        else:
            diarization = diarization_output

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker
            ))

        return segments

    def get_speaker_stats(
        self,
        segments: list[DiarizationSegment]
    ) -> dict[str, float]:
        """Calculate speaking time per speaker.

        Args:
            segments: List of diarization segments

        Returns:
            Dictionary mapping speaker ID to total speaking time in seconds
        """
        stats: dict[str, float] = {}
        for segment in segments:
            if segment.speaker not in stats:
                stats[segment.speaker] = 0.0
            stats[segment.speaker] += segment.duration
        return stats


def align_transcription_with_diarization(
    transcription_segments: list,
    diarization_segments: list[DiarizationSegment]
) -> list[dict]:
    """Align transcription segments with speaker labels.

    Uses overlap-based assignment: each transcription segment is assigned
    to the speaker with the most overlap.

    Args:
        transcription_segments: List of TranscriptionSegment objects
        diarization_segments: List of DiarizationSegment objects

    Returns:
        List of dictionaries with transcription data and speaker_id
    """
    aligned = []

    for trans_seg in transcription_segments:
        # Find overlapping diarization segments
        overlaps: dict[str, float] = {}

        for diar_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(trans_seg.start, diar_seg.start)
            overlap_end = min(trans_seg.end, diar_seg.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                if diar_seg.speaker not in overlaps:
                    overlaps[diar_seg.speaker] = 0.0
                overlaps[diar_seg.speaker] += overlap

        # Assign to speaker with most overlap
        if overlaps:
            speaker_id = max(overlaps, key=overlaps.get)
        else:
            speaker_id = "UNKNOWN"

        aligned.append({
            "start": trans_seg.start,
            "end": trans_seg.end,
            "speaker_id": speaker_id,
            "text": trans_seg.text,
            "avg_logprob": trans_seg.avg_logprob,
            "no_speech_prob": trans_seg.no_speech_prob,
            "words": trans_seg.words,
            "refined_by_openai": getattr(trans_seg, "refined_by_openai", False)
        })

    return aligned
