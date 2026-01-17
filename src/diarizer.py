"""Speaker diarization using pyannote.audio."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DiarizationSegment:
    """A segment with speaker label."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class Diarizer:
    """Wrapper for pyannote speaker diarization."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str | None = None
    ):
        self.model_name = model_name
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

    @property
    def pipeline(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token
            )

            # Use MPS on Apple Silicon if available
            if torch.backends.mps.is_available():
                self._pipeline.to(torch.device("mps"))
            elif torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))

        return self._pipeline

    def diarize(self, audio_path: str | Path) -> list[DiarizationSegment]:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of DiarizationSegment objects with speaker labels
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        diarization_output = self.pipeline(str(audio_path))

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
