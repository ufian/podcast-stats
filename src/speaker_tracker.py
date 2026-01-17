"""Speaker tracking and embedding matching."""

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class Speaker:
    """A known speaker profile."""

    id: str
    name: str
    embedding: list[float]
    episodes: list[str] = field(default_factory=list)
    total_speaking_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "embedding": self.embedding,
            "episodes": self.episodes,
            "total_speaking_time_sec": self.total_speaking_time_sec
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Speaker":
        return cls(
            id=data["id"],
            name=data["name"],
            embedding=data["embedding"],
            episodes=data.get("episodes", []),
            total_speaking_time_sec=data.get("total_speaking_time_sec", 0.0)
        )


@dataclass
class SpeakerMatch:
    """A potential match between a detected speaker and a known speaker."""

    known_speaker: Speaker
    similarity: float

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence match (>0.85)."""
        return self.similarity >= 0.85


class SpeakerTracker:
    """Manages speaker profiles and matching."""

    def __init__(
        self,
        speakers_path: str | Path = "data/speakers.json",
        embedding_model: str = "pyannote/embedding",
        hf_token: str | None = None,
        similarity_threshold: float = 0.75,
        high_confidence_threshold: float = 0.85,
        device: str = "auto"
    ):
        self.speakers_path = Path(speakers_path)
        self.embedding_model = embedding_model
        self.device = device
        # Use provided token, fall back to env var if empty/None
        self.hf_token = hf_token if hf_token else os.environ.get("HF_TOKEN")
        self.similarity_threshold = similarity_threshold
        self.high_confidence_threshold = high_confidence_threshold

        self._embedding_pipeline = None
        self.speakers: list[Speaker] = []
        self._load_speakers()

    def _load_speakers(self) -> None:
        """Load speakers from JSON file."""
        if self.speakers_path.exists():
            with open(self.speakers_path) as f:
                data = json.load(f)
                self.speakers = [
                    Speaker.from_dict(s) for s in data.get("speakers", [])
                ]
        else:
            self.speakers = []

    def save_speakers(self) -> None:
        """Save speakers to JSON file."""
        self.speakers_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.speakers_path, "w") as f:
            json.dump(
                {"speakers": [s.to_dict() for s in self.speakers]},
                f,
                indent=2,
                ensure_ascii=False
            )

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
    def embedding_pipeline(self):
        """Lazy load the embedding model."""
        if self._embedding_pipeline is None:
            from pyannote.audio import Model, Inference

            model = Model.from_pretrained(
                self.embedding_model,
                token=self.hf_token
            )

            device = self._get_device()
            if device.type != "cpu":
                model.to(device)

            self._embedding_pipeline = Inference(model, window="whole")

        return self._embedding_pipeline

    def extract_embedding(
        self,
        audio_path: str | Path,
        start: float | None = None,
        end: float | None = None
    ) -> np.ndarray:
        """Extract speaker embedding from audio.

        Args:
            audio_path: Path to audio file
            start: Optional start time in seconds
            end: Optional end time in seconds

        Returns:
            Embedding vector as numpy array
        """
        from pyannote.core import Segment

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if start is not None and end is not None:
            excerpt = Segment(start, end)
            embedding = self._embedding_pipeline.crop(str(audio_path), excerpt)
        else:
            embedding = self._embedding_pipeline(str(audio_path))

        return embedding

    def extract_embeddings_for_speakers(
        self,
        audio_path: str | Path,
        diarization_segments: list
    ) -> dict[str, np.ndarray]:
        """Extract embeddings for each detected speaker.

        Combines all segments for each speaker to get a robust embedding.

        Args:
            audio_path: Path to audio file
            diarization_segments: List of DiarizationSegment objects

        Returns:
            Dictionary mapping speaker label to embedding
        """
        from pydub import AudioSegment
        import tempfile

        # Group segments by speaker
        speaker_segments: dict[str, list] = {}
        for seg in diarization_segments:
            if seg.speaker not in speaker_segments:
                speaker_segments[seg.speaker] = []
            speaker_segments[seg.speaker].append(seg)

        embeddings = {}
        audio = AudioSegment.from_file(str(audio_path))

        for speaker, segments in tqdm(speaker_segments.items(), desc="Extracting embeddings",
                                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            # Combine segments for this speaker (up to 60 seconds)
            combined = AudioSegment.empty()
            total_duration = 0
            max_duration = 60000  # 60 seconds in ms

            for seg in sorted(segments, key=lambda s: s.duration, reverse=True):
                if total_duration >= max_duration:
                    break
                seg_audio = audio[int(seg.start * 1000):int(seg.end * 1000)]
                combined += seg_audio
                total_duration += len(seg_audio)

            # Export to temp file and extract embedding
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                combined.export(f.name, format="wav")
                temp_path = f.name

            try:
                embedding = self.embedding_pipeline(temp_path)
                embeddings[speaker] = embedding
            finally:
                Path(temp_path).unlink(missing_ok=True)

        return embeddings

    def find_matches(
        self,
        embedding: np.ndarray
    ) -> list[SpeakerMatch]:
        """Find known speakers matching an embedding.

        Args:
            embedding: Speaker embedding to match

        Returns:
            List of SpeakerMatch objects, sorted by similarity (descending)
        """
        matches = []

        for speaker in self.speakers:
            known_embedding = np.array(speaker.embedding)
            similarity = self._cosine_similarity(embedding, known_embedding)

            if similarity >= self.similarity_threshold:
                matches.append(SpeakerMatch(
                    known_speaker=speaker,
                    similarity=similarity
                ))

        # Sort by similarity (highest first)
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def add_speaker(
        self,
        name: str,
        embedding: np.ndarray,
        episode_id: str | None = None,
        speaking_time: float = 0.0
    ) -> Speaker:
        """Add a new speaker to the database.

        Args:
            name: Speaker name
            embedding: Speaker embedding
            episode_id: Optional first episode ID
            speaking_time: Speaking time in this episode

        Returns:
            The new Speaker object
        """
        speaker_id = f"spk_{uuid.uuid4().hex[:6]}"
        speaker = Speaker(
            id=speaker_id,
            name=name,
            embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            episodes=[episode_id] if episode_id else [],
            total_speaking_time_sec=speaking_time
        )
        self.speakers.append(speaker)
        self.save_speakers()
        return speaker

    def update_speaker(
        self,
        speaker_id: str,
        episode_id: str | None = None,
        speaking_time: float = 0.0,
        new_embedding: np.ndarray | None = None
    ) -> Speaker | None:
        """Update a speaker's profile.

        Args:
            speaker_id: ID of speaker to update
            episode_id: Episode to add to their list
            speaking_time: Speaking time to add
            new_embedding: Optional new embedding to average with existing

        Returns:
            Updated Speaker object, or None if not found
        """
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                if episode_id and episode_id not in speaker.episodes:
                    speaker.episodes.append(episode_id)
                speaker.total_speaking_time_sec += speaking_time

                if new_embedding is not None:
                    # Average embeddings (simple approach)
                    old_emb = np.array(speaker.embedding)
                    new_emb = np.asarray(new_embedding).flatten()
                    averaged = (old_emb + new_emb) / 2
                    # Normalize
                    averaged = averaged / np.linalg.norm(averaged)
                    speaker.embedding = averaged.tolist()

                self.save_speakers()
                return speaker

        return None

    def rename_speaker(self, speaker_id: str, new_name: str) -> Speaker | None:
        """Rename a speaker.

        Args:
            speaker_id: ID of speaker to rename
            new_name: New name for the speaker

        Returns:
            Updated Speaker object, or None if not found
        """
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                speaker.name = new_name
                self.save_speakers()
                return speaker
        return None

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        """Get a speaker by ID."""
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                return speaker
        return None

    def list_speakers(self) -> list[Speaker]:
        """Get all known speakers."""
        return self.speakers.copy()
