"""Main processing pipeline orchestrating transcription, diarization, and speaker tracking."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml
from pydub import AudioSegment

from .diarizer import Diarizer, DiarizationSegment, align_transcription_with_diarization
from .openai_transcriber import OpenAITranscriber
from .speaker_tracker import Speaker, SpeakerMatch, SpeakerTracker
from .transcriber import LocalTranscriber, TranscriptionSegment


@dataclass
class ProcessingResult:
    """Result of processing an episode."""

    episode_id: str
    source_file: str
    processed_at: str
    duration_sec: float
    speakers: list[str]
    segments: list[dict[str, Any]]
    transcription_stats: dict[str, Any]
    speaker_mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_file": self.source_file,
            "processed_at": self.processed_at,
            "duration_sec": self.duration_sec,
            "speakers": self.speakers,
            "transcription_stats": self.transcription_stats,
            "segments": self.segments
        }

    def save(self, output_dir: str | Path = "data/episodes") -> Path:
        """Save the episode data to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.episode_id}.json"
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return output_path


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""

    # API keys
    openai_api_key: str = ""
    hf_token: str = ""

    # Whisper settings
    whisper_model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "float16"
    language: str = "ru"

    # Confidence thresholds
    avg_logprob_threshold: float = -1.0
    no_speech_prob_threshold: float = 0.5
    openai_for_code_switching: bool = True

    # Speaker matching
    similarity_threshold: float = 0.75
    high_confidence_threshold: float = 0.85

    # Paths
    speakers_path: str = "data/speakers.json"
    episodes_path: str = "data/episodes"

    # Diarization
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/embedding"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        openai_cfg = data.get("openai", {})
        hf_cfg = data.get("huggingface", {})
        transcription = data.get("transcription", {})
        speaker_matching = data.get("speaker_matching", {})
        diarization = data.get("diarization", {})

        return cls(
            openai_api_key=openai_cfg.get("api_key", ""),
            hf_token=hf_cfg.get("token", ""),
            whisper_model=transcription.get("whisper_model", "large-v3"),
            device=transcription.get("device", "auto"),
            compute_type=transcription.get("compute_type", "float16"),
            language=transcription.get("language", "ru"),
            avg_logprob_threshold=transcription.get("avg_logprob_threshold", -1.0),
            no_speech_prob_threshold=transcription.get("no_speech_prob_threshold", 0.5),
            openai_for_code_switching=transcription.get("openai_for_code_switching", True),
            similarity_threshold=speaker_matching.get("similarity_threshold", 0.75),
            high_confidence_threshold=speaker_matching.get("high_confidence_threshold", 0.85),
            diarization_model=diarization.get("model", "pyannote/speaker-diarization-3.1"),
            embedding_model=diarization.get("embedding_model", "pyannote/embedding"),
        )


class Pipeline:
    """Main processing pipeline."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        config_path: str | Path = "config.yaml",
        speaker_callback: Callable[[str, list[SpeakerMatch], float], tuple[str, str]] | None = None
    ):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
            config_path: Path to config YAML file (used if config not provided)
            speaker_callback: Callback for speaker identification.
                Takes (detected_speaker_label, matches, speaking_time) and returns
                (speaker_id, speaker_name)
        """
        if config is None:
            config = PipelineConfig.from_yaml(config_path)
        self.config = config
        self.speaker_callback = speaker_callback

        self._transcriber: LocalTranscriber | None = None
        self._openai_transcriber: OpenAITranscriber | None = None
        self._diarizer: Diarizer | None = None
        self._speaker_tracker: SpeakerTracker | None = None

    @property
    def transcriber(self) -> LocalTranscriber:
        if self._transcriber is None:
            self._transcriber = LocalTranscriber(
                model_name=self.config.whisper_model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                language=self.config.language
            )
        return self._transcriber

    @property
    def openai_transcriber(self) -> OpenAITranscriber:
        if self._openai_transcriber is None:
            self._openai_transcriber = OpenAITranscriber(
                api_key=self.config.openai_api_key
            )
        return self._openai_transcriber

    @property
    def diarizer(self) -> Diarizer:
        if self._diarizer is None:
            self._diarizer = Diarizer(
                model_name=self.config.diarization_model,
                hf_token=self.config.hf_token
            )
        return self._diarizer

    @property
    def speaker_tracker(self) -> SpeakerTracker:
        if self._speaker_tracker is None:
            self._speaker_tracker = SpeakerTracker(
                speakers_path=self.config.speakers_path,
                embedding_model=self.config.embedding_model,
                hf_token=self.config.hf_token,
                similarity_threshold=self.config.similarity_threshold,
                high_confidence_threshold=self.config.high_confidence_threshold
            )
        return self._speaker_tracker

    def process(
        self,
        audio_path: str | Path,
        duration_limit: float | None = None,
        openai_all: bool = False,
        progress_callback: Callable[[str], None] | None = None
    ) -> ProcessingResult:
        """Process an audio file.

        Args:
            audio_path: Path to the audio file
            duration_limit: Optional limit on processing duration (seconds)
            openai_all: If True, use OpenAI for all transcription
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with all transcription and speaker data
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        def log(msg: str):
            if progress_callback:
                progress_callback(msg)

        # Generate episode ID from filename
        episode_id = self._generate_episode_id(audio_path)

        # Get audio duration
        audio = AudioSegment.from_file(str(audio_path))
        total_duration_sec = len(audio) / 1000.0

        # Handle duration limit
        processing_path = audio_path
        if duration_limit is not None and duration_limit < total_duration_sec:
            log(f"Limiting processing to first {duration_limit:.0f} seconds")
            processing_path = self._extract_audio_segment(
                audio_path, 0, duration_limit
            )
            total_duration_sec = duration_limit

        try:
            # Step 1: Diarization
            log("Starting speaker diarization...")
            diarization_segments = self.diarizer.diarize(processing_path)
            log(f"Found {len(set(s.speaker for s in diarization_segments))} speakers")

            # Step 2: Transcription
            if openai_all:
                log("Transcribing with OpenAI API...")
                segments = self._transcribe_with_openai(processing_path, diarization_segments)
            else:
                log("Transcribing with local Whisper...")
                segments = self._transcribe_hybrid(processing_path, log)

            # Step 3: Align transcription with diarization
            log("Aligning transcription with speaker labels...")
            aligned_segments = align_transcription_with_diarization(
                segments, diarization_segments
            )

            # Step 4: Speaker identification
            log("Extracting speaker embeddings...")
            speaker_mapping = self._identify_speakers(
                processing_path, diarization_segments, episode_id, log
            )

            # Apply speaker mapping to segments
            final_segments = []
            for seg in aligned_segments:
                detected_speaker = seg["speaker_id"]
                mapped_speaker = speaker_mapping.get(detected_speaker, detected_speaker)

                final_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker_id": mapped_speaker,
                    "text": seg["text"],
                    "confidence": self._calculate_confidence(
                        seg["avg_logprob"], seg["no_speech_prob"]
                    ),
                    "refined_by_openai": seg.get("refined_by_openai", False)
                })

            # Calculate stats
            openai_stats = self.openai_transcriber.get_usage_stats() if self._openai_transcriber else {}
            refined_count = sum(1 for s in final_segments if s.get("refined_by_openai"))

            result = ProcessingResult(
                episode_id=episode_id,
                source_file=audio_path.name,
                processed_at=datetime.now().isoformat(),
                duration_sec=total_duration_sec,
                speakers=list(set(s["speaker_id"] for s in final_segments)),
                segments=final_segments,
                transcription_stats={
                    "total_segments": len(final_segments),
                    "openai_refined_segments": refined_count,
                    "openai_minutes": openai_stats.get("total_minutes", 0),
                    "openai_cost_usd": openai_stats.get("total_cost_usd", 0)
                },
                speaker_mapping=speaker_mapping
            )

            log("Processing complete!")
            return result

        finally:
            # Clean up temp file if we created one
            if processing_path != audio_path and processing_path.exists():
                processing_path.unlink()

    def _generate_episode_id(self, audio_path: Path) -> str:
        """Generate episode ID from filename."""
        name = audio_path.stem
        # Remove common prefixes/suffixes and clean up
        name = re.sub(r'[^\w\-]', '_', name)
        return f"ep_{name}"

    def _extract_audio_segment(
        self,
        audio_path: Path,
        start: float,
        end: float
    ) -> Path:
        """Extract a segment of audio to a temporary file."""
        import tempfile

        audio = AudioSegment.from_file(str(audio_path))
        segment = audio[int(start * 1000):int(end * 1000)]

        # Export as WAV with 16kHz sample rate for pyannote compatibility
        segment = segment.set_frame_rate(16000).set_channels(1)

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )
        segment.export(temp_file.name, format="wav")
        return Path(temp_file.name)

    def _transcribe_hybrid(
        self,
        audio_path: Path,
        log: Callable[[str], None]
    ) -> list[TranscriptionSegment]:
        """Transcribe using hybrid local + OpenAI approach."""
        segments = list(self.transcriber.transcribe(audio_path))
        log(f"Local transcription: {len(segments)} segments")

        # Check which segments need OpenAI fallback
        fallback_count = 0
        refined_segments = []

        for seg in segments:
            needs_fallback = seg.needs_fallback(
                avg_logprob_threshold=self.config.avg_logprob_threshold,
                no_speech_prob_threshold=self.config.no_speech_prob_threshold,
                check_code_switching=self.config.openai_for_code_switching
            )

            if needs_fallback:
                try:
                    refined_text = self.openai_transcriber.transcribe_segment(
                        audio_path, seg.start, seg.end
                    )
                    seg.text = refined_text
                    seg.refined_by_openai = True
                    fallback_count += 1
                except Exception as e:
                    log(f"OpenAI fallback failed for segment {seg.start:.1f}-{seg.end:.1f}: {e}")
                    seg.refined_by_openai = False
            else:
                seg.refined_by_openai = False

            refined_segments.append(seg)

        if fallback_count > 0:
            log(f"Refined {fallback_count} segments with OpenAI")

        return refined_segments

    def _transcribe_with_openai(
        self,
        audio_path: Path,
        diarization_segments: list[DiarizationSegment]
    ) -> list[TranscriptionSegment]:
        """Transcribe entirely with OpenAI API."""
        text = self.openai_transcriber.transcribe_file(audio_path)

        # Create pseudo-segments based on diarization timing
        # This is a simplified approach - OpenAI doesn't give word timestamps
        segments = []
        for diar_seg in diarization_segments:
            segments.append(TranscriptionSegment(
                start=diar_seg.start,
                end=diar_seg.end,
                text="",  # Will be filled by alignment
                avg_logprob=0.0,
                no_speech_prob=0.0
            ))
            segments[-1].refined_by_openai = True

        # For full OpenAI mode, we just return the whole text as one segment
        # and let alignment handle speaker assignment
        if len(diarization_segments) > 0:
            segments = [TranscriptionSegment(
                start=diarization_segments[0].start,
                end=diarization_segments[-1].end,
                text=text,
                avg_logprob=0.0,
                no_speech_prob=0.0
            )]
            segments[0].refined_by_openai = True

        return segments

    def _identify_speakers(
        self,
        audio_path: Path,
        diarization_segments: list[DiarizationSegment],
        episode_id: str,
        log: Callable[[str], None]
    ) -> dict[str, str]:
        """Identify speakers by matching embeddings.

        Returns mapping from detected speaker labels to known speaker IDs.
        """
        speaker_mapping: dict[str, str] = {}

        # Get speaking time per speaker
        speaker_times = self.diarizer.get_speaker_stats(diarization_segments)

        # Extract embeddings
        embeddings = self.speaker_tracker.extract_embeddings_for_speakers(
            audio_path, diarization_segments
        )

        for detected_speaker, embedding in embeddings.items():
            speaking_time = speaker_times.get(detected_speaker, 0)
            matches = self.speaker_tracker.find_matches(embedding)

            if self.speaker_callback:
                # Use callback for interactive identification
                speaker_id, speaker_name = self.speaker_callback(
                    detected_speaker, matches, speaking_time
                )

                if speaker_id.startswith("new:"):
                    # Create new speaker
                    new_speaker = self.speaker_tracker.add_speaker(
                        name=speaker_name,
                        embedding=embedding,
                        episode_id=episode_id,
                        speaking_time=speaking_time
                    )
                    speaker_mapping[detected_speaker] = new_speaker.id
                    log(f"Created new speaker: {speaker_name} ({new_speaker.id})")
                else:
                    # Use existing speaker
                    self.speaker_tracker.update_speaker(
                        speaker_id,
                        episode_id=episode_id,
                        speaking_time=speaking_time,
                        new_embedding=embedding
                    )
                    speaker_mapping[detected_speaker] = speaker_id
                    speaker = self.speaker_tracker.get_speaker(speaker_id)
                    if speaker:
                        log(f"Matched {detected_speaker} to {speaker.name}")
            else:
                # Auto mode: use high confidence matches, keep label for others
                if matches and matches[0].similarity >= self.config.high_confidence_threshold:
                    match = matches[0]
                    self.speaker_tracker.update_speaker(
                        match.known_speaker.id,
                        episode_id=episode_id,
                        speaking_time=speaking_time,
                        new_embedding=embedding
                    )
                    speaker_mapping[detected_speaker] = match.known_speaker.id
                    log(f"Auto-matched {detected_speaker} to {match.known_speaker.name}")
                else:
                    # Keep original label
                    speaker_mapping[detected_speaker] = detected_speaker

        return speaker_mapping

    def _calculate_confidence(
        self,
        avg_logprob: float,
        no_speech_prob: float
    ) -> float:
        """Calculate a normalized confidence score."""
        # Convert log prob to probability-like score
        # avg_logprob typically ranges from -2 to 0
        logprob_score = max(0, min(1, (avg_logprob + 2) / 2))
        # no_speech_prob is already 0-1, invert it
        speech_score = 1 - no_speech_prob
        # Average the two
        return round((logprob_score + speech_score) / 2, 2)


def parse_duration(duration_str: str) -> float:
    """Parse duration string like '5m' or '30s' to seconds."""
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(s|m|h)?$', duration_str.lower())
    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")

    value = float(match.group(1))
    unit = match.group(2) or 's'

    multipliers = {'s': 1, 'm': 60, 'h': 3600}
    return value * multipliers[unit]
