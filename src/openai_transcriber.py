"""OpenAI API transcription fallback for low-confidence segments."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm


@dataclass
class OpenAISegment:
    """A transcribed segment from OpenAI verbose JSON response."""

    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


class OpenAITranscriber:
    """Wrapper for OpenAI Whisper API transcription."""

    # OpenAI Whisper API pricing: $0.006 per minute
    COST_PER_MINUTE_USD = 0.006

    def __init__(self, api_key: str | None = None, model: str = "whisper-1"):
        # Use provided key, fall back to env var if empty/None
        self.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set it in config.yaml under openai.api_key"
            )
        self.model = model
        self._client: OpenAI | None = None
        self.total_minutes_processed = 0.0
        self.total_cost_usd = 0.0

    @property
    def client(self) -> OpenAI:
        """Lazy load the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def transcribe_segment(
        self,
        audio_path: str | Path,
        start: float,
        end: float,
        language: str = "ru",
        prompt: str | None = None
    ) -> str:
        """Transcribe a specific segment of an audio file.

        Args:
            audio_path: Path to the audio file
            start: Start time in seconds
            end: End time in seconds
            language: Language code
            prompt: Optional prompt to guide transcription

        Returns:
            Transcribed text
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract the segment
        audio = AudioSegment.from_file(str(audio_path))
        segment = audio[int(start * 1000):int(end * 1000)]

        # Calculate duration and cost
        duration_minutes = (end - start) / 60.0
        segment_cost = duration_minutes * self.COST_PER_MINUTE_USD

        # Export to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            segment.export(f.name, format="mp3")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                if prompt is None:
                    prompt = "Подкаст на русском языке с техническими терминами на английском"

                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    prompt=prompt
                )

            # Track usage
            self.total_minutes_processed += duration_minutes
            self.total_cost_usd += segment_cost

            return transcript.text.strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str = "ru",
        prompt: str | None = None
    ) -> str:
        """Transcribe an entire audio file.

        Args:
            audio_path: Path to the audio file
            language: Language code
            prompt: Optional prompt to guide transcription

        Returns:
            Transcribed text
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get duration for cost tracking
        audio = AudioSegment.from_file(str(audio_path))
        duration_minutes = len(audio) / 60000.0  # pydub uses milliseconds
        segment_cost = duration_minutes * self.COST_PER_MINUTE_USD

        # OpenAI API has a 25MB file size limit
        # For larger files, we need to split them
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)

        if file_size_mb > 24:  # Leave some margin
            return self._transcribe_large_file(audio_path, language, prompt)

        with open(audio_path, "rb") as audio_file:
            if prompt is None:
                prompt = "Подкаст на русском языке с техническими терминами на английском"

            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language,
                prompt=prompt
            )

        # Track usage
        self.total_minutes_processed += duration_minutes
        self.total_cost_usd += segment_cost

        return transcript.text.strip()

    def _transcribe_large_file(
        self,
        audio_path: Path,
        language: str,
        prompt: str | None
    ) -> str:
        """Handle files larger than OpenAI's 25MB limit by chunking."""
        audio = AudioSegment.from_file(str(audio_path))

        # Split into 10-minute chunks (well under 25MB for MP3)
        chunk_duration_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
        chunks = []

        for i in range(0, len(audio), chunk_duration_ms):
            chunk = audio[i:i + chunk_duration_ms]
            chunks.append(chunk)

        transcripts = []
        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                chunk.export(f.name, format="mp3")
                temp_path = f.name

            try:
                with open(temp_path, "rb") as audio_file:
                    if prompt is None:
                        prompt = "Подкаст на русском языке с техническими терминами на английском"

                    transcript = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=language,
                        prompt=prompt
                    )
                    transcripts.append(transcript.text.strip())

                # Track usage for this chunk
                chunk_duration_minutes = len(chunk) / 60000.0
                self.total_minutes_processed += chunk_duration_minutes
                self.total_cost_usd += chunk_duration_minutes * self.COST_PER_MINUTE_USD

            finally:
                Path(temp_path).unlink(missing_ok=True)

        return " ".join(transcripts)

    def transcribe_with_timestamps(
        self,
        audio_path: str | Path,
        chunk_duration_sec: float = 60.0,
        language: str = "ru",
        prompt: str | None = None
    ) -> list[OpenAISegment]:
        """Transcribe audio file in chunks and return segments with timestamps.

        Uses OpenAI's verbose_json response format to get segment-level timestamps.

        Args:
            audio_path: Path to the audio file
            chunk_duration_sec: Duration of each chunk in seconds (default 60s)
            language: Language code
            prompt: Optional prompt to guide transcription

        Returns:
            List of OpenAISegment objects with timestamps
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = AudioSegment.from_file(str(audio_path))
        total_duration_ms = len(audio)
        chunk_duration_ms = int(chunk_duration_sec * 1000)

        if prompt is None:
            prompt = "Подкаст на русском языке с техническими терминами на английском"

        all_segments: list[OpenAISegment] = []
        chunk_starts = list(range(0, total_duration_ms, chunk_duration_ms))

        for chunk_start_ms in tqdm(chunk_starts, desc="OpenAI transcribe",
                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            chunk_end_ms = min(chunk_start_ms + chunk_duration_ms, total_duration_ms)
            chunk = audio[chunk_start_ms:chunk_end_ms]

            # Calculate time offset for this chunk
            time_offset = chunk_start_ms / 1000.0

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                chunk.export(f.name, format="mp3")
                temp_path = f.name

            try:
                with open(temp_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=language,
                        prompt=prompt,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"]
                    )

                # Track usage for this chunk
                chunk_duration_minutes = (chunk_end_ms - chunk_start_ms) / 60000.0
                self.total_minutes_processed += chunk_duration_minutes
                self.total_cost_usd += chunk_duration_minutes * self.COST_PER_MINUTE_USD

                # Parse segments from response (OpenAI returns Pydantic models)
                if hasattr(response, 'segments') and response.segments:
                    for seg in response.segments:
                        all_segments.append(OpenAISegment(
                            start=time_offset + getattr(seg, 'start', 0),
                            end=time_offset + getattr(seg, 'end', 0),
                            text=getattr(seg, 'text', "").strip(),
                            avg_logprob=getattr(seg, 'avg_logprob', 0.0),
                            no_speech_prob=getattr(seg, 'no_speech_prob', 0.0)
                        ))
                else:
                    # Fallback if no segments in response
                    all_segments.append(OpenAISegment(
                        start=time_offset,
                        end=time_offset + (chunk_end_ms - chunk_start_ms) / 1000.0,
                        text=getattr(response, 'text', "").strip()
                    ))

            finally:
                Path(temp_path).unlink(missing_ok=True)

        return all_segments

    def get_usage_stats(self) -> dict:
        """Get cumulative usage statistics."""
        return {
            "total_minutes": round(self.total_minutes_processed, 2),
            "total_cost_usd": round(self.total_cost_usd, 4)
        }
