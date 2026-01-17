"""Local Whisper transcription using faster-whisper."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from faster_whisper import WhisperModel


@dataclass
class TranscriptionSegment:
    """A transcribed segment with metadata."""

    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[dict] | None = None
    refined_by_openai: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def needs_fallback(
        self,
        avg_logprob_threshold: float = -1.0,
        no_speech_prob_threshold: float = 0.5,
        check_code_switching: bool = True
    ) -> bool:
        """Check if this segment should be sent to OpenAI for refinement."""
        if self.avg_logprob < avg_logprob_threshold:
            return True
        if self.no_speech_prob > no_speech_prob_threshold:
            return True
        if check_code_switching and has_code_switching(self.text):
            return True
        return False


def has_code_switching(text: str) -> bool:
    """Detect if text contains mix of Cyrillic and Latin characters.

    This indicates code-switching between Russian and English,
    which local Whisper often handles poorly.
    """
    has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    return has_cyrillic and has_latin


class LocalTranscriber:
    """Wrapper for faster-whisper local transcription."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        language: str = "ru"
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        """Lazy load the Whisper model."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        vad_filter: bool = True,
        word_timestamps: bool = True
    ) -> Iterator[TranscriptionSegment]:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
            vad_filter: Use VAD to filter out non-speech
            word_timestamps: Include word-level timestamps

        Yields:
            TranscriptionSegment objects for each detected segment
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )

        for segment in segments:
            words = None
            if word_timestamps and segment.words:
                words = [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability
                    }
                    for w in segment.words
                ]

            yield TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                avg_logprob=segment.avg_logprob,
                no_speech_prob=segment.no_speech_prob,
                words=words
            )

    def transcribe_segment(
        self,
        audio_path: str | Path,
        start: float,
        end: float
    ) -> str:
        """Transcribe a specific segment of an audio file.

        This is used for re-transcribing segments that need refinement.
        Note: This is less efficient than transcribing the whole file,
        so it's mainly used for testing or when OpenAI fallback is unavailable.
        """
        from pydub import AudioSegment
        import tempfile

        audio = AudioSegment.from_file(str(audio_path))
        segment = audio[int(start * 1000):int(end * 1000)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            segment.export(f.name, format="wav")
            temp_path = f.name

        try:
            segments, _ = self.model.transcribe(
                temp_path,
                language=self.language,
                vad_filter=False,
                word_timestamps=False
            )
            texts = [s.text.strip() for s in segments]
            return " ".join(texts)
        finally:
            Path(temp_path).unlink(missing_ok=True)
