# Podcast Transcription & Speaker Diarization System

## Overview
Python CLI tool to transcribe Russian podcasts (with English terms) and track speakers across episodes.

**Target**: Apple Silicon Mac | **Output**: JSON files | **Speaker ID**: Semi-automatic

---

## Tech Stack

| Component | Library | Reason |
|-----------|---------|--------|
| Transcription (primary) | `faster-whisper` | Optimized Whisper, MPS support via PyTorch |
| Transcription (fallback) | `openai` API | Better accuracy for low-confidence segments, better code-switching |
| Diarization | `pyannote.audio` | SOTA open-source, language-agnostic |
| Speaker embeddings | `pyannote/embedding` | Extract voice profiles for matching |
| CLI | `click` | Clean command structure |
| Audio processing | `pydub` | MP3 handling |

---

## Project Structure

```
podcast-stats/
├── src/
│   ├── __init__.py
│   ├── transcriber.py       # Whisper wrapper (local)
│   ├── openai_transcriber.py # OpenAI API fallback
│   ├── diarizer.py          # pyannote wrapper
│   ├── speaker_tracker.py   # Embedding matching
│   ├── pipeline.py          # Orchestrates processing
│   └── cli.py               # CLI entry point
├── data/
│   ├── speakers.json        # Known speaker profiles
│   └── episodes/            # Per-episode transcripts
├── mp3/                     # Input files
├── config.yaml              # API keys (in .gitignore)
├── config.example.yaml      # Template for config
├── .gitignore
├── requirements.txt
└── pyproject.toml
```

---

## Data Schemas

### speakers.json
```json
{
  "speakers": [
    {
      "id": "spk_001",
      "name": "Иван Иванов",
      "embedding": [0.123, -0.456, ...],
      "episodes": ["ep_rt_podcast983"],
      "total_speaking_time_sec": 1234.5
    }
  ]
}
```

### episodes/{episode_id}.json
```json
{
  "episode_id": "ep_rt_podcast983",
  "source_file": "rt_podcast983.mp3",
  "processed_at": "2024-01-17T10:30:00",
  "duration_sec": 3600,
  "speakers": ["spk_001", "spk_002"],
  "transcription_stats": {
    "total_segments": 150,
    "openai_refined_segments": 23,
    "openai_cost_usd": 0.05
  },
  "segments": [
    {
      "start": 0.0,
      "end": 5.5,
      "speaker_id": "spk_001",
      "text": "Привет всем, это Radio-T...",
      "confidence": 0.92,
      "refined_by_openai": false
    },
    {
      "start": 5.5,
      "end": 12.3,
      "speaker_id": "spk_002",
      "text": "Да, сегодня обсудим Kubernetes и Docker...",
      "confidence": 0.65,
      "refined_by_openai": true
    }
  ]
}
```

---

## Configuration

### config.yaml (in .gitignore - contains secrets)
```yaml
openai:
  api_key: "sk-..."       # Your OpenAI API key
  model: "whisper-1"

huggingface:
  token: "hf_..."         # Your Hugging Face token for pyannote

transcription:
  confidence_threshold: 0.7
  openai_for_code_switching: true

speaker_matching:
  similarity_threshold: 0.75
  high_confidence_threshold: 0.85
```

### config.example.yaml (tracked in git - template)
```yaml
openai:
  api_key: ""             # Get from https://platform.openai.com/api-keys
  model: "whisper-1"

huggingface:
  token: ""               # Get from https://huggingface.co/settings/tokens

transcription:
  confidence_threshold: 0.7
  openai_for_code_switching: true

speaker_matching:
  similarity_threshold: 0.75
  high_confidence_threshold: 0.85
```

### .gitignore
```
config.yaml
data/
mp3/
__pycache__/
*.pyc
.venv/
```

---

## OpenAI API Fallback Strategy

**Problem**: Local Whisper struggles with:
1. Low-confidence segments (unclear speech, overlapping)
2. Code-switching (English terms in Russian speech like "API", "Docker", "deployment")

**Solution**: Hybrid approach using OpenAI API for difficult segments

### When to Use OpenAI API:
1. **Low confidence score** - Whisper provides `no_speech_prob` and `avg_logprob` per segment
   - If `avg_logprob < -1.0` or `no_speech_prob > 0.5` → send to OpenAI
2. **Detected code-switching** - Heuristic: segment contains mix of Cyrillic and Latin characters
3. **Manual flag** - CLI option `--openai-all` to process entire file via OpenAI

### Implementation Flow:
```
Audio Segment
     │
     ▼
┌─────────────────┐
│ Local Whisper   │
│ (faster-whisper)│
└────────┬────────┘
         │
    ┌────┴────┐
    │ Check   │
    │confidence│
    └────┬────┘
         │
   Low confidence OR      High confidence
   code-switching?        ─────────────────► Use result
         │
         ▼
┌─────────────────┐
│ OpenAI API      │
│ (whisper-1)     │
└────────┬────────┘
         │
         ▼
    Use OpenAI result
    (mark as "refined")
```

### Cost Estimation:
- OpenAI Whisper API: $0.006 per minute of audio
- 1-hour podcast: ~$0.36 if fully processed via API
- With hybrid approach: typically 10-20% of segments need fallback → ~$0.04-0.07 per episode

---

## CLI Commands

```bash
# Process new episode (full)
python -m src.cli process mp3/rt_podcast983.mp3

# Process only first 5 minutes (for testing)
python -m src.cli process mp3/rt_podcast983.mp3 --duration 5m

# Process with full OpenAI (no local Whisper)
python -m src.cli process mp3/rt_podcast983.mp3 --openai-all

# List known speakers
python -m src.cli speakers list

# Rename speaker
python -m src.cli speakers rename spk_001 "Иван Иванов"

# Show statistics
python -m src.cli stats

# Show specific speaker stats
python -m src.cli stats --speaker spk_001
```

### Duration Flag
The `--duration` flag accepts formats like:
- `5m` - 5 minutes
- `30s` - 30 seconds
- `1h` - 1 hour
- `90` - 90 seconds (default unit)

Useful for:
- Testing pipeline before full processing
- Quick validation of transcription quality
- Debugging speaker diarization

---

## Semi-Automatic Speaker Workflow

1. **Process episode** → diarization produces SPEAKER_00, SPEAKER_01, etc.
2. **Extract embeddings** for each detected speaker
3. **Compare** against known speakers (cosine similarity)
4. **Interactive prompt** for matches:
   ```
   Detected speaker SPEAKER_00 (spoke 15:30)
   Possible match: "Иван Иванов" (similarity: 0.82)
   [1] Confirm match
   [2] Different existing speaker
   [3] New speaker (enter name)
   >
   ```
5. **Update** speakers.json and episode transcript

---

## Implementation Steps

### Phase 0: Environment Setup
1. Initialize git repo and connect to `git@github.com:ufian/podcast-stats.git`
2. Create `.gitignore` with secrets and data exclusions
3. Create Python virtual environment with pyenv
4. Install base dependencies
5. Delete redundant `PLAN.md` from project root

### Phase 1: Core Infrastructure
4. Create project structure, `pyproject.toml`, and `config.yaml`
5. Implement audio utilities (MP3 loading, duration limiting, segment extraction)
6. Implement `transcriber.py` - Local Whisper wrapper with confidence metrics
7. Implement `openai_transcriber.py` - OpenAI API wrapper for fallback
8. Implement `diarizer.py` - pyannote pipeline wrapper

### Phase 2: Basic Pipeline (Test with 5 min)
9. Implement basic `pipeline.py` - transcription + diarization
10. Implement basic `cli.py` with `process` command and `--duration` flag
11. **TEST**: Process first 5 minutes of sample podcast
12. Verify transcription output and speaker segments

### Phase 3: Hybrid Transcription
13. Add confidence scoring and code-switching detection
14. Implement segment-level fallback logic (local → OpenAI)
15. **TEST**: Verify OpenAI fallback works for low-confidence segments

### Phase 4: Speaker Tracking
16. Implement `speaker_tracker.py` - embedding extraction and matching
17. Create JSON storage utilities for speakers and episodes
18. Add interactive speaker confirmation prompts
19. **TEST**: Process second 5-minute segment, verify speaker matching

### Phase 5: Full Pipeline & Statistics
20. Add `--openai-all` flag for full OpenAI processing
21. Add statistics calculation (speaking time, episode count)
22. Add stats CLI command with filtering
23. **TEST**: Process full podcast episode

---

## Key Implementation Details

### Local Whisper with Confidence Metrics
```python
model = WhisperModel("large-v3", device="auto", compute_type="float16")
segments, info = model.transcribe(
    audio_path,
    language="ru",
    vad_filter=True,
    word_timestamps=True
)

# Check confidence per segment
for segment in segments:
    needs_fallback = (
        segment.avg_logprob < -1.0 or
        segment.no_speech_prob > 0.5 or
        has_code_switching(segment.text)  # Mix of Cyrillic/Latin
    )
    if needs_fallback:
        # Extract audio slice and send to OpenAI
        refined_text = openai_transcribe(audio_path, segment.start, segment.end)
        segment.text = refined_text
        segment.refined = True
```

### OpenAI API Fallback
```python
from openai import OpenAI

def openai_transcribe(audio_path: str, start: float, end: float) -> str:
    client = OpenAI()

    # Extract segment audio
    segment_audio = extract_audio_segment(audio_path, start, end)

    # Transcribe via API
    with open(segment_audio, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ru",
            prompt="Подкаст на русском языке с техническими терминами на английском"
        )
    return transcript.text
```

### Code-Switching Detection
```python
import re

def has_code_switching(text: str) -> bool:
    """Detect if text contains mix of Cyrillic and Latin characters."""
    has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', text))
    has_latin = bool(re.search(r'[a-zA-Z]', text))
    return has_cyrillic and has_latin
```

### pyannote Configuration
```python
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=config["huggingface"]["token"]
)
diarization = pipeline(audio_path)
```

### Speaker Matching Threshold
- Cosine similarity > 0.75 → suggest as match
- Cosine similarity > 0.85 → high confidence match

---

## Dependencies

```
faster-whisper>=1.0.0
pyannote.audio>=3.1
openai>=1.0.0           # OpenAI API for fallback transcription
torch>=2.0
pydub>=0.25
click>=8.0
numpy>=1.24
pyyaml>=6.0             # Config file parsing
```

---

## Verification Plan

1. **Unit tests**: Test each component in isolation
2. **Integration test**: Process sample MP3 end-to-end
3. **Manual verification**:
   - Run `python -m src.cli process mp3/rt_podcast983.mp3`
   - Verify JSON output in `data/episodes/`
   - Check speaker matching prompts work
   - Run `python -m src.cli stats` and verify output

---

## Notes

- **Setup**: Copy `config.example.yaml` to `config.yaml` and fill in your API keys
- **Hugging Face token**: Get from https://huggingface.co/settings/tokens (accept pyannote model terms)
- **OpenAI API key**: Get from https://platform.openai.com/api-keys
- **First run**: Will download ~3GB of models (Whisper large-v3 + pyannote)
- **Processing time**: ~10-15 min for 1-hour podcast on M1/M2
- **OpenAI costs**: ~$0.04-0.07 per episode with hybrid approach (10-20% fallback)
