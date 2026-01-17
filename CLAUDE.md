# Podcast Transcription System

## Project Overview

CLI tool for transcribing Russian podcasts with speaker diarization and tracking.

## Documentation

- **Implementation Plan**: See [.claude/podcast-transcription-plan.md](.claude/podcast-transcription-plan.md) for the full technical plan

## Setup

1. Create virtual environment:
   ```bash
   pyenv local venv  # or: python -m venv .venv && source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy config and add API keys:
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your keys
   ```

4. Accept HuggingFace model terms (required for pyannote):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/embedding

## Quick Reference

### Tech Stack
- `faster-whisper` - Local transcription (Whisper large-v3)
- `openai` API - Fallback for low-confidence segments
- `pyannote.audio` - Speaker diarization (v4.x)
- `click` - CLI framework

### Key Commands
```bash
# Process episode (interactive speaker matching)
python -m src.cli process mp3/<file>.mp3

# Process first 5 minutes (testing)
python -m src.cli process mp3/<file>.mp3 --duration 5m

# Auto mode (skip speaker prompts)
python -m src.cli process mp3/<file>.mp3 --auto

# Use OpenAI for all transcription
python -m src.cli process mp3/<file>.mp3 --openai-all

# Speaker management
python -m src.cli speakers list
python -m src.cli speakers rename <id> <name>
python -m src.cli speakers show <id>

# Statistics
python -m src.cli stats
python -m src.cli stats --speaker <id>
```

### Configuration (config.yaml)
```yaml
openai:
  api_key: "sk-..."        # OpenAI API key
huggingface:
  token: "hf_..."          # HuggingFace token
transcription:
  whisper_model: "large-v3"
  compute_type: "int8"     # Use int8 for Apple Silicon
```

### Project Structure
```
src/
├── transcriber.py       # Local Whisper
├── openai_transcriber.py # OpenAI fallback
├── diarizer.py          # Speaker diarization
├── speaker_tracker.py   # Speaker matching
├── pipeline.py          # Main orchestration
└── cli.py               # CLI entry point
data/
├── speakers.json        # Speaker profiles
└── episodes/            # Transcripts
```

## Notes

- First run downloads ~3GB of models
- Processing: ~10-15 min for 1-hour podcast on M1/M2
- OpenAI costs: ~$0.04-0.07 per episode (hybrid mode)
