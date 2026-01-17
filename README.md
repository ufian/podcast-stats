# Podcast Transcription System

CLI tool for transcribing Russian podcasts with automatic speaker diarization and tracking across episodes.

## Features

- **Hybrid transcription**: Local Whisper (faster-whisper) with OpenAI API fallback for low-confidence segments
- **Speaker diarization**: Automatic speaker detection using pyannote.audio
- **Speaker tracking**: Match speakers across episodes using voice embeddings
- **Code-switching support**: Handles Russian speech with English technical terms
- **Cost-efficient**: Only uses OpenAI API for difficult segments (~$0.04-0.07 per hour)

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2) or CUDA-capable GPU
- ~3GB disk space for models

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:ufian/podcast-stats.git
   cd podcast-stats
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys:
   ```bash
   cp config.example.yaml config.yaml
   ```

   Edit `config.yaml` and add your keys:
   - `openai.api_key`: Get from https://platform.openai.com/api-keys
   - `huggingface.token`: Get from https://huggingface.co/settings/tokens

5. Accept HuggingFace model terms (required):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding)

## Usage

### Process a podcast episode

```bash
# Full episode with interactive speaker matching
python -m src.cli process mp3/episode.mp3

# First 5 minutes only (for testing)
python -m src.cli process mp3/episode.mp3 --duration 5m

# Auto mode (skip speaker prompts)
python -m src.cli process mp3/episode.mp3 --auto

# Use OpenAI for all transcription (higher quality, higher cost)
python -m src.cli process mp3/episode.mp3 --openai-all
```

### Manage speakers

```bash
# List all known speakers
python -m src.cli speakers list

# Rename a speaker
python -m src.cli speakers rename SPEAKER_00 "John Doe"

# Show speaker details
python -m src.cli speakers show spk_abc123
```

### View statistics

```bash
# Overall statistics
python -m src.cli stats

# Statistics for specific speaker
python -m src.cli stats --speaker spk_abc123

# Output as JSON
python -m src.cli stats --json
```

## Output Format

Transcripts are saved to `data/episodes/{episode_id}.json`:

```json
{
  "episode_id": "ep_podcast123",
  "source_file": "podcast123.mp3",
  "duration_sec": 3600,
  "speakers": ["spk_001", "spk_002"],
  "segments": [
    {
      "start": 0.0,
      "end": 5.5,
      "speaker_id": "spk_001",
      "text": "Hello everyone...",
      "confidence": 0.92,
      "refined_by_openai": false
    }
  ]
}
```

## Configuration

See `config.yaml` for all options:

| Option | Description | Default |
|--------|-------------|---------|
| `transcription.whisper_model` | Whisper model size | `large-v3` |
| `transcription.compute_type` | Compute precision | `int8` |
| `transcription.language` | Transcription language | `ru` |
| `transcription.openai_for_code_switching` | Use OpenAI for mixed language | `true` |
| `speaker_matching.similarity_threshold` | Min similarity for match | `0.75` |

## Performance

- Processing time: ~10-15 minutes per hour of audio (on M1/M2)
- First run downloads ~3GB of models
- OpenAI costs: ~$0.04-0.07 per hour with hybrid approach

## License

MIT
