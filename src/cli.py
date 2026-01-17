"""CLI interface for podcast transcription system."""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

# Suppress PyTorch warnings about std() on small tensors
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
# Suppress lightning checkpoint warnings
warnings.filterwarnings("ignore", message=".*ModelCheckpoint.*")
warnings.filterwarnings("ignore", message=".*automatically upgraded.*")
warnings.filterwarnings("ignore", message=".*Found keys that are not in the model.*")

import click

from .pipeline import Pipeline, PipelineConfig, ProcessingResult, parse_duration
from .speaker_tracker import SpeakerMatch, SpeakerTracker


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def create_speaker_callback() -> Callable:
    """Create an interactive speaker identification callback."""

    def callback(
        detected_speaker: str,
        matches: list[SpeakerMatch],
        speaking_time: float
    ) -> tuple[str, str]:
        """Interactive prompt for speaker identification."""
        click.echo()
        click.echo(click.style(f"Detected speaker: {detected_speaker}", bold=True))
        click.echo(f"  Speaking time: {format_duration(speaking_time)}")

        if matches:
            click.echo()
            click.echo("Possible matches:")
            for i, match in enumerate(matches[:3], 1):
                confidence = "HIGH" if match.is_high_confidence else "medium"
                click.echo(
                    f"  [{i}] {match.known_speaker.name} "
                    f"(similarity: {match.similarity:.2f}, {confidence})"
                )

            click.echo(f"  [{len(matches[:3]) + 1}] Different existing speaker")
            click.echo(f"  [{len(matches[:3]) + 2}] New speaker")

            choice = click.prompt(
                "Select option",
                type=int,
                default=1 if matches[0].is_high_confidence else len(matches[:3]) + 2
            )

            if 1 <= choice <= len(matches[:3]):
                selected = matches[choice - 1]
                return (selected.known_speaker.id, selected.known_speaker.name)
            elif choice == len(matches[:3]) + 1:
                # Different existing speaker - list all
                return _select_existing_speaker()
            else:
                # New speaker
                return _create_new_speaker()
        else:
            click.echo("No matching speakers found.")
            click.echo("  [1] Select existing speaker")
            click.echo("  [2] New speaker")

            choice = click.prompt("Select option", type=int, default=2)

            if choice == 1:
                return _select_existing_speaker()
            else:
                return _create_new_speaker()

    return callback


def _select_existing_speaker() -> tuple[str, str]:
    """Prompt user to select from existing speakers."""
    tracker = SpeakerTracker()
    speakers = tracker.list_speakers()

    if not speakers:
        click.echo("No existing speakers. Creating new speaker.")
        return _create_new_speaker()

    click.echo()
    click.echo("Existing speakers:")
    for i, speaker in enumerate(speakers, 1):
        episodes = len(speaker.episodes)
        time_str = format_duration(speaker.total_speaking_time_sec)
        click.echo(f"  [{i}] {speaker.name} ({episodes} episodes, {time_str})")

    choice = click.prompt("Select speaker", type=int)
    if 1 <= choice <= len(speakers):
        selected = speakers[choice - 1]
        return (selected.id, selected.name)
    else:
        return _create_new_speaker()


def _create_new_speaker() -> tuple[str, str]:
    """Prompt user to create a new speaker."""
    name = click.prompt("Enter speaker name")
    return (f"new:{name}", name)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Podcast transcription and speaker diarization tool."""
    pass


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--duration", "-d",
    help="Limit processing duration (e.g., '5m', '30s', '1h')"
)
@click.option(
    "--openai-all",
    is_flag=True,
    help="Use OpenAI API for all transcription"
)
@click.option(
    "--auto",
    is_flag=True,
    help="Auto-mode: skip interactive speaker prompts"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for transcripts"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default="config.yaml",
    help="Path to config file"
)
@click.option(
    "--num-speakers", "-n",
    type=int,
    help="Exact number of speakers (speeds up diarization)"
)
@click.option(
    "--min-speakers",
    type=int,
    help="Minimum number of speakers"
)
@click.option(
    "--max-speakers",
    type=int,
    help="Maximum number of speakers"
)
def process(audio_file: str, duration: str | None, openai_all: bool,
            auto: bool, output: str | None, config: str,
            num_speakers: int | None, min_speakers: int | None, max_speakers: int | None):
    """Process an audio file for transcription and speaker diarization.

    Example:
        python -m src.cli process mp3/rt_podcast983.mp3
        python -m src.cli process mp3/rt_podcast983.mp3 --duration 5m
    """
    audio_path = Path(audio_file)

    # Parse duration limit
    duration_limit = None
    if duration:
        try:
            duration_limit = parse_duration(duration)
            click.echo(f"Processing limited to {format_duration(duration_limit)}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    # Set up speaker callback
    speaker_callback = None if auto else create_speaker_callback()

    # Initialize pipeline
    pipeline_config = PipelineConfig.from_yaml(config)
    if output:
        pipeline_config.episodes_path = output

    pipeline = Pipeline(config=pipeline_config, speaker_callback=speaker_callback)

    def progress(msg: str):
        click.echo(f"  {msg}")

    click.echo(f"Processing: {audio_path.name}")
    click.echo()

    try:
        result = pipeline.process(
            audio_path,
            duration_limit=duration_limit,
            openai_all=openai_all,
            progress_callback=progress,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        # Save result
        output_path = result.save(pipeline_config.episodes_path)

        click.echo()
        click.echo(click.style("Summary:", bold=True))
        click.echo(f"  Episode ID: {result.episode_id}")
        click.echo(f"  Duration: {format_duration(result.duration_sec)}")
        click.echo(f"  Speakers: {len(result.speakers)}")
        click.echo(f"  Segments: {result.transcription_stats['total_segments']}")

        if result.transcription_stats.get('openai_refined_segments', 0) > 0:
            click.echo(
                f"  OpenAI refined: {result.transcription_stats['openai_refined_segments']} segments"
            )
            click.echo(
                f"  OpenAI cost: ${result.transcription_stats['openai_cost_usd']:.4f}"
            )

        click.echo()
        click.echo(f"Saved to: {output_path}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        raise


@cli.group()
def speakers():
    """Manage speaker profiles."""
    pass


@speakers.command("list")
def list_speakers():
    """List all known speakers."""
    tracker = SpeakerTracker()
    speakers_list = tracker.list_speakers()

    if not speakers_list:
        click.echo("No speakers found.")
        return

    click.echo(click.style("Known speakers:", bold=True))
    click.echo()

    for speaker in speakers_list:
        episodes = len(speaker.episodes)
        time_str = format_duration(speaker.total_speaking_time_sec)
        click.echo(f"  {speaker.id}: {speaker.name}")
        click.echo(f"    Episodes: {episodes}")
        click.echo(f"    Total speaking time: {time_str}")
        click.echo()


@speakers.command("rename")
@click.argument("speaker_id")
@click.argument("new_name")
def rename_speaker(speaker_id: str, new_name: str):
    """Rename a speaker.

    Example:
        python -m src.cli speakers rename spk_001 "Иван Иванов"
    """
    tracker = SpeakerTracker()
    speaker = tracker.rename_speaker(speaker_id, new_name)

    if speaker:
        click.echo(f"Renamed {speaker_id} to '{new_name}'")
    else:
        click.echo(f"Speaker not found: {speaker_id}", err=True)
        sys.exit(1)


@speakers.command("show")
@click.argument("speaker_id")
def show_speaker(speaker_id: str):
    """Show details for a specific speaker."""
    tracker = SpeakerTracker()
    speaker = tracker.get_speaker(speaker_id)

    if not speaker:
        click.echo(f"Speaker not found: {speaker_id}", err=True)
        sys.exit(1)

    click.echo(click.style(f"Speaker: {speaker.name}", bold=True))
    click.echo(f"  ID: {speaker.id}")
    click.echo(f"  Episodes: {len(speaker.episodes)}")
    click.echo(f"  Total speaking time: {format_duration(speaker.total_speaking_time_sec)}")

    if speaker.episodes:
        click.echo()
        click.echo("  Episode list:")
        for ep in speaker.episodes:
            click.echo(f"    - {ep}")


@cli.command()
@click.option("--speaker", "-s", help="Filter by speaker ID")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def stats(speaker: str | None, as_json: bool):
    """Show transcription statistics.

    Example:
        python -m src.cli stats
        python -m src.cli stats --speaker spk_001
    """
    episodes_dir = Path("data/episodes")
    speakers_file = Path("data/speakers.json")

    # Load episode data
    episodes = []
    if episodes_dir.exists():
        for ep_file in episodes_dir.glob("*.json"):
            with open(ep_file) as f:
                episodes.append(json.load(f))

    # Load speakers
    speakers_list = []
    if speakers_file.exists():
        with open(speakers_file) as f:
            data = json.load(f)
            speakers_list = data.get("speakers", [])

    if not episodes and not speakers_list:
        click.echo("No data found.")
        return

    if speaker:
        # Filter by speaker
        _show_speaker_stats(speaker, episodes, speakers_list, as_json)
    else:
        # Overall stats
        _show_overall_stats(episodes, speakers_list, as_json)


def _show_overall_stats(episodes: list, speakers_list: list, as_json: bool):
    """Show overall statistics."""
    total_duration = sum(ep.get("duration_sec", 0) for ep in episodes)
    total_segments = sum(ep.get("transcription_stats", {}).get("total_segments", 0) for ep in episodes)
    total_openai = sum(ep.get("transcription_stats", {}).get("openai_refined_segments", 0) for ep in episodes)
    total_cost = sum(ep.get("transcription_stats", {}).get("openai_cost_usd", 0) for ep in episodes)

    stats_data = {
        "total_episodes": len(episodes),
        "total_speakers": len(speakers_list),
        "total_duration_sec": total_duration,
        "total_segments": total_segments,
        "openai_refined_segments": total_openai,
        "total_openai_cost_usd": total_cost
    }

    if as_json:
        click.echo(json.dumps(stats_data, indent=2))
        return

    click.echo(click.style("Overall Statistics", bold=True))
    click.echo()
    click.echo(f"  Episodes processed: {len(episodes)}")
    click.echo(f"  Known speakers: {len(speakers_list)}")
    click.echo(f"  Total audio: {format_duration(total_duration)}")
    click.echo(f"  Total segments: {total_segments}")

    if total_openai > 0:
        click.echo(f"  OpenAI refined: {total_openai} segments")
        click.echo(f"  Total OpenAI cost: ${total_cost:.4f}")

    if episodes:
        click.echo()
        click.echo("Recent episodes:")
        for ep in sorted(episodes, key=lambda e: e.get("processed_at", ""), reverse=True)[:5]:
            click.echo(f"    {ep.get('episode_id', 'unknown')} - {ep.get('source_file', 'unknown')}")


def _show_speaker_stats(speaker_id: str, episodes: list, speakers_list: list, as_json: bool):
    """Show statistics for a specific speaker."""
    # Find speaker
    speaker = None
    for s in speakers_list:
        if s.get("id") == speaker_id:
            speaker = s
            break

    if not speaker:
        click.echo(f"Speaker not found: {speaker_id}", err=True)
        sys.exit(1)

    # Calculate per-episode stats
    episode_stats = []
    for ep in episodes:
        if speaker_id in ep.get("speakers", []):
            # Calculate speaking time in this episode
            speaking_time = sum(
                seg.get("end", 0) - seg.get("start", 0)
                for seg in ep.get("segments", [])
                if seg.get("speaker_id") == speaker_id
            )
            episode_stats.append({
                "episode_id": ep.get("episode_id"),
                "speaking_time_sec": speaking_time
            })

    stats_data = {
        "speaker_id": speaker_id,
        "name": speaker.get("name"),
        "total_episodes": len(speaker.get("episodes", [])),
        "total_speaking_time_sec": speaker.get("total_speaking_time_sec", 0),
        "episodes": episode_stats
    }

    if as_json:
        click.echo(json.dumps(stats_data, indent=2))
        return

    click.echo(click.style(f"Speaker: {speaker.get('name')}", bold=True))
    click.echo()
    click.echo(f"  ID: {speaker_id}")
    click.echo(f"  Episodes: {len(speaker.get('episodes', []))}")
    click.echo(f"  Total speaking time: {format_duration(speaker.get('total_speaking_time_sec', 0))}")

    if episode_stats:
        click.echo()
        click.echo("  Per-episode speaking time:")
        for ep_stat in episode_stats:
            click.echo(f"    {ep_stat['episode_id']}: {format_duration(ep_stat['speaking_time_sec'])}")


if __name__ == "__main__":
    cli()
