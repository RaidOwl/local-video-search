import subprocess
import tempfile
import os
from typing import List


class TranscriptSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text.strip()


class Transcriber:
    """Transcribe audio from video files using faster-whisper."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"Loading Whisper model '{self._model_size}'...")
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type
            )
            print("Whisper model loaded.")
        return self._model

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio track from video to a WAV file."""
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
                audio_path
            ],
            capture_output=True,
            timeout=300
        )
        return result.returncode == 0

    def transcribe(self, video_path: str) -> List[TranscriptSegment]:
        """Transcribe a video file and return timestamped segments."""
        model = self._load_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            if not self.extract_audio(video_path, audio_path):
                print(f"Failed to extract audio from {video_path}")
                return []

            segments, _ = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}
            )

            return [
                TranscriptSegment(seg.start, seg.end, seg.text)
                for seg in segments
                if seg.text.strip()
            ]
        finally:
            os.unlink(audio_path)


class TranscriptionService:
    """Service wrapper for Transcriber."""

    def __init__(self, model_size: str = "base"):
        self.transcriber = Transcriber(model_size=model_size)

    def transcribe_video(self, video_path: str) -> List[TranscriptSegment]:
        return self.transcriber.transcribe(video_path)
