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

    # Sentence-window chunking targets (in words). Whisper's VAD only breaks on
    # ~½-second pauses, so in continuous narration it emits 40-60 word run-ons.
    # A single topic mention buried in such a chunk dilutes to a weak reranker
    # score. We instead re-chunk from the word stream into ~sentence units so
    # each indexed point is focused (precise reranking) and carries accurate
    # per-chunk start/end timestamps (accurate seek). MIN avoids tiny fragments;
    # MAX force-breaks a run-on sentence that never punctuates.
    MIN_CHUNK_WORDS = 6
    MAX_CHUNK_WORDS = 30

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8",
                 beam_size: int = 1, condition_on_previous_text: bool = False):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._condition_on_previous_text = condition_on_previous_text
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

    def transcribe(self, video_path: str, progress_callback=None) -> List[TranscriptSegment]:
        """Transcribe a video file and return timestamped segments.

        If ``progress_callback`` is provided it is called as transcription
        advances with ``(seconds_done, total_seconds)`` so callers can report
        progress to the user.
        """
        model = self._load_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            if not self.extract_audio(video_path, audio_path):
                print(f"Failed to extract audio from {video_path}")
                return []

            segments, info = model.transcribe(
                audio_path,
                beam_size=self._beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=True,
                condition_on_previous_text=self._condition_on_previous_text,
            )

            total = getattr(info, "duration", 0.0) or 0.0
            words: List[tuple] = []        # (start, end, text) per word
            raw: List[TranscriptSegment] = []  # fallback if no word timestamps
            # faster-whisper yields segments lazily, so iterating drives the work
            # and lets us report progress from each segment's end timestamp.
            for seg in segments:
                if seg.text.strip():
                    raw.append(TranscriptSegment(seg.start, seg.end, seg.text))
                for w in (getattr(seg, "words", None) or []):
                    if w.word and w.word.strip():
                        words.append((w.start, w.end, w.word))
                if progress_callback and total > 0:
                    try:
                        progress_callback(min(seg.end, total), total)
                    except Exception:
                        pass
            if progress_callback and total > 0:
                try:
                    progress_callback(total, total)
                except Exception:
                    pass
            # Re-chunk from words into sentence units; fall back to Whisper's own
            # segments only if word timestamps were unavailable.
            return self._chunk_words(words) if words else raw
        finally:
            os.unlink(audio_path)

    def _chunk_words(self, words: List[tuple]) -> List[TranscriptSegment]:
        """Group word-level timestamps into ~sentence chunks.

        Closes a chunk at sentence-ending punctuation once it has at least
        MIN_CHUNK_WORDS words, or force-closes at MAX_CHUNK_WORDS so a run-on
        sentence can't grow unbounded. Each chunk's start/end come from its
        first/last word, so timestamps stay accurate.
        """
        chunks: List[TranscriptSegment] = []
        buf: List[tuple] = []
        for start, end, txt in words:
            buf.append((start, end, txt))
            ends_sentence = txt.strip().endswith(('.', '!', '?', '…'))
            if (ends_sentence and len(buf) >= self.MIN_CHUNK_WORDS) or len(buf) >= self.MAX_CHUNK_WORDS:
                chunks.append(self._mk_chunk(buf))
                buf = []
        if buf:
            # Fold a sub-MIN trailing remainder ("...see you in the next" + "one")
            # back into the previous chunk instead of emitting a meaningless tiny
            # fragment that pollutes search. Only stands alone if it's the only chunk.
            if len(buf) < self.MIN_CHUNK_WORDS and chunks:
                prev = chunks[-1]
                tail = self._mk_chunk(buf)
                chunks[-1] = TranscriptSegment(
                    prev.start, tail.end, (prev.text + " " + tail.text).strip()
                )
            else:
                chunks.append(self._mk_chunk(buf))
        return chunks

    @staticmethod
    def _mk_chunk(buf: List[tuple]) -> TranscriptSegment:
        text = "".join(t for _, _, t in buf).strip()
        return TranscriptSegment(buf[0][0], buf[-1][1], text)


class TranscriptionService:
    """Service wrapper for Transcriber."""

    def __init__(self, model_size: str = "base", beam_size: int = 1,
                 condition_on_previous_text: bool = False):
        self.transcriber = Transcriber(
            model_size=model_size,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
        )

    def transcribe_video(self, video_path: str, progress_callback=None) -> List[TranscriptSegment]:
        return self.transcriber.transcribe(video_path, progress_callback=progress_callback)
