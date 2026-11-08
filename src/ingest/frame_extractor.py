import subprocess
import os
import re
from pathlib import Path
from threading import Lock
from typing import Callable, List, Optional, Tuple
import asyncio


# ── Hardware acceleration backends ──────────────────────────────────────────
# Each tuple is (name, display_label, description).
HWACCEL_OPTIONS: list[tuple[str, str, str]] = [
    ("off",    "Off (CPU only)",         "Standard software decode — works everywhere."),
    ("auto",   "Auto-detect",            "Use an available CUDA or VAAPI device, otherwise software decode."),
    ("cuda",   "NVIDIA CUDA / NVDEC",    "NVIDIA GPUs — uses cuvid decoders. Fastest on DGX Spark / GB10."),
    ("vaapi",  "VAAPI (AMD / Intel)",    "Linux VAAPI — for AMD GPUs (Radeon, RDNA, ROCm)."),
    ("drm",    "DRM (Direct Rendering)", "Linux DRM — lightweight kernel-level decode."),
    ("vulkan", "Vulkan",                 "Cross-platform GPU decode via Vulkan video."),
]


def _vaapi_devices() -> list[str]:
    try:
        return [
            os.path.join("/dev/dri", name)
            for name in sorted(os.listdir("/dev/dri"))
            if name.startswith("renderD")
        ]
    except FileNotFoundError:
        return []
    except OSError as e:
        print(f"[FFmpeg] Could not inspect VAAPI devices: {e}", flush=True)
        return []


def _device_usable(device: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-init_hw_device", device,
                "-f", "lavfi", "-i", "color=c=black:s=16x16:r=1",
                "-frames:v", "1", "-f", "null", "-",
            ],
            capture_output=True, text=True, errors="replace", timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[FFmpeg] Could not probe {device}: {e}", flush=True)
        return False
    if result.returncode:
        print(f"[FFmpeg] {device} unavailable: {result.stderr.strip()}", flush=True)
        return False
    return True


def _probe_hwaccel() -> tuple[str, Optional[str]]:
    """Find a usable CUDA or VAAPI device in the API's own environment."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, errors="replace", timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[FFmpeg] Could not list hardware accelerators: {e}", flush=True)
        return "off", None
    if result.returncode:
        print(f"[FFmpeg] Could not list hardware accelerators: {result.stderr.strip()}", flush=True)
        return "off", None

    available = {line.strip() for line in result.stdout.splitlines()
                 if line.strip() and not line.startswith("Hardware")}
    has_cuda_device = any(os.path.exists(path) for path in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/dxg"))
    if "cuda" in available and has_cuda_device and _device_usable("cuda=lvs:0"):
        return "cuda", None
    if "vaapi" in available:
        for device in _vaapi_devices():
            if _device_usable(f"vaapi=lvs:{device}"):
                return "vaapi", device
    return "off", None


_HWACCEL_CACHE: Optional[tuple[str, Optional[str]]] = None
_HWACCEL_LOCK = Lock()


def _auto_hwaccel() -> tuple[str, Optional[str]]:
    global _HWACCEL_CACHE
    if _HWACCEL_CACHE is None:
        with _HWACCEL_LOCK:
            if _HWACCEL_CACHE is None:
                _HWACCEL_CACHE = _probe_hwaccel()
    return _HWACCEL_CACHE


def _resolve_hwaccel(hwaccel: str) -> str:
    """Resolve ``auto`` to a concrete backend."""
    return _auto_hwaccel()[0] if hwaccel == "auto" else hwaccel


def _hwaccel_decode_args(hwaccel: str) -> list[str]:
    """Return the FFmpeg flags needed for GPU-accelerated decode.

    Returns an empty list when acceleration is off/unavailable.
    """
    resolved = _resolve_hwaccel(hwaccel)
    if resolved in ("off", "vdpau"):
        return []

    if resolved == "cuda":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

    if resolved == "vaapi":
        devices = _vaapi_devices() if hwaccel != "auto" else []
        render = _auto_hwaccel()[1] if hwaccel == "auto" else next(iter(devices), None)
        args = ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]
        if render:
            args.extend(["-vaapi_device", render])
        return args

    # drm, vulkan — straightforward hwaccel names
    return ["-hwaccel", resolved, "-hwaccel_output_format", resolved]


def _hwaccel_frame_filter(hwaccel: str, inner_filter: str) -> str:
    """Wrap a filter string with hwdownload when GPU hardware frames are in use.

    The ``select='gt(scene,...)'`` and ``scale`` filters operate on CPU memory,
    so GPU-decoded frames must be transferred back to CPU first. This returns
    the appropriate filter chain or the raw ``inner_filter`` when on CPU.
    """
    resolved = _resolve_hwaccel(hwaccel)
    if resolved in ("off", "vdpau"):
        return inner_filter

    if resolved == "cuda":
        # cuda -> hwdownload -> format(nv12) -> cpu filter
        return f"hwdownload,format=nv12,{inner_filter}"

    if resolved == "vaapi":
        # vaapi frames are in vaapi pixel formats — download via hwdownload
        # then feed into the CPU filter.
        return f"hwdownload,format=nv12,{inner_filter}"

    # drm, vulkan — same pattern
    return f"hwdownload,format=nv12,{inner_filter}"


class FrameExtractor:
    """Extract frames from videos using FFmpeg."""

    def __init__(self, interval: int = 5, scene_threshold: float = 0.4, min_gap: float = 2.0, threads: int = 0, hwaccel: str = "off"):
        """
        Args:
            interval: Seconds between extracted grid frames (the coverage floor)
            scene_threshold: FFmpeg scene score (0-1) above which a cut is a keyframe.
                Lower = more cuts detected. 0 disables scene detection.
            min_gap: Minimum seconds between any two extracted frames, used to thin
                out near-duplicate grid/scene timestamps and bound captioning cost.
            threads: Max CPU threads FFmpeg may use per decode. 0 lets FFmpeg use all
                cores; a positive cap keeps the host responsive during analysis.
            hwaccel: Hardware acceleration backend. One of: off, auto, cuda, vaapi,
                drm, vulkan. ``auto`` probes and picks the best available backend.
        """
        self.interval = interval
        self.scene_threshold = scene_threshold
        self.min_gap = min_gap
        self.threads = max(0, int(threads))
        self.hwaccel = hwaccel
        self._use_software_decode = False
        self.hwaccel_warning: Optional[str] = None

    def _threads_args(self) -> List[str]:
        return ['-threads', str(self.threads)] if self.threads > 0 else []

    def resolved_hwaccel(self) -> str:
        return "off" if self._use_software_decode else _resolve_hwaccel(self.hwaccel)

    def _run_decode(
        self, command: Callable[[str], list[str]], *, timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess[str]:
        mode = "off" if self._use_software_decode else self.hwaccel
        resolved = _resolve_hwaccel(mode)
        result = subprocess.run(command(mode), capture_output=True, text=True, errors="replace", timeout=timeout)
        if result.returncode and resolved not in ("off", "vdpau"):
            print(
                f"[FFmpeg] {resolved} decoding failed; retrying in software: "
                f"{result.stderr.strip()[-1000:]}",
                flush=True,
            )
            result = subprocess.run(command("off"), capture_output=True, text=True, errors="replace", timeout=timeout)
            if result.returncode == 0:
                self._use_software_decode = True
                self.hwaccel_warning = (
                    f"{resolved.upper()} decoding failed; using software decoding. See API logs for details."
                )
            else:
                self.hwaccel_warning = (
                    f"{resolved.upper()} decoding and its software fallback failed. See API logs for details."
                )
        if result.returncode:
            print(f"[FFmpeg] Decoding failed: {result.stderr.strip()[-1000:]}", flush=True)
        return result

    def _frame_command(
        self, video_path: str | Path, timestamp: float, output_path: str | Path, mode: str,
    ) -> list[str]:
        return [
            'ffmpeg', '-y',
            *self._threads_args(),
            *_hwaccel_decode_args(mode),
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-q:v', '2',
            '-vf', _hwaccel_frame_filter(mode, 'scale=448:-1'),
            str(output_path),
        ]

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        prefix: str = "frame"
    ) -> List[Tuple[str, float]]:
        """
        Extract frames from a video.

        Samples on a fixed time grid (the coverage floor) AND at scene cuts, so a
        subject that's only on screen briefly between grid points still gets a
        frame. Returns ``(frame_path, timestamp)`` pairs; the timestamp is the
        real seek time (NOT index*interval), since scene frames are irregular.

        Returns:
            List of ``(frame_path, timestamp_seconds)`` tuples, ordered by time.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video duration
        duration = self._get_duration(video_path)
        if duration <= 0:
            return []

        timestamps = self._build_timestamps(video_path, duration)

        frames: List[Tuple[str, float]] = []
        for i, timestamp in enumerate(timestamps):
            output_path = output_dir / f"{prefix}_{i:04d}.jpg"

            result = self._run_decode(
                lambda mode: self._frame_command(video_path, timestamp, output_path, mode)
            )
            if result.returncode == 0 and output_path.exists():
                frames.append((str(output_path), round(timestamp, 2)))

        return frames

    def _build_timestamps(self, video_path: Path, duration: float) -> List[float]:
        """Merge the regular grid with detected scene cuts, then thin by min_gap.

        The grid guarantees we never sample more sparsely than ``interval`` (so this
        is never worse than the old uniform sampling); scene cuts add frames at shot
        boundaries where new subjects/text typically appear. Timestamps closer than
        ``min_gap`` are collapsed (keeping the earlier one) to avoid near-duplicate
        captions and bound cost.
        """
        grid = []
        t = 0.0
        while t < duration:
            grid.append(t)
            t += self.interval

        cuts = self._detect_scene_cuts(video_path, duration) if self.scene_threshold > 0 else []
        merged = sorted(set(grid) | {c for c in cuts if 0.0 < c < duration})

        thinned: List[float] = []
        last = -1e9
        for ts in merged:
            if ts - last >= self.min_gap:
                thinned.append(ts)
                last = ts
        return thinned

    def _detect_scene_cuts(self, video_path: Path, duration: float) -> List[float]:
        """Return timestamps (s) of scene cuts via a single FFmpeg analysis pass.

        When ``hwaccel`` is enabled, GPU-accelerated decode is used to speed up
        the full-stream decode, with frames transferred back to CPU for the scene
        detection filter. Degrades gracefully: any failure/timeout returns an
        empty list, leaving the regular grid as the sole source of frames.
        """
        def command(mode: str) -> list[str]:
            return [
                'ffmpeg', '-hide_banner',
                *self._threads_args(),
                *_hwaccel_decode_args(mode),
                '-i', str(video_path),
                '-vf', _hwaccel_frame_filter(mode, f"select='gt(scene,{self.scene_threshold})',showinfo"),
                '-an', '-f', 'null', '-',
            ]
        try:
            result = self._run_decode(command, timeout=max(120.0, duration * 3.0))
        except (OSError, subprocess.SubprocessError) as e:
            print(f"[FFmpeg] Scene detection failed: {e}", flush=True)
            return []

        cuts: List[float] = []
        for line in result.stderr.splitlines():
            m = re.search(r'pts_time:([0-9.]+)', line)
            if m:
                try:
                    cuts.append(float(m.group(1)))
                except ValueError:
                    pass
        return cuts
    
    def extract_frame_at(
        self,
        video_path: str,
        timestamp: float,
        output_path: str
    ) -> Optional[str]:
        """Extract a single frame at a specific timestamp."""
        result = self._run_decode(
            lambda mode: self._frame_command(video_path, timestamp, output_path, mode)
        )
        if result.returncode == 0 and Path(output_path).exists():
            return output_path
        return None
    
    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=30)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return float(data.get('format', {}).get('duration', 0))
        except Exception:
            pass
        return 0.0


async def extract_frames_async(
    video_path: str,
    output_dir: str,
    interval: int = 5,
    hwaccel: str = "off",
) -> List[Tuple[str, float]]:
    """Async wrapper for frame extraction."""
    loop = asyncio.get_event_loop()
    extractor = FrameExtractor(interval=interval, hwaccel=hwaccel)
    return await loop.run_in_executor(
        None,
        extractor.extract_frames,
        video_path,
        output_dir
    )