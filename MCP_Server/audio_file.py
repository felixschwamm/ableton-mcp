"""Minimal WAV inspection for rendered audio (no numpy dependency).

Live records PCM (16/24/32-bit) or 32-bit float WAV, sometimes as
WAVE_FORMAT_EXTENSIBLE. The stdlib ``wave`` module rejects float files, so
this reads the RIFF chunks directly.
"""

from __future__ import annotations

import math
import struct
import sys
from array import array
from typing import Any, Dict

_PCM = 1
_FLOAT = 3
_EXTENSIBLE = 0xFFFE


def wav_stats(path: str) -> Dict[str, Any]:
    """Return sample_rate, channels, bits, duration_s and peak_dbfs of a WAV."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")

    fmt = None
    samples = b""
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == _EXTENSIBLE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    if fmt is None:
        raise ValueError("WAV has no fmt chunk")

    tag, channels, rate, bits = fmt
    frame_bytes = channels * bits // 8
    usable = len(samples) - len(samples) % (bits // 8)
    samples = samples[:usable]

    if tag == _FLOAT and bits == 32:
        values = array("f", samples)
        full_scale = 1.0
    elif tag == _FLOAT and bits == 64:
        values = array("d", samples)
        full_scale = 1.0
    elif tag == _PCM and bits == 16:
        values = array("h", samples)
        full_scale = 32768.0
    elif tag == _PCM and bits == 24:
        # Widen each 3-byte sample to 4 bytes (value << 8) so array can read it.
        n = len(samples) // 3
        wide = bytearray(4 * n)
        wide[1::4] = samples[0::3]
        wide[2::4] = samples[1::3]
        wide[3::4] = samples[2::3]
        values = array("i", bytes(wide))
        full_scale = 2147483648.0
    elif tag == _PCM and bits == 32:
        values = array("i", samples)
        full_scale = 2147483648.0
    else:
        raise ValueError("Unsupported WAV format (tag %d, %d bit)" % (tag, bits))

    if sys.byteorder != "little":
        values.byteswap()

    peak = (max(max(values), -min(values)) / full_scale) if len(values) else 0.0
    return {
        "sample_rate": rate,
        "channels": channels,
        "bits": bits,
        "duration_s": (len(samples) / frame_bytes / rate) if frame_bytes and rate else 0.0,
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(peak) if peak > 0 else float("-inf"),
    }
