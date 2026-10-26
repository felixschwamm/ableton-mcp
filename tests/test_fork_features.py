"""Tests for the fork's additions to the Live-side script and server:
deferred replies (device list after load, playhead read-back), meters, clip
automation, parameters by display value, single-track render routing and
WAV analysis. Live is faked; no Ableton, no network.
"""

import math
import queue
import struct
import types

import pytest

from test_remote_script_bridge import _load_script_module, make_instance
from test_resample_capture import FakeRouting, FakeSong, FakeTrack


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def _resolve(script_module, result):
    """Drive a handler result through _Deferred ticks like Live would."""
    while isinstance(result, script_module._Deferred):
        result = result.finish()
    return result


# ------------------------------------------------------------------ deferred replies

def test_reply_waits_for_deferred_result(script_module):
    inst = make_instance(script_module)
    scheduled = []
    inst.schedule_message = lambda ticks, fn: scheduled.append((ticks, fn))
    replies = queue.Queue()

    inst._reply_when_ready(script_module._Deferred(lambda: {"done": True}, 2), replies)
    assert replies.empty()
    ticks, fn = scheduled.pop()
    assert ticks == 2
    fn()
    assert replies.get_nowait() == {"status": "success", "result": {"done": True}}


def test_device_list_is_read_after_live_applies_the_load(script_module):
    """Regression (upstream #134): the reply listed no devices because the
    chain was read in the same tick as browser.load_item."""
    track = types.SimpleNamespace(devices=[types.SimpleNamespace(name="Saturator")])
    inst = make_instance(script_module)
    pending = inst._after_devices_change(track, ["Saturator"], {"loaded": True})

    # Still unchanged on the first tick: keeps waiting.
    pending = pending.finish()
    assert isinstance(pending, script_module._Deferred)
    track.devices.insert(0, types.SimpleNamespace(name="Analog"))
    result = _resolve(script_module, pending)

    assert result["devices_after"] == ["0: Analog", "1: Saturator"]
    assert result["new_devices"] == ["Analog"]


def test_device_list_wait_gives_up(script_module):
    track = types.SimpleNamespace(devices=[types.SimpleNamespace(name="Analog")])
    inst = make_instance(script_module)
    result = _resolve(script_module, inst._after_devices_change(track, ["Analog"], {}))
    assert result["devices_after"] == ["0: Analog"]
    assert result["new_devices"] == []


def test_playhead_reply_reads_the_applied_position(script_module):
    inst = make_instance(script_module)
    inst._song = types.SimpleNamespace(current_song_time=164.0)
    pending = inst._set_current_song_time(73.5)
    assert _resolve(script_module, pending) == {"current_song_time": 73.5}


# ------------------------------------------------------------------ meters

def test_meter_levels_for_track_and_main(script_module):
    track = types.SimpleNamespace(name="Synth", output_meter_left=0.5,
                                  output_meter_right=0.25, output_meter_level=0.5)
    main = types.SimpleNamespace(name="Main", output_meter_left=0.0,
                                 output_meter_right=0.0, output_meter_level=0.0)
    inst = make_instance(script_module)
    inst._song = types.SimpleNamespace(tracks=[track], master_track=main, is_playing=True)

    assert inst._get_meter_levels(0)["left"] == 0.5
    assert inst._get_meter_levels(-1)["track_name"] == "Main"
    with pytest.raises(IndexError):
        inst._get_meter_levels(3)


# ------------------------------------------------------------------ display values

class FreqParam(object):
    """20 Hz .. 20 kHz on a 0..1 log scale, shown the way Live does."""
    name = "F1 Freq"
    min = 0.0
    max = 1.0
    is_quantized = False

    def __init__(self):
        self.value = 0.5

    def str_for_value(self, v):
        hz = 20.0 * (1000.0 ** v)
        return "%.2f kHz" % (hz / 1000.0) if hz >= 1000 else "%.0f Hz" % hz


class TypeParam(object):
    name = "F1 Type"
    min = 0.0
    max = 2.0
    is_quantized = True
    value_items = ["LP12", "BP12", "HP12"]
    value = 0.0

    def str_for_value(self, v):
        return self.value_items[int(v)]


def test_display_value_bisection_hits_the_target(script_module):
    inst = make_instance(script_module)
    param = FreqParam()
    value = inst._value_for_display(param, "800 Hz")
    assert param.str_for_value(value) == "800 Hz"
    value = inst._value_for_display(param, "1.2 kHz")
    assert param.str_for_value(value) == "1.20 kHz"


def test_display_value_matches_quantized_choice(script_module):
    inst = make_instance(script_module)
    assert inst._value_for_display(TypeParam(), "bp12") == 1.0
    with pytest.raises(ValueError):
        inst._value_for_display(TypeParam(), "Notch")


def test_display_number_units(script_module):
    inst = make_instance(script_module)
    assert inst._display_number("0.5 s") == 500.0
    assert inst._display_number("250 ms") == 250.0
    assert inst._display_number("-6.0 dB") == -6.0
    assert inst._display_number("-inf dB") == float("-inf")
    assert inst._display_number("C3") is None


# ------------------------------------------------------------------ clip automation

class FakeEnvelope(object):
    def __init__(self):
        self.steps = []

    def insert_step(self, time, duration, value):
        self.steps.append((time, duration, value))


class FakeAutomationClip(object):
    name = "Pad"

    def __init__(self):
        self.envelopes = {}
        self.cleared = []

    def automation_envelope(self, param):
        return self.envelopes.get(param.name)

    def create_automation_envelope(self, param):
        self.envelopes[param.name] = FakeEnvelope()
        return self.envelopes[param.name]

    def clear_envelope(self, param):
        self.cleared.append(param.name)
        self.envelopes.pop(param.name, None)

    def clear_all_envelopes(self):
        self.cleared.append("all")
        self.envelopes.clear()


def _automation_instance(script_module, clip, arrangement=False):
    device = types.SimpleNamespace(parameters=[FreqParam()])
    slot = types.SimpleNamespace(has_clip=True, clip=clip)
    track = types.SimpleNamespace(devices=[device], clip_slots=[slot],
                                  arrangement_clips=[clip])
    inst = make_instance(script_module)
    inst._song = types.SimpleNamespace(tracks=[track])
    return inst


def test_set_clip_automation_writes_sorted_breakpoints(script_module):
    clip = FakeAutomationClip()
    inst = _automation_instance(script_module, clip)
    points = [{"time": 8.0, "value": 1.5}, {"time": 0.0, "display": "800 Hz"}]

    result = inst._set_clip_automation(0, 0, 0, 0, points)

    steps = clip.envelopes["F1 Freq"].steps
    assert [s[0] for s in steps] == [0.0, 8.0]
    assert all(s[1] == 0.0 for s in steps)
    assert steps[1][2] == 1.0  # clamped to max
    assert result["first_display"] == "800 Hz"


def test_set_clip_automation_on_arrangement_clip_replaces_envelope(script_module):
    clip = FakeAutomationClip()
    clip.envelopes["F1 Freq"] = FakeEnvelope()
    inst = _automation_instance(script_module, clip)

    inst._set_clip_automation(0, 0, 0, 0, [{"time": 0.0, "value": 0.2}], arrangement=True)

    assert clip.cleared == ["F1 Freq"]
    assert len(clip.envelopes["F1 Freq"].steps) == 1


def test_clear_clip_automation_all_or_one(script_module):
    clip = FakeAutomationClip()
    inst = _automation_instance(script_module, clip)
    assert inst._clear_clip_automation(0, 0)["cleared"] == "all"
    assert inst._clear_clip_automation(0, 0, 0, 0)["cleared"] == "F1 Freq"


# ------------------------------------------------------------------ single-track render

class FakeChannel(object):
    def __init__(self, display_name):
        self.display_name = display_name


def test_render_routes_capture_input_from_the_source_track(script_module):
    synth = FakeTrack("OnSight Synth")
    song = FakeSong([synth])

    def create_audio_track(_index):
        capture = FakeTrack()
        capture.available_input_routing_types = [
            FakeRouting("Ext. In"), FakeRouting("Resampling"), FakeRouting("OnSight Synth")]
        capture.available_input_routing_channels = [
            FakeChannel("Pre FX"), FakeChannel("Post FX"), FakeChannel("Post Mixer")]
        capture.input_routing_channel = capture.available_input_routing_channels[0]
        song.tracks.append(capture)

    song.create_audio_track = create_audio_track
    inst = make_instance(script_module)
    inst._song = song
    inst.schedule_message = lambda _delay, fn: fn()

    inst._start_resample_capture(0.0, 32.0, source_track_index=0)
    capture = song.tracks[-1]

    assert capture.name == "MCP Render OnSight Synth"
    assert capture.input_routing_type.display_name == "OnSight Synth"
    assert capture.input_routing_channel.display_name == "Post FX"
    assert capture.current_monitoring_state == 2


def test_render_rejects_bad_source_track(script_module):
    inst = make_instance(script_module)
    inst._song = FakeSong([FakeTrack("Synth")])
    with pytest.raises(IndexError):
        inst._start_resample_capture(0.0, 16.0, source_track_index=5)


# ------------------------------------------------------------------ WAV analysis

def _write_wav(path, tag, bits, frames, rate=48000, channels=2):
    if tag == 3:
        data = struct.pack("<%df" % len(frames), *frames)
    elif bits == 24:
        data = b"".join(struct.pack("<i", int(v * 8388607))[:3] for v in frames)
    else:
        data = struct.pack("<%dh" % len(frames), *[int(v * 32767) for v in frames])
    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


@pytest.mark.parametrize("tag,bits", [(1, 16), (1, 24), (3, 32)])
def test_wav_stats_reads_peak_and_duration(tmp_path, tag, bits):
    from MCP_Server.audio_file import wav_stats

    frames = [0.0, 0.0, 0.5, -0.25] * 12000  # 24000 stereo frames = 0.5 s
    path = tmp_path / "render.wav"
    _write_wav(path, tag, bits, frames)

    stats = wav_stats(str(path))

    assert stats["sample_rate"] == 48000
    assert stats["duration_s"] == pytest.approx(0.5)
    assert stats["peak_dbfs"] == pytest.approx(20 * math.log10(0.5), abs=0.01)


def test_silent_render_is_flagged(tmp_path):
    import MCP_Server.server as server

    path = tmp_path / "silent.wav"
    _write_wav(path, 3, 32, [0.0] * 1000)
    assert "WARNING: the render is silent" in server._describe_render(str(path))


# ------------------------------------------------------------------ render tool (server)

class ScriptedConnection(object):
    """Answers each command type from a list of canned replies, in order."""

    def __init__(self, replies):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.sent = []

    def send_command(self, command_type, params=None):
        self.sent.append(command_type)
        options = self.replies[command_type]
        return options.pop(0) if len(options) > 1 else options[0]


def test_render_track_to_audio_waits_copies_and_cleans_up(tmp_path, monkeypatch):
    import asyncio
    import MCP_Server.server as server
    import MCP_Server.script_handshake as sh
    import MCP_Server.dataset.consent as consent
    import MCP_Server.telemetry_decorator as td

    monkeypatch.setattr(sh, "require_capability", lambda name: None)
    monkeypatch.setattr(consent, "maybe_consent_notice", lambda: "")
    monkeypatch.setattr(td, "get_telemetry",
                        lambda: types.SimpleNamespace(record_event=lambda *a, **k: None),
                        raising=False)
    recorded = tmp_path / "Recorded" / "take.wav"
    recorded.parent.mkdir()
    _write_wav(recorded, 3, 32, [0.5, -0.5] * 4800)
    conn = ScriptedConnection({
        "start_resample_capture": [{"estimated_seconds": 1.0}],
        "get_resample_capture_status": [
            {"status": "recording", "progress": 0.5},
            {"status": "done", "reason": "completed", "file_path": str(recorded)},
        ],
        "cancel_resample_capture": [{"cancelled": True}],
    })
    monkeypatch.setattr(server, "get_ableton_connection", lambda: conn)

    out = asyncio.run(server.render_track_to_audio(
        None, track_index=0, start_beat=0.0, length_beats=2.0,
        target_path=str(tmp_path / "out") + "/"))

    assert (tmp_path / "out" / "take.wav").exists()
    assert "peak -6.0 dBFS" in out
    assert conn.sent[-1] == "cancel_resample_capture"
