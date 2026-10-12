"""Tests for the OES-512 synthetic telemetry pipeline (all inputs SYNTHETIC)."""

import asyncio
import struct

import numpy as np
import pytest

import main as eng


def _parser(tmp_path, **kwargs):
  return eng.OES512TelemetryParser(log_filepath=str(tmp_path / "alerts.log"), **kwargs)


def test_frame_round_trip(tmp_path):
  producer = eng.OES512TelemetryProducer()
  packet = producer.pack_frame()
  assert len(packet) == eng.TOTAL_FRAME_SIZE
  ok, frame, msg = asyncio.run(_parser(tmp_path).parse_frame(packet))
  assert ok and msg == "OK"
  assert frame.sequence_id == 1
  assert len(frame.raw_channels) == 512 and len(frame.filtered_channels) == 512
  assert frame.coherence_metrics is None  # window not yet filled
  expected = producer.generate_synthetic_channels(1 / 250.0)
  np.testing.assert_allclose(frame.raw_channels, expected, rtol=1e-6, atol=1e-4)


def test_rejects_bad_frames(tmp_path):
  parser = _parser(tmp_path)
  packet = bytearray(eng.OES512TelemetryProducer().pack_frame())
  assert asyncio.run(parser.parse_frame(bytes(packet[:-1])))[2] == "Incomplete buffer"
  bad_magic = b"XXXXX" + bytes(packet[5:])
  assert asyncio.run(parser.parse_frame(bad_magic))[2] == "Invalid magic header"
  packet[eng.HEADER_SIZE] ^= 0xFF
  assert asyncio.run(parser.parse_frame(bytes(packet)))[2] == "Checksum mismatch"


def test_producer_is_deterministic():
  a, b = eng.OES512TelemetryProducer(), eng.OES512TelemetryProducer()
  for _ in range(5):
    pa, pb = a.pack_frame(), b.pack_frame()
    # Payloads match; only the wall-clock header timestamp may differ.
    assert pa[eng.HEADER_SIZE:-eng.TAIL_SIZE] == pb[eng.HEADER_SIZE:-eng.TAIL_SIZE]


def test_notch_attenuates_50hz():
  dsp = eng.OES512DSPPipeline(num_channels=4)
  fs = 250.0
  out = []
  for n in range(2000):
    out.append(dsp.process_sample([np.sin(2 * np.pi * 50.0 * n / fs)] * 4))
  tail = np.asarray(out[1000:])
  assert np.max(np.abs(tail)) < 0.05


def test_bandpass_keeps_10hz():
  dsp = eng.OES512DSPPipeline(num_channels=2)
  fs = 250.0
  out = [dsp.process_sample([np.sin(2 * np.pi * 10.0 * n / fs)] * 2) for n in range(2000)]
  assert 0.8 < np.max(np.abs(np.asarray(out[1000:]))) < 1.2


def test_metrics_definitions():
  rng = np.random.default_rng(0)
  window = 64
  engine = eng.CoherenceEngineMetrics(num_channels=64, window_size=window)
  common = rng.normal(size=window)
  result = None
  for i in range(window):
    result = engine.update_and_evaluate(np.full(64, common[i]))
  assert result is not None
  assert result["coherence"] == pytest.approx(1.0)
  assert result["instability"] == pytest.approx(0.0, abs=1e-9)
  assert result["drift"] == 0.0

  independent = eng.CoherenceEngineMetrics(num_channels=64, window_size=window)
  for row in rng.normal(size=(window, 64)):
    result = independent.update_and_evaluate(row)
  assert result["coherence"] < 0.3


def test_constant_channels_do_not_produce_nan():
  engine = eng.CoherenceEngineMetrics(num_channels=32, window_size=8)
  result = None
  for _ in range(8):
    result = engine.update_and_evaluate(np.zeros(32))
  assert result == {"coherence": 0.0, "instability": 0.0, "drift": 0.0}


def test_alert_thresholds():
  manager = eng.OES512AlertManager()
  assert manager.check_thresholds({"coherence": 0.5, "instability": 1.0, "drift": 1.0}) == []
  alerts = manager.check_thresholds({"coherence": 0.1, "instability": 20.0, "drift": 30.0})
  assert len(alerts) == 3
  assert "High Coherence" in manager.check_thresholds(
      {"coherence": 0.9, "instability": 0.0, "drift": 0.0}
  )[0]


def test_webhook_is_off_by_default(tmp_path):
  parser = _parser(tmp_path)
  assert parser.dispatcher.webhook_url is None
  asyncio.run(parser.dispatcher.dispatch_webhook(1, {"coherence": 0}, "x"))  # no-op


def test_end_to_end_alerts_only_after_synthetic_step(tmp_path):
  log = tmp_path / "alerts.log"
  asyncio.run(
      eng.main(["--fast", "--packets", "900", "--window", "256", "--log", str(log)])
  )
  lines = log.read_text().splitlines()[2:]
  assert lines, "the synthetic step should raise at least one alert"
  first_frame = int(lines[0].split("|")[1].strip().lstrip("#"))
  assert first_frame > 700


def test_producer_validation():
  with pytest.raises(ValueError):
    eng.OES512TelemetryProducer(sample_rate_hz=0)
  assert struct.calcsize(eng.PAYLOAD_FORMAT) == 512 * 4
