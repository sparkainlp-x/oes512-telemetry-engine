import asyncio
import datetime
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import scipy.signal as signal
import urllib.request

# Binary frame configuration for OES-512
MAGIC_HEADER = b"OES52"  # OES-512 signature
HEADER_FORMAT = "!5sIQ"  # Magic (5b), Seq ID (uint32), Timestamp (uint64)
NUM_CHANNELS = 512
PAYLOAD_FORMAT = f"!{NUM_CHANNELS}f"  # 512 x float32 channels
TAIL_FORMAT = "!I"  # Checksum (uint32)

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)
TAIL_SIZE = struct.calcsize(TAIL_FORMAT)
TOTAL_FRAME_SIZE = HEADER_SIZE + PAYLOAD_SIZE + TAIL_SIZE


@dataclass
class OES512Frame:
  sequence_id: int
  timestamp_us: int
  raw_channels: List[float]
  filtered_channels: List[float] | None = None
  coherence_metrics: Dict[str, float] | None = None
  alerts: List[str] | None = None


def compute_checksum(data: bytes) -> int:
  return sum(data) & 0xFFFFFFFF


class OES512DSPPipeline:
  """Stateful real-time SciPy filter pipeline scaled for 512 streaming channels."""

  def __init__(
      self,
      num_channels: int = NUM_CHANNELS,
      fs: float = 250.0,
      notch_freq: float = 50.0,
      notch_q: float = 30.0,
      bp_low: float = 0.5,
      bp_high: float = 45.0,
  ):
    self.num_channels = num_channels
    self.fs = fs

    # 1. 50 Hz Notch Filter Setup
    b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs=fs)
    self.sos_notch = signal.tf2sos(b_notch, a_notch)
    zi_notch = signal.sosfilt_zi(self.sos_notch)
    self.zi_notch = np.repeat(
        zi_notch[:, :, np.newaxis], self.num_channels, axis=2
    )

    # 2. Bandpass Filter Setup (0.5Hz - 45Hz Butterworth)
    self.sos_bp = signal.butter(
        4, [bp_low, bp_high], btype="bandpass", fs=fs, output="sos"
    )
    zi_bp = signal.sosfilt_zi(self.sos_bp)
    self.zi_bp = np.repeat(zi_bp[:, :, np.newaxis], self.num_channels, axis=2)

  def process_sample(self, raw_channels: List[float]) -> List[float]:
    x = np.array(raw_channels, dtype=np.float64)[np.newaxis, :]
    y_notch, self.zi_notch = signal.sosfilt(
        self.sos_notch, x, axis=0, zi=self.zi_notch
    )
    y_bp, self.zi_bp = signal.sosfilt(
        self.sos_bp, y_notch, axis=0, zi=self.zi_bp
    )
    return y_bp.squeeze().tolist()


class CoherenceEngineMetrics:
  """Evaluates structural system metrics (Coherence, Instability, Drift)

  from the 512-channel rolling telemetry buffer.
  """

  def __init__(
      self, num_channels: int = NUM_CHANNELS, window_size: int = 512
  ):
    self.num_channels = num_channels
    self.window_size = window_size
    self.buffer = np.zeros((window_size, num_channels), dtype=np.float64)
    self.index = 0
    self.is_filled = False
    self.baseline_profile = None

  def update_and_evaluate(
      self, filtered_sample: List[float]
  ) -> Dict[str, float] | None:
    self.buffer[self.index] = filtered_sample
    self.index = (self.index + 1) % self.window_size
    if self.index == 0:
      self.is_filled = True

    if not self.is_filled:
      return None

    ordered_data = np.concatenate(
        (self.buffer[self.index :], self.buffer[: self.index]), axis=0
    )

    # 1. Coherence Index (sample subset of 32 channels for correlation efficiency)
    subset_data = ordered_data[:, ::16]
    corr_matrix = np.corrcoef(subset_data, rowvar=False)
    off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
    coherence_index = float(np.mean(np.abs(off_diag)))

    # 2. Instability (Standard deviation of channel power variations)
    channel_powers = np.var(ordered_data, axis=0)
    instability = float(np.std(channel_powers))

    # 3. Drift (Euclidean shift from initial baseline profile)
    current_profile = np.mean(ordered_data, axis=0)
    if self.baseline_profile is None:
      self.baseline_profile = current_profile
    drift = float(np.linalg.norm(current_profile - self.baseline_profile))

    return {
        "coherence": round(coherence_index, 4),
        "instability": round(instability, 4),
        "drift": round(drift, 4),
    }


class OES512AlertManager:
  """Monitors Coherence metrics against nominal operational boundaries."""

  def __init__(
      self,
      coherence_warn_low: float = 0.20,
      coherence_warn_high: float = 0.85,
      instability_max: float = 15.0,
      drift_max: float = 25.0,
  ):
    self.coherence_warn_low = coherence_warn_low
    self.coherence_warn_high = coherence_warn_high
    self.instability_max = instability_max
    self.drift_max = drift_max

  def check_thresholds(self, metrics: Dict[str, float]) -> List[str]:
    alerts = []
    coh = metrics["coherence"]
    inst = metrics["instability"]
    drift = metrics["drift"]

    if coh < self.coherence_warn_low:
      alerts.append(
          f"[WARNING] Low Coherence detected ({coh:.3f} <"
          f" {self.coherence_warn_low})"
      )
    elif coh > self.coherence_warn_high:
      alerts.append(
          f"[WARNING] High Coherence saturation ({coh:.3f} >"
          f" {self.coherence_warn_high})"
      )

    if inst > self.instability_max:
      alerts.append(
          f"[CRITICAL] Instability threshold exceeded ({inst:.3f} >"
          f" {self.instability_max})"
      )

    if drift > self.drift_max:
      alerts.append(
          f"[CRITICAL] System drift limit breached ({drift:.3f} >"
          f" {self.drift_max})"
      )

    return alerts


class OES512AlertDispatcher:
  """Handles local file logging and asynchronous HTTP POST webhook dispatches."""

  def __init__(
      self,
      log_filepath: str = "oes512_alerts.log",
      webhook_url: str | None = None,
  ):
    self.log_filepath = log_filepath
    self.webhook_url = webhook_url

    if not os.path.exists(self.log_filepath):
      with open(self.log_filepath, "w", encoding="utf-8") as f:
        f.write(
            "TIMESTAMP_UTC | FRAME_ID | METRICS (Coh, Inst, Drift) |"
            " ALERT_MESSAGE\n"
        )
        f.write("-" * 80 + "\n")

  def log_to_file(
      self, sequence_id: int, metrics: Dict[str, float], alert_msg: str
  ):
    utc_now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S,%f"
    )[:-3]
    metric_str = (
        f"Coh={metrics['coherence']:.3f}, Inst={metrics['instability']:.3f},"
        f" Drift={metrics['drift']:.3f}"
    )
    log_entry = (
        f"{utc_now} | #{sequence_id:04d} | {metric_str} | {alert_msg}\n"
    )

    try:
      with open(self.log_filepath, "a", encoding="utf-8") as f:
        f.write(log_entry)
    except IOError as e:
      print(f"[ERROR] Failed to write to alert log file: {e}")

  async def dispatch_webhook(
      self, sequence_id: int, metrics: Dict[str, float], alert_msg: str
  ):
    if not self.webhook_url:
      return

    utc_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "source": "OES-512 Telemetry Engine",
        "timestamp_utc": utc_now,
        "sequence_id": sequence_id,
        "metrics": metrics,
        "alert_message": alert_msg,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        self.webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
      await asyncio.to_thread(urllib.request.urlopen, req, timeout=2.5)
    except Exception as e:
      print(f"[WEBHOOK ERROR] Failed to deliver alert to endpoint: {e}")


class OES512TelemetryProducer:
  """Simulates 250 Hz acquisition across 512 high-density channels."""

  def __init__(self, sample_rate_hz: float = 250.0):
    self.sample_rate = sample_rate_hz
    self.interval = 1.0 / sample_rate_hz
    self.seq_id = 0

  def generate_synthetic_channels(self, t: float) -> List[float]:
    channels = []
    for ch in range(NUM_CHANNELS):
      freq = 8.0 + (ch % 15) * 0.1
      anomaly_spike = 60.0 if (self.seq_id > 400 and ch % 32 == 0) else 0.0
      signal_val = (
          math.sin(2 * math.pi * freq * t + (ch * 0.02)) * 20.0
          + anomaly_spike
      )
      noise = math.sin(2 * math.pi * 50.0 * t) * 10.0
      channels.append(round(signal_val + noise, 4))
    return channels

  def pack_frame(self) -> bytes:
    self.seq_id += 1
    now_us = int(time.time() * 1_000_000)
    t = now_us / 1_000_000.0

    header = struct.pack(HEADER_FORMAT, MAGIC_HEADER, self.seq_id, now_us)
    channels = self.generate_synthetic_channels(t)
    payload = struct.pack(PAYLOAD_FORMAT, *channels)
    checksum = compute_checksum(header + payload)
    tail = struct.pack(TAIL_FORMAT, checksum)
    return header + payload + tail

  async def stream(self, queue: asyncio.Queue, max_packets: int = 500):
    for _ in range(max_packets):
      packet = self.pack_frame()
      await queue.put(packet)
      await asyncio.sleep(self.interval)


class OES512TelemetryParser:
  """Demuxes frames, evaluates metrics, checks alerts, and triggers dispatchers."""

  def __init__(
      self,
      fs: float = 250.0,
      webhook_url: str | None = "http://localhost:8080/oes-webhook",
  ):
    self.dsp = OES512DSPPipeline(num_channels=NUM_CHANNELS, fs=fs)
    self.coherence_engine = CoherenceEngineMetrics(
        num_channels=NUM_CHANNELS, window_size=512
    )
    self.alert_manager = OES512AlertManager()
    self.dispatcher = OES512AlertDispatcher(
        log_filepath="oes512_alerts.log", webhook_url=webhook_url
    )

  async def parse_frame(
      self, buffer: bytes
  ) -> Tuple[bool, OES512Frame | None, str]:
    if len(buffer) < TOTAL_FRAME_SIZE:
      return False, None, "Incomplete buffer"

    header_bytes = buffer[:HEADER_SIZE]
    payload_bytes = buffer[HEADER_SIZE : HEADER_SIZE + PAYLOAD_SIZE]
    tail_bytes = buffer[
        HEADER_SIZE + PAYLOAD_SIZE : HEADER_SIZE + PAYLOAD_SIZE + TAIL_SIZE
    ]

    magic, seq_id, ts_us = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != MAGIC_HEADER:
      return False, None, "Invalid magic header"

    if struct.unpack(TAIL_FORMAT, tail_bytes)[0] != compute_checksum(
        header_bytes + payload_bytes
    ):
      return False, None, "Checksum mismatch"

    raw_channels = list(struct.unpack(PAYLOAD_FORMAT, payload_bytes))
    filtered_channels = self.dsp.process_sample(raw_channels)
    coherence_metrics = self.coherence_engine.update_and_evaluate(
        filtered_channels
    )

    alerts = []
    if coherence_metrics:
      alerts = self.alert_manager.check_thresholds(coherence_metrics)
      for alert in alerts:
        self.dispatcher.log_to_file(seq_id, coherence_metrics, alert)
        asyncio.create_task(
            self.dispatcher.dispatch_webhook(seq_id, coherence_metrics, alert)
        )

    frame = OES512Frame(
        sequence_id=seq_id,
        timestamp_us=ts_us,
        raw_channels=raw_channels,
        filtered_channels=filtered_channels,
        coherence_metrics=coherence_metrics,
        alerts=alerts,
    )
    return True, frame, "OK"

  async def consume(self, queue: asyncio.Queue):
    while True:
      packet = await queue.get()
      if packet is None:
        queue.task_done()
        break

      valid, frame, err = await self.parse_frame(packet)
      if valid and frame:
        if frame.coherence_metrics:
          cm = frame.coherence_metrics
          print(
              f"[OES-512 #{frame.sequence_id:04d}] Coh: {cm['coherence']:.3f} |"
              f" Inst: {cm['instability']:.3f} | Drift: {cm['drift']:.3f}"
          )
          for alert in frame.alerts:
            print(f"  -> {alert}")
        else:
          print(
              f"[Buffering #{frame.sequence_id:04d}] Filling 512x512 history"
              " window..."
          )
      else:
        print(f"[ERR] {err}")

      queue.task_done()


async def main():
  queue = asyncio.Queue(maxsize=200)
  producer = OES512TelemetryProducer(sample_rate_hz=250.0)
  parser = OES512TelemetryParser(
      fs=250.0, webhook_url="http://localhost:8080/oes-webhook"
  )

  print(
      "--- Starting OES-512 Telemetry Pipeline with File Logger & Async Webhook"
      " ---"
  )
  producer_task = asyncio.create_task(producer.stream(queue, max_packets=500))
  consumer_task = asyncio.create_task(parser.consume(queue))

  await producer_task
  await queue.put(None)
  await consumer_task
  print(
      "--- Processing Complete. Local audit saved to oes512_alerts.log ---"
  )


if __name__ == "__main__":
  asyncio.run(main())
