# OES-512 High-Density Telemetry & Coherence Engine

An asynchronous Python telemetry ingestion, digital signal processing (DSP), and system dynamics monitoring engine scaled for 512-channel high-density data streams.

## Features
- **Binary Frame Streaming:** Ingestion and demuxing of 512-channel frames with checksum validation.
- **Stateful SciPy DSP:** Real-time 50 Hz notch filter and Butterworth bandpass filtering using Second-Order Sections (SOS).
- **Coherence Engine:** Rolling buffer analysis computing live structural metrics (**Coherence, Instability, and Drift**).
- **Audit Logging & Webhooks:** Local file auditing (`oes512_alerts.log`) combined with non-blocking asynchronous HTTP POST webhook dispatches.

## Installation & Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   python main.py
   ```
