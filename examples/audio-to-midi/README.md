# Pop2Piano on RBLN NPU

Turn pop-music audio into a piano-cover MIDI on a Rebellions NPU.

- `run_pop2piano.py` — CLI: audio file (or a HuggingFace sample clip) → `.mid`.
- `app.py` — local Gradio web demo: upload audio, pick an arranger, get a downloadable MIDI plus a piano preview.

## Install

```bash
source ~/optimum-rbln/.venv/bin/activate
pip install gradio
```

The MIDI preview uses FluidSynth when a SoundFont is available
(`~/.local/share/soundfonts/FluidR3_GM.sf2` by default); otherwise it falls
back to pretty_midi's built-in sine synth. FluidSynth is optional.

## Run the web demo

```bash
python app.py                         # loads ./pop2piano if present, else compiles once and caches it
# options:
python app.py --batch_size 4 --enc_max_seq_len 256 --server_port 7860
```

The first launch with a given `(batch_size, enc/dec length)` compiles the model
and saves it under `--model_dir` (default `pop2piano`); later launches load that
artifact and start fast.

## Accessing it from your laptop (the demo runs on a remote server)

`app.py` binds to `127.0.0.1` on the remote host by default, so it is not
reachable directly. Pick one of the following.

### Option A — SSH port forwarding (recommended)

Run the demo on the remote server as above, then from **your laptop** open a
tunnel and browse to it locally:

```bash
ssh -L 7860:localhost:7860 <user>@<remote-host>
# keep that session open, then open http://localhost:7860 in your browser
```

Nothing is exposed publicly and no firewall changes are needed. If you use a
different `--server_port`, forward that port instead.

### Option B — Gradio share link

Let Gradio create a temporary public `*.gradio.live` URL (it tunnels out, so it
works even when inbound ports are firewalled):

```bash
python app.py --share
```

The link is printed in the console and lasts for the session. Requires outbound
internet from the server (Gradio downloads a small tunnel binary on first use).

### Option C — bind to all interfaces

If your laptop can already reach the server's IP (same network or VPN):

```bash
python app.py --server_name 0.0.0.0 --server_port 7860
# browse to http://<remote-host-ip>:7860
```

Only do this on a trusted network — it serves the demo to anyone who can reach
that address.
