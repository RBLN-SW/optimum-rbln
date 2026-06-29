"""Local Gradio demo for Pop2Piano on RBLN NPU: audio in -> piano MIDI out.

Run:
    pip install gradio
    python app.py                       # loads ./pop2piano if present, else compiles once
    python app.py --batch_size 4 --share

The first launch with a given (batch_size, enc/dec length) compiles the model and
caches it under --model_dir; later launches load that artifact and start fast.

MIDI preview is rendered with FluidSynth when a SoundFont is available
(`~/.local/share/soundfonts/FluidR3_GM.sf2` by default), otherwise it falls back
to pretty_midi's built-in sine synth.
"""

import base64
import ctypes
import ctypes.util
import glob
import os
import tempfile

import fire
import librosa
import numpy as np


SOUNDFONT = os.path.expanduser("~/.local/share/soundfonts/FluidR3_GM.sf2")
_FLUID_LIBDIR = os.path.expanduser("~/.local/lib/fluidsynth/")
DEFAULT_SPREAD = 1.0
PROJECT_URL = "https://sweetcocoa.github.io/pop2piano_samples/"
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def _enable_fluidsynth():
    """Make a root-less FluidSynth install loadable, returning True on success.

    Preloads libinstpatch into the global namespace and points ctypes'
    `find_library` at our versioned libfluidsynth so pyfluidsynth can import
    without the system package or LD_LIBRARY_PATH.
    """
    lib = os.path.join(_FLUID_LIBDIR, "libfluidsynth.so.3")
    if not os.path.exists(lib):
        return False
    try:
        ctypes.CDLL(os.path.join(_FLUID_LIBDIR, "libinstpatch-1.0.so.2"), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    _orig = ctypes.util.find_library
    ctypes.util.find_library = lambda name: lib if (name and "fluidsynth" in name) else _orig(name)
    try:
        import fluidsynth  # noqa: F401

        return True
    except Exception:
        return False


HAS_FLUIDSYNTH = _enable_fluidsynth()


def _peak_normalize(audio):
    return audio / (np.abs(audio).max() + 1e-8) if audio.size else audio


def _render_piano(midi, fs=22050):
    """pretty_midi -> (float32 waveform in [-1, 1], backend_name)."""
    backend = "sine"
    audio = None
    if HAS_FLUIDSYNTH:
        try:
            sf2 = SOUNDFONT if os.path.exists(SOUNDFONT) else None
            audio = midi.fluidsynth(fs=fs, synthesizer=sf2)
            backend = "fluidsynth"
        except Exception:
            audio = None
    if audio is None:
        audio = midi.synthesize(fs=fs)
    return _peak_normalize(audio).astype(np.float32), backend


def piano_preview(midi, fs=22050):
    """Mono piano render as (sample_rate, int16 waveform, backend_name)."""
    audio, backend = _render_piano(midi, fs)
    return fs, (audio * 32767).astype(np.int16), backend


def prepare_original(original, orig_sr, fs=22050):
    """Mono original resampled to `fs` and peak-normalized, for stereo mixing."""
    orig = original.astype(np.float32)
    if orig_sr != fs:
        orig = librosa.resample(orig, orig_sr=orig_sr, target_sr=fs)
    return _peak_normalize(orig)


def mix_stereo(orig, piano, fs, spread):
    """Linear-pan original toward the left and piano toward the right by `spread`.

    spread=0 places both dead-center (mono blend); spread=1 hard-pans original
    fully left and piano fully right. Returns (sample_rate, int16 (n_samples, 2)).
    """
    n = max(orig.shape[0], piano.shape[0])
    o = np.zeros(n, dtype=np.float32)
    p = np.zeros(n, dtype=np.float32)
    o[: orig.shape[0]] = orig
    p[: piano.shape[0]] = piano
    s = float(np.clip(spread, 0.0, 1.0))
    left = o * (1 + s) / 2 + p * (1 - s) / 2
    right = o * (1 - s) / 2 + p * (1 + s) / 2
    stereo = np.clip(np.stack([left, right], axis=1), -1.0, 1.0)
    return fs, (stereo * 32767).astype(np.int16)


def midi_player_html(midi_path):
    """A html-midi-player widget: plays the MIDI with a piano-roll that scrolls in sync.

    The MIDI is inlined as a data URI so the value is self-contained (cacheable as
    plain text). The web component itself is loaded once via the page <head>.
    """
    with open(midi_path, "rb") as f:
        src = "data:audio/midi;base64," + base64.b64encode(f.read()).decode()
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        '<script src="https://cdn.jsdelivr.net/combine/'
        "npm/tone@14.7.77,npm/@magenta/music@1.23.1/es6/core.js,"
        'npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>'
        "<style>body{margin:0;font-family:sans-serif}midi-player{width:100%}"
        "midi-visualizer{display:block;margin-top:6px}</style></head><body>"
        f'<midi-player src="{src}" sound-font></midi-player>'
        f'<midi-visualizer type="piano-roll" src="{src}"></midi-visualizer>'
        "<script>Promise.all(["
        "customElements.whenDefined('midi-player'),"
        "customElements.whenDefined('midi-visualizer')]).then(function(){"
        "var p=document.querySelector('midi-player'),v=document.querySelector('midi-visualizer');"
        "if(p&&v&&p.addVisualizer)p.addVisualizer(v);});</script></body></html>"
    )
    # Fully entity-encode the document for the srcdoc attribute. Encoding < and >
    # (not strictly required for an attribute value) keeps the literal "<script"
    # out of the gr.HTML value, so Gradio's script-tag check stays quiet; the
    # browser decodes srcdoc back into a real iframe document where scripts run.
    srcdoc = (
        doc.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return (
        f'<iframe srcdoc="{srcdoc}" style="width:100%;height:340px;border:0" '
        'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def main(
    model_id: str = "sweetcocoa/pop2piano",
    model_dir: str = "pop2piano",
    batch_size: int = 4,
    enc_max_seq_len: int = 256,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    import gradio as gr
    from transformers import Pop2PianoProcessor

    from optimum.rbln import RBLNPop2PianoForConditionalGeneration

    if os.path.isdir(model_dir):
        print(f"Loading compiled model from '{model_dir}' ...", flush=True)
        model = RBLNPop2PianoForConditionalGeneration.from_pretrained(model_dir, export=False)
    else:
        print(f"Compiling '{model_id}' (one-time) -> '{model_dir}' ...", flush=True)
        model = RBLNPop2PianoForConditionalGeneration.from_pretrained(
            model_id,
            export=True,
            rbln_batch_size=batch_size,
            rbln_enc_max_seq_len=enc_max_seq_len,
            rbln_dec_max_seq_len=256,
        )
        model.save_pretrained(model_dir)

    processor = Pop2PianoProcessor.from_pretrained(model_id)
    composers = sorted(
        model.generation_config.composer_to_feature_token.keys(),
        key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0),
    )
    decode_batch = model.rbln_config.batch_size

    def transcribe(audio_path, composer):
        if audio_path is None:
            raise gr.Error("Please upload an audio file.")
        import time

        array, sr = librosa.load(audio_path, sr=None)
        if array.ndim > 1:
            array = array.mean(axis=0)
        inputs = processor(audio=array, sampling_rate=sr, return_tensors="pt")

        t0 = time.time()
        token_ids = model.generate(input_features=inputs["input_features"], composer=composer)
        gen_s = time.time() - t0

        midi = processor.batch_decode(token_ids=token_ids, feature_extractor_output=inputs)["pretty_midi_objects"][0]
        work = tempfile.mkdtemp()
        out_path = os.path.join(work, "piano_cover.mid")
        midi.write(out_path)
        player = midi_player_html(out_path)

        fs = 22050
        piano, backend = _render_piano(midi, fs)
        orig = prepare_original(array, sr, fs)
        sources = (orig, piano, fs)
        panned = mix_stereo(orig, piano, fs, DEFAULT_SPREAD)

        n_seg = int(inputs["input_features"].shape[0])
        n_notes = sum(len(inst.notes) for inst in midi.instruments)
        stats = (
            f"- Input length: **{len(array) / sr:.1f}s**\n"
            f"- Beat-synced segments: **{n_seg}** (decoded in batches of {decode_batch})\n"
            f"- Generated notes: **{n_notes}**, duration **{midi.get_end_time():.1f}s**\n"
            f"- NPU generation time: **{gen_s:.2f}s**\n"
            f"- A/B render: **{backend}**"
        )
        return panned, player, out_path, stats, sources

    def repan(sources, spread):
        if not sources:
            return gr.update()
        orig, piano, fs = sources
        return mix_stereo(orig, piano, fs, spread)

    with gr.Blocks(title="Pop2Piano on RBLN NPU") as demo:
        gr.Markdown(
            "# 🎹 Pop2Piano on RBLN NPU\n"
            "Turn pop-music audio into a piano-cover MIDI on the Rebellions NPU. "
            "Upload audio, pick an *arranger*, then press **Transcribe**."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["upload"], type="filepath", label="Input audio")
                composer_in = gr.Dropdown(choices=composers, value=composers[0], label="Arranger (composer)")
                run_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                mix_out = gr.Audio(label="A/B preview — original vs piano", type="numpy")
                pan_slider = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_SPREAD,
                    step=0.05,
                    label="Stereo spread (0 = centered blend, 1 = original hard-left / piano hard-right)",
                )
                gr.Markdown("**Piano cover** — press play; the piano roll scrolls with playback")
                player_out = gr.HTML()
                midi_out = gr.File(label="Download MIDI (.mid)")
                stats_out = gr.Markdown()

        sources_state = gr.State()
        outputs = [mix_out, player_out, midi_out, stats_out, sources_state]
        run_btn.click(transcribe, [audio_in, composer_in], outputs)
        pan_slider.release(repan, [sources_state, pan_slider], mix_out)

        examples = _gather_examples(composers[0])
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[audio_in, composer_in],
                outputs=outputs,
                fn=transcribe,
                cache_examples=True,
            )

        gr.Markdown(f"Project page & audio samples: [{PROJECT_URL}]({PROJECT_URL})")

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=gr.themes.Soft(),
        allowed_paths=[EXAMPLE_DIR],
    )


def _gather_examples(default_composer):
    """Example rows for the UI: the full-length local hanroro clip(s)."""
    return [
        [path, default_composer]
        for path in sorted(glob.glob(os.path.join(EXAMPLE_DIR, "hanroro*.mp3")))
    ]


if __name__ == "__main__":
    fire.Fire(main)
