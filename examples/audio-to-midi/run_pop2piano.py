import os

import fire
import librosa

from optimum.rbln import RBLNPop2PianoForConditionalGeneration


def load_audio(audio_path: str = None):
    """Returns (waveform: 1-D np.ndarray, sampling_rate: int, name: str).

    With no `audio_path`, downloads a real pop-music clip from the HuggingFace
    `sweetcocoa/pop2piano_ci` dataset.
    """
    if audio_path is not None:
        array, sampling_rate = librosa.load(audio_path, sr=None)
        return array, sampling_rate, os.path.splitext(os.path.basename(audio_path))[0]

    from datasets import load_dataset

    ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")
    samples = ds[0]["audio"].get_all_samples()
    array = samples.data.numpy()
    if array.ndim > 1:  # stereo -> mono
        array = array.mean(axis=0)
    return array, samples.sample_rate, "pop2piano_ci_0"


def main(
    model_id: str = "sweetcocoa/pop2piano",
    from_transformers: bool = False,
    # rbln config
    batch_size: int = 1,
    enc_max_seq_len: int = 256,
    # input / generation
    audio_path: str = None,
    composer: str = "composer1",
    output_dir: str = ".",
):
    from transformers import Pop2PianoProcessor

    # `batch_size` is how many beat-synced segments the decoder processes per chunk;
    # a larger value decodes more segments in parallel (the last chunk is padded).
    if from_transformers:
        model = RBLNPop2PianoForConditionalGeneration.from_pretrained(
            model_id=model_id,
            export=True,
            rbln_batch_size=batch_size,
            rbln_enc_max_seq_len=enc_max_seq_len,
            rbln_dec_max_seq_len=256,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNPop2PianoForConditionalGeneration.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    processor = Pop2PianoProcessor.from_pretrained(model_id)

    # audio -> log-mel features (essentia beat-tracking happens here, on host)
    array, sampling_rate, name = load_audio(audio_path)
    inputs = processor(audio=array, sampling_rate=sampling_rate, return_tensors="pt")
    print(f"Input: '{name}' -> {tuple(inputs['input_features'].shape)} (segments, frames, d_model)")

    # NPU generation (composer conditioning + encoder + decoder); segments looped at batch 1.
    # Generation params come from the model's generation_config (greedy by default), matching
    # the upstream Pop2Piano usage `generate(input_features=..., composer=...)`.
    token_ids = model.generate(
        input_features=inputs["input_features"],
        composer=composer,
    )

    # tokens -> MIDI
    midi = processor.batch_decode(token_ids=token_ids, feature_extractor_output=inputs)["pretty_midi_objects"][0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.mid")
    midi.write(output_path)

    num_notes = sum(len(inst.notes) for inst in midi.instruments)
    print(f"Saved piano cover: {output_path} ({num_notes} notes, {midi.get_end_time():.1f}s)")


if __name__ == "__main__":
    fire.Fire(main)
