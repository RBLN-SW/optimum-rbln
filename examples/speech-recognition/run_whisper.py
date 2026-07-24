import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoProcessor, pipeline

from optimum.rbln import RBLNWhisperForConditionalGeneration


# You can compile the model ahead of time with the CLI and then load the
# artifacts here by passing the output directory as --model-id:
#
#   optimum-rbln-cli --model-id openai/whisper-tiny -o whisper-tiny \
#       --batch_size 1


def prepare_shortform(model_id, batch_size):
    processor = AutoProcessor.from_pretrained(model_id)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    input_features_list = []

    for i in range(batch_size):
        input_features = processor(
            ds[i]["audio"]["array"],
            sampling_rate=ds[i]["audio"]["sampling_rate"],
            truncation=False,
            return_tensors="pt",
        ).input_features
        input_features_list.append(input_features)
    input_features = torch.cat(input_features_list, dim=0)
    return processor, input_features


def prepare_longform(model_id, batch_size):
    processor = AutoProcessor.from_pretrained(model_id)
    ds = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

    input = processor(
        ds[0]["audio"]["array"],
        sampling_rate=ds[0]["audio"]["sampling_rate"],
        truncation=False,
        padding="longest",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_features = input.input_features.repeat(batch_size, 1, 1)
    attention_mask = input.attention_mask.repeat(batch_size, 1)

    return processor, input_features, attention_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--return-token-timestamps", action="store_true")
    parser.add_argument("--long-form", action="store_true")
    parser.add_argument("--pipe", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # set kwargs
    rbln_kwargs = {}
    gen_kwargs = {"return_timestamps": True}
    if args.return_token_timestamps:
        gen_kwargs.update({"return_token_timestamps": True})
        rbln_kwargs.update({"rbln_token_timestamps": True})

    # compile or load model
    if os.path.isdir(args.model_id):
        model = RBLNWhisperForConditionalGeneration.from_pretrained(args.model_id)
    else:
        model = RBLNWhisperForConditionalGeneration.from_pretrained(
            args.model_id,
            rbln_batch_size=args.batch_size,
            **rbln_kwargs,
        )

    # generation strategy
    # 1. short_form
    # 2. long_form
    # 3. pipe
    if not args.long_form and not args.pipe:
        processor, input_features = prepare_shortform(args.model_id, args.batch_size)
        outputs = model.generate(
            input_features=input_features,
            **gen_kwargs,
        )

        generated_ids = outputs["sequences"] if isinstance(outputs, dict) else outputs
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        print("---RBLN Shortform Generate Result ---")
        for i, transcription in enumerate(transcriptions):
            print(f"transcription {i} : {transcription}")
            if args.return_token_timestamps:
                print(f"token_timestamps {i} : {outputs['token_timestamps'][i]}")

    if args.long_form:
        processor, input_features, attention_mask = prepare_longform(args.model_id, args.batch_size)
        outputs = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        generated_ids = outputs.get("sequences") if isinstance(outputs, dict) else outputs
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        print("---RBLN Longform Generate Result ---")
        for i, transcription in enumerate(transcriptions):
            print(f"transcription {i} : {transcription}")

    if args.pipe:
        processor = AutoProcessor.from_pretrained(args.model_id)
        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps=True,
            batch_size=args.batch_size,
        )
        generate_kwargs = {"repetition_penalty": 1.3, "num_beams": 1}

        with torch.no_grad():
            outputs = pipe(sample, generate_kwargs=generate_kwargs)
        print("---RBLN Pipeline Result ---")
        print("--Text--")
        print(outputs["text"])
        print("--Chunks--")
        print(outputs["chunks"])


if __name__ == "__main__":
    main()
