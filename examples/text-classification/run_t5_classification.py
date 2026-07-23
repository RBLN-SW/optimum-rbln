import os

import fire
from transformers import AutoTokenizer, T5EncoderModel

from optimum.rbln import RBLNT5EncoderModel


def main(
    model_id: str = "google-t5/t5-small",
    batch_size: int = 1,
    from_transformers: bool = False,
    prompt: str | None = "Studies have been shown that owning a dog is good for you",
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    t5_encoder_model = T5EncoderModel.from_pretrained(model_id)

    if from_transformers:
        model = RBLNT5EncoderModel.from_model(
            model=t5_encoder_model,
            rbln_batch_size=batch_size,
        )
        model.save_pretrained(os.path.basename(model_id))
    else:
        model = RBLNT5EncoderModel.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )

    target_sentences = [prompt] * batch_size
    inputs = tokenizer(target_sentences, return_tensors="pt", padding="max_length", max_length=512)

    outputs = model(**inputs)
    print(outputs)


if __name__ == "__main__":
    fire.Fire(main)
