# Chronos-2 forecasting

Install the forecasting extra alongside the RBLN compiler and runtime:

```bash
pip install 'optimum-rbln[chronos]' --extra-index-url https://download.pytorch.org/whl/cpu
```

Load `amazon/chronos-2` explicitly with `RBLNChronos2Model`. The default configuration
compiles batch 1 at the checkpoint's maximum context 8192 and forecast horizon 1024.
Normalization, patching and inverse scaling use native Chronos preprocessing on CPU;
all learned layers and the quantile prediction head execute on the NPU.

```python
import torch
from optimum.rbln import RBLNChronos2Model

model = RBLNChronos2Model.from_pretrained("amazon/chronos-2", export=True)
context = torch.sin(torch.arange(8192, dtype=torch.float32) / 10).unsqueeze(0)
forecast = model(context=context).quantile_preds  # [1, 21, 1024]
model.save_pretrained("compiled_chronos2")
model = RBLNChronos2Model.from_pretrained("compiled_chronos2")
```

Smaller batches and shorter histories are padded to the compiled capacities.
Groups share information only when their `group_ids` are equal. NaNs represent
missing observations and unknown future covariates; explicit masks are supported.
Inputs longer than the compiled context retain their most recent observations.

Compile the forecast length your application needs. Chronos predicts complete
16-step patches, so length 36 compiles three patches and direct forward returns
48 steps. A native `Chronos2Pipeline` returns the requested 36 steps:

```python
from chronos import Chronos2Pipeline

model = RBLNChronos2Model.from_pretrained(
    "amazon/chronos-2",
    export=True,
    rbln_batch_size=4,
    rbln_context_length=512,
    rbln_prediction_length=36,
)
pipeline = Chronos2Pipeline(model)
inputs = torch.sin(torch.arange(257, dtype=torch.float32) / 10).reshape(1, 1, -1)
forecast = pipeline.predict(inputs, prediction_length=36, context_length=512, batch_size=4)
```

Each compiled model accepts its exact future-patch count. Passing a different
`num_output_patches` raises an error; compile another configuration for that
forecast length. Additional future tokens change attention results, so cropping
a larger forecast is not equivalent. Compiled inference returns quantile predictions;
training targets and attention-weight outputs are unsupported.
