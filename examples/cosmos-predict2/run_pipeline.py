from diffusers.utils import export_to_video
from optimum.rbln.diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import RBLNCosmos2_5_PredictBasePipeline
from PIL import Image

# pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
#     "nvidia/Cosmos-Predict2.5-2B",
#     revision="diffusers/base/pre-trained",
#     export=True,
#     rbln_config={
#         "text_encoder": {
#             "visual": {
#                 "max_seq_len": 2048,
#             },
#             "tensor_parallel_size": 8,
#             # "kvcache_partition_len": 16_384,
#             "max_seq_len": 512,
#         },
#         # "transformer": {
#         #     "num_devices": 8,
#         # },
#         # "safety_checker": {"device": 8, "qwen3guard": {"tensor_parallel_size": 1}},
#     },
# )

# pipe.save_pretrained("cosmos-predict2.5_text_encoder")

pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
    # "cosmos-predict2.5_text_encoder",
    # "cosmos-predict2.5_core_compile_te_tm_vae",
    "cosmos-predict2.5_core_compile_vae",
    export=False,
    rbln_config={
        # "text_encoder": {
        #     "visual": {
        #         "device": 16,
        #     },
        #     "device": [16, 17, 18, 19, 20, 21, 22, 23],
        # },
        # "transformer": {
        #     "device": [24, 25, 26, 27, 28, 29, 30, 31],
        # },
        "vae": {
            "device_map": {"encoder": 0, "decoder": 1},
        },
    },
)

input_text = "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
image_path = "bus_terminal.jpg"
input_image = Image.open(image_path)

video = pipe(image=input_image, prompt=input_text, height=704, width=1280).frames[0]
export_to_video(video, "bus_terminal.mp4", fps=16)
print(f"Saved {len(video)} frames to output.mp4")
