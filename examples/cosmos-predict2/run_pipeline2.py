from optimum.rbln.diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import RBLNCosmos2_5_PredictBasePipeline


pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
    "nvidia/Cosmos-Predict2.5-2B",
    revision="diffusers/base/pre-trained",
    export=True,
    rbln_config={
        "text_encoder": {
            "visual": {
                "max_seq_lens": 8192,
            },
            "tensor_parallel_size": 8,
            # "kvcache_partition_len": 16_384,
            "max_seq_len": 512,
        },
        "safety_checker": {"device": 8, "qwen3guard": {"tensor_parallel_size": 1}},
    },
)

pipe.save_pretrained("cosmos-predict2.5_vae")

# pipe = RBLNCosmos2_5_PredictBasePipeline.from_pretrained(
#     "cosmos-predict2.5",
#     export=False,
#     rbln_config = {
#         "text_encoder": {
#             "visual": {
#                 "device": 0,
#             },
#             "device": [0,1,2,3,4,5,6,7],
#         },
#     }
# )
