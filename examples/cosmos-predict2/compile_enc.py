import importlib

import rebel
import set_environments  # noqa: F401
import torch
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import override


CACHE_SIZE_0 = [
    # [1, 3, 1, 704* 1280], # first cache
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 1, 176, 320],
    [1, 192, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 1, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
]

CACHE_SIZE_N = [
    # [1, 3, 2, 704, 1280] , # first cache
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 1, 176, 320],
    [1, 192, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 1, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
]


class WrapperE0(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache = self.encoder(x, feat_cache=[], feat_idx=feat_idx, generate_cache=True)
        return out, feat_cache


class RBLNWrapperE0(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.cache_dims = CACHE_SIZE_0

    def forward(self, x, *args):
        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache = self.encoder(x, feat_cache=[], feat_idx=feat_idx, generate_cache=True)

        uncached = feat_cache[0]

        # post-process: update rbln cache tensors
        dummy_outs = []
        position = torch.tensor(
            0, dtype=torch.int16
        )  # 0 is dummy value -> first output of next chunk have to slice out this frame
        axis = torch.tensor(2, dtype=torch.int16)
        # axis = torch.tensor(0, dtype=torch.int16)
        for cache, feat_cache_item, cache_dim in zip(list(args[1:]), feat_cache[1:], self.cache_dims):
            n, c, d, h, w = feat_cache_item.shape
            feat_cache_item = feat_cache_item.reshape(n, c, d, -1)
            if cache_dim[2] == 2:
                # feat_cache_item = torch.concat([torch.zeros_like(feat_cache_item), feat_cache_item], dim=2) # NOTE(seinpark): for 2-frame cache, pad one frame ealrlier
                feat_cache_item = torch.nn.functional.pad(feat_cache_item, (0, 0, 1, 0))  # pad one frame earlier

            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_item, position, axis)
            dummy_outs.append(dummy_out)
            print(cache.shape, feat_cache_item.shape)
        return out, uncached, dummy_outs


class WrapperEN(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, *feat_cache):
        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache = self.encoder(x, feat_cache=list(feat_cache), feat_idx=feat_idx)
        print(out.shape)
        return out, feat_cache


class RBLNWrapperEN(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.cache_dims = CACHE_SIZE_N

    def forward(self, x, uncached, *feat_cache):
        # num_cache_dim = uncached.shape[2]
        feat_cache_reshaped = []
        feat_cache_reshaped.append(uncached)
        print(uncached.shape)

        # pre-process: reshape rbln cache tensors to torch layout
        for cache, cache_dim in zip(list(feat_cache), self.cache_dims):
            reshaped_cache = cache.reshape(*cache_dim)  # n c d (hw) -> n c d h w
            feat_cache_reshaped.append(reshaped_cache)
            print(reshaped_cache.shape)

        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache_e1 = self.encoder(x, feat_cache=feat_cache_reshaped, feat_idx=feat_idx)

        # post-process: update rbln cache tensors
        uncached = feat_cache_e1[0]
        dummy_outs = []
        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(2, dtype=torch.int16)
        print("E1 Updating caches...")
        for cache, feat_cache_e1_item in zip(list(feat_cache), feat_cache_e1[1:]):
            n, c, d, h, w = feat_cache_e1_item.shape
            feat_cache_e1_item = feat_cache_e1_item.reshape(n, c, d, -1)
            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_e1_item, position, axis)
            dummy_outs.append(dummy_out)
            print(cache.shape, feat_cache_e1_item.shape)
        return out, uncached, dummy_outs


def update_config(config):
    config.model.config.precision = "float32"
    config.model.config.text_encoder_config.model_config.model_config.attn_implementation = "sdpa"
    config.model.config.text_encoder_config.model_config.model_config.vision_config.attn_implementation = "sdpa"


def get_dummy_inputs(input_info, fill=0, static_tensors={}, meta_tensor_names=[]):
    dummy = []
    for name, shape, dtype in input_info:
        if name in static_tensors:
            tensor = static_tensors[name]
            print(f"Using static tensor for {name} with shape {list(tensor.shape)} and dtype {tensor.dtype}")
            assert shape == list(tensor.shape), f"Expected compiled shape: {shape}, tensor shape: {list(tensor.shape)}"
            if shape != list(tensor.shape):
                raise RuntimeError(f"Different shape for dummy inputs. ({shape} != {list(tensor.shape)})")
            if getattr(torch, dtype) != tensor.dtype:
                raise RuntimeError(f"Different dtype for dummy inputs ({dtype} != {tensor.dtype})")
            dummy.append(tensor)
        else:
            if name in meta_tensor_names:
                device = "meta"
            else:
                device = "cpu"

            dummy.append(
                torch.fill(torch.empty(*shape, dtype=getattr(torch, dtype), device=torch.device(device)), fill)
                if len(shape) > 0
                else torch.tensor(fill, dtype=getattr(torch, dtype), device=torch.device(device))
            )
    return tuple(dummy)


CHUNK_SIZE = 4
PADDED_FRAME_OF_FIRST = 2


def main():
    experiment_name = "Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt"
    s3_checkpoint_dir = "/mnt/shared_data/groups/sw_dev/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/15a82a2ec231bc318692aa0456a36537c806e7d4/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
    enable_fsdp = False
    load_ema_to_reg = True
    instantiate_ema = True
    config_file = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
    experiment_opts = ["~data_train"]

    config_module = config_file.replace("/", ".").replace(".py", "")
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    update_config(config)

    # Override checkpoint path if provided
    if s3_checkpoint_dir:
        config.checkpoint.load_path = str(s3_checkpoint_dir)

    if load_ema_to_reg:
        config.model.config.ema.enabled = False

    if instantiate_ema is False and config.model.config.ema.enabled:
        config.model.config.ema.enabled = False
    # config.model.config.net.atten_backend = "torch"

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    if not enable_fsdp:
        # disable fsdp
        config.model.config.fsdp_shard_size = 1

    tokenizer = instantiate(config.model.config.tokenizer)
    tokenizer.model.model.encoder = tokenizer.model.model.encoder.to(dtype=torch.float32)

    # # x = torch.load("vae_encoder_input.pt").to("cpu", dtype=torch.float32)
    x = torch.zeros(1, 3, 93, 704, 1280)

    INPUT_0 = torch.randn(1, 3, 1, 704, 1280)
    INPUT_N = torch.randn(1, 3, CHUNK_SIZE, 704, 1280)

    # tokenizer.model.model.clear_cache()
    # encoder_e0 = WrapperE0(tokenizer.model.model.encoder)
    # encoder_e0.eval()

    # encoder_en = WrapperEN(tokenizer.model.model.encoder)
    # encoder_en.eval()

    # print("forward enc 0")
    # output, cache_0 = encoder_e0(INPUT_0, )
    # print("forward enc N")
    # output, cache_n = encoder_en(INPUT_N, *cache_0)
    # import pdb; pdb.set_trace()
    # cache_shapes = []
    # for c in CACHE_SIZE_0:
    #     cache_shapes.append(c.shape)

    # compiled_e0 = rebel.compile_from_torch(
    #     encoder_e0,
    #     input_info=[("x", [1, 3, 16, 704, 1280], "float32"),],
    # )

    # ###### RBLN compile start ######

    from rebel.compile_context import CompileContext

    tokenizer.model.model.clear_cache()
    encoder_e0 = RBLNWrapperE0(tokenizer.model.model.encoder)
    encoder_e0.eval()

    context = CompileContext(use_weight_sharing=False)

    enc0_input_info = [
        ("x", list(INPUT_0.shape), "float32"),
    ]
    enc0_input_info.append(("uncached", [1, 3, PADDED_FRAME_OF_FIRST, 704, 1280], "float32"))  # uncached from E0

    for i, shape in enumerate(CACHE_SIZE_0):
        shape = [*shape[:3], shape[-2] * shape[-1]]  # N C D HW
        enc0_input_info.append((f"feat_cache_{i}", shape, "float32"))

    enc0_example_inputs = get_dummy_inputs(enc0_input_info, fill=0)

    # Mark encoder's static tensors (cache states)
    static_tensors = {}
    for (name, _, _), tensor in zip(enc0_input_info, enc0_example_inputs):
        if "feat_cache" in name:
            static_tensors[name] = tensor
            context.mark_static_address(tensor)

    enc1_input_info = [
        ("x", [1, 3, CHUNK_SIZE, 704, 1280], "float32"),
    ]
    enc1_input_info.append(("uncached", [1, 3, PADDED_FRAME_OF_FIRST, 704, 1280], "float32"))  # uncached from E0

    for i, shape in enumerate(CACHE_SIZE_N):
        shape = [*shape[:3], shape[-2] * shape[-1]]  # N C D HW
        enc1_input_info.append((f"feat_cache_{i}", shape, "float32"))

    enc1_example_inputs = get_dummy_inputs(enc1_input_info, fill=0, static_tensors=static_tensors)
    # Mark decoder's static tensors (self kv states)
    for (name, _, _), tensor in zip(enc1_input_info, enc1_example_inputs):
        if "feat_cache" in name:
            context.mark_static_address(tensor)

    # print("================================================================================================================================")
    # print("Compiling E0...")
    # compiled_e0 = rebel.compile_from_torch(
    #     encoder_e0,
    #     input_info=enc0_input_info,
    #     example_inputs=enc0_example_inputs,
    #     compile_context=context,
    # )

    # runtime_e0 = compiled_e0.create_runtime(device=0, tensor_type="pt")

    # out = runtime_e0(x[:, :, :1])
    # out_rbln = out[0]
    # import pdb; pdb.set_trace()
    # feat_cache_rbln = out[1:]

    print(
        "================================================================================================================================"
    )
    print("Compiling E1...")
    encoder_en = RBLNWrapperEN(tokenizer.model.model.encoder)

    compiled_en = rebel.compile_from_torch(
        encoder_en,
        input_info=enc1_input_info,
        example_inputs=enc1_example_inputs,
        compile_context=context,
    )


if __name__ == "__main__":
    main()
