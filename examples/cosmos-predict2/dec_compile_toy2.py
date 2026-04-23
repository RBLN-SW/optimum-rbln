import rebel
import torch
from torch.nn.attention import sdpa_kernel
from optimum.rbln import RBLNQwen2_5_VLForConditionalGeneration
# import set_environments  # noqa: F401

from utils import (
    fake_cuda,
    _wrap_tensor_factory,
    _wrap_load,
    _wrap_factory_func,
    _wrap_autocast,
    _wrap_sdpa_kernel,
    _wrap_module_to,
    _FakeCudaModule,
)

original_Tensor_cuda = torch.Tensor.cuda
original_tensor = torch.tensor
original_torch_load = torch.load
original_autocast = torch.amp.autocast
original_cuda = torch.cuda
original_sdpa_kernel = sdpa_kernel

original_factory_func = []
for _name in ["zeros", "ones", "randn", "rand", "empty", "full", "empty_like"]:
    if hasattr(torch, _name):
        original_factory_func.append(getattr(torch, _name))
        setattr(torch, _name, _wrap_factory_func(getattr(torch, _name)))

if hasattr(torch.nn.Module, "to"):
    original_torch_nn_module_to = torch.nn.Module.to
    torch.nn.Module.to = _wrap_module_to(torch.nn.Module.to)

torch.Tensor.cuda = fake_cuda
torch.tensor = _wrap_tensor_factory(torch.tensor)
torch.load = _wrap_load(torch.load)
torch.amp.autocast = _wrap_autocast(torch.amp.autocast)
torch.cuda = _FakeCudaModule()
torch.nn.attention.sdpa_kernel = _wrap_sdpa_kernel(sdpa_kernel)


CACHE_SIZE_0 = [    
    # padded
    [1, 16, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 192, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
]

CACHE_SIZE_N = [
    [1, 16, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 384, 2, 88, 160],
    [1, 192, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 384, 2, 176, 320],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 192, 2, 352, 640],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
    [1, 96, 2, 704, 1280],
]


class WrapperD0(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache = self.decoder(x[:, :, :1], feat_cache=[], feat_idx=feat_idx, generate_cache=True)
        return out, feat_cache

class RBLNWrapperD0(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.cache_dims = CACHE_SIZE_0

    def forward(self, x, *args):
        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache = self.decoder(x, feat_cache=[], feat_idx=feat_idx, generate_cache=True)
        # uncached = feat_cache[0]
        
        # post-process: update rbln cache tensors
        dummy_outs = []
        position = torch.tensor(0, dtype=torch.int16) # 0 is dummy value -> first output of next chunk have to slice out this frame
        axis = torch.tensor(2, dtype=torch.int16)
        # for cache, feat_cache_item, cache_dim in zip(list(args), feat_cache[1:], self.cache_dims):
        for cache, feat_cache_item, cache_dim in zip(list(args), feat_cache, self.cache_dims):
            n, c, d, h, w = feat_cache_item.shape
            feat_cache_item = feat_cache_item.reshape(n, c, d, -1)
            # feat_cache_item = torch.concat([torch.zeros_like(feat_cache_item), feat_cache_item], dim=2) # NOTE(seinpark): for 2-frame cache, pad one frame ealrlier
            feat_cache_item = torch.nn.functional.pad(feat_cache_item, (0,0,1,0)) # pad one frame earlier
                
            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_item, position, axis)
            dummy_outs.append(dummy_out)
            # print(cache.shape, feat_cache_item.shape)
        return out, dummy_outs
        return out, uncached, dummy_outs

class WrapperDN(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x, *feat_cache):
        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache = self.decoder(x, feat_cache=list(feat_cache), feat_idx=feat_idx)
        return out, feat_cache

class RBLNWrapperDN(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.cache_dims = CACHE_SIZE_N

    def forward(self, x, *feat_cache):
        # num_cache_dim = uncached.shape[2]
        feat_cache_reshaped = []

        # pre-process: reshape rbln cache tensors to torch layout
        for cache, cache_dim in zip(list(feat_cache), self.cache_dims):
            cache = cache.reshape(*cache_dim) # n c d (hw) -> n c d h w
            feat_cache_reshaped.append(cache)
            # print(cache.shape)

        feat_idx = torch.zeros(1, dtype=torch.int32)
        out, feat_cache_dn = self.decoder(x, feat_cache=feat_cache_reshaped, feat_idx=feat_idx)
        
        # post-process: update rbln cache tensors
        dummy_outs = []
        position = torch.tensor(0, dtype=torch.int16)
        axis = torch.tensor(2, dtype=torch.int16)
        print("E1 Updating caches...")
        for cache, feat_cache_dn_item in zip(list(feat_cache), feat_cache_dn):
            n, c, d, h, w = feat_cache_dn_item.shape
            feat_cache_dn_item = feat_cache_dn_item.reshape(n, c, d, -1)
            dummy_out = torch.ops.rbln_custom_ops.rbln_cache_update(cache, feat_cache_dn_item, position, axis)
            dummy_outs.append(dummy_out)
            # print(cache.shape, feat_cache_dn_item.shape)
        return out, dummy_outs


def get_dummy_inputs(
    input_info, fill=0, static_tensors = {}, meta_tensor_names = []
):
    dummy = []
    for name, shape, dtype in input_info:
        if name in static_tensors:
            tensor = static_tensors[name]
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


def main():
    from rbln_models.wan2pt1 import Wan2pt1VAEInterface
    config = {'chunk_duration': 81, 'load_mean_std': False, 'name': 'wan2pt1_tokenizer', '_target_': Wan2pt1VAEInterface, 'temporal_window': 16}
    tokenizer = config["_target_"](config)
    
    tokenizer.model.model.decoder = tokenizer.model.model.decoder.to(dtype=torch.float32)
    tokenizer.model.model.conv2 = tokenizer.model.model.conv2.to(dtype=torch.float32)

    z = torch.randn(1, 16, 24, 88, 160)
    x = tokenizer.model.model.conv2(z)
    
    from rebel.compile_context import CompileContext
    
    tokenizer.model.model.clear_cache()
    decoder_e0 = RBLNWrapperD0(tokenizer.model.model.decoder)
    decoder_e0.eval()
    
    context = CompileContext(use_weight_sharing=False)
    
    dec0_input_info = [("x", list(x[:, :, :1].shape), "float32"),]
    for i, shape in enumerate(CACHE_SIZE_0):
        shape = [*shape[:3], shape[-2]*shape[-1]]  # N C D HW
        dec0_input_info.append((f"feat_cache_{i}", shape, "float32"))

    dec0_example_inputs = get_dummy_inputs(dec0_input_info, fill=0)

    # Mark encoder's static tensors (cache states)
    static_tensors = {}
    for (name, _, _), tensor in zip(dec0_input_info, dec0_example_inputs):
        if "feat_cache" in name:
            static_tensors[name] = tensor
            context.mark_static_address(tensor)

    dec1_input_info = [("x", list(x[:, :, 1:2].shape), "float32"),]
    
    for i, shape in enumerate(CACHE_SIZE_N):
        shape = [*shape[:3], shape[-2]*shape[-1]]  # N C D HW
        dec1_input_info.append((f"feat_cache_{i}", shape, "float32"))

    dec1_example_inputs = get_dummy_inputs(dec1_input_info, fill=0, static_tensors=static_tensors)

    # Mark decoder's static tensors (self kv states)
    for (name, _, _), tensor in zip(dec1_input_info, dec1_example_inputs):
        if "feat_cache" in name:
            context.mark_static_address(tensor)

    # print("================================================================================================================================")
    # print("Compiling E0...")
    # compiled_e0 = rebel.compile_from_torch(
    #     decoder_e0,
    #     input_info=dec0_input_info,
    #     example_inputs=dec0_example_inputs,
    #     compile_context=context,
    # )
    # compiled_e0.save("decoder_e0_opt.rbln")
    
    print("================================================================================================================================")
    print("Compiling E1...")
    decoder_dn = RBLNWrapperDN(tokenizer.model.model.decoder)

    compiled_dn = rebel.compile_from_torch(
        decoder_dn,
        input_info=dec1_input_info,
        example_inputs=dec1_example_inputs,
        compile_context=context,
    )
    compiled_dn.save("decoder_dn_opt.rbln")


if __name__ == "__main__":
    main()