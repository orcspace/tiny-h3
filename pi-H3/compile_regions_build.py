"""Build dynamic, weight-free regional packages for the ZeroGPU demo."""

import copy
import os
from contextlib import ExitStack
from unittest.mock import patch
from types import MethodType
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import spaces
import torch
from huggingface_hub import HfApi
from spaces.zero.torch.aoti import aoti_compile_and_save
from transformers.utils.output_capturing import maybe_install_capturing_hooks

from lakonlab.ops.sol_attn.aot import versions
from vision_attention import vision_attention


REGIONS = []


class RefinerLayerRegion(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model.forward(features, attention_kwargs=None)


class TextLayerRegion(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, position_embeddings):
        return self.model.forward(features, position_embeddings, attention_mask=None, use_cache=False)


class VisionLayerRegion(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, cu_seqlens, position_embeddings):
        return self.model.forward(features, cu_seqlens, position_embeddings)


def examples(kind, module):
    if kind == 'dit':
        tokens = torch.export.Dim('tokens', min=256, max=131072)
        levels = torch.export.Dim('levels', min=2, max=16)
        hidden = module.attn.to_q.in_features
        rank = module.adaln_proj.linear.in_features
        # The released pruned model uses rank-8 indexed time modulation.
        args = (torch.randn(1, 4096, hidden, device='cuda', dtype=torch.bfloat16),
                torch.zeros(2, rank, device='cuda', dtype=torch.bfloat16),
                torch.zeros(4096, device='cuda', dtype=torch.long),
                _rope(torch.zeros(4096, 3, device='cuda', dtype=torch.long)),
                dict(sol_prefix=torch.tensor(1)))
        shapes = ({1: tokens}, {0: levels}, {0: tokens}, ({0: tokens}, {0: tokens}), dict(sol_prefix={}))
    elif kind.startswith('sol_refiner'):
        from lakonlab.models.architectures.sol_h3.model import DIM, NUM_REGISTER_TOKENS

        x = torch.randn(1, 12, 48, 84, DIM, dtype=torch.bfloat16, device='cuda')
        g = x.new_zeros(1, 3, NUM_REGISTER_TOKENS + 1, DIM)
        shifted = kind == 'sol_refiner_shifted'
        extra = next(args for _, args in _sol_refiner.block_args(x.shape, g.shape, x.device)
                     if args[0] == shifted)
        _, global_indices, window_indices = extra
        frames = torch.export.Dim('frames', min=2, max=256)
        height = torch.export.Dim('height', min=16, max=256)
        width = torch.export.Dim('width', min=16, max=256)
        blocks = torch.export.Dim('blocks', min=1, max=64)
        global_shapes = tuple(
            {0: blocks, 1: torch.export.Dim.AUTO} if rows.ndim == 2 else {0: torch.export.Dim.AUTO}
            for rows in global_indices)
        window_shapes = tuple({0: torch.export.Dim.AUTO} for _ in window_indices)
        args = (x, g, *extra)
        shapes = ({1: frames, 2: height, 3: width}, {1: blocks}, None, global_shapes, window_shapes)
    elif kind.startswith('refiner'):
        single = kind.endswith('_single')
        tokens = None if single else torch.export.Dim('text_tokens', min=2, max=8192)
        args = (torch.randn(1, 1 if single else 256, module.model.attn.to_q.in_features,
                            device='cuda', dtype=torch.bfloat16),)
        shapes = ({1: tokens} if tokens is not None else {},)
    elif kind.startswith('text'):
        single = kind.endswith('_single')
        tokens = None if single else torch.export.Dim('text_tokens', min=2, max=8192)
        length = 1 if single else 256
        hidden, head_dim = module.model.self_attn.q_proj.in_features, module.model.self_attn.head_dim
        features = torch.randn(1, length, hidden, device='cuda', dtype=torch.bfloat16)
        rotary = (torch.ones(1, length, head_dim, device='cuda', dtype=torch.bfloat16),
                  torch.zeros(1, length, head_dim, device='cuda', dtype=torch.bfloat16))
        shape = {} if single else {1: tokens}
        args, shapes = (features, rotary), (shape, (shape, shape))
    elif kind.startswith('vision_'):
        images = int(kind[-1])
        tokens = images * torch.export.Dim('image_patches', min=2, max=8192)
        hidden, head_dim = module.model.attn.qkv.in_features, module.model.attn.head_dim
        features = torch.randn(256, hidden, device='cuda', dtype=torch.bfloat16)
        cu_seqlens = torch.arange(images + 1, device='cuda', dtype=torch.int32) * (256 // images)
        phase = _vision_rope(torch.zeros(256, 2, device='cuda', dtype=torch.long))
        phase = torch.cat((phase, phase), dim=-1)
        rotary = (phase.cos(), phase.sin())
        args, shapes = (features, cu_seqlens, rotary), ({0: tokens}, {}, ({0: tokens}, {0: tokens}))
    elif kind == 'audio':
        time = torch.export.Dim('audio_time', min=64, max=640)
        args = (torch.zeros(2, module.conv_pre.in_channels, 200, device='cuda', dtype=torch.float32),)
        shapes = ({2: time},)
    else:
        vae = _vae
        if kind == 'video_decoder':
            with spaces.aoti_capture(module) as call:
                vae.decode(torch.zeros(1, 24, 37, 48, 84, device='cuda', dtype=torch.float16))
            args = call.args
            batch = torch.export.Dim('tile_batch', min=1, max=28)
            shapes = ({0: batch},)
        else:
            with spaces.aoti_capture(module) as call:
                vae.encode(torch.zeros(1, 3, 1, 768, 1344, device='cuda', dtype=torch.float16))
            args = tuple(tensor.contiguous() for tensor in call.args)
            shapes = ({},)
    return args, {}, shapes


def layout(kind, module):
    scalar = lambda v: isinstance(v, (bool, int, float, str, type(None))) or (
        isinstance(v, (tuple, list, dict)) and all(map(scalar, v.values() if isinstance(v, dict) else v)))
    modules = [*module.modules()]
    if kind == 'dit' or kind.startswith('sol_refiner'):
        modules.append(module.attn.processor)
    settings = [(type(m).__qualname__, sorted((k, repr(v)) for k, v in vars(m).items() if scalar(v) and not k.startswith('_forward_') and not (kind.startswith('text') and k == 'layer_idx'))) for m in modules]
    tensors = [(name, tuple(t.shape), str(t.dtype)) for name,t in module.state_dict().items()]
    source = hashlib.sha256()
    for path in sorted((Path(__file__).parent/'lakonlab').rglob('*.py')):
        source.update(path.read_bytes())
    source.update(Path(__file__).read_bytes())
    source.update(Path(__file__).with_name('vision_attention.py').read_bytes())
    return hashlib.sha256(repr((kind, settings, tensors, source.hexdigest(), versions())).encode()).hexdigest()[:24]


@spaces.GPU(size='xlarge', duration=600)
def compile_region(index, directory):
    kind, module = REGIONS[index]
    print('COMPILING_REGION', kind, directory, flush=True)
    args, kwargs, dynamic_shapes = examples(kind, module)
    with torch.no_grad(), ExitStack() as stack:
        for child in module.modules():
            stack.enter_context(patch.object(child, '_forward_hooks', {}))
        exported = torch.export.export(module, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes,
                                       strict=not kind.startswith('sol_refiner'),
                                       prefer_deferred_runtime_asserts_over_guards=True)
        aoti_compile_and_save(directory, exported)
    print('COMPILED_REGION', kind, directory, flush=True)


@spaces.GPU(size='xlarge', duration=120)
def warm_runtime_kernels(directory, widths):
    import triton

    from lakonlab.ops.activation import silu_mul_
    from lakonlab.ops.quant.quant import quantize_fp8_rowwise

    with triton.knobs.cache.scope():
        triton.knobs.cache.dir = directory
        for width in widths:
            quantize_fp8_rowwise(torch.zeros(17, width, device='cuda', dtype=torch.bfloat16))
        for gate_first in (True, False):
            silu_mul_(torch.zeros(17, 8192, device='cuda', dtype=torch.bfloat16), gate_first)
        torch.cuda.synchronize()



def prepare_regions(pipes, text_pipe):
    text_pipe.text_encoder.set_attn_implementation('sdpa')
    maybe_install_capturing_hooks(text_pipe.text_encoder.model.language_model)
    for block in text_pipe.text_encoder.model.visual.blocks:
        block.attn.forward = MethodType(vision_attention, block.attn)
    regions, bindings = [], []
    for pipe in pipes:
        regions.extend(('dit', block) for block in pipe.transformer.transformer_blocks)
        for block in pipe.transformer.token_refiner.refiner_blocks:
            variants = [RefinerLayerRegion(block) for _ in range(2)]
            regions.extend(zip(('refiner_single', 'refiner'), variants))
            def refiner_call(hidden_states, attention_kwargs=None, variants=variants):
                return variants[hidden_states.size(1) != 1](hidden_states)
            bindings.append((block, 'forward', refiner_call))
        refiner = getattr(pipe.transformer, 'dwd_pixel_refiner', None)
        if refiner is not None:
            from lakonlab.models.architectures.sol_h3.model import NUM_LAYERS

            regions.extend(('sol_refiner_shifted' if index % 2 else 'sol_refiner', block)
                           for index, block in enumerate(refiner.blocks, NUM_LAYERS - len(refiner.blocks)))
    for block in text_pipe.text_encoder.model.language_model.layers:
        variants = [TextLayerRegion(block) for _ in range(2)]
        regions.extend(zip(('text_single', 'text'), variants))
        def text_call(features, position_embeddings=None, attention_mask=None, position_ids=None,
                      past_key_values=None, use_cache=False, variants=variants, **kwargs):
            return variants[features.size(1) != 1](features, position_embeddings)
        bindings.append((block, 'forward', text_call))
    for block in text_pipe.text_encoder.model.visual.blocks:
        variants = [VisionLayerRegion(copy.copy(block)) for _ in range(2)]
        regions.extend((f'vision_{index + 1}', region) for index, region in enumerate(variants))
        def vision_call(features, cu_seqlens, position_embeddings=None, variants=variants, **kwargs):
            return variants[cu_seqlens.numel() - 2](features, cu_seqlens, position_embeddings)
        bindings.append((block, 'forward', vision_call))
    regions.extend([('audio', pipes[0].audio_vae.decoder),
                    ('video_decoder', pipes[0].vae.decoder), ('video_encoder', pipes[0].vae.encoder)])
    return regions, bindings


def build_regions(pipes, text_pipe):
    global _vae, _rope, _vision_rope, _sol_refiner
    _vae = pipes[0].vae
    _rope = pipes[0].transformer.rope
    _vision_rope = text_pipe.text_encoder.model.visual.rotary_pos_emb
    _sol_refiner = pipes[2].transformer.dwd_pixel_refiner
    REGIONS[:] = prepare_regions(pipes, text_pipe)[0]
    cache = Path('/tmp/pi-h3-aoti')
    cache.mkdir(exist_ok=True)
    repo = os.environ['SPACE_ID']
    api = HfApi()
    repo_args = dict(repo_id=repo, repo_type='space', revision='sm120-artifacts')
    existing = set(api.list_repo_files(**repo_args))
    done = set()
    for index, (kind, module) in enumerate(REGIONS):
        key = layout(kind, module)
        if key not in done:
            remote = 'artifacts-v1/regions/' + key
            if remote + '/root/package.pt2' not in existing:
                directory = cache/key
                compile_region(index, str(directory))
                api.upload_folder(**repo_args, folder_path=directory, path_in_repo=remote,
                                  commit_message='Build pi-H3 1.0 SM120 '+kind)
            done.add(key)
    from lakonlab.ops.quant import FrozenFP8Linear

    widths = sorted({module.in_features for root in [text_pipe.text_encoder, *(pipe.transformer for pipe in pipes)]
                     for module in root.modules() if isinstance(module, FrozenFP8Linear)})
    with TemporaryDirectory(prefix='pih3-runtime-kernels-') as directory:
        warm_runtime_kernels(directory, widths)
        api.upload_folder(**repo_args, folder_path=directory, path_in_repo='artifacts-v1/attention/triton_cache',
                          ignore_patterns=['**/*.lock'], commit_message='Cache pi-H3 1.0 runtime Triton kernels')
    print(f'SM120 regional packages ready: {len(done)}', flush=True)
