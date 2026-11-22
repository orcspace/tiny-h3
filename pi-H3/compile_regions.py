"""Load weight-free SM120 artifacts without invoking a compiler."""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from spaces.zero.torch.aoti import aoti_load_from_module_dir

from compile_regions_build import layout, prepare_regions
from lakonlab.ops.sol_attn import load_sol_attention_aot


def prepare_compiled_models(pipes, text_pipe):
    regions, bindings = prepare_regions(pipes, text_pipe)
    root = Path(__file__).resolve().with_name('artifacts-v1')
    if not root.is_dir():
        root = Path(snapshot_download(
            os.environ['SPACE_ID'], repo_type='space', revision='sm120-artifacts',
            allow_patterns=['artifacts-v1/**'])) / 'artifacts-v1'
    groups = {}
    for kind, module in regions:
        groups.setdefault(layout(kind, module), []).append(module)
    missing = [key for key in groups if not (root/'regions'/key/'root/package.pt2').is_file()]
    if missing or not (root/'attention/manifest.json').is_file():
        raise RuntimeError(f'Compiled SM120 artifacts are missing from {root}. Missing regions: {missing}')
    for key, modules in groups.items():
        aoti_load_from_module_dir(modules, root/'regions'/key/'root')
        for module in modules:
            module.forward.weights.update(dict(module.named_buffers()))
    load_sol_attention_aot(root/'attention')
    for module, name, call in bindings:
        setattr(module, name, call)
    encoder = pipes[0].vae.encoder
    compiled_encoder = encoder.forward
    encoder.forward = lambda features: compiled_encoder(features.contiguous())
    print(f'Loaded {len(groups)} SM120 regional packages and the Sol BSA bundle', flush=True)
