import hashlib
import json
import math
import os
import random
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import spaces
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from gradio.context import LocalContext
from gradio.utils import get_upload_folder
from PIL import Image, ImageOps
from lakonlab.models.architectures import RGBColorEncoder
from lakonlab.models.architectures.diffusers.attention import set_attention_backends
from lakonlab.models.architectures.diffusers.pretrained import AutoencoderKLMiniMaxH3
from lakonlab.models.architectures.diffusers.minimax_h3 import MiniMaxH3SolAttnProcessor
from lakonlab.models.diffusions.schedulers import FlowEulerODEScheduler
from lakonlab.ops import quantize_frozen_linears
from lakonlab.ops.quant import FrozenFP8Linear
from lakonlab.pipelines.pipeline_minimax_h3 import MiniMaxH3BlocksMod, MiniMaxH3DWDBlocks
from lakonlab.pipelines.utils import load_lakonlab_adapter
from lakonlab.ui.gradio.create_img_to_video import create_interface_img_to_video
from lakonlab.utils.io_utils import rgb_to_yuv420p, save_video_audio
from starlette.requests import Request as StarletteRequest

# gradio's component_server route (polled by the workflow canvas) passes a raw starlette Request to
# get_function_with_locals, while ZeroGPU's patch of that function reads request.request; hand it a gr.Request
_get_function_with_locals = gr.utils.get_function_with_locals


def _get_function_with_locals_gr_request(*args, request=None, **kwargs):
    if isinstance(request, StarletteRequest):
        request = gr.Request(request=request)
    return _get_function_with_locals(*args, request=request, **kwargs)


gr.utils.get_function_with_locals = _get_function_with_locals_gr_request

# ZeroGPU forks its GPU worker on the first call and reuses it, and the worker keeps the gradio app (Blocks) of the
# call that forked it. With three apps in this process (demo, studio, workflow), the progress of a later call from
# another app went to the forking app's queue, so the bar never moved. The contextualizer ZeroGPU sends with every
# call now names the calling app, looked up in apps registered before the first fork (see launch_demo).
_BLOCKS_BY_ID = {}
try:
    import spaces.zero.gradio as _zerogpu_gradio
except ImportError:
    _zerogpu_gradio = None
if _zerogpu_gradio is not None:
    class _WorkerContextualizer(_zerogpu_gradio.WorkerContextualizer):
        def __init__(self):
            super().__init__()
            self.blocks_id = id(LocalContext.blocks.get(None))

        def __call__(self):
            super().__call__()
            if self.blocks_id in _BLOCKS_BY_ID:
                LocalContext.blocks.set(_BLOCKS_BY_ID[self.blocks_id])

    _zerogpu_gradio.WorkerContextualizer = _WorkerContextualizer

ADAPTER_REPO = 'Lakonik/pi-H3'
ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_SEED = 2**31 - 1
ASPECT_RATIOS = ('21:9', '16:9', '4:3', '1:1', '3:4', '9:16')

DEFAULT_PROMPT = ('integrated_multimodal_description: [Shot 1] Live-action, handheld smartphone selfie video, a close '
                  "medium shot with the camera held at arm's length slightly above eye level, a young man in an orange "
                  'beanie and a green rain jacket fills the left 40 percent of the frame and looks straight into the '
                  'lens. Behind him, a lakeside campsite glows at golden hour, with a yellow tent at the right edge of '
                  'the background and a dark treeline above the shore. The camera shakes slightly. The young man with '
                  'a cheerful, breathy voice (S1) says: <d>[English] Day two of camping, and the vibes are '
                  'immaculate.</d> While he speaks without noticing, a large brown bear steps out of the treeline in '
                  'the right background and walks along the shore toward the camera, growing larger in the frame. It '
                  'stops directly behind him and lowers its huge head onto his shoulder nearest the right edge of the '
                  'frame, so that its snout reaches into the right half of the frame beside his ear. His smile freezes '
                  'in place and his eyes slide sideways toward the bear without his head moving.\n\noverall_soundscape: '
                  'Wind hisses across the lake while gravel crunches under heavy, slow paws. The bear lets out a long, '
                  'warm huff beside the microphone and his nylon jacket rustles as he goes rigid.\n\nnon_diegetic_music: '
                  'N/A')


with open(os.path.join(ROOT, 'demo/examples/minimax_h3/examples.json')) as _examples_handle:
    EXAMPLES = json.load(_examples_handle)
EXAMPLE_FILES = {os.path.basename(example[key]) for example in EXAMPLES
                 for key in ('in_image', 'in_last_image') if example.get(key)}


def canvas_size(aspect_ratio, num_pixels, multiple=32):
    """(height, width) of the aspect ratio at num_pixels, both rounded up to the multiple, so 1 MP gives H3's native
    768x1344 for 16:9 and 672x1536 for 21:9."""
    aspect_width, aspect_height = map(float, aspect_ratio.split(':'))
    height = (num_pixels * aspect_height / aspect_width) ** 0.5
    width = (num_pixels * aspect_width / aspect_height) ** 0.5
    return math.ceil(height / multiple) * multiple, math.ceil(width / multiple) * multiple


def build_demo(*, compile_artifacts=False):
    args = SimpleNamespace(
        crf=23, aot=True, attention='sol', dense_backend='native', sol_prefix_backend='native',
        sol_tau=1.0, sol_thresh_type='exact', sol_dense_blocks=2, sol_kv_splits=1)
    refiner_backend = 'native'

    # load base pipeline with batched VAE decoding
    base_pipe = MiniMaxH3BlocksMod().init_pipeline('multimodalart/MiniMax-H3-Pruned')
    vae_spec = base_pipe.get_component_spec('vae')
    vae_spec.type_hint = AutoencoderKLMiniMaxH3
    base_pipe.update_components(vae=vae_spec.load(dtype=torch.float16))
    base_pipe.load_components(
        names=['text_encoder', 'tokenizer', 'processor', 'audio_vae', 'transformer'],
        dtype=torch.bfloat16, trust_remote_code=True)
    base_pipe.vae.decode_tile_batch_size = 28
    base_pipe.text_encoder.requires_grad_(False)
    quantize_frozen_linears(base_pipe.text_encoder, exclude_keys=['lm_head'])  # layer by layer, before .cuda()
    base_pipe.transformer.requires_grad_(False)  # keep physical CPU addresses for sharing the base weights

    scheduler_2step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=12.0, max_raw_t=0.93, final_step_size_scale=0.6)
    audio_scheduler_2step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=3.0, max_raw_t=0.93, final_step_size_scale=0.6)
    scheduler_4step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=12.0, max_raw_t=0.96, final_step_size_scale=0.6)
    audio_scheduler_4step = FlowEulerODEScheduler(
        num_train_timesteps=1, shift=3.0, max_raw_t=0.96, final_step_size_scale=0.6)
    components = {name: getattr(base_pipe, name) for name in ['vae', 'audio_vae', 'transformer']}

    # the text encoder runs in its own pipeline, so it can sit on a second GPU
    blocks = MiniMaxH3BlocksMod()
    dwd_blocks = MiniMaxH3DWDBlocks()
    text_blocks = MiniMaxH3BlocksMod.from_blocks_dict(
        {name: blocks.sub_blocks[name] for name in ('before_encode', 'text_encoder')})
    blocks.sub_blocks.pop('text_encoder')
    dwd_blocks.sub_blocks.pop('text_encoder')

    text_pipe = text_blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    text_pipe.update_components(
        text_encoder=base_pipe.text_encoder, tokenizer=base_pipe.tokenizer, processor=base_pipe.processor)

    pipe_2step = blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_2step.update_components(**components, scheduler=scheduler_2step, audio_scheduler=audio_scheduler_2step)
    load_lakonlab_adapter(pipe_2step, ADAPTER_REPO, subfolder='pi_h3_2step_adapter')

    pipe_4step = blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_4step.update_components(**components, scheduler=scheduler_4step, audio_scheduler=audio_scheduler_4step)
    load_lakonlab_adapter(pipe_4step, ADAPTER_REPO, subfolder='pi_h3_4step_adapter')

    pipe_2step_dwd = dwd_blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_2step_dwd.update_components(**components, scheduler=scheduler_2step, audio_scheduler=audio_scheduler_2step)
    load_lakonlab_adapter(pipe_2step_dwd, ADAPTER_REPO, subfolder='pi_h3_2step_dwd_adapter')

    pipe_4step_dwd = dwd_blocks.init_pipeline('multimodalart/MiniMax-H3-Pruned')
    pipe_4step_dwd.update_components(**components, scheduler=scheduler_4step, audio_scheduler=audio_scheduler_4step)
    load_lakonlab_adapter(pipe_4step_dwd, ADAPTER_REPO, subfolder='pi_h3_4step_dwd_adapter')

    pipe_2step.transformer.requires_grad_(False)
    pipe_4step.transformer.requires_grad_(False)
    pipe_2step_dwd.transformer.requires_grad_(False)
    pipe_4step_dwd.transformer.requires_grad_(False)

    # a DWD adapter is trained on top of its frozen piFlow adapter and shares its LoRA layers
    for dwd_pipe, piflow_pipe in ((pipe_2step_dwd, pipe_2step), (pipe_4step_dwd, pipe_4step)):
        for name, m in list(dwd_pipe.transformer.named_modules()):
            if name.endswith(('.lora_A', '.lora_B')):
                dwd_pipe.transformer.set_submodule(name, piflow_pipe.transformer.get_submodule(name))

    # the base transformer is quantized in place; every pipe takes its fp8 linears wherever it holds the base weight,
    # then quantizes the linears its adapter replaced
    quant_exclude_keys = ['lora', 'adaln_proj', 'norm_out', 'proj_out']
    base_linears = {m.weight.data_ptr(): m for m in base_pipe.transformer.modules() if isinstance(m, torch.nn.Linear)}
    quantize_frozen_linears(base_pipe.transformer, exclude_keys=quant_exclude_keys)
    fp8_linears = {ptr: m for ptr, m in base_linears.items() if isinstance(m, FrozenFP8Linear)}
    pipes = [pipe_2step, pipe_4step, pipe_2step_dwd, pipe_4step_dwd]
    for _pipe in pipes:
        for name, m in list(_pipe.transformer.named_modules()):
            if isinstance(m, torch.nn.Linear) and m.weight.data_ptr() in fp8_linears:
                _pipe.transformer.set_submodule(name, fp8_linears[m.weight.data_ptr()])
        quantize_frozen_linears(_pipe.transformer, exclude_keys=quant_exclude_keys)
    print('Staged FP8 weights and loaded all four adapters', flush=True)
    for _pipe in pipes:
        _pipe.transformer.to('cuda').eval()
    base_pipe.text_encoder.to('cuda')
    base_pipe.vae.to('cuda')
    base_pipe.audio_vae.to('cuda')
    print('Registered model weights with ZeroGPU', flush=True)

    del base_pipe, components, base_linears, fp8_linears

    for _pipe in (pipe_2step, pipe_4step):
        set_attention_backends(_pipe.transformer, backend=args.dense_backend)
    for _pipe in (pipe_2step_dwd, pipe_4step_dwd):
        set_attention_backends(
            _pipe.transformer, backend=args.dense_backend, overrides={'self.dwd_pixel_refiner': refiner_backend})
    for _pipe in pipes:
        for block in _pipe.transformer.transformer_blocks[args.sol_dense_blocks:]:
            processor = MiniMaxH3SolAttnProcessor()
            processor.tau, processor.thresh_type = args.sol_tau, args.sol_thresh_type
            processor.kv_splits = args.sol_kv_splits
            block.attn.set_processor(processor)
            block.attn.set_attention_backend(args.sol_prefix_backend)
    if compile_artifacts:
        from compile_regions_build import build_regions

        build_regions(pipes, text_pipe)
    from compile_regions import prepare_compiled_models

    prepare_compiled_models(pipes, text_pipe)

    color_encoder = RGBColorEncoder(
        use_affine_norm=True, mean=(-0.03, -0.088, -0.188), std=(0.458, 0.448, 0.45)).to('cuda')

    @spaces.GPU(size='xlarge')
    def generate(
            seed, prompt, in_image, in_last_image, aspect_ratio, megapixels, duration, steps, remove_decoder,
            progress=gr.Progress(track_tqdm=True)):
        height, width = canvas_size(aspect_ratio, megapixels * 1e6)
        # H3 generates 17n + 5 frames at 24 fps, from 124 (5.2 s) to 345 (14.4 s)
        num_frames = min(max(round((duration * 24 - 5) / 17), 7), 20) * 17 + 5
        steps = 2 if steps < 3 else 4
        if remove_decoder:
            pipe = pipe_2step_dwd if steps == 2 else pipe_4step_dwd
        else:
            pipe = pipe_2step if steps == 2 else pipe_4step
        # keyframes are center-cropped and resized onto the requested canvas
        image = None if in_image is None else ImageOps.fit(
            in_image, (width, height), Image.Resampling.LANCZOS)
        last_image = None if in_last_image is None else ImageOps.fit(
            in_last_image, (width, height), Image.Resampling.LANCZOS)
        text = text_pipe(
            prompt=prompt, image=image, last_image=last_image, height=height, width=width,
            num_frames=num_frames, output=['prompt_embeds', 'text_token_tags'])
        out = pipe(
            prompt_embeds=text['prompt_embeds'].to('cuda'),
            text_token_tags=text['text_token_tags'],
            image=image,
            last_image=last_image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
            output=['dwd_pixels' if remove_decoder else 'videos', 'audio', 'sampling_rate'],
            **({} if remove_decoder else dict(output_type='pt')),  # the DWD pipeline has no decode step
        )
        if remove_decoder:  # RGB (C, T, H, W) in [-1, 1] to (T, C, H, W) in [0, 1]
            out['videos'] = color_encoder.decode(out.pop('dwd_pixels')).div_(2).add_(0.5).clamp_(
                min=0, max=1).transpose(1, 2)
        # generation settings for reproduction, as JSON in the mp4 comment tag (keyframes as hashes of the fitted images)
        settings = dict(
            seed=seed, prompt=prompt, steps=steps, remove_decoder=remove_decoder, aspect_ratio=aspect_ratio,
            megapixels=megapixels, height=height, width=width, num_frames=num_frames, fps=24,
            first_frame=None if image is None else hashlib.sha256(image.tobytes()).hexdigest(),
            last_frame=None if last_image is None else hashlib.sha256(last_image.tobytes()).hexdigest(),
            adapter=f'{ADAPTER_REPO}/pi_h3_{steps}step{"_dwd" if remove_decoder else ""}_adapter',
            attention=args.attention, dense_backend=args.dense_backend, refiner_backend=refiner_backend, aot=args.aot,
            **({k: v for k, v in vars(args).items() if k.startswith('sol_') and k not in ('sol_aot', 'sol_precompile')}
               if args.attention == 'sol' else {}))
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        save_video_audio(
            rgb_to_yuv420p(out['videos'][0]).cpu().numpy(), out['audio'][0].float().cpu().numpy(), output_path,
            None, 24, out['sampling_rate'], pix_fmt='yuv420p',
            options={'preset': 'veryfast', 'crf': str(args.crf)}, metadata={'comment': json.dumps(settings)})
        return output_path

    return build_app(generate)


TABS_HTML = (
    '<style>.html-container:has(.pih3-head){padding:0!important}'
    '.pih3-head{display:flex;align-items:center;gap:20px;flex-wrap:wrap}.pih3-head h1{margin:0!important}'
    '.pih3-tabs{display:flex;gap:4px;padding:4px;border:1px solid var(--border-color-primary);border-radius:10px}'
    '.pih3-tabs a{padding:6px 16px;border-radius:7px;font-weight:600;'
    'text-decoration:none!important;color:var(--body-text-color)!important}'
    '.pih3-tabs a.active{background:var(--body-text-color);color:var(--background-fill-primary)!important}</style>'
    '<div class="pih3-head prose"><h1>pi-H3 Demo 1.0</h1><div class="pih3-tabs"><a class="active">Classic UI</a>'
    '<a href="/studio" onclick="this.href=\'/studio\'+location.search">Studio</a>'
    '<a href="/workflow/?ui=canvas" onclick="this.href=\'/workflow/?ui=canvas\'+location.search.replace(\'?\',\'&\')">'
    'Workflow</a></div></div>')

# The workflow page: the same three tabs in the canvas toolbar, and the canvas only. gr.Workflow has no option to
# turn off its App view, so its toggle is hidden and a URL without ?ui= (which opens the App view) gets ui=canvas.
# Its own Hugging Face sign-in is hidden too: mounted under /workflow its OAuth callback does not resolve, the
# pi-H3 node runs on the visitor's ZeroGPU quota without it, and saving needs write access to this Space anyway.
WORKFLOW_HEAD = """<style>
.view-toggle, .login-btn, .save-space-btn, .toolbar-user-wrap { display: none !important; }
.pih3-wf-tabs { display: flex; gap: 4px; padding: 3px; margin-left: 8px; border: 1px solid var(--border-color-primary);
                border-radius: 9px; }
.pih3-wf-tabs a { padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; text-decoration: none;
                  color: var(--body-text-color); }
.pih3-wf-tabs a.active { background: var(--body-text-color); color: var(--background-fill-primary); }
</style>
<script>
(() => {
  const query = new URLSearchParams(location.search);
  if (!query.has('ui')) {
    query.set('ui', 'canvas');
    location.replace(location.pathname + '?' + query);
    return;
  }
  query.delete('ui');
  const rest = query.toString() ? '?' + query : '';
  const tabs = `<div class="pih3-wf-tabs"><a href="/demo/${rest}">Classic UI</a><a href="/studio${rest}">Studio</a>`
             + '<a class="active">Workflow</a></div>';
  // the canvas renders (and may re-render) its toolbar after load, so keep the tabs right after its title, outside
  // the title's rename button (links inside a button do not reliably navigate)
  new MutationObserver(() => {
    const title = document.querySelector('.toolbar-left .name-text');
    const host = title && (title.closest('.name-btn') || title);
    if (host && !host.parentElement.querySelector(':scope > .pih3-wf-tabs')) host.insertAdjacentHTML('afterend', tabs);
  }).observe(document.documentElement, {childList: true, subtree: true});
})();
</script>"""


def build_interface(generate):
    with gr.Blocks(analytics_enabled=False,
                   title='pi-H3 Demo'
                   ) as demo:

        gr.HTML(TABS_HTML, padding=False)
        md_txt = 'The fastest MiniMax H3 video model: 2 or 4 steps, direct pixel output without a video VAE decoder. ' \
                 'Accelerated by **pi-Flow V2** and **[DWD](https://hanshengchen.com/dwd/)** ' \
                 '(**D**ecoding **W**ithout a **D**ecoder). ' \
                 'Read the [blog post](https://hanshengchen.com/blog/pi-h3/) for more details.\n' \
                 '<br> MiniMax H3 **prompting guide**: [https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)'
        gr.Markdown(md_txt)

        create_interface_img_to_video(
            generate,
            prompt=DEFAULT_PROMPT,
            steps=2, min_steps=2, max_steps=4, steps_slider_step=2, remove_decoder=True,
            aspect_ratios=ASPECT_RATIOS,
            examples=EXAMPLES,
            args=['last_seed', 'prompt', 'in_image', 'in_last_image', 'aspect_ratio', 'megapixels', 'duration',
                  'steps', 'remove_decoder'])
    return demo



# --------------------------------------------------------------------------- helpers


def _resolve_seed(seed, randomize) -> int:
    return random.randint(0, MAX_SEED) if randomize else min(max(int(float(seed)), 0), MAX_SEED)


def _check_choice(value, choices, name):
    if value not in choices:
        raise gr.Error(f"{name} must be one of: {', '.join(choices)}.")
    return value


def _requested_path(value) -> str:
    """Turn whatever the studio/API client sends into a path."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise gr.Error(f'Expected an image, got {type(value).__name__}')
    if path := value.get('path'):
        return path
    url = value.get('url') or ''
    if '/gradio_api/file=' in url:
        return unquote(url.split('/gradio_api/file=', 1)[1])
    if url.startswith(('http://', 'https://')):
        raise gr.Error('Upload the image instead of passing a URL.')
    return unquote(urlparse(url).path)


def _local_path(value) -> str:
    """A path this process may open: only files inside gradio's upload folder,
    where /gradio_api/upload puts uploads."""
    path = os.path.realpath(_requested_path(value))
    folder = os.path.realpath(get_upload_folder())
    if os.path.commonpath([path, folder]) != folder:
        raise gr.Error('Upload the image first.')
    return path


def _open_rgb(value):
    if not value:
        return None
    return ImageOps.exif_transpose(Image.open(_local_path(value))).convert('RGB')


def _publish_mp4(path: str) -> str:
    """Move the finished mp4 into gradio's upload folder so the studio and the workflow canvas
    can fetch it by URL."""
    folder = get_upload_folder()
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f'pi_h3_{os.urandom(8).hex()}.mp4')
    os.replace(path, target)
    return target


def _as_media(path: str) -> dict:
    """The media shape used by gr.Workflow inputs and outputs."""
    import gradio_client.utils as client_utils
    return {
        'path': path,
        'url': f'/gradio_api/file={client_utils.encode_file_path(path)}',
        'is_file': True,
    }


def _run(fn, *args):
    """One GPU call with the studio's error surface: messages, not stack traces."""
    try:
        return fn(*args)
    except (RuntimeError, ValueError) as error:
        message = str(error).split('\n')[0]
        if 'ZeroGPU' in message and ('limit' in message or 'quota' in message):
            message += (' The run was billed anonymously: open the Space from its huggingface.co page '
                        'while signed in, so it uses your own ZeroGPU quota — the anonymous allowance '
                        'is tiny and shared.')
        raise gr.Error(message)


def _parse_bool(value) -> bool:
    """The canvas passes DWD as a text reference; accept any spelling."""
    return value if isinstance(value, bool) else str(value).strip().lower() in ('true', '1', 'yes', 'on')


# ------------------------------------------------------------------ studio and workflow (contributed by AK)
# A second UI over the same generate(): a gr.Server studio page at /studio and a gr.Workflow canvas at /workflow,
# next to the Blocks demo at /demo. Images arrive as uploaded file paths and are opened the way gr.Image(type='pil',
# image_mode='RGB') opens them (EXIF transpose, then RGB), so both UIs feed generate() identical inputs.


def build_app(generate):
    """The Space's ASGI app: the Blocks demo, the studio and the workflow canvas over one generate()."""

    def generate_studio(seed, prompt, in_image, in_last_image, aspect_ratio, megapixels, duration, steps,
                        remove_decoder):
        """One generate() call -> (published mp4 path, run info note). ZeroGPU bills the visitor whose token came
        with the calling page's request."""
        start = time.perf_counter()
        path = _run(
            generate, seed, prompt, _open_rgb(in_image), _open_rgb(in_last_image), aspect_ratio, megapixels,
            duration, steps, remove_decoder)
        height, width = canvas_size(aspect_ratio, megapixels * 1e6)
        note = (f"{2 if steps < 3 else 4} steps · {'DWD (no decoder)' if remove_decoder else 'VAE decoded'} · "
                f'{width}x{height} · {time.perf_counter() - start:.1f} s')
        return _publish_mp4(path), note

    def pi_h3_generate(prompt: str = '', in_image: str = '', in_last_image: str = '', seed: str = '42',
                       aspect_ratio: str = '16:9', duration: str = '5.0', steps: str = '2',
                       remove_decoder: str = 'true') -> dict:
        """Generate one video with pi-H3 from a prompt and optional first/last keyframes.

        gr.Workflow passes the node's inputs positionally, so the parameters follow the order of the
        op_pi_h3_generate inputs in workflow.json."""
        _check_choice(aspect_ratio, ASPECT_RATIOS, 'Aspect ratio')
        if not (prompt or '').strip() and not in_image:
            raise gr.Error('Write a prompt or add a first frame.')
        path, _ = generate_studio(
            min(max(int(float(seed)), 0), MAX_SEED), prompt, in_image, in_last_image, aspect_ratio, 1.0,
            float(duration) if duration else 5.0, int(float(steps)) if steps else 2, _parse_bool(remove_decoder))
        return _as_media(path)

    server = gr.Server(title='pi-H3 Studio')

    @server.get('/')
    def _root(request: StarletteRequest):  # noqa: ANN201
        """The Blocks demo is the default page; keep the query (HF passes the theme and sign-in token there)."""
        query = request.url.query
        return RedirectResponse('/demo/' + ('?' + query if query else ''))

    @server.get('/studio', response_class=HTMLResponse)
    def _studio_page():  # noqa: ANN201
        with open(os.path.join(ROOT, 'studio.html'), encoding='utf-8') as handle:
            return handle.read()

    @server.get('/examples/{name}', response_class=FileResponse)
    def _example_file(name: str):  # noqa: ANN201
        """The example keyframes, fetched by the studio page and uploaded by the visitor like any file."""
        if name not in EXAMPLE_FILES or os.path.sep in name:
            raise gr.Error('Unknown example.')
        return FileResponse(os.path.join(ROOT, 'demo/examples/minimax_h3', name))

    @server.api(name='generate', concurrency_limit=1)
    def api_generate(prompt: str = '', seed: float = 0, randomize: bool = True, in_image: str = '',
                     in_last_image: str = '', aspect_ratio: str = '16:9', megapixels: float = 1.0,
                     duration: float = 5.0, steps: float = 2, remove_decoder: bool = True) -> dict:
        """Generate one video with pi-H3 (2 or 4 steps, optional VAE-free DWD decoding)."""
        _check_choice(aspect_ratio, ASPECT_RATIOS, 'Aspect ratio')
        if not (prompt or '').strip() and not in_image:
            raise gr.Error('Write a prompt or add a first frame.')
        seed_value = _resolve_seed(seed, randomize)
        path, note = generate_studio(
            seed_value, prompt, in_image, in_last_image, aspect_ratio, megapixels, duration, steps, remove_decoder)
        return {'video_url': _as_media(path)['url'], 'info': note, 'seed': seed_value}

    workflow = gr.Workflow(graph=os.path.join(ROOT, 'workflow.json'), bind={'pi_h3_generate': pi_h3_generate})
    workflow.title = 'pi-H3 Workflow'
    return SimpleNamespace(server=server, mounts={
        '/demo': (build_interface(generate), dict(
            allowed_paths=[os.path.join(ROOT, 'demo/examples/minimax_h3')],
            css_paths=os.path.join(ROOT, 'lakonlab/ui/gradio/style.css'),
            head_paths=os.path.join(ROOT, 'lakonlab/ui/gradio/head.html'))),
        '/workflow': (workflow, dict(head=WORKFLOW_HEAD))})


def launch_demo(app, port=None):
    # The @server.api endpoints are deferred until Server.launch(), which also fires the ZeroGPU registration hook
    # (pack weights + startup report) through the patched Blocks.launch path. Launch on a loopback port for
    # registration, then serve the same app via uvicorn.
    port = port or int(os.environ.get('GRADIO_SERVER_PORT', '7860'))
    os.environ['GRADIO_SERVER_PORT'] = '7861'
    try:
        app.server.launch(prevent_thread_lock=True, server_name='127.0.0.1', server_port=7861, quiet=True,
                          show_error=False, ssr_mode=False)
    finally:
        os.environ['GRADIO_SERVER_PORT'] = str(port)
    try:
        app.server.close()
    except Exception:
        pass
    # mount only after the registration launch above, so the mounted queues stay on the public server's event loop
    server = app.server
    for path, (blocks, kwargs) in app.mounts.items():
        server = gr.mount_gradio_app(server, blocks, path=path, ssr_mode=False, **kwargs)
    for blocks in [app.server.get_blocks()] + [blocks for blocks, _ in app.mounts.values()]:
        _BLOCKS_BY_ID[id(blocks)] = blocks

    import uvicorn
    uvicorn.run(server, host='0.0.0.0', port=port)


if __name__ == "__main__":
    launch_demo(build_demo())
