import os
import av
import numpy as np
import torch
import torch.nn.functional as F

LAKONLAB_CACHE = os.environ.get('LAKONLAB_CACHE', os.path.expanduser('~/.cache/lakonlab'))


def hf_model_loader(model_class, repo_id, **kwargs):
    return model_class.from_pretrained(repo_id, **kwargs)


def hf_modular_model_loader(blocks, repo_id, **kwargs):
    model = blocks.init_pipeline(repo_id)
    model.load_components(**kwargs)
    if model.null_component_names:
        raise RuntimeError(f'Failed to load components: {model.null_component_names}')
    return model

def rgb_to_yuv420p(video):
    """Convert float RGB video [T,C,H,W] in [0, 1] to uint8 planar YUV420p [T,H*3/2,W] (the Y plane, then the U and
    V planes), with the BT.601 limited-range matrix FFmpeg's swscale applies to rgb24 input by default. Running it on
    the GPU spares the CPU conversion and halves the host copy."""
    yuv = torch.einsum(
        'ij,tjhw->tihw',
        video.new_tensor([
            [65.481, 128.553, 24.966],
            [-37.797, -74.203, 112.0],
            [112.0, -93.786, -18.214]
        ]),
        video
    ) + video.new_tensor([16.0, 128.0, 128.0]).view(1, 3, 1, 1)
    y, uv = yuv[:, 0], F.avg_pool2d(yuv[:, 1:], 2)
    t, h, w = y.shape
    return torch.cat([y, uv.reshape(t, h // 2, w)], dim=1).round().clamp(0, 255).to(torch.uint8)


def save_video_audio(video, audio, filepath, file_client, fps, sample_rate, pix_fmt='rgb24', options=None,
                     metadata=None):
    """Save uint8 video and float audio [C,S] as an H.264/AAC MP4. The video is RGB [T,H,W,C] with
    ``pix_fmt='rgb24'``, or planar YUV420p [T,H*3/2,W] (see ``rgb_to_yuv420p``) with ``pix_fmt='yuv420p'``.
    ``options`` are libx264 options with string values, e.g. ``{'preset': 'veryfast', 'crf': '18'}`` (libx264's own
    defaults, preset medium and crf 23, if None). ``metadata`` are container tags with string values, e.g.
    ``{'comment': '...'}``. The index is placed before the
    media data (faststart), so playback can start while the file downloads. Faststart rewrites the finished file,
    so only local and S3 paths are supported: a local file is written in place, an S3 one uploads a temporary file."""
    path = str(filepath)
    try:
        with av.open(path, mode='w', format='mp4', options={'movflags': 'faststart'}) as container:
            container.metadata.update(metadata or {})
            video_stream = container.add_stream(
                'libx264', rate=fps, options=options)
            video_stream.width = video.shape[2]
            video_stream.height = video.shape[1] if pix_fmt == 'rgb24' else video.shape[1] * 2 // 3
            video_stream.pix_fmt = 'yuv420p'
            audio_stream = container.add_stream('aac', rate=sample_rate)
            audio_stream.layout = {1: 'mono', 2: 'stereo'}[audio.shape[0]]
            for frame in video:
                for packet in video_stream.encode(av.VideoFrame.from_ndarray(frame, format=pix_fmt)):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(audio, dtype=np.float32), format='fltp', layout=audio_stream.layout)
            frame.sample_rate = sample_rate
            frame.pts = 0
            for packet in audio_stream.encode(frame):
                container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
    finally:
        pass
