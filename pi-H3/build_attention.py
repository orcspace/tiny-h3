"""Build the prerelease SM120 attention bundle on ZeroGPU."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import gradio as gr
import spaces
from huggingface_hub import HfApi


@spaces.GPU(size='xlarge', duration=600)
def build(directory):
    from tools.export_sol_attention import main
    with patch.object(sys, 'argv', ['export_sol_attention.py', directory]):
        main()


if __name__ == '__main__':
    with TemporaryDirectory(prefix='pih3-attention-') as directory:
        build(directory)
        assert (Path(directory) / 'sol_sm120_bsa.so').is_file()
        HfApi().upload_folder(
            repo_id=os.environ['SPACE_ID'], repo_type='space', revision='sm120-artifacts',
            folder_path=directory, path_in_repo='artifacts-v1/attention',
            commit_message='Build pi-H3 1.0 SM120 Sol attention')
    with gr.Blocks() as demo:
        gr.Markdown('pi-H3 1.0 SM120 attention bundle saved.')
    demo.launch()
