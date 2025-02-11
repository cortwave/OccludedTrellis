import os
import shutil
from typing import *

import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict
from gradio_litmodel3d import LitModel3D
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

from occluded_trellis import apply_occluded_patch

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
        image: Image.Image,
        mask: Image.Image,
        seed: int,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int,
        req: gr.Request,
) -> Tuple[dict, str]:
    """
    Convert an image to a 3D model.

    Args:
        image (Image.Image): The input image.
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    outputs = pipeline.run(
        image,
        mask,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


def extract_glb(
        state: dict,
        mesh_simplify: float,
        texture_size: int,
        req: gr.Request,
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.

    Returns:
        str: The path to the extracted Gaussian file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def extract_mask(editor_content):
    bg = editor_content.get("background", None)
    layers = editor_content.get("layers", [])

    if bg is None and not layers:
        raise ValueError("No background or layers provided")

    mask = layers[0]
    mask = np.array(mask)
    mask = np.where(mask[..., 3] != 0, 0, 255).astype(
        np.uint8)
    mask = np.stack([mask] * 3, -1)
    mask = Image.fromarray(mask)
    return [bg, mask]


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/), [Occluded version](https://github.com/cortwave/OccludedTrellis)
    * Upload an image, draw a mask and click "Generate" to create a 3D asset. If the image has alpha channel, it be used as the mask. Otherwise, we use `rembg` to remove the background.
    * If you find the generated 3D asset satisfactory, click "Extract GLB" to extract the GLB file and download it.
    """)

    with gr.Row():
        with gr.Column():
            im_ed = gr.ImageMask(label="Image Prompt",
                                 format="png",
                                 image_mode="RGBA",
                                 type="pil",
                                 sources=["upload", "clipboard"])
            image_prompt = gr.Image(visible=False, type="pil", image_mode="RGBA", format="png")
            mask = gr.Image(visible=False, type="pil")
            with gr.Accordion(label="Generation Settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)

            generate_btn = gr.Button("Generate")

            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)

            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
            gr.Markdown("""
                        *NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
                        """)

        with gr.Column():
            video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=300)
            model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=300)

            with gr.Row():
                download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)

    output_buf = gr.State()

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        extract_mask,
        inputs=im_ed,
        outputs=[image_prompt, mask]
    ).then(
        image_to_3d,
        inputs=[image_prompt, mask, seed, ss_guidance_strength, ss_sampling_steps,
                slat_guidance_strength, slat_sampling_steps],
        outputs=[output_buf, video_output],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_glb],
    )

    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_glb],
    )

# Launch the Gradio app
if __name__ == "__main__":
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline = apply_occluded_patch(pipeline)
    pipeline.cuda()
    demo.launch()
