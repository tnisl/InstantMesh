import os
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile
import time
from huggingface_hub import hf_hub_download


if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)

OUTPUT_DIR = '/kaggle/working/InstantMesh/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def images_to_video(images, output_path, fps=30):
    """Save rendered frames to mp4 robustly on Kaggle.

    images: Tensor with shape (N, C, H, W), values expected in [0, 1].
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []

    images = images.detach().float().cpu().clamp(0, 1)
    for i in range(images.shape[0]):
        frame = images[i].permute(1, 2, 0).numpy()
        frame = (frame * 255.0).round().astype(np.uint8)
        frames.append(frame)

    frames = np.stack(frames, axis=0)

    # Kaggle/Gradio sometimes fails with codec='h264'. Try stable codecs first.
    last_error = None
    for codec in ("libx264", "mpeg4", None):
        try:
            kwargs = {"fps": fps}
            if codec is not None:
                kwargs["codec"] = codec
            imageio.mimwrite(output_path, frames, **kwargs)
            print(f"[images_to_video] saved: {output_path} codec={codec}", flush=True)
            return output_path
        except Exception as e:
            last_error = e
            print(f"[images_to_video] codec={codec} failed: {e}", flush=True)

    raise RuntimeError(f"Could not save video: {last_error}")


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = 'configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
    cache_dir=model_cache_dir
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model", cache_dir=model_cache_dir)
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device0)

# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model", cache_dir=model_cache_dir)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()

print('Loading Finished!')


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background):

    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image

def generate_mvs(input_image, sample_steps, sample_seed):
    """Generate Zero123++ multiview image and return file paths.

    Returning file paths is more stable than returning PIL objects through
    Gradio Queue on Kaggle. The first path is the original 3x2 grid used by
    make3d(); the second path is only for UI display as a 2x3 grid.
    """
    print("[generate_mvs] start", flush=True)
    seed_everything(int(sample_seed))

    try:
        if input_image is None:
            raise gr.Error("No processed image available.")

        generator = torch.Generator(device=device0).manual_seed(int(sample_seed))

        with torch.inference_mode():
            z123_image = pipeline(
                input_image,
                num_inference_steps=int(sample_steps),
                generator=generator,
            ).images[0].convert("RGB")

        print("[generate_mvs] raw z123 image:", z123_image.size, z123_image.mode, flush=True)

        # PIL size is (W, H); numpy shape is (H, W, C). Expected: (960, 640, 3).
        raw_np = np.array(z123_image, dtype=np.uint8, copy=True)
        print("[generate_mvs] numpy:", raw_np.shape, raw_np.dtype, flush=True)

        h, w, c = raw_np.shape
        if c != 3 or h % 3 != 0 or w % 2 != 0:
            raise ValueError(f"Unexpected multiview shape: {raw_np.shape}; expected 3x2 RGB grid")

        view_h = h // 3
        view_w = w // 2

        # Original Zero123++ layout: 3 rows x 2 cols.
        # UI layout: 2 rows x 3 cols.
        views = raw_np.reshape(3, view_h, 2, view_w, c).transpose(0, 2, 1, 3, 4)
        views = views.reshape(6, view_h, view_w, c)
        show_np = views.reshape(2, 3, view_h, view_w, c).transpose(0, 2, 1, 3, 4)
        show_np = np.ascontiguousarray(show_np.reshape(2 * view_h, 3 * view_w, c))

        run_id = f"mvs_seed{int(sample_seed)}_{int(time.time())}"
        raw_path = os.path.join(OUTPUT_DIR, f"{run_id}_raw_3x2.png")
        show_path = os.path.join(OUTPUT_DIR, f"{run_id}_show_2x3.png")

        z123_image.save(raw_path)
        Image.fromarray(show_np).save(show_path)

        print("[generate_mvs] saved raw:", raw_path, flush=True)
        print("[generate_mvs] saved show:", show_path, flush=True)
        print("[generate_mvs] done", flush=True)

        return raw_path, show_path

    except Exception as e:
        import traceback
        print("[generate_mvs] ERROR:", flush=True)
        traceback.print_exc()
        raise gr.Error(f"Multi-view generation failed: {str(e)}")

def make_mesh(mesh_fpath, planes):
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")

    print("[make_mesh] extract mesh", flush=True)
    with torch.inference_mode():
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]

        print("[make_mesh] save glb:", mesh_glb_fpath, flush=True)
        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)

        print("[make_mesh] save obj:", mesh_fpath, flush=True)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)

    print(f"[make_mesh] mesh saved: {mesh_fpath}", flush=True)
    return mesh_fpath, mesh_glb_fpath


def make3d(multiview_path):
    """Build video, OBJ and GLB from saved raw 3x2 multiview image path."""
    print("[make3d] start", flush=True)
    print("[make3d] multiview_path:", multiview_path, flush=True)

    try:
        if multiview_path is None or not os.path.exists(multiview_path):
            raise gr.Error("Generate multi-views first.")

        pil_img = Image.open(multiview_path).convert("RGB")
        images_np = np.array(pil_img, dtype=np.float32, copy=True) / 255.0
        print("[make3d] input numpy:", images_np.shape, images_np.dtype, flush=True)

        h, w, c = images_np.shape
        if c != 3 or h % 3 != 0 or w % 2 != 0:
            raise ValueError(f"Unexpected multiview shape for 3D: {images_np.shape}; expected 3x2 RGB grid")

        images = torch.from_numpy(images_np).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        print("[make3d] views tensor:", images.shape, flush=True)

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
        render_cameras = get_render_cameras(
            batch_size=1,
            radius=4.5,
            elevation=20.0,
            is_flexicubes=IS_FLEXICUBES,
        ).to(device1)

        images = images.unsqueeze(0).to(device1)
        images = v2.functional.resize(
            images,
            (320, 320),
            interpolation=3,
            antialias=True,
        ).clamp(0, 1)

        run_id = os.path.basename(multiview_path).replace("_raw_3x2.png", "")
        mesh_fpath = os.path.join(OUTPUT_DIR, f"{run_id}.obj")
        video_fpath = os.path.join(OUTPUT_DIR, f"{run_id}.mp4")

        with torch.inference_mode():
            print("[make3d] forward_planes", flush=True)
            planes = model.forward_planes(images, input_cameras)

            print("[make3d] render video", flush=True)
            # Lower values are more stable on Kaggle VRAM than the original chunk_size=20/render_size=384.
            chunk_size = 5 if IS_FLEXICUBES else 1
            render_size = 320

            frames = []
            for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
                if IS_FLEXICUBES:
                    frame = model.forward_geometry(
                        planes,
                        render_cameras[:, i:i + chunk_size],
                        render_size=render_size,
                    )["img"]
                else:
                    frame = model.synthesizer(
                        planes,
                        cameras=render_cameras[:, i:i + chunk_size],
                        render_size=render_size,
                    )["images_rgb"]

                frames.append(frame.detach().cpu())

            frames = torch.cat(frames, dim=1)
            print("[make3d] frames:", frames.shape, flush=True)

        print("[make3d] save video", flush=True)
        images_to_video(frames[0], video_fpath, fps=30)

        # Release video frames before mesh extraction/export.
        del frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

        print("[make3d] done", flush=True)
        return video_fpath, mesh_fpath, mesh_glb_fpath

    except Exception as e:
        import traceback
        print("[make3d] ERROR:", flush=True)
        traceback.print_exc()
        raise gr.Error(f"3D generation failed: {str(e)}")


import gradio as gr

_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/TencentARC/InstantMesh' target='_blank'><b>InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</b></a></h2>

**InstantMesh** is a feed-forward framework for efficient 3D mesh generation from a single image based on the LRM/Instant3D architecture.

Code: <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.07191' target='_blank'>ArXiv</a>.

❗️❗️❗️**Important Notes:**
- Our demo can export a .obj mesh with vertex colors or a .glb mesh now. If you prefer to export a .obj mesh with a **texture map**, please refer to our <a href='https://github.com/TencentARC/InstantMesh?tab=readme-ov-file#running-with-command-line' target='_blank'>Github Repo</a>.
- The 3D mesh generation results highly depend on the quality of generated multi-view images. Please try a different **seed value** if the result is unsatisfying (Default: 42).
'''

_CITE_ = r"""
If InstantMesh is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/InstantMesh' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/InstantMesh?style=social)](https://github.com/TencentARC/InstantMesh)
---
📝 **Citation**

If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

📋 **License**

Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

📧 **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>bluestyle928@gmail.com</b>.
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGBA", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=75,
                        value=75,
                        step=5
                    )

            with gr.Row():
                submit = gr.Button("Generate Multi-views", elem_id="generate", variant="primary")
                build3d = gr.Button("Build 3D from Multi-views", variant="secondary")

            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=20
                )

        with gr.Column():

            with gr.Row():

                with gr.Column():
                    mv_show_images = gr.Image(
                        label="Generated Multi-views",
                        type="filepath",
                        width=379,
                        interactive=False
                    )

                with gr.Column():
                    output_video = gr.Video(
                        label="video", format="mp4",
                        width=379,
                        autoplay=True,
                        interactive=False
                    )

            with gr.Row():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(
                        label="Output Model (OBJ Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        #width=768,
                        interactive=False,
                    )
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")

            with gr.Row():
                gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

    gr.Markdown(_CITE_)
    mv_images = gr.State()

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    )

    build3d.click(
        fn=make3d,
        inputs=[mv_images],
        outputs=[output_video, output_model_obj, output_model_glb],
    )

demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=43839, debug = True)
