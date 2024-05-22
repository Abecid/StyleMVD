import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from shutil import copy
from datetime import datetime
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers import DDPMScheduler
from diffusers.utils import numpy_to_pil
import numpy as np
import cv2
from PIL import Image

from configs import parse_config
from ip_adapter import IPAdapterXL
from pipeline.prompt import azim_text_prompt, azim_neg_text_prompt
from pipeline.utils import center_crop
from pipeline.pipeline_controlnet_sd_xl import StableStyleMVDPipeline, get_conditioning_images, get_canny_conditioning_images


def seed_everything(seed=2023):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")


opt = parse_config()
seed_everything(opt.seed)

if opt.mesh_config_relative:
	mesh_path = join(dirname(opt.config), opt.mesh)
else:
	mesh_path = abspath(opt.mesh)

if opt.output:
	output_root = abspath(opt.output)
else:
	output_root = dirname(opt.config)

output_name_components = []
if opt.prefix and opt.prefix != "":
	output_name_components.append(opt.prefix)
if opt.timeformat and opt.timeformat != "":
	output_name_components.append(datetime.now().strftime(opt.timeformat))
if opt.use_mesh_name:
	mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
	output_name_components.append(mesh_name)
output_name = "_".join(output_name_components)
output_dir = join(output_root, output_name)

if not isdir(output_dir):
	os.makedirs(output_dir)
else:
	print(f"Results exist in the output directory, use time string to avoid name collision.")
	exit(0)

print(f"Saving to {output_dir}")

copy(opt.config, os.path.join(output_dir, "config.yaml"))

logging_config = {
	"output_dir":output_dir, 
	"log_interval":opt.log_interval,
	"view_fast_preview": opt.view_fast_preview,
	"tex_fast_preview": opt.tex_fast_preview,
	}


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

if opt.cond_type == "canny":
    controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
elif opt.cond_type == "depth":
    controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)


# load StyleMVD pipeline
mvd_pipe = StableStyleMVDPipeline(**pipe.components)
mvd_pipe.set_config(
    mesh_path=mesh_path,
	mesh_transform={"scale":opt.mesh_scale},
	mesh_autouv=opt.mesh_autouv,
	camera_azims=opt.camera_azims,
	top_cameras=not opt.no_top_cameras,
	texture_size=opt.latent_tex_size,
	render_rgb_size=opt.rgb_view_size,
	texture_rgb_size=opt.rgb_tex_size,
	height=opt.latent_view_size*8,
	width=opt.latent_view_size*8,
	max_batch_size=48,
	controlnet_conditioning_end_scale= opt.conditioning_scale_end,
	guidance_rescale = opt.guidance_rescale,
	multiview_diffusion_end=opt.mvd_end,
	shuffle_background_change=opt.shuffle_bg_change,
	shuffle_background_end=opt.shuffle_bg_end,
	ref_attention_end=opt.ref_attention_end,
	logging_config=logging_config,
	cond_type=opt.cond_type,
)
mvd_pipe.initialize_pipeline(
    mesh_path=mesh_path,
    mesh_transform={"scale":opt.mesh_scale},
    mesh_autouv=opt.mesh_autouv,
    camera_azims=opt.camera_azims,
    camera_centers=None,
    top_cameras=not opt.no_top_cameras,
    ref_views=[],
    latent_size=mvd_pipe.height//8,
    render_rgb_size=mvd_pipe.render_rgb_size,
    texture_size=mvd_pipe.texture_size,
    texture_rgb_size=mvd_pipe.texture_rgb_size,
    max_batch_size=mvd_pipe.max_batch_size,
    logging_config=logging_config
)


# generate prompt based on camera pose 
prompt_list = [azim_text_prompt(opt.prompt, pose) for pose in mvd_pipe.camera_poses]
negative_prompt_list = [azim_neg_text_prompt(opt.negative_prompt, pose) for pose in mvd_pipe.camera_poses]
print(prompt_list)


# control image
if opt.cond_type == "depth":
    conditioning_images, masks = get_conditioning_images(mvd_pipe.uvp, mvd_pipe.height, cond_type=mvd_pipe.cond_type)
elif opt.cond_type == "canny":
    conditioning_images, masks = get_canny_conditioning_images(mvd_pipe.uvp, mvd_pipe.height)


# load style image
style_image = Image.open(opt.style_img)
style_image.resize((512, 512))
# style_image = center_crop(style_image, 512)
style_image.save(os.path.join(output_dir, "style_img.png"))


# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(mvd_pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])


# generate image
images = ip_model.generate(pil_image=style_image,
                            prompt=prompt_list,
                            negative_prompt=negative_prompt_list,
                            scale=opt.ip_adapter_scale,
                            guidance_scale=opt.guidance_scale,
                            num_samples=1,
                            num_inference_steps=opt.num_inference_steps,
                            seed=opt.seed,
                            image=conditioning_images,
                            controlnet_conditioning_scale=opt.conditioning_scale,
                            )
