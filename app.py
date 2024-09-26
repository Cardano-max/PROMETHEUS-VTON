import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI
from pydantic import BaseModel, Field
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add GroundingDINO to the system path
logger.info("Adding GroundingDINO to system path...")
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# Import necessary modules
logger.info("Importing modules...")
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops, slconfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict
from segment_anything import build_sam, SamPredictor
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from diffusers import DDPMScheduler, AutoencoderKL
from huggingface_hub import hf_hub_download
from utils_mask import get_mask_location
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
import apply_net

logger.info("All modules imported successfully.")

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load models
logger.info("Loading models...")

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    logger.info(f"Loading model from {repo_id}...")
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = slconfig.SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    logger.info(f"Model loaded from {cache_file}")
    _ = model.eval()
    return model

# Load GroundingDINO and SAM models
sam_checkpoint = 'sam_vit_h_4b8939.pth'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

logger.info("GroundingDINO and SAM models loaded successfully.")

# Load other models and components
logger.info("Loading other models and components...")
base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", revision=None, use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", revision=None, use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

logger.info("All models and components loaded successfully.")

# Set up pipeline
logger.info("Setting up TryonPipeline...")
pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder
logger.info("TryonPipeline set up successfully.")

# Utility functions
def detect_clothing(img, has_hat, has_gloves, human_img):
    logger.info("Detecting clothing...")
    # ... (rest of the function)

def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, use_grounding, has_hat, has_gloves, denoise_steps, seed):
    logger.info("Starting try-on process...")
    # ... (rest of the function)

# Set up Gradio interface
logger.info("Setting up Gradio interface...")
gr.component._pydantic_to_pythonic = lambda x: x

image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    
    with gr.Row():
        with gr.Column():
            logger.info("Creating image input component...")
            imgs = gr.Image(source='upload', type="pil", label='Human. Upload image for auto-masking', tool='sketch')
            
            logger.info("Creating checkbox components...")
            is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)", value=False)
            is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing", value=False)
            use_grounding = gr.Checkbox(label='Yes', info='Use Grounded Segment Anything to generate mask (better than auto-masking)', value=True)
            has_hat = gr.Checkbox(label='Yes', info='Look for a hat to mask in the outfit', value=False)
            has_gloves = gr.Checkbox(label='Yes', info='Look for gloves to mask in the outfit', value=False)
            
            logger.info("Setting up image examples...")
            example = gr.Examples(inputs=imgs, examples=human_ex_list, examples_per_page=10)

        with gr.Column():
            logger.info("Creating garment image input...")
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            
            logger.info("Creating prompt input...")
            prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            
            logger.info("Setting up garment examples...")
            example = gr.Examples(inputs=garm_img, examples_per_page=8, examples=garm_list_path)

        with gr.Column():
            logger.info("Creating output image components...")
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)
            image_out = gr.Image(label="Output", elem_id="output-img", show_share_button=False)

    with gr.Column():
        logger.info("Creating try-on button and advanced settings...")
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=20, step=1)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)

    logger.info("Setting up try-on function...")
    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, use_grounding, has_hat, has_gloves, denoise_steps, seed], outputs=[image_out, masked_img], api_name='tryon')

logger.info("Launching Gradio interface...")
image_blocks.launch(share=True)