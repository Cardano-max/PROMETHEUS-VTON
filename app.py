import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict

from segment_anything.build_sam import build_sam
from segment_anything.predictor import SamPredictor

from huggingface_hub import hf_hub_download

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL

import spaces
from utils_mask import get_mask_location
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cpu')

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    _ = model.eval()
    return model

sam_checkpoint = 'sam_vit_h_4b8939.pth'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

def detect(image, image_source, text_prompt, model, box_threshold=0.3, text_threshold=0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    return annotated_frame, boxes

def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.cpu()

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 1
    mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask)

base_path = 'yisol/IDM-VTON'


# Load example images
example_path = os.path.join(os.path.dirname(__file__), 'example')

# human_list = os.listdir(os.path.join(example_path, "human"))
# human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

# garm_list = os.listdir(os.path.join(example_path, "cloth"))
# garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]



def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

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

def detect_clothing(img, has_hat, has_gloves, human_img):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = img.convert("RGB").resize((768, 1024))
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    hat = ', hat' if has_hat else ''
    gloves = ', gloves' if has_gloves else ''

    annotated_frame, detected_boxes = detect(image_transformed, image, text_prompt=f"clothing clothes tops bottoms{hat}{gloves}", model=groundingdino_model)
    segmented_frame_masks = segment(image, sam_predictor, boxes=detected_boxes)
    mask = segmented_frame_masks[0][0].cpu().numpy()

    for i in range(1, len(segmented_frame_masks)):
        mask += segmented_frame_masks[i][0].cpu().numpy()

    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask2, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask2 = mask2.resize((768, 1024))
    image_mask_pil = Image.fromarray(np.asarray(mask2) - mask)
    return image_mask_pil

@spaces.GPU
def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, use_grounding, has_hat, has_gloves, denoise_steps, seed):
    device = "cuda"
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    if garm_img is None:
        return None, None, "Please upload a garment image."

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    
    if dict is None or dict.get("background") is None:
        return None, None, "Please upload a human image."

    human_img_orig = dict["background"].convert("RGB")
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    elif use_grounding:
        mask = detect_clothing(dict["background"], has_hat, has_gloves, human_img)
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt = "a photo of " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                prompt = [prompt]
                negative_prompt = [negative_prompt]
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0)[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray

# human_list = os.listdir(os.path.join(example_path, "human"))
# human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    img = load_image(ex_human)
    if img:
        human_ex_list.append(img)
        print(f"Processed human image: {ex_human}")

# garm_list = os.listdir(os.path.join(example_path, "cloth"))
# garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

garm_ex_list = []
for garm_path in garm_list_path:
    img = load_image(garm_path)
    if img:
        garm_ex_list.append(img)
        print(f"Processed garment image: {garm_path}")

# Gradio interface
image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image.")
    with gr.Row():
        with gr.Column():
            imgs = gr.Image(type="pil", label='Human Image')
            with gr.Row():
                is_checked = gr.Checkbox(label="Use auto-generated mask")
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Use auto-crop & resizing")
            with gr.Row():
                use_grounding = gr.Checkbox(label='Use Grounded Segment Anything', value=True)
            with gr.Row():
                has_hat = gr.Checkbox(label='Include hat')
                has_gloves = gr.Checkbox(label='Include gloves')

        with gr.Column():
            garm_img = gr.Image(type="pil", label="Garment Image")
            prompt = gr.Textbox(label="Garment Description", placeholder="e.g., Short Sleeve Round Neck T-shirt")

        with gr.Column():
            masked_img = gr.Image(label="Masked image output")
        with gr.Column():
            image_out = gr.Image(label="Output")

    with gr.Column():
        try_button = gr.Button("Try-on")
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Slider(minimum=20, maximum=40, value=20, step=1, label="Denoising Steps")
                seed = gr.Number(label="Seed", value=42)
        error_output = gr.Textbox(label="Error Messages")

    try_button.click(fn=start_tryon, 
                     inputs=[imgs, garm_img, prompt, is_checked, is_checked_crop, use_grounding, has_hat, has_gloves, denoise_steps, seed], 
                     outputs=[image_out, masked_img, error_output])

if __name__ == "__main__":
    image_blocks.launch(share=True)