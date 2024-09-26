import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
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
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
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
    args = SLConfig.fromfile(cache_config_file)
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

# Define example lists
logger.info("Defining example lists...")
garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

human_ex_list = human_list_path

logger.info(f"Found {len(garm_list_path)} garment examples and {len(human_ex_list)} human examples.")

# Utility functions
def detect_clothing(img, has_hat, has_gloves, human_img):
    logger.info("Detecting clothing...")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_source = img.convert("RGB").resize((768,1024))
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    hat = ', hat' if has_hat else ''
    gloves = ', gloves' if has_gloves else ''
    
    prediction = predict(
        model=groundingdino_model, 
        image=image_transformed, 
        caption=f"clothing clothes tops bottoms{hat}{gloves}",
        box_threshold=0.3,
        text_threshold=0.25
    )
    
    # Unpack the prediction
    detected_boxes = prediction[0]  # Assuming boxes are the first element
    
    segmented_frame_masks = segment(image, sam_predictor, boxes=detected_boxes)
    
    mask = segmented_frame_masks[0][0].cpu().numpy()

    for i in range(1, len(segmented_frame_masks)):
        mask += segmented_frame_masks[i][0].cpu().numpy()
        logger.info(f'Added {i} total mask(s) to initial')
    
    keypoints = openpose_model(human_img.resize((384,512)))
    model_parse, _ = parsing_model(human_img.resize((384,512)))
    mask2, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask2 = mask2.resize((768,1024))
    image_mask_pil = Image.fromarray((np.asarray(mask2)-mask).astype(np.uint8))
    return image_mask_pil

def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    return masks.cpu()

def start_tryon(human_img_dict, garm_img, garment_des, is_checked, is_checked_crop, use_grounding, has_hat, has_gloves, denoise_steps, seed):
    logger.info("Starting try-on process...")
    device = "cuda"
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

        # Convert seed to integer
    seed = int(seed) if seed is not None else None
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None


    garm_img = garm_img.convert("RGB").resize((768,1024))
    
    # Check if human_img_dict is a dictionary and extract the image
    if isinstance(human_img_dict, dict) and 'image' in human_img_dict:
        human_img_orig = human_img_dict['image'].convert("RGB")
    else:
        logger.error("Invalid input format for human image")
        return None, None

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
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))

    if is_checked:
        logger.info('Automasking...')
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    elif use_grounding:
        mask = detect_clothing(human_img, has_hat, has_gloves, human_img)
    elif isinstance(human_img_dict, dict) and 'mask' in human_img_dict:
        mask = human_img_dict['mask'].convert("L").resize((768, 1024))
    else:
        mask = Image.fromarray(np.array(human_img.convert("L").resize((768, 1024))) > 128)

    mask_gray = (1-transforms.ToTensor()(mask)) * transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(human_img))
    mask_gray = transforms.ToPILImage()((mask_gray+1.0)/2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
        
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
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

            pose_img =  transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(pose_img)).unsqueeze(0).to(device,torch.float16)
            garm_tensor =  transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(garm_img)).unsqueeze(0).to(device,torch.float16)
            generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
            images = pipe(
                prompt_embeds=prompt_embeds.to(device,torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength = 1.0,
                pose_img = pose_img,
                text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                cloth = garm_tensor,
                mask_image=mask,
                image=human_img, 
                height=1024,
                width=768,
                ip_adapter_image = garm_img.resize((768,1024)),
                guidance_scale=2.0,
            )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray

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
            seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42, precision=0)  # precision=0 ensures integer
    logger.info("Setting up try-on function...")
    try_button.click(
        fn=start_tryon, 
        inputs=[
            imgs,  # This is now directly the human image
            garm_img, 
            prompt, 
            is_checked,
            is_checked_crop,
            use_grounding, 
            has_hat, 
            has_gloves, 
            denoise_steps, 
            seed
        ], 
        outputs=[image_out, masked_img], 
        api_name='tryon'
    )
logger.info("Launching Gradio interface...")
image_blocks.launch(share=True)