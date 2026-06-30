###############################################################################
# IMPORTS
###############################################################################
import os, argparse, numpy as np, cv2
from PIL import Image
import torch

# Diffusers pipeline imports
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
)

# Optional SDXL ControlNet pipeline (may not exist in older diffusers versions)
try:
    from diffusers import StableDiffusionXLControlNetPipeline
    SDXL_CN_AVAILABLE = True
except Exception:
    SDXL_CN_AVAILABLE = False


###############################################################################
# DEVICE SELECTION (Apple M1/M2, CUDA, CPU) + DEFAULT dtype
###############################################################################
def device_and_dtype():
    """
    Determines the best available device and an appropriate torch dtype.
    - Apple Silicon → 'mps' with float32
    - NVIDIA CUDA → 'cuda' with float16 for faster inference
    - CPU → float32
    """
    if torch.backends.mps.is_available():
        # Ensures fallback to CPU doesn't cause crashes in kernels
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps", torch.float32

    # CUDA available → use fp16 for speed
    return ("cuda" if torch.cuda.is_available() else "cpu",
            torch.float16 if torch.cuda.is_available() else torch.float32)


###############################################################################
# BASIC IMAGE I/O HELPERS
###############################################################################
def load_rgb(path): 
    """Load an image as RGB (3-channel)."""
    return Image.open(path).convert("RGB")

def load_mask(path): 
    """Load a single-channel grayscale mask for blending."""
    return Image.open(path).convert("L")

def save_img(img, path):
    """Ensure output folder exists and save an image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


###############################################################################
# PIPELINE SETUP: TEXT-TO-IMAGE (SDXL or SD1.5)
###############################################################################
def setup_text2img(model_id, device, dtype):
    """
    Loads a Stable Diffusion text-to-image pipeline.
    Uses SDXL if model ID contains 'stable-diffusion-xl',
    otherwise uses SD1.5 or other SD variants.
    """
    if "stable-diffusion-xl" in model_id:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )

    pipe = pipe.to(device)

    # Try enabling memory optimizations (not available everywhere)
    try: 
        pipe.enable_attention_slicing()
    except Exception: 
        pass

    return pipe


###############################################################################
# PIPELINE SETUP: InstructPix2Pix EDITING
###############################################################################
def setup_edit(device, dtype):
    """Loads the Instruct-Pix2Pix editing pipeline."""
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=dtype,
        safety_checker=None,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    try: 
        pipe.enable_attention_slicing()
    except Exception: 
        pass

    return pipe


###############################################################################
# UTILITY: Generate Canny Edge Maps From an Image
###############################################################################
def canny_from_image(image_pil, low=100, high=200):
    """Runs Canny edge detector and returns a PIL grayscale image."""
    img = np.array(image_pil.convert("RGB"))
    edges = cv2.Canny(img, low, high)
    return Image.fromarray(edges)


###############################################################################
# PIPELINE SETUP: CONTROLNET (SD 1.5 or SDXL, T2I or IMG2IMG)
###############################################################################
def setup_controlnet(model_base, controlnet_model_id, device, dtype, img2img=False, sdxl=False):
    """
    Loads a ControlNet pipeline. Works for:
    - SD1.5 ControlNet
    - SDXL ControlNet (if installed)
    - Text→Image
    - Image→Image
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=dtype,
        use_safetensors=True
    )

    # SDXL ControlNet?
    if sdxl and SDXL_CN_AVAILABLE:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_base,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True
        )
    else:
        # SD 1.5 IMAGE-TO-IMAGE ControlNet
        if img2img:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                model_base,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True
            )
        # SD 1.5 TEXT-TO-IMAGE ControlNet
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_base,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True
            )

    pipe = pipe.to(device)

    try: 
        pipe.enable_attention_slicing()
    except Exception: 
        pass

    return pipe


###############################################################################
# OPTIONAL: LOAD LoRA WEIGHTS IF PROVIDED
###############################################################################
def attach_lora_if_any(pipe, lora_path):
    """
    If a LoRA folder is provided, load and fuse the LoRA weights into the model.
    """
    if not lora_path:
        return pipe

    try:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
        print(f"[OK] Fused LoRA from {lora_path}")
    except Exception as e:
        print(f"[WARN] Could not load LoRA: {e}")

    return pipe


###############################################################################
# MODE 1: Text → Image (T2I)
###############################################################################
def run_t2i(args):
    device, dtype = device_and_dtype()

    pipe = setup_text2img(args.model, device, dtype)
    pipe = attach_lora_if_any(pipe, args.lora_path)

    # Seed for reproducibility
    generator = (torch.Generator(device=device).manual_seed(args.seed)
                 if args.seed is not None else None)

    # Perform inference
    out = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        generator=generator
    ).images[0]

    save_img(out, args.output)
    print(f"[OK] saved -> {args.output}")


###############################################################################
# MODE 2: InstructPix2Pix Editing
###############################################################################
def run_edit(args):
    device, dtype = device_and_dtype()
    pipe = setup_edit(device, dtype)

    image = load_rgb(args.input_image)
    generator = (torch.Generator(device=device).manual_seed(args.seed)
                 if args.seed is not None else None)

    out = pipe(
        prompt=args.instruction,
        image=image,
        num_inference_steps=args.steps,
        image_guidance_scale=args.img_guidance,
        guidance_scale=args.cfg,
        generator=generator
    ).images[0]

    # Optional mask blending
    if args.mask:
        mask = load_mask(args.mask)
        m = np.asarray(mask).astype(np.float32) / 255.0
        base = np.asarray(image).astype(np.float32)
        edited = np.asarray(out).astype(np.float32)

        # Alpha blend edited image with original image
        blended = (m[..., None] * edited + (1 - m[..., None]) * base).astype(np.uint8)
        out = Image.fromarray(blended)

    save_img(out, args.output)
    print(f"[OK] saved -> {args.output}")


###############################################################################
# MODE 3: ControlNet Text→Image
###############################################################################
def run_t2i_cn(args):
    device, dtype = device_and_dtype()
    sdxl = "stable-diffusion-xl" in args.model.lower()

    pipe = setup_controlnet(
        model_base=args.model,
        controlnet_model_id=args.controlnet_model,
        device=device,
        dtype=dtype,
        img2img=False,
        sdxl=sdxl
    )

    pipe = attach_lora_if_any(pipe, args.lora_path)

    # Load control image or compute canny edges
    if args.control_image:
        control = load_rgb(args.control_image).convert("L")
    else:
        raise SystemExit("--control_image required for t2i_cn. You can also use --auto_canny.")

    if args.auto_canny:
        control = canny_from_image(
            load_rgb(args.control_image),
            args.canny_low, args.canny_high
        )

    generator = (torch.Generator(device=device).manual_seed(args.seed)
                 if args.seed is not None else None)

    out = pipe(
        prompt=args.prompt,
        image=control,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        generator=generator
    ).images[0]

    save_img(out, args.output)
    print(f"[OK] saved -> {args.output}")


###############################################################################
# MODE 4: ControlNet Image→Image
###############################################################################
def run_img2img_cn(args):
    device, dtype = device_and_dtype()
    sdxl = "stable-diffusion-xl" in args.model.lower()

    pipe = setup_controlnet(
        model_base=args.model,
        controlnet_model_id=args.controlnet_model,
        device=device,
        dtype=dtype,
        img2img=True,
        sdxl=sdxl
    )

    image = load_rgb(args.input_image)

    if args.control_image:
        control = load_rgb(args.control_image).convert("L")
    else:
        # If --auto_canny enabled → compute edges from input image
        control = canny_from_image(image, args.canny_low, args.canny_high) if args.auto_canny else None

        if control is None:
            raise SystemExit("Provide --control_image or enable --auto_canny for img2img_cn.")

    generator = (torch.Generator(device=device).manual_seed(args.seed)
                 if args.seed is not None else None)

    out = pipe(
        prompt=args.prompt,
        image=image,
        control_image=control,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        strength=args.strength,
        generator=generator
    ).images[0]

    save_img(out, args.output)
    print(f"[OK] saved -> {args.output}")


###############################################################################
# ARGUMENT PARSER (CLI INTERFACE)
###############################################################################
def build_parser():
    """
    Builds CLI with 4 subcommands:
    - t2i
    - edit
    - t2i_cn
    - img2img_cn
    """
    p = argparse.ArgumentParser("Instruct-Imagen (Open Replica) — macOS M1 + ControlNet")
    sub = p.add_subparsers(dest="mode", required=True)

    # ----------------------- TEXT → IMAGE -----------------------
    p_t2i = sub.add_parser("t2i")
    p_t2i.add_argument("--prompt", required=True)
    p_t2i.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Default SDXL; for M1 use 'runwayml/stable-diffusion-v1-5' for speed.")
    p_t2i.add_argument("--height", type=int, default=768)
    p_t2i.add_argument("--width", type=int, default=768)
    p_t2i.add_argument("--steps", type=int, default=30)
    p_t2i.add_argument("--cfg", type=float, default=6.5)
    p_t2i.add_argument("--seed", type=int, default=42)
    p_t2i.add_argument("--lora_path", default=None)
    p_t2i.add_argument("--output", default="outputs/t2i.png")

    # ----------------------- INSTRUCT PIX2PIX -----------------------
    p_edit = sub.add_parser("edit")
    p_edit.add_argument("--instruction", required=True)
    p_edit.add_argument("--input_image", required=True)
    p_edit.add_argument("--steps", type=int, default=40)
    p_edit.add_argument("--cfg", type=float, default=7.0)
    p_edit.add_argument("--img_guidance", type=float, default=1.5)
    p_edit.add_argument("--seed", type=int, default=42)
    p_edit.add_argument("--mask", default=None)
    p_edit.add_argument("--output", default="outputs/edit.png")

    # ----------------------- CONTROLNET T2I -----------------------
    p_t2i_cn = sub.add_parser("t2i_cn")
    p_t2i_cn.add_argument("--prompt", required=True)
    p_t2i_cn.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p_t2i_cn.add_argument("--controlnet_model", default="lllyasviel/control_v11p_sd15_canny")
    p_t2i_cn.add_argument("--control_image", default=None)
    p_t2i_cn.add_argument("--auto_canny", action="store_true")
    p_t2i_cn.add_argument("--canny_low", type=int, default=100)
    p_t2i_cn.add_argument("--canny_high", type=int, default=200)
    p_t2i_cn.add_argument("--height", type=int, default=512)
    p_t2i_cn.add_argument("--width", type=int, default=512)
    p_t2i_cn.add_argument("--steps", type=int, default=30)
    p_t2i_cn.add_argument("--cfg", type=float, default=7.0)
    p_t2i_cn.add_argument("--seed", type=int, default=42)
    p_t2i_cn.add_argument("--lora_path", default=None)
    p_t2i_cn.add_argument("--output", default="outputs/t2i_cn.png")

    # ----------------------- CONTROLNET IMG2IMG -----------------------
    p_i2i_cn = sub.add_parser("img2img_cn")
    p_i2i_cn.add_argument("--prompt", required=True)
    p_i2i_cn.add_argument("--input_image", required=True)
    p_i2i_cn.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p_i2i_cn.add_argument("--controlnet_model", default="lllyasviel/control_v11p_sd15_canny")
    p_i2i_cn.add_argument("--control_image", default=None)
    p_i2i_cn.add_argument("--auto_canny", action="store_true")
    p_i2i_cn.add_argument("--canny_low", type=int, default=100)
    p_i2i_cn.add_argument("--canny_high", type=int, default=200)
    p_i2i_cn.add_argument("--strength", type=float, default=0.75,
                          help="0-1, how much to deviate from input")
    p_i2i_cn.add_argument("--steps", type=int, default=30)
    p_i2i_cn.add_argument("--cfg", type=float, default=7.0)
    p_i2i_cn.add_argument("--seed", type=int, default=42)
    p_i2i_cn.add_argument("--output", default="outputs/img2img_cn.png")

    return p


###############################################################################
# MAIN DISPATCH FUNCTION
###############################################################################
def main():
    args = build_parser().parse_args()

    if args.mode == "t2i":
        run_t2i(args)
    elif args.mode == "edit":
        run_edit(args)
    elif args.mode == "t2i_cn":
        run_t2i_cn(args)
    elif args.mode == "img2img_cn":
        run_img2img_cn(args)


###############################################################################
# ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()

