# Instruct-Imagen (Open Replica) — macOS M1 Ready

This project **replicates the *behavior* of CVPR'24 *Instruct-Imagen*** using open-source components you can run on a **MacBook Pro M1**:

- **Text → Image** with Stable Diffusion XL (or fall back to SD 1.5 for lower memory)
- **Instruction-based Editing** with Instruct-Pix2Pix
- **Optional LoRA Fine-tuning** on a small instruction dataset (SD 1.5 recommended on M1)

> The original Google *Instruct-Imagen* and *Imagen* code/weights are **not public**. This repo is a practical, runnable approximation for your experiments and presentation.

---

## 0) macOS M1 Setup

- Install Python 3.10 or 3.11 (via Homebrew or pyenv).
- Create a venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```
- Install deps:
```bash
pip install -r requirements.txt
```
- Set MPS fallback (avoids crashes when ops are missing on Metal):
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

> **Tip:** On M1, half-precision is limited; we default to **float32** on MPS.

---

## 1) Quick Start

### A) Text → Image (SDXL)
```bash
python main_mps.py t2i   --prompt "A cozy living room in watercolor style"   --height 768 --width 768 --steps 30 --cfg 6.5   --output outputs/t2i_sdxl.png
```

If SDXL is too slow/memory-heavy on your M1, use SD 1.5:
```bash
python main_mps.py t2i   --prompt "A cozy living room in watercolor style"   --model runwayml/stable-diffusion-v1-5   --height 512 --width 512 --steps 25 --cfg 7.0   --output outputs/t2i_sd15.png
```

### B) Instruction-based Editing (Instruct-Pix2Pix)
Put an image at `examples/input.jpg`, then:
```bash
python main_mps.py edit   --instruction "Add a small Christmas tree near the window"   --input_image examples/input.jpg   --steps 40 --cfg 7.0 --img_guidance 1.5   --output outputs/edit.png
```
Optional masked blend (white=edit, black=keep):
```bash
python main_mps.py edit   --instruction "Turn the sky into a dramatic sunset"   --input_image examples/input.jpg   --mask examples/mask.png   --output outputs/edit_mask.png
```

---

## 2) LoRA Fine-tuning (Small-scale)

On M1, training is feasible with **SD 1.5** (smaller), batch size 1.

### Prepare tiny dataset
Folder structure:
```
data/raw/0001/input.jpg        # optional (for edit)
data/raw/0001/target.jpg       # required
data/raw/0001/instruction.txt  # e.g., "Add a red hat to the dog."
```
Create JSONL:
```bash
python dataset_prep.py --root data/raw --out data/train.jsonl
```

### Train LoRA (SD 1.5 recommended)
```bash
python train_lora_mps.py   --train_jsonl data/train.jsonl   --base_model runwayml/stable-diffusion-v1-5   --epochs 1 --bs 1 --lr 5e-5   --output_dir lora_out
```

### Use LoRA at inference (text→image path)
```bash
python main_mps.py t2i   --prompt "A wooden table with a red hat"   --model runwayml/stable-diffusion-v1-5   --lora_path lora_out/lora_epoch_1   --height 512 --width 512   --output outputs/t2i_lora.png
```

---

## 3) Notes Mapping to the Paper

- **Instruction following** → Instruct-Pix2Pix path (edit mode).
- **Text encoder + diffusion U-Net** → provided by the chosen SD pipeline.
- **Multi-modal control** → text + input image (+ optional mask). Reference-image conditioning can be added later.
- **Instruction tuning** → LoRA training on (input, instruction, target) triples.

This gives you a **runnable** and **presentable** reproduction of the *concepts* in the paper on Mac M1.

---

## 4) Troubleshooting

- **Slow / OOM**: prefer SD 1.5, use 512×512, steps 20–30.
- **Black/blank outputs**: reduce CFG (e.g., 5–7), or steps (25–30).
- **Instruct-Pix2Pix artifacts**: increase `--img_guidance` (1.5–2.0).



---

## 5) ControlNet (NEW)

### A) Text→Image with ControlNet (Canny, SD 1.5 recommended on M1)
Provide a **control image** (a photo to edge-detect) and enable canny:
```bash
python main_mps.py t2i_cn \
  --prompt "Detailed skyline at sunset, watercolor" \
  --model runwayml/stable-diffusion-v1-5 \
  --controlnet_model lllyasviel/control_v11p_sd15_canny \
  --control_image examples/input.jpg --auto_canny \
  --height 512 --width 512 --steps 30 --cfg 7.0 \
  --output outputs/t2i_cn.png
```

### B) Img2Img with ControlNet (structure-preserving edit)
Use your input as both the image to edit and the structure provider (Canny):
```bash
python main_mps.py img2img_cn \
  --prompt "Turn it into a rainy cyberpunk night scene" \
  --input_image examples/input.jpg \
  --model runwayml/stable-diffusion-v1-5 \
  --controlnet_model lllyasviel/control_v11p_sd15_canny \
  --auto_canny --steps 30 --cfg 7.0 --strength 0.75 \
  --output outputs/img2img_cn.png
```

> For SDXL ControlNet, supply an SDXL control model (e.g., a canny SDXL checkpoint) and keep image sizes modest on M1.
