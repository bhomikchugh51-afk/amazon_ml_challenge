📦 Amazon ML Challenge 2024 — Product Attribute Extraction from Images

Extract structured product attributes (weight, voltage, wattage, volume) directly from product images — moving from a classic OCR baseline to a vision-language model pipeline with LoRA fine-tuning.

📋 Overview

Given a product image and a target attribute (e.g. item_weight, voltage), the goal is to return a structured value + unit — e.g. {"value": 500, "unit": "g"}. This is the core task from the Amazon ML Challenge 2024, a real-world entity-extraction problem where the answer isn't always in clean text — it's printed on packaging, labels, and product photos in inconsistent formats.

This project explores three approaches of increasing sophistication:

OCR + regex baseline — the honest starting point to beat
Vision-language model (zero-shot) — a multimodal model reasoning directly over the image
LoRA fine-tuning — adapting the VLM on labeled data for better accuracy
🧠 Approach
1. OCR Baseline

Uses Tesseract OCR to extract raw text from the image, then applies attribute-specific regex patterns (e.g. matching \d+\s*(g|kg|lb|oz) for weight) to pull out a value + unit.

2. Vision-Language Model (Zero-Shot)

Replaces OCR with a small open vision-language model (Qwen2-VL-2B-Instruct), prompted to look at the image and return the requested attribute as structured JSON:

{"value": number, "unit": string}  or  {"value": null}
3. LoRA Fine-Tuning

Outlines a parameter-efficient fine-tuning setup using PEFT/LoRA, targeting the model's attention projection layers (q_proj, v_proj) to adapt it on the challenge's labeled dataset without full fine-tuning cost.

🛠️ Tech Stack
transformers — vision-language model loading & inference
peft — LoRA fine-tuning
accelerate, bitsandbytes — efficient model loading/training
Pillow — image handling
pytesseract — OCR baseline
🚀 Getting Started
Prerequisites
Python 3.9+
A CUDA-capable GPU recommended for the vision-language model steps
Installation
bash
git clone <your-repo-url>
cd amazon-ml-challenge-2024
pip install transformers peft accelerate bitsandbytes pillow pytesseract

Tesseract also needs to be installed at the system level (e.g. apt install tesseract-ocr on Linux, or via Homebrew on macOS).

Run

Open 09_amazon_ml_challenge_2024.ipynb in Jupyter and run cells in order:

OCR baseline (ocr_baseline())
Vision-language inference (predict())
LoRA fine-tuning setup (adapt with your own dataset)
📁 Project Structure
├── 09_amazon_ml_challenge_2024.ipynb   # Main notebook
└── README.md
⚠️ Current Status

This notebook is a working scaffold, not a fully trained/evaluated pipeline:

✅ OCR baseline implemented and runnable
✅ Vision-language zero-shot inference implemented and runnable
🚧 LoRA fine-tuning is outlined but not yet wired into a training loop — needs a Trainer/SFTTrainer loop over (image, prompt, target_json) tuples
🚧 No evaluation/accuracy numbers yet — needs a benchmark comparing OCR vs zero-shot VLM vs fine-tuned VLM
🔭 Roadmap
 Implement the full LoRA fine-tuning loop on the official challenge dataset
 Evaluate and compare OCR-only vs VLM zero-shot vs VLM fine-tuned on a fixed eval split
 Add post-processing for unit normalization (e.g. kg → g)
 Submit to Amazon ML Challenge 2025
📄 License

This project is for educational/portfolio purposes. Feel free to fork and adapt.

Built as a hands-on exploration of vision-language models and parameter-efficient fine-tuning for structured information extraction.
