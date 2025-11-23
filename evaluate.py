import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import models, transforms
from PIL import Image
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm
import yaml
import argparse
# ================= evaluation=================

# VGG for Style Loss
VGG_STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
VGG_STYLE_WEIGHTS = {'relu1_1': 1.0, 'relu2_1': 0.8, 'relu3_1': 0.5, 'relu4_1': 0.3, 'relu5_1': 0.1}

# GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================


# --- CLIP Score  ---

def load_clip_model():
    """Load OpenAI CLIP Model and Preprocessor"""
    print(f"Loading CLIP (ViT-B/32) on {DEVICE}...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess

def calculate_clip_score(model, preprocess, img_pil_a, img_pil_b):
    """
    Calculates the cosine similarity between two images in CLIP feature space (CLIP-I Score).
    Used for style similarity.
    """
    img_a_tensor = preprocess(img_pil_a).unsqueeze(0).to(DEVICE)
    img_b_tensor = preprocess(img_pil_b).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat_a = model.encode_image(img_a_tensor)
        feat_b = model.encode_image(img_b_tensor)

        feat_a /= feat_a.norm(dim=-1, keepdim=True)
        feat_b /= feat_b.norm(dim=-1, keepdim=True)

        similarity = (feat_a @ feat_b.T).item()

    return similarity

# --- SSIM  ---

def calculate_ssim(gen_img_np, content_img_pil):
    """
    Calculates Structural Similarity Index (SSIM) between generated and content image.
    Used for content preservation metric.
    """
    target_size = (gen_img_np.shape[1], gen_img_np.shape[0]) # (W, H)
    if content_img_pil.size != target_size:
        # Resize content image to match generated image size for SSIM calculation
        content_img_pil = content_img_pil.resize(target_size, Image.BICUBIC)
    
    content_img_np = np.array(content_img_pil)
    
    # Ensure both are HWC RGB for comparison
    if content_img_np.ndim == 2:
        content_img_np = np.stack([content_img_np]*3, axis=-1)
    elif content_img_np.shape[2] == 4:
         content_img_np = content_img_np[..., :3]
        
    return calc_ssim(content_img_np, gen_img_np, channel_axis=2, data_range=255)


# --- VGG Style Loss (Gram Matrix)  ---

def image_loader_vgg(image_pil, imsize=512):
    """
    Preprocesses PIL image for VGG-19, including ImageNet normalization.
    """
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = loader(image_pil).unsqueeze(0)
    return image.to(DEVICE, torch.float)

def gram_matrix(input):
    """
    Computes the Gram matrix for a feature map.
    """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)  
    G = torch.mm(features, features.t())
    # Normalize by the number of elements in each feature map
    return G.div(a * b * c * d)


class VGGFeatureExtractor(nn.Module):
    """
    VGG-19 wrapper to extract features from specified layers.
    """
    def __init__(self, style_layers):
        super(VGGFeatureExtractor, self).__init__()
        # Load VGG-19 pretrained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.feature_map_extractor = nn.Sequential()
        i = 0
        for layer in vgg:
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv{}_1'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu{}_1'.format(i)
                layer = nn.ReLU(inplace=False)
            
            self.feature_map_extractor.add_module(name, layer)
            if name in style_layers:
                # Stop after reaching the deepest style layer needed
                if name == style_layers[-1]:
                    break

    def forward(self, x):
        features = {}
        for name, layer in self.feature_map_extractor._modules.items():
            x = layer(x)
            if name in VGG_STYLE_LAYERS:
                features[name] = x
        return features


def calculate_vgg_style_loss(extractor, target_img_pil, style_grams, imsize=512):
    """
    Calculates the total VGG Style Loss for one target image.
    """
    # Preprocess target image for VGG
    target_tensor = image_loader_vgg(target_img_pil, imsize)
    
    with torch.no_grad():
        target_features = extractor(target_tensor)

    style_loss = 0.0
    for name in VGG_STYLE_LAYERS:
        target_gram = gram_matrix(target_features[name])
        style_gram = style_grams[name]
        
        # Mean Squared Error (MSE) between Gram matrices
        layer_loss = F.mse_loss(target_gram, style_gram)
        
        # Weighted accumulation
        style_loss += VGG_STYLE_WEIGHTS[name] * layer_loss

    return style_loss.item()

def load_config(config_path):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    
def main():
    parser = argparse.ArgumentParser(description="Style Diffusion Evaluation Script.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    
    try:
        cfg = load_config(args.config)
        cfg['device'] = torch.device(cfg.get('device', 'cpu') if torch.cuda.is_available() else "cpu")
        print(f"Configuration loaded from {args.config}. Device: {cfg['device']}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    
    # Generated content path
    generated_dir = cfg.get('generated_dir', "./output/test_run/content_stylized")
    
    # Original content path
    content_dir = cfg.get('content_path', "./data/content")
    
    # Original style path
    style_image_path = cfg.get('style_path', "./data/style/van_gogh/000.jpg")
        
        
    # 1. prepare model
    clip_model, clip_preprocess = load_clip_model()
    vgg_extractor = VGGFeatureExtractor(VGG_STYLE_LAYERS).to(DEVICE).eval()
    
    if not os.path.exists(style_image_path):
        print(f"Error: Style reference image not found at {style_image_path}")
        return
    
    style_img_pil = Image.open(style_image_path).convert("RGB")
    
    # GRAM MATRIX (VGG Style Loss)
    style_vgg_tensor = image_loader_vgg(style_img_pil, imsize=512)
    with torch.no_grad():
        style_features = vgg_extractor(style_vgg_tensor)
        style_grams = {name: gram_matrix(feat) for name, feat in style_features.items()}

    # GRAB images
    gen_files = sorted(glob.glob(os.path.join(generated_dir, "*.jpg")) + 
                       glob.glob(os.path.join(generated_dir, "*.png")))
    
    if not gen_files:
        print("No generated images found.")
        return

    ssim_scores = []
    clip_scores = []
    style_losses = []

    print(f"Starting evaluation for {len(gen_files)} images...")

    # 2. evaluate all images
    for gen_path in tqdm(gen_files, desc="Evaluating Images"):
        filename = os.path.basename(gen_path)
        
        # orginal image and generated image pair
        base = os.path.splitext(filename)[0]
        content_path = next((os.path.join(content_dir, base + ext) 
                             for ext in [".jpg", ".png", ".jpeg"] 
                             if os.path.exists(os.path.join(content_dir, base + ext))), None)
        
        if not content_path:
            print(f"\nSkipping {filename}: Content image not found in {content_dir}.")
            continue

        img_gen = Image.open(gen_path).convert("RGB")
        img_content = Image.open(content_path).convert("RGB")
        img_gen_np = np.array(img_gen)

        # --- A. SSIM Score --
        ssim_val = calculate_ssim(img_gen_np, img_content)
        
        # --- B. CLIP-I Score  ---
        clip_val = calculate_clip_score(clip_model, clip_preprocess, img_gen, style_img_pil)

        # --- C. VGG Style Loss  ---
        # 使用 512x512 作为 VGG 评估的通用尺寸
        style_loss_val = calculate_vgg_style_loss(vgg_extractor, img_gen, style_grams, imsize=512)

        ssim_scores.append(ssim_val)
        clip_scores.append(clip_val)
        style_losses.append(style_loss_val)

    # 3. 输出统计结果
    if ssim_scores:
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        avg_clip = sum(clip_scores) / len(clip_scores)
        avg_style_loss = sum(style_losses) / len(style_losses)
        
        print("\n" + "="*50)
        print("  Style Diffusion Combined Evaluation Results")
        print("="*50)
        print(f"Images Processed: {len(ssim_scores)}")
        print("\n[内容保留指标 structure similarity SSIM]")
        print(f"Avg SSIM:            {avg_ssim:.4f} (Higher is better structure preservation)")
        print("\n[风格一致性指标 style similarity CLIP-I Score and  Style Loss]")
        print(f"Avg CLIP-I Score:    {avg_clip:.4f} (Higher is better style semantic match)")
        print(f"Avg VGG Style Loss:  {avg_style_loss:.4f} (Lower is better style texture match)")
        print("="*50)
    else:
        print("No valid image pairs evaluated.")

if __name__ == "__main__":
    main()