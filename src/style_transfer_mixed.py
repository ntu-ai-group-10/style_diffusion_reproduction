import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import copy
import logging
import clip
from models.improved_ddpm.script_util import i_DDPM

from src.helper import rgb_to_luma_601, prepare_image_as_tensor
from src.diffusion import ddim_deterministic, get_linear_alphas_cumprod
from src.style_transfer import style_reconstruction_loss, style_disentanglement_loss

def style_diffusion_fine_tuning_mixed(
    style1_color: torch.Tensor,
    style1_gray: torch.Tensor,
    style1_latent: torch.Tensor,
    style1_weight: float,
    style2_color: torch.Tensor,
    style2_gray: torch.Tensor,
    style2_latent: torch.Tensor,
    style2_weight: float,
    content_gray: list,
    content_latent: list,
    model: nn.Module,
    alphas_cumprod: list,
    clip_model,
    clip_preprocess,
    t_trans: int,
    s_rev: int,
    k: int,
    k_s: int,
    lr: float,
    lr_multiplier: float,
    lambda_style: float,
    lambda_l1: float,
    lambda_dir: float,
    device: str,
    logger=None,
):
    if logger is not None:
        logger.info(f"Starting mixed style transfer fine-tuning (Weights: {style1_weight}/{style2_weight})...")

    # Initialize fine-tuned model
    model_finetuned = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model_finetuned.parameters(), lr=lr)
    lambda_lr = lambda epoch: lr_multiplier ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # Freeze CLIP model
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Pre-calculate CLIP embeddings for BOTH styles to save time
    # We need the "Target" embeddings to be the weighted average of Style 1 and Style 2
    with torch.no_grad():
        preprocess_normalize = T.Compose([
            T.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + 
            clip_preprocess.transforms[:2] + 
            clip_preprocess.transforms[4:]
        )

        # Encode Style 1
        s1_c_emb = clip_model.encode_image(preprocess_normalize(style1_color)).detach()
        s1_g_emb = clip_model.encode_image(preprocess_normalize(style1_gray)).detach()

        # Encode Style 2
        s2_c_emb = clip_model.encode_image(preprocess_normalize(style2_color)).detach()
        s2_g_emb = clip_model.encode_image(preprocess_normalize(style2_gray)).detach()

        # Create the MIXED target embeddings
        target_style_color_emb = (style1_weight * s1_c_emb) + (style2_weight * s2_c_emb)
        target_style_gray_emb  = (style1_weight * s1_g_emb) + (style2_weight * s2_g_emb)

    # Training loop
    for iter in range(k):
        if logger is not None:
            logger.info(f"Starting fine-tuning iteration {iter+1}...")

        # --- PHASE 1: Style Reconstruction (Optimize for both styles) ---
        for i in range(k_s):
            
            # Setup DDIM steps
            ddim_timesteps_backward = np.linspace(0, t_trans, s_rev, dtype=int)[::-1]
            
            #clear gradients at the start of iteration
            optimizer.zero_grad()

            # --- Path A: Style 1 ---
            x_t_s1 = style1_latent.clone().to(device)
            # Run diffusion step for Style 1
            for step in range(len(ddim_timesteps_backward)-1):
                x_prev_s1 = ddim_deterministic(
                    x_start=x_t_s1, model=model_finetuned, alphas_cumprod=alphas_cumprod,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device, requires_grad=True
                )
                if step == len(ddim_timesteps_backward)-2: # Final step
                    loss_s1 = style_reconstruction_loss(x_prev_s1, style1_color.detach().to(device))
                    weighted_loss_s1 = (loss_s1 * style1_weight * lambda_style)
                    weighted_loss_s1.backward()

                x_t_s1 = x_prev_s1.detach() # Detach for next step logic, but we kept graph in x_prev_s1

            # --- Path B: Style 2 ---
            x_t_s2 = style2_latent.clone().to(device)
            # Run diffusion step for Style 2
            for step in range(len(ddim_timesteps_backward)-1):
                x_prev_s2 = ddim_deterministic(
                    x_start=x_t_s2, model=model_finetuned, alphas_cumprod=alphas_cumprod,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device, requires_grad=True
                )
                if step == len(ddim_timesteps_backward)-2: # Final step
                    loss_s2 = style_reconstruction_loss(x_prev_s2, style2_color.detach().to(device))
                    weighted_loss_s2 = (loss_s2 * style2_weight * lambda_style)
                    weighted_loss_s2.backward()
                
                x_t_s2 = x_prev_s2.detach()

            # Now that both gradients are accumulated, we update the weights
            optimizer.step()

        # --- PHASE 2: Style Disentanglement (Optimize for Mixed Style) ---
        for i in range(len(content_latent)):
            x_t = content_latent[i].clone().to(device)
            ddim_timesteps_backward = np.linspace(0, t_trans, s_rev, dtype=int)[::-1]

            for step in range(len(ddim_timesteps_backward)-1):
                x_t_prev = ddim_deterministic(
                    x_start=x_t, model=model_finetuned, alphas_cumprod=alphas_cumprod,
                    ddim_timesteps=[ddim_timesteps_backward[step], ddim_timesteps_backward[step+1]],
                    device=device, requires_grad=True
                )
                
                # Get CLIP embedding of current estimate
                I_content_color_emb = clip_model.encode_image(preprocess_normalize(x_t_prev))
                
                # Get CLIP embedding of reference content (gray)
                with torch.no_grad():
                    I_content_gray_emb = clip_model.encode_image(preprocess_normalize(content_gray[i])).detach()

                # Calculate loss against the mixed targets we computed earlier
                loss_sd = style_disentanglement_loss(
                    I_content_color_emb, 
                    I_content_gray_emb, 
                    target_style_color_emb,
                    target_style_gray_emb, 
                    lambda_l1, 
                    lambda_dir
                )

                optimizer.zero_grad()
                loss_sd.backward()
                optimizer.step()

                x_t = x_t_prev.detach()

        scheduler.step()

    if logger is not None:
        logger.info("Mixed style transfer fine-tuning completed.")
    return model_finetuned

if __name__ == "__main__":
    #Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # CHECKPOINT_PATH = "models/checkpoints/256x256_diffusion_uncond.pt"
    CHECKPOINT_PATH = "models/checkpoints/512x512_diffusion.pt"
    IMAGE_SIZE = 512

    DIFFUSION_NUM_TIMESTEPS = 1000
    DIFFUSION_BETA_START = 0.0001
    DIFFUSION_BETA_END = 0.02

    T_TRANS = 601
    S_FOR = 40
    S_REV = 6

    K = 5
    K_S = 50
    LR = 0.000004
    LR_MULTIPLIER = 1.2
    LAMBDA_STYLE = 1
    LAMBDA_L1 = 10
    LAMBDA_DIR = 6

    N_CONTENT_SAMPLE = 50

    CONTENT_GRAY_PATH = "output/test_run_van_gogh/content_processed/"
    CONTENT_LATENT_PATH = "output/test_run_van_gogh/content_latents/"

    STYLE1_COLOR_PATH = "data/style/van_gogh/000.jpg"
    STYLE1_GRAY_PATH = "output/test_run_van_gogh/style_processed/style.pt"
    STYLE1_LATENT_PATH = "output/test_run_van_gogh/style_latents/style.pt"
    STYLE1_WEIGHT = 0.5

    STYLE2_COLOR_PATH = "data/style/monet/000.jpg"
    STYLE2_GRAY_PATH = "output/test_run_monet/style_processed/style.pt"
    STYLE2_LATENT_PATH = "output/test_run_monet/style_latents/style.pt"
    STYLE2_WEIGHT = 0.5

    SAMPLE_CONTENT_ID = 10

    OUTPUT_DIR = "output/"
    OUTPUT_PREFIX = "style_transfer_mixed__"
    OUTPUT_STYLIZED_DIR = "output/van_gogh_monet_mixed/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(OUTPUT_DIR, f"style_transfer_mixed.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Main")
    logger.info(f"Starting example usage of style transfer mixed...")

    #load model
    model = i_DDPM("IMAGENET", IMAGE_SIZE)
    init_ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(init_ckpt)
    model.to(DEVICE)

    #get alphas_cumprod
    alphas_cumprod = get_linear_alphas_cumprod(
        timesteps=DIFFUSION_NUM_TIMESTEPS,
        beta_start=DIFFUSION_BETA_START,
        beta_end=DIFFUSION_BETA_END
    )

    #get sample style color, gray, and latent
    style1_color = Image.open(STYLE1_COLOR_PATH)
    style1_color = prepare_image_as_tensor(style1_color, image_size=IMAGE_SIZE, device=DEVICE)
    style1_gray = torch.load(STYLE1_GRAY_PATH, map_location=DEVICE, weights_only=True)
    style1_latent = torch.load(STYLE1_LATENT_PATH, map_location=DEVICE, weights_only=True)

    style2_color = Image.open(STYLE2_COLOR_PATH)
    style2_color = prepare_image_as_tensor(style2_color, image_size=IMAGE_SIZE, device=DEVICE)
    style2_gray = torch.load(STYLE2_GRAY_PATH, map_location=DEVICE, weights_only=True)
    style2_latent = torch.load(STYLE2_LATENT_PATH, map_location=DEVICE, weights_only=True)
    
    #get sample content color, gray, and latent
    content_filenames = [f for f in os.listdir(CONTENT_LATENT_PATH) if f.lower().endswith('.pt')]
    content_filenames.sort()
    sample_content_filenames = content_filenames[:N_CONTENT_SAMPLE]
    logger.info(f"Sample content image filename: {sample_content_filenames[SAMPLE_CONTENT_ID]}")

    content_gray = []
    for file in sample_content_filenames:
        content_gray.append(
            torch.load(os.path.join(CONTENT_GRAY_PATH, file), map_location=DEVICE, weights_only=True)            
        )

    content_latent = []
    for file in sample_content_filenames:
        content_latent.append(
            torch.load(os.path.join(CONTENT_LATENT_PATH, file), map_location=DEVICE, weights_only=True)
        )

    #test reverse diffusion to reconstruct style
    ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    style1_x0_est = ddim_deterministic(style1_latent, model, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    style1_x0_est_image = style1_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style1_x0_est_image = ((style1_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style1_x0_est_image = (style1_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(style1_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style1_image_after_reverse_diffusion.jpg"))
    style2_x0_est = ddim_deterministic(style2_latent, model, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    style2_x0_est_image = style2_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    style2_x0_est_image = ((style2_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    style2_x0_est_image = (style2_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(style2_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "style2_image_after_reverse_diffusion.jpg"))

    #test reverse diffusion to reconstruct sample content
    ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    content_x0_est = ddim_deterministic(content_latent[SAMPLE_CONTENT_ID], model, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    content_x0_est_image = content_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_x0_est_image = ((content_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_x0_est_image = (content_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(content_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_reverse_diffusion.jpg"))

    #load clip model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

    #apply style diffusion fine-tuning
    model_finetuned = style_diffusion_fine_tuning_mixed(
        style1_color,
        style1_gray,
        style1_latent,
        STYLE1_WEIGHT,
        style2_color,
        style2_gray,
        style2_latent,
        STYLE2_WEIGHT,
        content_gray,
        content_latent,
        model,
        alphas_cumprod,
        clip_model,
        clip_preprocess,
        T_TRANS,
        S_REV,
        K,
        K_S,
        LR,
        LR_MULTIPLIER,
        LAMBDA_STYLE,
        LAMBDA_L1,
        LAMBDA_DIR,
        DEVICE,
        logger=logger,
    )
    torch.save(model_finetuned.state_dict(), os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}finetuned_style_model.pt"))
    # model_finetuned = i_DDPM("IMAGENET", IMAGE_SIZE)
    # finetuned_ckpt = torch.load(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}finetuned_style_model.pt"), weights_only=True)
    # model_finetuned.load_state_dict(finetuned_ckpt)
    # model_finetuned.eval().to(DEVICE)

    #test reverse diffusion to reconstruct sample_content with finetuned model
    ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
    # ddim_timesteps_backward = np.linspace(0, T_TRANS, S_FOR, dtype=int)
    ddim_timesteps_backward = ddim_timesteps_backward[::-1]
    content_x0_est = ddim_deterministic(content_latent[SAMPLE_CONTENT_ID], model_finetuned, alphas_cumprod, ddim_timesteps_backward, DEVICE, logger=logger)
    content_x0_est_image = content_x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
    content_x0_est_image = ((content_x0_est_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
    content_x0_est_image = (content_x0_est_image * 255).astype(np.uint8)
    Image.fromarray(content_x0_est_image).save(os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + "content_image_after_reverse_diffusion_with_finetuned_model.jpg"))

    #generate stylized image
    logger.info("Generating stylized images using fine-tuned model...")

    stylized_output_path = OUTPUT_STYLIZED_DIR
    os.makedirs(stylized_output_path, exist_ok=True)
    for i in range(len(content_latent)):
        x_t = content_latent[i].clone().to(DEVICE)
        ddim_timesteps_backward = np.linspace(0, T_TRANS, S_REV, dtype=int)
        ddim_timesteps_backward = ddim_timesteps_backward[::-1]
        x0_est = ddim_deterministic(x_t, model_finetuned, alphas_cumprod, ddim_timesteps_backward, device=DEVICE)

        stylized_image = x0_est.squeeze(0).permute(1, 2, 0).cpu().numpy()
        stylized_image = ((stylized_image + 1) / 2).clip(0, 1)  # scale back to [0,1]
        stylized_image = (stylized_image * 255).astype(np.uint8)
        Image.fromarray(stylized_image).save(os.path.join(stylized_output_path, f"{content_filenames[i].split('.')[0]}.jpg"))
    
    logger.info("Stylized images generated.")
