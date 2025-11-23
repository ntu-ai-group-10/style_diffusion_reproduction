"""
prepare_dataset.py
-------------------
Quick dataset setup for StyleDiffusion reproduction.

Creates:
data/
 ├── content/
 │     ├── 000.jpg ...
 └── style/
       ├── van_gogh/
       ├── monet/
       └── ukiyoe/
Usage:
    python src/prepare_dataset.py --n_content 50 --size 512
"""
from tqdm import tqdm
import os, argparse, random
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
from datasets import load_dataset   # huggingface 'datasets' library

# Define styles to download, stored in personal gdrive
STYLE_URLS = {
    "van_gogh": [
        "https://drive.google.com/uc?export=download&id=11jswdbZIc2OutOQ3s71x6xx6PYYDbnKp", # Starry Night
        "https://drive.google.com/uc?export=download&id=1qgxoBVwz79uWD9acVtP8wIk-NBi3xPn3", # Sunflowers
    ],
    "monet": [
        "https://drive.google.com/uc?export=download&id=16tIRnZQT9tZIADCF_NbQpyDuySYKPJUc", # Impression, Sunrise
        "https://drive.google.com/uc?export=download&id=1r5dHwmzTism7fVQA94QGvbqPmpmTXvwI", # Beach at Pourville 
    ],
    "ukiyoe": [
        "https://drive.google.com/uc?export=download&id=1JcnqgKk-L1pFneS2laKHdrAymtHfop5G", # The Great Wave off Kanagawa
        "https://drive.google.com/uc?export=download&id=1FtcFKxst7vkoF_JGrxXpRMpC50XHwhiD", # Red Fuji
    ],
}

def download_style_images(output_dir, size=256):
    for style, urls in STYLE_URLS.items():
        outdir = Path(output_dir) / "style" / style
        outdir.mkdir(parents=True, exist_ok=True)
        for i, url in enumerate(urls):
            r = requests.get(url)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((size, size))
            img.save(outdir / f"{i:03d}.jpg")
    print(f"✅ Saved example styles to {output_dir}/style/")

def download_coco_images(output_dir, n_images=100, size=256):
    """Download subset of COCO training images via HuggingFace."""
    ds = load_dataset("detection-datasets/coco", split="train[:1%]")
    os.makedirs(output_dir, exist_ok=True)
    subset = random.sample(range(len(ds)), min(n_images, len(ds)))
    for i in subset:
        img = ds[i]['image'].convert("RGB")
        img = img.resize((size, size))
        img.save(f"{output_dir}/{i:04d}.jpg")
    print(f"✅ Saved {len(subset)} content images to {output_dir}/")

def download_flickr_images(output_dir, n_images=None, size=512):
    """
    Download images from Adyakanta/test_flickr30k via HuggingFace.
    Only saves images (resized), ignores captions/metadata.
    
    Args:
        output_dir (str): Path to save images.
        n_images (int, optional): Number of images to download. If None, download all.
        size (int): Size to resize images to (square). Set to None to keep original size.
    """
    print(f"⏳ Loading Flickr30k dataset (split='test')...")
    try:
        ds = load_dataset("Adyakanta/test_flickr30k", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = len(ds)
    print(f"Dataset loaded. Total available images: {total_samples}")

    if n_images is not None:
        limit = min(n_images, total_samples)
        indices = random.sample(range(total_samples), limit)
        print(f"Sampling {limit} random images...")
    else:
        indices = range(total_samples)
        print(f"Processing all {total_samples} images...")

    count = 0
    for i in tqdm(indices, desc="Saving images"):
        try:
            item = ds[i]
            
            img_id_raw = str(item['img_id'])
            if img_id_raw.endswith('.jpg'):
                img_name = img_id_raw
            else:
                img_name = f"{img_id_raw}.jpg"
            
            img = item['image'].convert("RGB") #make sure RGB
            
            # Resize if size specified
            if size is not None:
                img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            # 保存路径
            save_path = os.path.join(output_dir, img_name)
            img.save(save_path, quality=95)
            
            count += 1
            
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
            continue

    print(f"✅ Successfully saved {count} content images to {output_dir}/")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Root output directory")
    parser.add_argument("--n_content", type=int, default=100)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    download_style_images(args.out, size=args.size)
    # download_coco_images(os.path.join(args.out, "content_coco"),
    #                      n_images=args.n_content, size=args.size)
    download_flickr_images(os.path.join(args.out, "content_flickr"),
                           n_images=args.n_content, size=args.size)
    print("🎨 Dataset preparation complete!")



if __name__ == "__main__":
    main()