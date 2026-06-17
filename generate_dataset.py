import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import shutil

# Paths
REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

def augment_image(img, save_path):
    """One image నుండి multiple augmented images create చేయి"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    variations = []

    # Original
    variations.append(img)

    # Rotate
    variations.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    variations.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # Flip
    variations.append(cv2.flip(img, 1))

    # Brightness change
    bright = ImageEnhance.Brightness(pil_img).enhance(1.5)
    variations.append(cv2.cvtColor(np.array(bright), cv2.COLOR_RGB2BGR))

    dark = ImageEnhance.Brightness(pil_img).enhance(0.6)
    variations.append(cv2.cvtColor(np.array(dark), cv2.COLOR_RGB2BGR))

    # Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    variations.append(blurred)

    return variations

def create_fake(img):
    """Real image నుండి fake document create చేయి"""
    fake_versions = []

    # Text region tamper చేయి
    h, w = img.shape[:2]
    tampered = img.copy()

    # Random rectangle add చేయి (tampering simulate)
    x1, y1 = w//4, h//4
    x2, y2 = 3*w//4, h//2
    color = (
        int(np.mean(img[y1:y2, x1:x2, 0])) + 30,
        int(np.mean(img[y1:y2, x1:x2, 1])),
        int(np.mean(img[y1:y2, x1:x2, 2]))
    )
    cv2.rectangle(tampered, (x1, y1), (x2, y2), color, -1)
    fake_versions.append(tampered)

    # JPEG compression artifact
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    compressed = cv2.imdecode(enc, 1)
    fake_versions.append(compressed)

    # Color shift
    shifted = img.copy()
    shifted[:, :, 0] = np.clip(shifted[:, :, 0].astype(int) + 40, 0, 255)
    fake_versions.append(shifted)

    # Noise add చేయి
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    noisy = cv2.add(img, noise)
    fake_versions.append(noisy)

    return fake_versions

def process_images(source_folder):
    """Source folder లో images process చేయి"""
    images = []
    for f in os.listdir(source_folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            path = os.path.join(source_folder, f)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                images.append((f, img))
    return images

print("🔄 Dataset generating...")

# Source images (నువ్వు real folder లో పెట్టిన images)
source_images = process_images(REAL_DIR)

if len(source_images) == 0:
    print("⚠️ real/ folder లో images పెట్టు!")
else:
    real_count = 0
    fake_count = 0

    for filename, img in source_images:
        name = os.path.splitext(filename)[0]

        # Real augmented images
        real_variations = augment_image(img, REAL_DIR)
        for i, var in enumerate(real_variations):
            save_path = os.path.join(REAL_DIR, f"{name}_real_{i}.jpg")
            cv2.imwrite(save_path, var)
            real_count += 1

        # Fake versions create చేయి
        fake_versions = create_fake(img)
        for i, fake in enumerate(fake_versions):
            fake_augs = augment_image(fake, FAKE_DIR)
            for j, aug in enumerate(fake_augs):
                save_path = os.path.join(FAKE_DIR, f"{name}_fake_{i}_{j}.jpg")
                cv2.imwrite(save_path, aug)
                fake_count += 1

    print(f"✅ Real images: {real_count}")
    print(f"✅ Fake images: {fake_count}")
    print(f"✅ Total: {real_count + fake_count} images ready!")