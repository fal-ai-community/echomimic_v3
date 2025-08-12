# pip install retina-face
# we recommand tensorflow==2.15
import sys

import numpy as np
import torch
from PIL import Image
from retinaface.pre_trained_models import get_model

retinaface_model = get_model(
    "resnet50_2020-07-20",
    max_size=2048,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def get_mask_coord(image_path):

    img = Image.open(image_path).convert("RGB")
    img = np.array(img)[:, :, ::-1]
    if img is None:
        raise ValueError(f"Exception while loading {img}")

    height, width, _ = img.shape

    facial_areas = retinaface_model.predict_jsons(img)

    if len(facial_areas) == 0:
        print(f"{image_path} has no face detected!")
        return None
    else:

        face = facial_areas[0]
        x, y, x2, y2 = face["bbox"]

        return y, y2, x, x2, height, width


if __name__ == "__main__":
    image_path = sys.argv[1]
    y, y2, x, x2, height, width = get_mask_coord(image_path)
    print(y, y2, x, x2, height, width)
    # Draw rectangle on image
    from PIL import ImageDraw
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([x, y, x2, y2], outline="red", width=5)
    image.save("result.jpg")
