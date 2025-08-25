from PIL import Image, ImageOps
import numpy as np

IMG_SIZE = (28, 28)

def preprocess_image(file) -> np.ndarray:
    img = Image.open(file).convert("L")
    img = ImageOps.invert(img)

    img_resized = img.resize((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", IMG_SIZE, color=0)
    canvas.paste(img_resized, ((IMG_SIZE[0]-20)//2, (IMG_SIZE[1]-20)//2))

    img_array = np.array(canvas).astype("float32") / 255.0
    img_array = 1.0 - img_array
    img_array = (img_array > 0.5).astype("float32")
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array