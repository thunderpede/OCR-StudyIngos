import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXTS_PATH = os.path.join(BASE_DIR, "texts/synthetic_texts.txt")
FONTS_PATH = os.path.join(BASE_DIR, 'fonts')
BACKGROUNDS_PATH = os.path.join(BASE_DIR, 'backgpounds')


with open(TEXTS_PATH, "r", encoding="utf-8") as file:
    texts = file.readlines()
    
texts = [line.strip() for line in texts if line.strip()]

fonts = [os.path.join(FONTS_PATH, path) for path in os.listdir(FONTS_PATH)]
backgrounds = [os.path.join(BACKGROUNDS_PATH, path) for path in os.listdir(BACKGROUNDS_PATH)]


colors = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [192, 192, 192],
    [128, 128, 128],
    [64, 64, 64],
    [255, 165, 0],
    [255, 192, 203],
    [173, 216, 230],
    [144, 238, 144],
    [210, 105, 30],
    [75, 0, 130],
    [220, 20, 60],
    [154, 205, 50],
    [139, 69, 19],
    [255, 140, 0],
    [72, 61, 139],
    [123, 104, 238],
    [0, 206, 209],
    [255, 215, 0],
    [147, 112, 219]
]

sizes = [128, 152]
opacities = list(range(100, 200))
