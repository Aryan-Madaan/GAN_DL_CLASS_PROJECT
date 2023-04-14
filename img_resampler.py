import os
entries = os.listdir("resized")
from PIL import Image

for i in entries:
    if i != ".DS_Store":
        name = i.split('.')[0]
        img_png = Image.open(f'resized/{i}')
        img_png = img_png.convert('RGB')
        img_png.save(f'new_resized/{name}.jpg')