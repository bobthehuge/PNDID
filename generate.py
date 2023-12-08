import cv2 as cv
import numpy as np
import os
import struct

from PIL import Image, ImageFont, ImageDraw
from distord import elastic_transform as deform

def load_fonts(folder, size):
    fonts_paths = [os.path.join(folder, file) for file in os.listdir(folder)]
    fonts = []

    for font_path in fonts_paths:
        try:
            fonts.append(ImageFont.truetype(font_path, size))
        except:
            print('Failed to load {} with size {}'.format(font_path, size))

    return fonts

def sample_font(font):
    imgs = []

    for i in range(10):
        img = Image.fromarray(np.zeros((28,28), dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        draw.text((7,0), str(i), 255, font=font)
        imgs.append((np.array(img),i))
        # imgs.append(deform(imgs[-1],40,5))

    return imgs

def write_dset(name: str, magic_id: int, dset) -> None:
    assert(len(dset) > 0)

    raw_imgs = open(name+'-img', 'wb')
    raw_lbls = open(name+'-lbl', 'wb')

    magic = struct.pack('>L', magic_id)
    count = struct.pack('>L', len(dset))
    size1 = struct.pack('>L', dset[0][0].shape[0])
    size2 = struct.pack('>L', dset[0][0].shape[1])

    raw_imgs.write(magic)
    raw_imgs.write(count)
    raw_imgs.write(size1)
    raw_imgs.write(size2)

    raw_lbls.write(magic)
    raw_lbls.write(count)

    for sample in dset:
        img, lbl = sample

        raw_imgs.write(img.reshape(-1))
        raw_lbls.write(lbl.to_bytes(1, 'big'))

    raw_imgs.close()
    raw_lbls.close()

fonts = load_fonts('fonts', 20)

kernel = np.ones((2,2), dtype=np.uint8)

total_set = []

for font in fonts:
    s1 = sample_font(font)
    s2 = [(cv.dilate(i,kernel,iterations=1),l) for (i,l) in s1]
    s3 = [(cv.dilate(i,kernel,iterations=2),l) for (i,l) in s1]
    total_set += s1+s2+s3

write_dset('t90k', 4242, total_set)
