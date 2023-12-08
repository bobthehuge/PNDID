import cv2
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import numpy as np
import struct

MAGIC = struct.pack('<L', 2049)
COUNT = struct.pack('<L', 10000)
NROWS = struct.pack('<L', 28)
NCOLS = struct.pack('<L', 28)

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

def clean(image):
    _,cl = cv2.threshold(image, 70, 255, cv2.THRESH_TOZERO)
    return cl

def expand_set(name: str, offset: int, count: int) -> None:
    fimg = open(name, 'rb')
    fimg.read(offset)

    fres = open(name, 'ab')
    img = np.zeros([28,28], dtype=np.uint8)

    for k in range(count):
        for i in range(28):
            for j in range(28):
                img[i][j] = int.from_bytes(fimg.read(1), 'big')

        d_img = elastic_transform(img, 34, 4)
        # fres.write(cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY).reshape(-1))
        fres.write(d_img.reshape(-1))

    fimg.close()
    fres.close()
