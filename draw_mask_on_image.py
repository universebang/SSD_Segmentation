import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw#
import PIL.ImageFont as ImageFont
from skimage import io, transform, filters
from matplotlib import pyplot as plt
from scipy import misc

def draw_mask_on_image_array(image, mask, color='red', alpha=0.7):
    """Draws mask on an image.
    Args:
        image: uint8 numpy array with shape (img_height, img_height, 3)
        mask: a float numpy array of shape (img_height, img_height) with
        values between 0 and 1
        color: color to draw the keypoints with. Default is red.
        alpha: transparency value between 0 and 1. (default: 0.7)
    Raises:
        ValueError: On incorrect data type for image or masks.
    """

    #mask = mask.astype(np.float32)

    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.float32:
        raise ValueError('`mask` not of type np.float32')
    if np.any(np.logical_or(mask > 1.0, mask < 0.0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
        np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))
    return image

'''
images_dir = 'C:/Users/klickmal/Desktop/try/1.jpg' #+ '/' + image_name
mask_dir = 'C:/Users/klickmal/Desktop/try/mask1.jpg' #+ '/' + image_name
print(images_dir)
image = io.imread(images_dir)
mask = io.imread(mask_dir)/255
mask = mask.astype(np.float32)
image_new = draw_mask_on_image_array(image, mask)

misc.imsave('C:/Users/klickmal/Desktop/try/' + '10.jpg', image_new)
plt.figure(figsize=(20,12))
plt.imshow(image_new)
plt.show()
'''
