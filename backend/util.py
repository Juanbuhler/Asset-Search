from PIL import Image
import numpy as np


def get_collage(image_paths, x_ims, x_size=1500, crop_center=True):
    # Load all images and determine their aspect ratios
    images = [Image.open(path) for path in image_paths]
    aspect_ratios = [img.width / img.height for img in images]
    avg_aspect_ratio = np.mean(aspect_ratios)

    # Calculate the number of rows needed (y_ims)
    total_images = len(images)
    y_ims = total_images // x_ims + (1 if total_images % x_ims != 0 else 0)

    # Calculate the size of each image in the collage
    x_imsize = int(x_size / x_ims)
    y_imsize = int(x_imsize / avg_aspect_ratio)

    # Calculate the total height of the collage
    y_size = y_imsize * y_ims

    # Create a new blank image for the collage
    collage = Image.new("RGB", (x_size, y_size))

    id = 0
    for j in range(y_ims):
        for i in range(x_ims):
            if id >= total_images:
                break
            img = images[id]
            img_aspect_ratio = img.width / img.height

            if crop_center:
                # Calculate dimensions to crop the image to fit exactly into the collage cell
                target_aspect_ratio = x_imsize / y_imsize
                if img_aspect_ratio > target_aspect_ratio:
                    # Crop the width (landscape)
                    new_width = int(target_aspect_ratio * img.height)
                    left = (img.width - new_width) // 2
                    right = left + new_width
                    top = 0
                    bottom = img.height
                else:
                    # Crop the height (portrait)
                    new_height = int(img.width / target_aspect_ratio)
                    top = (img.height - new_height) // 2
                    bottom = top + new_height
                    left = 0
                    right = img.width

                img = img.crop((left, top, right, bottom))
                img = img.resize((x_imsize, y_imsize))
                collage.paste(img, (i * x_imsize, j * y_imsize))
            else:
                # Resize image while maintaining aspect ratio
                if img_aspect_ratio > (x_imsize / y_imsize):
                    new_width = x_imsize
                    new_height = int(new_width / img_aspect_ratio)
                else:
                    new_height = y_imsize
                    new_width = int(new_height * img_aspect_ratio)

                img = img.resize((new_width, new_height))

                # Calculate position to paste the image to center it within the grid cell
                x_offset = (x_imsize - new_width) // 2
                y_offset = (y_imsize - new_height) // 2
                collage.paste(img, (i * x_imsize + x_offset, j * y_imsize + y_offset))

            id += 1

    return collage
