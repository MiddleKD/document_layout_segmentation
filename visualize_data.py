import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Define color code
category_map = {1:"text", 2:"title", 3:"list", 4:"tabel", 5:"figure"}
colors = {1: (255, 0, 0),
          2: (0, 255, 0),
          3: (0, 0, 255),
          4: (255, 255, 0),
          5: (0, 255, 255)}

# Function to viz the annotation
def markup(image, annotations):
    ''' Draws the segmentation, bounding box, and label of each annotation
    '''
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        # Draw segmentation
        draw.polygon(annotation['segmentation'],
                     fill=colors[annotation["category_id"]] + (64,))
        # # Draw bbox
        # draw.rectangle(
        #     (annotation['bbox'][0],
        #      annotation['bbox'][1],
        #      annotation['bbox'][0] + annotation['bbox'][2],
        #      annotation['bbox'][1] + annotation['bbox'][3]),
        #     outline=colors[samples['categories'][annotation['category_id'] - 1]['name']] + (255,),
        #     width=2
        # )
        # # Draw label
        # w, h = draw.textsize(text=samples['categories'][annotation['category_id'] - 1]['name'],
        #                      font=font)
        # if annotation['bbox'][3] < h:
        #     draw.rectangle(
        #         (annotation['bbox'][0] + annotation['bbox'][2],
        #          annotation['bbox'][1],
        #          annotation['bbox'][0] + annotation['bbox'][2] + w,
        #          annotation['bbox'][1] + h),
        #         fill=(64, 64, 64, 255)
        #     )
        #     draw.text(
        #         (annotation['bbox'][0] + annotation['bbox'][2],
        #          annotation['bbox'][1]),
        #         text=samples['categories'][annotation['category_id'] - 1]['name'],
        #         fill=(255, 255, 255, 255),
        #         font=font
        #     )
        # else:
        #     draw.rectangle(
        #         (annotation['bbox'][0],
        #          annotation['bbox'][1],
        #          annotation['bbox'][0] + w,
        #          annotation['bbox'][1] + h),
        #         fill=(64, 64, 64, 255)
        #     )
        #     draw.text(
        #         (annotation['bbox'][0],
        #          annotation['bbox'][1]),
        #         text=samples['categories'][annotation['category_id'] - 1]['name'],
        #         fill=(255, 255, 255, 255),
        #         font=font
        #     )
    return np.array(image)

root_dir = "/media/mlfavorfit/sda/publaynet"

import os
with open(os.path.join(root_dir, "train.json"), 'r') as fp:
    samples = json.load(fp)

# Visualize annotations
# font = ImageFont.truetype("examples/DejaVuSans.ttf", 15)
for image_id, obj in samples.items():
    fn = obj["file_name"]
    ann = obj["annotations"]
    if os.path.exists(os.path.join(root_dir, "train", fn)):
        print(fn)
        img = Image.open(os.path.join(root_dir, "train", fn))
        plt.imshow(markup(img, ann))
        plt.show()
plt.subplots_adjust(hspace=0, wspace=0)
