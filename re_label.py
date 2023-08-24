import json

# with open('/home/mlfavorfit/Downloads/publaynet/train.json', 'r') as fp:
#     samples = json.load(fp)

# images = {}
# for image in samples['images']:
#     images[image['id']] = {'file_name': image['file_name'], 'annotations': []}
# for ann in samples['annotations']:
#     temp = {}
#     temp["segmentation"] = ann["segmentation"][0]
#     temp["category_id"] = ann["category_id"]
#     images[ann['image_id']]['annotations'].append(temp)


# with open("temp.json", mode="w") as f:
#     json.dump(images, f, ensure_ascii=False, indent=4)


root_dir = "/media/mlfavorfit/sda/publaynet/all"
check_dir = "/media/mlfavorfit/sda/publaynet/val"

images = {}
import os
with open(os.path.join(root_dir, "train.json"), 'r') as fp:
    samples = json.load(fp)

for image_id, obj in samples.items():
    fn = obj["file_name"]
    ann = obj["annotations"]
    if os.path.exists(os.path.join(check_dir, "image", fn)):
        images[image_id] = obj

with open("temp.json", mode="w") as f:
    json.dump(images, f, ensure_ascii=False, indent=4)

print(len(images))