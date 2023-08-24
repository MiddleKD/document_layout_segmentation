from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as F
import json
import cv2
import numpy as np
Image.MAX_IMAGE_PIXELS = None

class PublaynetDataset(Dataset):
    def __init__(self, root_dir, resize_size = (640,640)):
        self.width, self.height = resize_size

        with open(os.path.join(root_dir, "label.json"), mode="r") as f:
            labels_json = json.load(f)
        
        self.images = []
        self.labels = []
        
        for image_id, obj in labels_json.items():
            self.images.append(os.path.join(root_dir, "image", obj["file_name"]))
            self.labels.append(obj["annotations"])
        
        # self.images = self.images[:10]
        # self.labels = self.labels[:10]

        self.transformer = transforms.Compose([
            transforms.Resize([self.height, self.width]),
            transforms.ToTensor()
        ])

        print(f"Dataset -> num_of_data:{len(self.images)}")


    def make_labeled_segmentation(self, annotations, input_img_size):
        width, height = input_img_size
        num_labels = 6

        # 각 레이블에 대한 마스크를 저장할 텐서 초기화
        masks = np.zeros((num_labels, height, width), dtype=np.float32)

        for annotation in annotations:
            polygon = annotation["segmentation"]
            polygon_pairs = np.array([(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)], dtype=np.int32)
            
            # 해당 category_id에 대한 마스크 채널을 업데이트
            cv2.fillPoly(masks[annotation["category_id"] - 1], [polygon_pairs], 1)  # category_id는 1부터 시작한다고 가정
        
        masks[-1] = np.where(np.sum(masks[:-1], axis=0) == 0, 1, 0)

        label_tensor = torch.tensor(masks, dtype=torch.float32)
        label_tensor = F.interpolate(label_tensor.unsqueeze(0), [160, 160], mode='bilinear')

        return label_tensor.squeeze(0).int().float()
        # mask = np.zeros((height, width), dtype=np.float32)
        # for annotation in annotations:
        #     polygon = annotation["segmentation"]
        #     polygon_pairs = np.array([(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)], dtype=np.int32)
            
        #     cv2.fillPoly(mask, [polygon_pairs], annotation["category_id"])
        
        # label_tensor = torch.tensor(mask, dtype=torch.float32)
        # label_tensor = F.interpolate(label_tensor.unsqueeze(0).unsqueeze(0), [160, 160], mode='bilinear')

        # return label_tensor.squeeze(0).int().float()    # label_tensor.squeeze(0).int()
            
        
    def __getitem__(self, idx):
        input = self.images[idx]
        annotations = self.labels[idx]

        input_img = Image.open(input)

        input_ts = self.transformer(input_img)
        label_ts = self.make_labeled_segmentation(annotations, input_img.size)

        return input_ts, label_ts


    def __len__(self):
        return len(self.images)
    
if __name__ == "__main__":
    temp_dataset = PublaynetDataset("/media/mlfavorfit/sda/publaynet/val", (640,640))
    
    input_ts, label_ts = temp_dataset[8000]

    print(np.unique(input_ts))
    print(np.unique(label_ts))

    print(input_ts.shape)
    print(label_ts.shape)
