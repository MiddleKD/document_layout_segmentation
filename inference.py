import torch
from torchvision.transforms import transforms
from model.beit_model import Extended_BEIT
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def visiualize_label(labeld_tensor):

    # label_map = {1:"image", 2:"label", 2:"text", 3:"table", 4:"logo", 5:"icon"}
    label_map = {0:"text", 1:"title", 2:"list", 3:"tabel", 4:"figure", 5:"bg"}
    color_map = {idx:plt.cm.viridis(cur) for idx, cur in enumerate(np.linspace(0,1,5))}

    colored_out = torch.zeros((3, 160,160))
    for label, color in color_map.items():
        mask = labeld_tensor == label
        for channel, color_value in enumerate(color[:3]):
            colored_out[channel, mask] = color_value

    plt.imshow(colored_out.permute(1,2,0))
    plt.show()

    print(labeld_tensor)
    print(labeld_tensor.shape)

def inference(model, input_ts):
    out = model(input_ts)
    
    print("Contained label: ", out.argmax(1).unique())
    out_labeld = out[0].argmax(0).float()

    return out_labeld

def run(img_path, model_ckpt):
    transformer = transforms.Compose([
            transforms.Resize([640, 640]),
            transforms.ToTensor()
        ])
    
    img = Image.open(img_path)
    img_tensor = transformer(img)

    model = Extended_BEIT()
    model.load_state_dict(torch.load(model_ckpt))

    out_labeled = inference(model, img_tensor.unsqueeze(0))

    visiualize_label(out_labeled)

    return out_labeled

if __name__ == "__main__":

    run(img_path="./test/2.jpg", model_ckpt="./ckpt/model_3.pth")
