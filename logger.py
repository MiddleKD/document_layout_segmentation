import wandb
from sklearn.metrics import confusion_matrix

def wnb_page_init(epochs, project, name, wnb_api_key):
    wandb.login(key=wnb_api_key)
    wandb.init(
        project=project,
        name=name,
        config={
        # "architecture": "efficientNet",
        "epochs": epochs,
        }
    )

def wnb_write(like_json):
    wandb.log(like_json)

def wnb_close():
    wandb.finish()

def wnb_write_conf_mat(epoch, labels, preds, num_class):
    wandb.log({f"conf_mat{epoch}" : wandb.plot.confusion_matrix(
                y_true=labels, 
                preds=preds, 
                class_names=range(num_class))})
    
def wnb_watch(model, criterion):
    wandb.watch(model, criterion, log="all", log_freq=1)


import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def save_tensor_image(output, save_dir):
    # label_map = {1:"image", 2:"label", 2:"text", 3:"table", 4:"logo", 5:"icon"}
    label_map = {0:"bg", 1:"text", 2:"title", 3:"list", 4:"tabel", 5:"figure"}
    color_map = {idx:plt.cm.viridis(cur) for idx, cur in enumerate(np.linspace(0,1,5))}

    colored_out = torch.zeros((3, 160,160))
    for label, color in color_map.items():
        mask = output == label
        for channel, color_value in enumerate(color[:3]):
            colored_out[channel, mask] = color_value

    img = Image.fromarray(np.array(colored_out*255).astype(np.uint8).transpose(1,2,0))

    if not save_dir.endswith(".jpg"):
        save_dir = save_dir + ".jpg"

    img.save(save_dir)
