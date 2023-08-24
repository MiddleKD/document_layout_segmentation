from torch.utils.data import DataLoader
from model.beit_model import Extended_BEIT
from tqdm import tqdm
import torch
import os

from datetime import datetime
from torch.nn import CrossEntropyLoss
from dataset import PublaynetDataset
import logger
from custom_lr_scheduler import CosineAnnealingWarmUpRestartsMaxReduce

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

def train(model, epoch, loader, optimizer, criterion, args):
    model.train()

    running_loss = 0.0
    for idx, (inputs, labels) in tqdm(enumerate(loader), leave=False, desc=f"Epoch:{epoch} train", total=len(loader)):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(loader)
    current_lr = optimizer.param_groups[0]['lr']

    print(f'Epoch:{epoch} Train Loss: {train_loss:.4f} Lr: {current_lr:.6f}')
    return train_loss, current_lr
    

def val(model, epoch, loader, criterion, args):
    model.eval()

    with torch.no_grad():
        running_loss = 0.0

        for idx, (inputs, labels) in tqdm(enumerate(loader), leave=False, desc=f"Epoch:{epoch} val", total=len(loader)):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            if idx % 1000 == 0:
                logger.save_tensor_image(outputs[0].argmax(0), f"./result/{epoch}_{idx}")

        val_loss = running_loss / len(loader)
    
    print(f'Epoch:{epoch} Val Loss: {val_loss:.4f}')
    return val_loss
    

def run(args):
    train_dataset = PublaynetDataset(os.path.join(args.data_path,"train"), resize_size=(640,640))
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=16, pin_memory=True, shuffle=True)

    val_dataset = PublaynetDataset(os.path.join(args.data_path,"val"), resize_size=(640,640))
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=16, pin_memory=True, shuffle=False)

    model = Extended_BEIT()
    model.to(args.device)

    criterion = CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=0.1, step_size_up=1000, step_size_down=1000, cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, nesterov=True)
    scheduler = CosineAnnealingWarmUpRestartsMaxReduce(optimizer, T_0=3, T_mult=2, eta_max=args.lr, T_up=1, gamma=0.5, last_epoch=-1)

    logger.wnb_page_init(epochs=args.epochs, project=args.project_name, name=f"{datetime.today().month}.{datetime.today().day}.{args.run_name}.start", wnb_api_key=os.getenv("WANDB_KEY"))
    logger.wnb_watch(model, criterion)

    for epoch in range(args.epochs):
        train_loss, current_lr = train(model, epoch, train_loader, optimizer, criterion, args)
        val_loss = val(model, epoch, val_loader, criterion, args)
        scheduler.step()

        logger.wnb_write({"train_loss":train_loss, "val_loss": val_loss, "lr": current_lr})
    
        if epoch % 3 == 0:
            torch.save(model.state_dict(), f"./ckpt/model_{epoch}.pth")

    logger.wnb_close()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script description")

    parser.add_argument('--device', default='cuda', type=str, help='Device to be used (default: cuda)')
    parser.add_argument('--epochs', default=30, type=int, help='Num of epochs (default: 30)')
    parser.add_argument('--lr', default=1e-2, type=float, help='Define learning rate (default: 1e-3)')
    parser.add_argument('--project_name', default="doc_layout", type=str, help='Define project name posted to wandb')
    parser.add_argument('--run_name', default="publaynet", type=str, help='Define project name posted to wandb')
    parser.add_argument('--data_path', default='/media/mlfavorfit/sda/publaynet', type=str, help='List of dataset paths')   
    args = parser.parse_args()

    run(args)
    
