import os
import warnings
import argparse
import json
import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from model import CSRNet
from utils import save_checkpoint
import dataset

warnings.filterwarnings("ignore")

# ------------------------------
# Argument Parser
parser = argparse.ArgumentParser(description='CSRNet Training')

parser.add_argument('train_json', metavar='TRAIN', help='path to training json')
parser.add_argument('test_json', metavar='TEST', help='path to test json')
parser.add_argument('--pretrained', '-p', default=None, type=str, help='path to pretrained model')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use')
parser.add_argument('--task', type=str, default='default_task', help='task name/id')
parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('--lr', type=float, default=1e-7, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.95, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience epochs')

args = parser.parse_args()

# ------------------------------
# Utility Functions
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# ------------------------------
# Main Function
def main():
    set_seed(int(time.time()))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load dataset lists
    with open(args.train_json, 'r') as f:
        train_list = json.load(f)
    with open(args.test_json, 'r') as f:
        val_list = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=f'runs/{args.task}')

    # Model
    model = CSRNet().to(device)
    

    # Loss & Optimizer
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = GradScaler()  # For AMP

    start_epoch = 0
    best_mae = float('inf')
    epochs_no_improve = 0  # For early stopping

    # Resume checkpoint if provided
    if args.pretrained and os.path.isfile(args.pretrained):
        print(f"Loading checkpoint '{args.pretrained}'")
        checkpoint = torch.load(args.pretrained)
        start_epoch = checkpoint['epoch']
        best_mae = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{args.pretrained}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No pretrained checkpoint found at '{args.pretrained}'")

    # DataLoader setup
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    train_loader = DataLoader(
        dataset.listDataset(train_list, shuffle=True, transform=transform,
                            train=True, seen=model.seen,
                            batch_size=args.batch_size, num_workers=4),
        batch_size=args.batch_size)

    val_loader = DataLoader(
        dataset.listDataset(val_list, shuffle=False, transform=transform, train=False),
        batch_size=args.batch_size)

    # ------------------------------
    # Evaluate Constant Average Baseline BEFORE training
    avg_count = compute_average_count(train_loader)
    baseline_mae, baseline_rmse = evaluate_constant_baseline(val_loader, avg_count)
    print(f"\nBaseline Results → MAE: {baseline_mae:.3f}, RMSE: {baseline_rmse:.3f}")
    print("="*60)

    # ------------------------------
    # Training Loop
    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train_loss = train(train_loader, model, criterion, optimizer, scaler, epoch, device)
        val_mae, val_rmse, avg_infer_time = validate(val_loader, model, device)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('Inference_time_ms/val', avg_infer_time * 1000, epoch)

        # Check improvements
        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.3f} | Val RMSE: {val_rmse:.3f} | Best MAE: {best_mae:.3f}")
        print(f"Avg Inference Time: {avg_infer_time*1000:.2f} ms/image")

        # Save checkpoint
        save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'best_prec1': best_mae,
    'optimizer': optimizer.state_dict(),
}, is_best, args.task)


        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"\nNo improvement for {args.early_stop_patience} epochs. Early stopping.")
            break

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nTraining completed in {total_duration/60:.2f} minutes ({total_duration:.2f} seconds).")

    writer.close()

    # Save Graphs
    save_graphs(f'runs/{args.task}')

# ------------------------------
# Compute Average Count Function
def compute_average_count(train_loader):
    print("\nComputing average count from training data...")
    total_count = 0
    num_images = 0
    for _, target in train_loader:
        count = target.sum().item()
        total_count += count
        num_images += 1
    avg_count = total_count / num_images
    print(f"Average Count per Image: {avg_count:.2f}")
    return avg_count

# ------------------------------
# Evaluate Constant Average Baseline
def evaluate_constant_baseline(val_loader, avg_count):
    print("\nEvaluating Constant Average Baseline...")
    mae = 0.0
    mse = 0.0
    for _, target in val_loader:
        gt_count = target.sum().item()
        mae += abs(avg_count - gt_count)
        mse += (avg_count - gt_count) ** 2
    mae /= len(val_loader)
    mse /= len(val_loader)
    rmse = math.sqrt(mse)
    print(f"Constant Baseline → MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return mae, rmse

# ------------------------------
# Training Function
def train(train_loader, model, criterion, optimizer, scaler, epoch, device):
    model.train()
    total_loss = 0.0
    start = time.time()

    for i, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.type(torch.FloatTensor).to(device)


        optimizer.zero_grad()
        with autocast(enabled=(device.type == 'cuda')):
            output = model(img)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if i % args.print_freq == 0:
            print(f'Epoch [{epoch+1}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

    end = time.time()
    print(f"Epoch [{epoch+1}] completed in {end - start:.2f} seconds")
    return total_loss / len(train_loader)

# ------------------------------
# Validation Function
def validate(val_loader, model, device):
    print("\nStarting validation...")
    model.eval()
    mae = 0.0
    mse = 0.0
    inference_times = []

    with torch.no_grad():
        for i, (img, target) in enumerate(val_loader):
            img = img.to(device)
            target_count = target.sum().type(torch.FloatTensor).to(device)

            start_infer = time.time()
            output = model(img)
            end_infer = time.time()

            pred_count = output.sum()

            # MAE
            mae += abs(pred_count - target_count)
            # MSE
            mse += (pred_count - target_count) ** 2
            # Inference time
            inference_times.append(end_infer - start_infer)

    mae /= len(val_loader)
    mse /= len(val_loader)
    rmse = math.sqrt(mse)

    avg_infer_time = sum(inference_times) / len(inference_times)

    print(f'Validation Results → MAE: {mae:.3f}, RMSE: {rmse:.3f}, Avg Inference Time: {avg_infer_time*1000:.2f} ms')
    return mae.item(), rmse, avg_infer_time

# ------------------------------
# Learning Rate Adjustment
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 30:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# ------------------------------
# Graph Saving Function
def save_graphs(log_dir):
    print("\nSaving training graphs as PNG images...")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Retrieve scalar metrics
    mae_events = ea.Scalars('MAE/val')
    rmse_events = ea.Scalars('RMSE/val')
    loss_events = ea.Scalars('Loss/train')

    epochs = [e.step for e in mae_events]
    mae_values = [e.value for e in mae_events]
    rmse_values = [e.value for e in rmse_events]
    loss_values = [e.value for e in loss_events]

    # MAE Graph
    plt.figure()
    plt.plot(epochs, mae_values, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Validation MAE over Epochs')
    plt.savefig(f'{log_dir}/validation_mae.png')
    plt.close()

    # RMSE Graph
    plt.figure()
    plt.plot(epochs, rmse_values, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Validation RMSE over Epochs')
    plt.savefig(f'{log_dir}/validation_rmse.png')
    plt.close()

    # Loss Graph
    plt.figure()
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    plt.savefig(f'{log_dir}/training_loss.png')
    plt.close()

    print(f"Graphs saved in: {log_dir}")

# ------------------------------
if __name__ == '__main__':
    main()
