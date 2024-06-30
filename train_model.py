import albumentations as A
import argparse
from dataset import Sentinel2_Dataset, create_splits
import numpy as np
import torch
from utils import *
from models import *
import os
import time
import glob
import matplotlib.pyplot as plt
import tifffile
import random
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', nargs=1, type=str, default="test")
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dft', type=bool, default=True)

    
    train_crop_size = 128
    num_augmentations = 4
    num_workers = 4
    pin_memory = True
    patience = 4
    dim = 9

    args = vars(parser.parse_args())
    experiment_name = args["name"]
    batch_size = args["batchsize"]
    n_epochs = args["epochs"]
    lr = args["lr"]
    seed = args["seed"]

    wandb.login()
    wandb.init(project="Master")
    wandb.log({
        "batchsize": batch_size,
        "epochs": n_epochs,
        "learningrate": lr,
        "seed": seed,
        "cropsize": train_crop_size,
        "num_augmentations": num_augmentations,
        "num_workes": num_workers,
        })
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    img_paths_train, img_paths_val, img_paths_test, label_paths_train, label_paths_val, label_paths_test = create_splits(seed=seed)

    train_transforms = A.Compose(
        [
            A.RandomCrop(train_crop_size, train_crop_size),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip()
        ]
    )

    train_dataset = Sentinel2_Dataset(img_paths_train, label_paths_train, transforms=train_transforms, num_augmentations=num_augmentations)
    val_dataset = Sentinel2_Dataset(img_paths_val, label_paths_val, transforms=None, num_augmentations=0)
    test_dataset = Sentinel2_Dataset(img_paths_test, label_paths_test, transforms=None, num_augmentations=0)

    # Create Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
 
    on_gpu = torch.cuda.is_available()

    early_patience = patience * 3
    log_path = f"trained_models/{experiment_name}"
    os.makedirs(log_path, exist_ok=True)

    model = UnetLarge(dim, 2)

    loss_func = XEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=patience,
    )

    model = model.cuda()


    save_dict = {"model_name": experiment_name}
    best_metric, best_metric_epoch, early_stopping_metric = 0, 0, 0
    early_stop_counter, best_val_early_stopping_metric = 0, 0
    early_stop_flag, start_time = False, time.time()

    for curr_epoch_num in range(1, n_epochs):
        ## Part 1: Training loop with Model weights updating
        model.train()
        torch.set_grad_enabled(True)

        losses, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
        end = time.time()
        for iter_num, (x_data, targets) in enumerate(train_loader):
            x_data, targets = x_data.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).long()
            optimizer.zero_grad()
            preds = model(x_data)
            loss = loss_func(preds, targets)
            loss.backward()
            optimizer.step()

            losses.update(loss.detach().item(), x_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        print(f"Ep: [{curr_epoch_num}] TotalT: {(time.time() - start_time) / 60:.1f} min"
            f", BatchT: {batch_time.avg:.3f}s, DataT: {data_time.avg:.3f}s, Loss: {losses.avg:.4f}")

        ## Part 2: Validation loop to get the current IoU
        model.eval()
        torch.set_grad_enabled(False)

        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
        tps, fps, fns = 0, 0, 0

        end = time.time()
        for iter_num, data in enumerate(val_loader):
            data_time.update(time.time() - end)
            x_data = data["image"].float()
            targets = data["mask"].long()
            x_data, targets = x_data.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            preds = model(x_data)
            loss = loss_func(preds, targets)
            losses.update(loss.detach().item(), x_data.size(0))

            preds = torch.softmax(preds, dim=1)[:, 1]
            preds = (preds > 0.5) * 1

            tp, fp, fn = tp_fp_fn(preds, targets)
            assert tp >= 0 and fp >= 0 and fn >= 0, f"Negative tp/fp/fn: tp: {tp}, fp: {fp}, fn: {fn}"
            tps += tp
            fps += fp
            fns += fn

            batch_time.update(time.time() - end)
            end = time.time()

        iou_global = tps / (tps + fps + fns)
        #wandb.log({"iou_global": iou_global})
        ## When validation IoU improved, save model. If not, increment the lr scheduler and early stopping counters
        early_stopping_metric = iou_global
        if curr_epoch_num > 6:
            scheduler.step(early_stopping_metric)

        print(f"Ep: [{curr_epoch_num}] ValT: {(batch_time.avg * len(val_loader)) / 60:.1f} min, BatchT: {batch_time.avg:.3f}s, "
            f"DataT: {data_time.avg:.3f}s, Loss: {losses.avg:.4f}, Global IoU: {iou_global:.4f} (val)")

        if early_stopping_metric > best_metric:
            best_metric, best_metric_epoch = early_stopping_metric, curr_epoch_num
            save_dict.update(
                {
                    "best_metric": best_metric,
                    "epoch_num": curr_epoch_num,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                }
            )

            old_model = glob.glob(f"{log_path}/best_iou*")  # delete old best model
            if len(old_model) > 0:
                os.remove(old_model[0])

            save_path = f"{log_path}/best_iou_{curr_epoch_num}_{best_metric:.4f}.pt"
            torch.save(save_dict, save_path)

        # Early Stopping (ES): Increment stop counter if metric didn't improve
        if early_stopping_metric < best_val_early_stopping_metric:
            early_stop_counter += 1  
        else:   
            # reset ES counters
            best_val_early_stopping_metric, early_stop_counter = early_stopping_metric, 0

        # ES: Stop when stop counter reaches ES patience
        if early_stop_counter > early_patience:
            print("Early Stopping")
            break
    torch.cuda.empty_cache()

    print(f"Best validation IoU of {best_metric:.5f} in epoch {best_metric_epoch}.")

    # load best model
    model_weights_path = glob.glob(f"trained_models/{experiment_name}/best_iou*")[0]

    model =  UnetLarge(dim, 2)
    model = model.cuda()
    model.eval()

    cp = torch.load(model_weights_path)
    model.load_state_dict(cp["model_state_dict"])
    print(f"Loaded {model_weights_path} at epoch {cp['epoch_num']}")

    tps, fps, fns = 0, 0, 0
    probs = []
    torch.set_grad_enabled(False)
    for data in test_loader:
        x_data = data["image"].float()
        targets = data["mask"].long()
        x_data, targets = x_data.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        preds = torch.softmax(model(x_data), dim=1)[:, 1]
        preds = (preds > 0.5) * 1
        
        tp, fp, fn = tp_fp_fn(preds, targets)
        assert tp >= 0 and fp >= 0 and fn >= 0, f"Negative tp/fp/fn: tp: {tp}, fp: {fp}, fn: {fn}"
        tps += tp
        fps += fp
        fns += fn
            
        probs.append(preds.cpu().numpy())
        
    iou_global = tps / (tps + fps + fns)
        
    last_prob = probs[-1].copy()
    probs = np.array(probs[:-1]).reshape((-1, 512, 512))

    probs = np.concatenate((probs, last_prob), 0)
    probs.shape
    print(f"Test set global IoU: {iou_global:.4f}")

    show_how_many = 20
    alpha = 0.5
    for k in range(show_how_many):
        rand_int = np.random.randint(len(test_dataset))
        label_path = test_dataset.mask_paths[rand_int]

        f, axarr = plt.subplots(1, 3, figsize=(30, 10))
        pred = probs[rand_int]
        pred = np.ma.masked_where(pred == 0, pred)

        label = tifffile.imread(label_path)
        label = np.ma.masked_where(label == 0, label)

        swirp = plt.imread(label_path.replace("LabelWater.tif", "SWIRP.png"))

        axarr[0].set_title("Ground Truth")
        axarr[0].imshow(label, cmap="cool", alpha=alpha)
        axarr[1].set_title("Prediction")
        axarr[1].imshow(pred, cmap="cool", alpha=alpha)
        axarr[2].set_title("SWIRP")
        axarr[2].imshow(swirp, cmap="cool", alpha=alpha)


        plt.tight_layout()
        plt.savefig(f'trained_models/{experiment_name}/{label_path.split("\\")[-2]}.png')