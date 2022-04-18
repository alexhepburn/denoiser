import os
import gc
import numpy as np

import torch
import torchvision
import torchvision.datasets as datasets

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from network import BF_CNN

from losses import MSE_Loss, NLPD_Loss, ssim, psnr, DISTS_Loss

from dataset import h5Dataset, NoiseDataset, Kodak

LOSSES = {
    'mse': MSE_Loss,
    'nlpd': NLPD_Loss,
    'dists': DISTS_Loss
}


def update_logs(loss, psnr, ssim, iteration, optimizer, mode='train'):
    if mode == 'train':
        lr = [group['lr'] for group in optimizer.param_groups][0]
        wandb.log({
            'batch': iteration,
            'train_loss': loss,
            'train_psnr_loss': psnr,
            'train_ssim_loss': ssim,
            'learning_rate': lr})
    else:
        wandb.log({
            'test_loss': loss,
            'test_psnr_loss': psnr,
            'test_ssim_loss': ssim})


def update_plots(in_image, out_image, iteration, mode='train'):
    in_grid = torchvision.utils.make_grid(in_image, nrow=2)
    out_grid = torchvision.utils.make_grid(out_image, nrow=2)
    if mode == 'train':
        wandb.log({
            'train_org': [wandb.Image(in_grid, caption="Training Images")],
            'train_recon': [wandb.Image(out_grid, caption="Reconstructed Training Images")],
            })
    else:
        wandb.log({
            'test_org': [wandb.Image(in_grid, caption="Test Images")],
            'test_recon': [wandb.Image(out_grid, caption="Reconstructed Test Images")],
            })

from torch.utils.data.distributed import DistributedSampler

def configure_datasets(cfg):
    if cfg.dataset.type == 'h5':
        dataset = h5Dataset(cfg.dataset.dir.train)
    elif cfg.dataset.type == 'noise':
        dataset = NoiseDataset(min=cfg.dataset.min, max=cfg.dataset.max)

    train_sampler = DistributedSampler(dataset)

    test_dataset = Kodak(cfg.dataset.dir.test)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_dataloader=torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        sampler=train_sampler)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.dataset.test_batch_size,
        num_workers=cfg.dataset.num_workers,
        sampler=test_sampler)
    
    return train_dataloader, test_dataloader, train_sampler, test_sampler

def get_noise(inputs):
    noise = torch.randn_like(inputs)
    n = noise.shape[0]
    noise_arr = (55/255 - 0.) * torch.rand(n) + 0. # max_noise=55, min_noise=0.
    for k in range(n):
        noise.data[k] = noise.data[k] * noise_arr[k]
    return noise

@hydra.main()
def main(cfg) -> None:
    device = 'cuda' if torch.cuda.is_available() and not cfg.no_cuda else 'cpu'
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    print(local_rank)
    if local_rank == 0:
        wandb.init(project='denoising', config=cfg)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    train_dataloader, test_dataloader, train_sampler, test_sampler = configure_datasets(cfg)

    model = BF_CNN(cfg.model)
    model.to(device)
    import torch.nn as nn
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 60, 70, 80, 90, 100], gamma=0.5)

    loss_fn = LOSSES[cfg.loss.name](**cfg.loss.kwargs)

    iteration = 0
    for epoch in range(cfg.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for i, inputs in enumerate(train_dataloader):
            inputs = inputs.to('cuda')
            noisy_input = inputs + get_noise(inputs)
            outputs = noisy_input - model(noisy_input)
            loss = loss_fn(outputs, inputs)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % cfg.loss_iter == 0:
                p = psnr(outputs, inputs)
                m = ssim(outputs, inputs)

                if local_rank == 0:
                    print(f'Train epoch {epoch}: ['
                        f'{i*len(inputs)}/{len(train_dataloader.dataset)}'
                        f' ({100. * i / len(train_dataloader):.0f}%)]'
                        f'\tLoss: {loss.item():.3f} |'
                        f'\tPSNR: {p:.3f} |'
                        f'\tSSIM: {m:.3f} |')
                    update_logs(loss.item(), p, m, iteration, optimizer)

            if iteration % cfg.eval_iter == 0:
                model.eval()
                test_losses, test_psnr, test_ssim = [], [], []
                for test_batch in test_dataloader:
                    test_batch = test_batch.to(device)
                    noisy_test = test_batch + get_noise(test_batch)
                    test_outputs = noisy_test - model(noisy_test)
                    test_losses.append(loss_fn(test_outputs, test_batch).item())
                    test_psnr.append(psnr(test_outputs, test_batch))
                    test_ssim.append(ssim(test_outputs, test_batch))

                if local_rank == 0:
                    update_logs(
                        np.mean(test_losses), np.mean(test_psnr), np.mean(test_ssim), iteration, optimizer, mode='test')
                    update_plots(noisy_test[0], test_outputs[0], iteration, mode='test')
                    torch.save(model.module.state_dict(), '%d-%d-checkpoint.pth'%(epoch, iteration))

                del test_batch
                del noisy_test
                del test_outputs

                model.train()
            gc.collect()

            del inputs
            del noisy_input
            del outputs

            iteration += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--config_dir", type=str)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    initialize_config_dir(config_dir=args.config_dir)
    cfg = compose(config_name=args.config_name)

    os.chdir(args.output_dir)

    main(cfg)   
