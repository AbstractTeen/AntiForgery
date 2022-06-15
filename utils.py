import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from color_space import *

def load_model_weights(model, path):
    pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

        c_trg_list.append(c_trg.cuda())
    return c_trg_list

def random_transform(img):
    T_list = [
        T.RandomHorizontalFlip(p=0.5),
        #T.RandomErasing(p=1, scale=(0.03, 0.10)),
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomVerticalFlip(p=0.5),
        T.RandomCrop((192,192)),
    ]

    T_compose = T.Compose([
        T.RandomChoice(T_list),
        T.Resize((256, 256)),
    ])

    return T_compose(img)

def compare(img1,img2):
    """input tensor, translate to np.array"""
    img1_np = img1.squeeze(0).cpu().numpy()
    img2_np = img2.squeeze(0).cpu().numpy()
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))

    ssim = structural_similarity(img1_np,img2_np,multichannel=True)
    psnr = peak_signal_noise_ratio(img1_np,img2_np)

    return ssim, psnr

def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter = 100):

    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()

    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    r = torch.ones_like(pert_a)

    X = denorm(X_nat.clone())

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(lab2rgb(X_lab))

        #X_new = random_transform(X_new)

        with torch.no_grad():
            gen_noattack, gen_feats_noattack = model(X_nat, c_trg[i%len(c_trg)])

        gen_stargan, gen_feats_stargan = model(X_new, c_trg[i%5])

        loss = -criterion(gen_stargan, gen_noattack)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return X_new, X_new - X


