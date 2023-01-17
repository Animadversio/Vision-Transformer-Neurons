import os
import json
from os.path import join
# import lpips from torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from core.utils.plot_utils import show_imgrid, save_imgrid, saveallforms, to_imgrid
from torchmetrics.functional import cosine_similarity, pairwise_cosine_similarity
from lpips import LPIPS
def avg_cosine_sim_mat(X):
    cosmat = pairwise_cosine_similarity(X,)
    idxs = torch.tril_indices(cosmat.shape[0], cosmat.shape[1], offset=-1)
    cosmat_vec = cosmat[idxs[0], idxs[1]]
    return cosmat, cosmat_vec.mean(), cosmat_vec


def load_evol_imgs(expdir):
    imgs = []
    for i in range(10):
        proto = plt.imread(join(expdir, f"evol_img_rep{i:02d}.png"))
        imgs.append(proto)
    imgtsrs = torch.tensor(imgs)
    imgtsrs = imgtsrs.permute(0, 3, 1, 2)
    return imgtsrs


saveroot = r"F:\insilico_exps\ViT-CNN-Mixer"
# expdir = r"F:\insilico_exps\ViT-CNN-Mixer\mixer_b16_224\L11_T98_U210"
# expdir = r"F:\insilico_exps\ViT-CNN-Mixer\resnetv2_50\L4B2_T3_3_U205"
# expdir = r"F:\insilico_exps\ViT-CNN-Mixer\vit_base_patch16_224_dino\L11_T98_U200"
#%%
import os
import glob
modelname = r"vit_base_patch16_224_dino"
modelname = r"mixer_b16_224"
modelname = r"resnetv2_50"
modelname = r"resnext50_32x4d_Robust"
sumdir = join(saveroot, modelname, "summary")
os.makedirs(sumdir, exist_ok=True)
expdirs = glob.glob(join(saveroot, modelname, "*T*"))
#%%
unit_list = []
cosine_vec_col = []
for expdir in expdirs:
    unit_label = expdir.split("\\")[-1]
    imgtsrs = load_evol_imgs(expdir)
    imgtsrs = (imgtsrs - 0.5) * 2
    cosine_mat, avg_cosine, cosine_vec = avg_cosine_sim_mat(imgtsrs.flatten(1))

    plt.figure(figsize=[6, 5])
    sns.heatmap(cosine_mat, annot=True, fmt=".2f")
    # sns.displot(cosine_mat)
    plt.title(f"{unit_label}\navg cosine sim: {avg_cosine:.2f}, med: {cosine_vec.median():.2f}")
    plt.axis("image")
    saveallforms(expdir, "cosine_mat", plt.gcf())
    saveallforms(sumdir, f"cosine_mat_{unit_label}", plt.gcf())
    plt.show()
    cosine_vec_col.append(cosine_vec)
    unit_list.append(unit_label)

cosine_vec_col = torch.stack(cosine_vec_col)
torch.save(cosine_vec_col, join(sumdir, "cosine_vec_col.pt"))
json.dump(unit_list, open(join(sumdir, "unit_list.json"), "w"))
#%%
# cosine_mat = pairwise_cosine_similarity(imgtsrs.flatten(1), )

#%%
