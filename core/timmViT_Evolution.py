import os

import timm

mmix = timm.create_model('mixer_b16_224_in21k', pretrained=True)
mmix.eval()
#%%
mres = timm.create_model('resnetv2_50', pretrained=True)
#%%
mvit = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
mvit.cuda().eval().requires_grad_(False)
#%%
from os.path import join
import torch
import numpy as np
from core.utils import upconvGAN, CholeskyCMAES
from core.utils.layer_hook_utils import featureFetcher_module
from core.utils.plot_utils import save_imgrid, show_imgrid, showimg
# from transformers import ViTFeatureExtractor, ViTModel
from torchmetrics.functional import pairwise_cosine_similarity
from PIL import Image
import requests
from torchvision.transforms import ToTensor, ToPILImage, Resize, Normalize, Compose
preprocess = Compose([
    Resize(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #ToTensor(),
])
G = upconvGAN()
G.eval().requires_grad_(False).cuda()

fetcher = featureFetcher_module()
#%%
def evol_score_fun(G, score_fun, generation=100):
    CMAoptim = CholeskyCMAES(4096, )
    codes = np.random.randn(1, 4096)
    score_traj = []
    gen_traj = []
    for i in range(generation):
        imgs = G.visualize_batch_np(codes).cuda()
        scores = score_fun(preprocess(imgs))
        codes = CMAoptim.step_simple(scores.numpy(), codes)
        score_traj.extend(scores)
        gen_traj.extend(i * np.ones_like(scores))
        print(scores.mean())

    gen_traj = torch.tensor(gen_traj)
    score_traj = torch.tensor(score_traj)
    code_mean = codes.mean(0, keepdims=True)
    img_avg = G.visualize_batch_np(code_mean)
    # show_imgrid(img_avg)
    code_mean = torch.tensor(code_mean)
    return img_avg, code_mean, score_traj, gen_traj


def grad_score_fun(G, score_fun, generation=100, batch_size=10):
    zs = torch.randn(batch_size, 4096).cuda()
    zs.requires_grad_(True)
    optimizer = torch.optim.Adam([zs], lr=5e-2)
    score_traj = []
    gen_traj = []
    for i in range(generation):
        img = G.visualize(zs)
        scores = score_fun(preprocess(img))
        loss = - scores.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        score_traj.append(scores.detach().cpu().numpy())
        gen_traj.append(i * np.ones_like(scores.detach().cpu().numpy()))
        print(loss.item())
    fetcher.cleanup()
    score_traj = np.stack(score_traj)
    gen_traj = np.stack(gen_traj)
    # show_imgrid(img_avg)
    return img, zs, score_traj, gen_traj


def grad_evol_pipeline(G, score_fun, grad_gen=100, grad_batch_size=10,
                        CMA_gen=100, repN=10, titlestr="", expdir=""):
    imgs, codes, score_trajs, gen_trajs = grad_score_fun(G, score_fun, generation=grad_gen, batch_size=grad_batch_size)
    show_imgrid(imgs, nrow=5)
    save_imgrid(imgs, join(expdir, "grad_evol_imgs.png"), nrow=5)
    plt.figure()
    plt.plot(gen_trajs, score_trajs, '-')
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(titlestr)
    plt.savefig(join(expdir, "grad_score_traj.png"))
    plt.show()
    torch.save({"codes": codes.detach(), "score_trajs": score_trajs, "gen_trajs": gen_trajs},
               join(expdir, "grad_evol_results.pt"))

    for repi in range(repN):
        img_avg, code_mean, score_traj, gen_traj = evol_score_fun(G, score_fun, generation=CMA_gen)
        save_imgrid(img_avg, join(expdir, f"evol_img_rep{repi:02d}.png"), nrow=5)
        torch.save({"code_mean": code_mean, "score_traj": score_traj, "gen_traj": gen_traj},
                   join(expdir, f"evol_results_rep{repi:02d}.pt"))
        plt.figure()
        plt.plot(gen_traj, score_traj, '.', color="k", alpha=0.1)
        sns.lineplot(x=gen_traj, y=score_traj, ci=95)
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title(f"{titlestr}\nRep {repi}")
        plt.savefig(join(expdir, f"evol_score_traj_rep{repi:02d}.png"))
        plt.show()

#%%
import os
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
saveroot = r"F:\insilico_exps\ViT-CNN-Mixer"
#%%
"""ViT experiments """
modelname = 'vit_base_patch16_224_dino'
mvit = timm.create_model(modelname, pretrained=True)
mvit.cuda().eval().requires_grad_(False)
modeldir = join(saveroot, modelname)
os.makedirs(modeldir, exist_ok=True)
#%%
layeri = 11
token_id = 98
for unit_id in range(150, 200, 5):
    expdir = join(modeldir, f"L{layeri}_T{token_id}_U{unit_id}")
    os.makedirs(expdir, exist_ok=True)

    def score_fun(imgs, token_id=token_id, unit_id=unit_id):
        fetcher.record_module(mvit.blocks[layeri], "act", ingraph=True)
        mvit(imgs)
        fetcher.cleanup()
        return fetcher["act"][:, token_id, unit_id]

    titlestr = f"{modelname} L{layeri} T{token_id} U{unit_id}"
    grad_evol_pipeline(G, score_fun, titlestr=titlestr, expdir=expdir)



#%%
""" ResNet experiments """
modelname = 'resnetv2_50'
mres = timm.create_model(modelname, pretrained=True)
mres.cuda().eval().requires_grad_(False)
modeldir = join(saveroot, modelname)
os.makedirs(modeldir, exist_ok=True)
#%
# chan_id = 400
for chan_id in range(205, 300, 5):
    x_id, y_id = 3, 3
    expdir = join(modeldir, f"L4B2_T{x_id}_{y_id}_U{chan_id}")
    titlestr = f"{modelname} L4B2 ch{chan_id} [{x_id}, {y_id}]"
    os.makedirs(expdir, exist_ok=True)


    def score_fun(imgs, ):
        fetcher.record_module(mres.stages[3].blocks[2], "act", ingraph=True)
        mres(imgs)
        fetcher.cleanup()
        return fetcher["act"][:, chan_id, x_id, y_id]

    grad_evol_pipeline(G, score_fun, titlestr=titlestr, expdir=expdir)
# imgs, zs, score_traj, gen_traj = grad_score_fun(G, score_fun, )
# show_imgrid(imgs, nrow=5)
#%%
# use the Agg backend
import matplotlib
matplotlib.use('Agg')
# change back to the default backend
# matplotlib.use('module://backend_interagg')
#%%
""" Mixer experiments """
modelname = 'mixer_b16_224'
mmix = timm.create_model(modelname, pretrained=True)
mmix.cuda().eval().requires_grad_(False)
modeldir = join(saveroot, modelname)
os.makedirs(modeldir, exist_ok=True)


token_id = 98
for unit_id in range(200, 300, 5):
    expdir = join(modeldir, f"L11_T{token_id}_U{unit_id}")
    titlestr = f"{modelname} L11 ch{unit_id} T[{token_id}]"
    os.makedirs(expdir, exist_ok=True)

    def score_fun(imgs, ):
        fetcher.record_module(mmix.blocks[11], "act", ingraph=True)
        mmix(imgs)
        fetcher.cleanup()
        return fetcher["act"][:, token_id, unit_id]

    grad_evol_pipeline(G, score_fun, titlestr=titlestr, expdir=expdir)
#%%
from core.utils.robustCNN_utils import load_pretrained_robust_model
""" RobustCNN experiments """
modelname = "resnext50_32x4d_Robust"
resAT = load_pretrained_robust_model("resnext50_32x4d")
resAT.cuda().eval().requires_grad_(False)
modeldir = join(saveroot, modelname)
os.makedirs(modeldir, exist_ok=True)

x_id, y_id = 3, 3
for chan_id in range(205, 300, 5):
    expdir = join(modeldir, f"L4B2_T{x_id}_{y_id}_U{chan_id}")
    titlestr = f"{modelname} L4B2 ch{chan_id} [{x_id}, {y_id}]"
    os.makedirs(expdir, exist_ok=True)

    def score_fun(imgs, ):
        fetcher.record_module(resAT.layer4[2], "act", ingraph=True)
        resAT(imgs)
        fetcher.cleanup()
        return fetcher["act"][:, chan_id, x_id, y_id]

    grad_evol_pipeline(G, score_fun, titlestr=titlestr, expdir=expdir)


#%%

imgs, codes, score_trajs, gen_trajs = grad_score_fun(G, score_fun, generation=100, batch_size=10)
show_imgrid(imgs, nrow=5)
save_imgrid(imgs, join(expdir, "grad_evol_imgs.png"), nrow=5)
plt.figure()
plt.plot(gen_trajs, score_trajs, '-')
plt.xlabel("Generation")
plt.ylabel("Score")
plt.title(f"{modelname} L{layeri} T{token_id} U{unit_id}")
plt.savefig(join(expdir, "grad_score_traj.png"))
plt.show()
torch.save({"codes":codes.detach(), "score_trajs": score_trajs, "gen_trajs": gen_trajs},
           join(expdir, "grad_evol_results.pt"))

#%%
for repi in range(10):
    img_avg, code_mean, score_traj, gen_traj = evol_score_fun(G, score_fun, generation=100)
    save_imgrid(img_avg, join(expdir, f"evol_img_rep{repi:02d}.png"), nrow=5)
    torch.save({"code_mean":code_mean, "score_traj": score_traj, "gen_traj": gen_traj},
               join(expdir, f"evol_results_rep{repi:02d}.pt"))
    plt.figure()
    plt.plot(gen_traj, score_traj, '.', color="k", alpha=0.1)
    sns.lineplot(x=gen_traj, y=score_traj, ci=95)
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(f"{modelname} L{layeri} T{token_id} U{unit_id}\nRep {repi}")
    plt.savefig(join(expdir, f"evol_score_traj_rep{repi:02d}.png"))
    plt.show()
#%%

#%% Dev
CMAoptim = CholeskyCMAES(4096, )
codes = np.random.randn(1, 4096)
for i in range(100):
    imgs = G.visualize_batch_np(codes).cuda()
    scores = score_fun(preprocess(imgs))
    codes = CMAoptim.step_simple(scores.numpy(), codes)
    print(scores.mean())

fetcher.cleanup()
img_avg = G.visualize_batch_np(codes.mean(0, keepdims=True))
show_imgrid(img_avg)




#%% SCRATCH ZONE
# fetcher.record_module(mvit.blocks[11].mlp.act, "act", ingraph=True)
def score_fun(imgs, token_id=98, unit_id=200):
    fetcher.record_module(mvit.blocks[11], "act", ingraph=False)
    mvit(imgs)
    fetcher.cleanup()
    return fetcher["act"][:, token_id, unit_id]


img_avg, code_mean, score_traj, gen_traj = evol_score_fun(G, score_fun, generation=50)
#%%
def score_fun(imgs, token_id=98, unit_id=200):
    fetcher.record_module(mvit.blocks[11], "act", ingraph=True)
    mvit(imgs)
    fetcher.cleanup()
    return fetcher["act"][:, token_id, unit_id]


imgs, codes, score_trajs, gen_trajs = grad_score_fun(G, score_fun, generation=100, batch_size=10)
show_imgrid(imgs, nrow=5)
#%%
img = torch.randn(1, 3, 224, 224).cuda()
out = mvit(img)
#%%
fetcher = featureFetcher_module()
#%%
fetcher.record_module(mvit.blocks[8], "act",)
with torch.no_grad():
    mvit(img)
fetcher.cleanup()
fetcher["act"].shape
#%%
#%%
fetcher.record_module(mvit.blocks[11], "act", ingraph=True)

def score_fun(imgs, token_id=98, unit_id=10):
    mvit(imgs)
    return fetcher["act"][:, token_id, unit_id]


z = torch.randn(10, 4096).cuda()
z.requires_grad_(True)
optimizer = torch.optim.Adam([z], lr=5e-2)
for i in range(100):
    img = G.visualize(z)
    # inputs = feature_extractor(img, return_tensors="pt")
    scores = score_fun(preprocess(img))
    loss = - scores.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
fetcher.cleanup()
show_imgrid(img, nrow=5)
#%%

fetcher.record_module(mvit.blocks[11].mlp.act, "act", ingraph=True)

def score_fun(imgs, token_id=98, unit_id=2000):
    mvit(imgs)
    return fetcher["act"][:, token_id, unit_id]

z = torch.randn(10, 4096).cuda()
z.requires_grad_(True)
optimizer = torch.optim.Adam([z], lr=5e-2)
for i in range(100):
    img = G.visualize(z)
    # inputs = feature_extractor(img, return_tensors="pt")
    scores = score_fun(preprocess(img))
    loss = - scores.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
fetcher.cleanup()
show_imgrid(img, nrow=5)