# from transformers import ViTImageProcessor, ViTModel
# import torch
# from datasets import load_dataset
#
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
#
# feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#
# inputs = feature_extractor(image, return_tensors="pt")
#%%
import numpy as np
import torch
from core.utils import upconvGAN, CholeskyCMAES
from core.utils.layer_hook_utils import featureFetcher_module
from core.utils.plot_utils import save_imgrid, show_imgrid, showimg
from transformers import ViTFeatureExtractor, ViTModel
from torchmetrics.functional import pairwise_cosine_similarity
from PIL import Image
import requests


def avg_cosine_sim_mat(X):
    cosmat = pairwise_cosine_similarity(X,)
    idxs = torch.tril_indices(cosmat.shape[0], cosmat.shape[1], offset=-1)
    cosmat_vec = cosmat[idxs[0], idxs[1]]
    return cosmat, cosmat_vec.mean()
#%%
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")
model.eval().requires_grad_(False).cuda()
#%%
target_module = model.encoder.layer[8].output
target_module = model.encoder.layer[8].intermediate
target_module = model.encoder.layer[8].attention
target_module = model.encoder.layer[8]
fetcher = featureFetcher_module()
fetcher.record_module(target_module, "L8",
          return_input=False, ingraph=False, store_device=None)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
fetcher.cleanup()
# fetcher["L8"].shape
#%%
G = upconvGAN()
G.eval().requires_grad_(False).cuda()
#%%

#%%
from torchvision.transforms import ToTensor, ToPILImage, Resize, Normalize, Compose
preprocess = Compose([
    Resize(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #ToTensor(),
])
#%%
"""Gradient based optimization"""
z = torch.randn(10, 4096).cuda()
z.requires_grad_(True)
optimizer = torch.optim.Adam([z], lr=5e-2)
#%%
fetcher = featureFetcher_module()
fetcher.record_module(model.encoder.layer[8], "score", ingraph=True,)
for i in range(100):
    img = G.visualize(z)
    # inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(preprocess(img))
    loss = - fetcher["score"][:, 500, 50].mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
fetcher.cleanup()
#%%
show_imgrid(img, nrow=5)
#%%
fetcher.record_module(model.encoder.layer[8], "score", ingraph=True, )
img_grad = img.detach().requires_grad_(True)
outputs = model(preprocess(img_grad))
loss = - fetcher["score"][:, 500, 50].mean()
loss.backward()
fetcher.cleanup()
#%%
grad_img = img_grad.grad
show_imgrid(grad_img, nrow=5)
#%%
grad_norm_map = grad_img.norm(dim=1, keepdim=True)
show_imgrid(grad_norm_map / grad_norm_map.quantile(0.995),nrow=5)
#%%
cosmat_gradnorm = pairwise_cosine_similarity(grad_norm_map.flatten(1))  # ~ 0.08
rid, cid = torch.tril_indices(10,10,offset=-1)
print("Gradient norm cosine similarity: ", cosmat_gradnorm[rid, cid].mean().item())
#%%
cosmat_imgs = pairwise_cosine_similarity(img.flatten(1))  # ~ 0.08
rid, cid = torch.tril_indices(10,10,offset=-1)
cosmat_imgs[rid, cid].mean()
print("ActMax Image cosine similarity: ", cosmat_imgs[rid, cid].mean().item())
#%%
cosmat_gradimg = pairwise_cosine_similarity(grad_img.flatten(1))  # ~ 0.08
rid, cid = torch.tril_indices(10,10,offset=-1)
cosmat_gradimg[rid, cid].mean()
print("Gradient Image cosine similarity: ", cosmat_gradimg[rid, cid].mean().item())

#%%
#%%
"""Gradient based optimization of pixel parametrization"""
pixs = 0.5 + 0.01 * torch.rand(5, 3, 256, 256).cuda()
pixs.requires_grad_(True)
optimizer = torch.optim.Adam([pixs], lr=1e-2)
fetcher = featureFetcher_module()
fetcher.record_module(model.encoder.layer[8], "score", ingraph=True,)
for i in range(50):
    # inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(preprocess(torch.clamp(pixs, 0, 1)))
    optimizer.zero_grad()
    loss = - fetcher["score"][:, 500, 50].mean()
    loss.backward()
    optimizer.step()
    print(loss.item())
fetcher.cleanup()
#%%
show_imgrid(pixs)
#%%
show_imgrid(pixs.grad)
#%%
pix_grad_mat = pixs.grad.flatten(start_dim=1)
pix_grad_norm_mat = pixs.grad.norm(dim=1).reshape(img_grad.shape[0], -1)
#%%
show_imgrid(pixs.grad.norm(dim=1, keepdim=True) / pixs.grad.norm(dim=1, keepdim=True).quantile(0.99))
#%%
torch.cosine_similarity(pix_grad_mat[0:1], pix_grad_mat)
#%%
pairwise_cosine_similarity(pix_grad_norm_mat)
#%%
"""Gradient free CholCMAES evoution of the units """
CMAoptim = CholeskyCMAES(4096, )
codes = np.random.randn(1, 4096)
fetcher.record_module(model.encoder.layer[8], "score", ingraph=True,)
for i in range(100):
    imgs = G.visualize_batch_np(codes)
    # inputs = feature_extractor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(preprocess(imgs.cuda()))
    scores = fetcher["score"][:, 500, 50].detach().numpy()
    codes = CMAoptim.step_simple(scores, codes)
    print(scores.mean())

fetcher.cleanup()
#%%
show_imgrid(imgs)
#%%
from core.utils import TorchScorer
scorer = TorchScorer("resnet50_linf8", 224)
#%%
scorer.select_unit(("resnet50_linf8",".layer4.Bottleneck2",20,4,4), allow_grad=True)
#%%
zs = 0.5 * torch.randn(10, 4096).cuda()
zs.requires_grad_(True)
optimizer = torch.optim.Adam([zs], lr=1E-2, )
#%
for i in range(100):
    imgs = G.visualize(zs)
    score = scorer.score_tsr_wgrad(imgs)
    optimizer.zero_grad()
    loss = - score.mean()
    loss.backward()
    optimizer.step()
    zs.data.add_((score==0)[:, None] * torch.randn_like(zs) * 0.01)
    print(loss.item())
#%%
show_imgrid(imgs,nrow=5)
#%%
imgs_grad = imgs.detach().requires_grad_(True)
loss = - scorer.score_tsr_wgrad(imgs_grad).mean()
loss.backward()
CNN_img_grad = imgs_grad.grad
CNN_img_gradnorm = CNN_img_grad.norm(dim=1, keepdim=True)
#%%
show_imgrid(CNN_img_gradnorm * 500, nrow=5)
#%%
show_imgrid(CNN_img_grad * 50, nrow=5)
#%%
CNN_img_cosmat, CNN_img_cosavg = avg_cosine_sim_mat(imgs.flatten(start_dim=1))
CNN_img_grad_cosmat, CNN_img_grad_cosavg = avg_cosine_sim_mat(CNN_img_grad.flatten(start_dim=1))
CNN_img_gradnorm_cosmat, CNN_img_gradnorm_cosavg = avg_cosine_sim_mat(CNN_img_gradnorm.flatten(start_dim=1))
print("CNN Image cosine similarity: ", CNN_img_cosavg.item())
print("CNN Image gradient cosine similarity: ", CNN_img_grad_cosavg.item())
print("CNN Image gradient norm cosine similarity: ", CNN_img_gradnorm_cosavg.item())