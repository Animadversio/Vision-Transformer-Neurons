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
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")
model.eval().requires_grad_(False)
#%%
import torch
from core.utils import upconvGAN, CholeskyCMAES
from core.utils.layer_hook_utils import featureFetcher_module
from core.utils.plot_utils import save_imgrid, show_imgrid, showimg
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
model.eval().requires_grad_(False).cuda()
#%%
from torchmetrics.functional import pairwise_cosine_similarity

#%%
from torchvision.transforms import ToTensor, ToPILImage, Resize, Normalize, Compose
preprocess = Compose([
    Resize(224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #ToTensor(),
])
#%%
z = torch.randn(5, 4096).cuda()
z.requires_grad_(True)
optimizer = torch.optim.Adam([z], lr=5e-2)
#%%

fetcher = featureFetcher_module()
fetcher.record_module(model.encoder.layer[8], "score", ingraph=True,)

for i in range(100):
    img = G.visualize(z)
    # inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(preprocess(img))
    optimizer.zero_grad()
    loss = - fetcher["score"][:, 500, 50].mean()
    loss.backward()
    optimizer.step()
    print(loss.item())
fetcher.cleanup()
#%%
show_imgrid(img)
#%%
fetcher.record_module(model.encoder.layer[8], "score", ingraph=True, )
img_grad = img.detach().requires_grad_(True)
outputs = model(preprocess(img_grad))
loss = - fetcher["score"][:, 500, 50].mean()
loss.backward()
fetcher.cleanup()
#%%
show_imgrid(img_grad.grad)
#%%
img_grad_mat = img_grad.grad.reshape(img_grad.shape[0], -1)
img_grad_norm_mat = img_grad.grad.norm(dim=1).reshape(img_grad.shape[0], -1)
#%%
grad_norm_map = img_grad.grad.norm(dim=1, keepdim=True)
show_imgrid(grad_norm_map / grad_norm_map.quantile(0.99))
#%%

torch.cosine_similarity(img_grad_mat[0:1], img_grad_mat[1:])  # ~ 0.08
#%%

pairwise_cosine_similarity(img_grad_norm_mat)  # ~ 0.08
#%%
#%%
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
