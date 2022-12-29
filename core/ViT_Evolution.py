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

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')
inputs = feature_extractor(images=image, return_tensors="pt")
#%%
from core.utils.layer_hook_utils import featureFetcher_module
from core.utils.plot_utils import save_imgrid
#%%
target_module = model.encoder.layer[8]
fetcher = featureFetcher_module()
fetcher.record_module(target_module, "L8",
          return_input=False, ingraph=False, store_device=None)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
fetcher.cleanup()
