""""
You might need to fuck around with huggingface loging tokens etc and ask meta for permission 
to download the models by clicking the link that will appear in the terminal.
"""

from transformers import pipeline
from transformers.image_utils import load_image
import numpy as np

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-vits16-pretrain-lvd1689m",
    task="image-feature-extraction",
)
features = feature_extractor(image)

print(np.shape(np.array(features)))



