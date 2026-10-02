"""
Generating LIME explanations for chest X-ray images.
"""

import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os


# LIME EXPLANATION
def generate_lime_image(pil_image, model):
    img = np.array(pil_image.resize((224, 224)))  # resize for LIME
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img,
        model.predict,
        top_labels=1,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=30,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp / 255.0, mask)

    lime_pil = Image.fromarray((lime_img * 255).astype(np.uint8))

    # -------------------------------------
    # UPSCALE => to make LIME image bigger
    # -------------------------------------
    lime_pil = lime_pil.resize((600, 600), Image.NEAREST)

    return lime_pil

