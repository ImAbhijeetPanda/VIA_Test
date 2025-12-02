import torch
import streamlit as st
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage
from torchvision.models.efficientnet import EfficientNet
import torch.nn as nn
import torch.serialization


# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TRANSFORMS
# ============================================================
class SafeColorJitter(transforms.ColorJitter):
    def __call__(self, img):
        img = super().__call__(img)
        img = F.to_tensor(img)
        img = torch.clamp(img, 0, 1)
        return F.to_pil_image(img)


normalize_transform = transforms.Normalize([0.5]*3, [0.5]*3)


class PairedTransform:
    def __init__(self, spatial_transform, color_jitter, normalize=True):
        self.spatial_transform = spatial_transform
        self.color_jitter = color_jitter
        self.normalize = normalize

    def __call__(self, before_img, after_img):
        seed = torch.randint(0, 9999999, ()).item()
        torch.manual_seed(seed)
        before_img = self.spatial_transform(before_img)
        torch.manual_seed(seed)
        after_img = self.spatial_transform(after_img)

        before_img = self.color_jitter(before_img)
        after_img  = self.color_jitter(after_img)

        to_tensor = transforms.ToTensor()
        b = to_tensor(before_img)
        a = to_tensor(after_img)

        if self.normalize:
            b = normalize_transform(b)
            a = normalize_transform(a)

        return b, a


val_transform = PairedTransform(
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]),
    SafeColorJitter(0.0, 0.0, 0.0)
)

train_transform = PairedTransform(
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
    ]),
    SafeColorJitter(0.2, 0.2, 0.2)
)

to_pil = ToPILImage()
softmax = torch.nn.Softmax(dim=1)


def denorm(t):
    t = (t * 0.5) + 0.5
    t = torch.clamp(t, 0, 1)
    return to_pil(t)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    torch.serialization.add_safe_globals([EfficientNet])
    model_path = "best_efficientnet_full.pth"   # MUST be in repo root
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device).eval()
    return model


model = load_model()


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Solar Panel Fault Detection", layout="wide")

st.title("Solar Panel Image Comparison Classifier")
st.markdown("Upload **before** and **after** images (or upload one image twice).")


col1, col2 = st.columns(2)

before_file = col1.file_uploader("Upload BEFORE image", type=["jpg", "png", "jpeg"])
after_file  = col2.file_uploader("Upload AFTER image",  type=["jpg", "png", "jpeg"])


# If user uploads only one file → use it for both
single_mode = False
if before_file and not after_file:
    after_file = before_file
    single_mode = True
elif after_file and not before_file:
    before_file = after_file
    single_mode = True


if before_file and after_file:
    before_pil = Image.open(before_file).convert("RGB")
    after_pil  = Image.open(after_file).convert("RGB")

    st.subheader("Original Images")
    oc1, oc2 = st.columns(2)
    oc1.image(before_pil, caption="BEFORE", use_column_width=True)
    oc2.image(after_pil, caption="AFTER", use_column_width=True)

    # ----------------------------------------------------------
    # VAL TRANSFORM + PREDICTION
    # ----------------------------------------------------------
    b_det, a_det = val_transform(before_pil, after_pil)

    with torch.no_grad():
        inp = torch.cat([b_det, a_det], dim=0).unsqueeze(0).to(device)
        logits = model(inp)
        probs = softmax(logits).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    label = "Positive" if pred == 1 else "Negative"

    st.subheader("Prediction Result")
    st.markdown(f"### **Predicted Class: `{label}`**")
    st.write(f"**Negative Probability:** {probs[0]:.4f}")
    st.write(f"**Positive Probability:** {probs[1]:.4f}")
    if single_mode:
        st.info("Only one image uploaded. Used it as BOTH before/after.")

    # ----------------------------------------------------------
    # SHOW deterministic val transform
    # ----------------------------------------------------------
    st.subheader("Deterministic Model Input (val_transform)")
    vc1, vc2 = st.columns(2)
    vc1.image(denorm(b_det), caption="Before (val-transformed)", use_column_width=True)
    vc2.image(denorm(a_det), caption="After (val-transformed)", use_column_width=True)

    # ----------------------------------------------------------
    # AUGMENTATIONS
    # ----------------------------------------------------------
    st.subheader("Data Augmentations (train_transform)")

    for i in range(3):
        b_aug, a_aug = train_transform(before_pil, after_pil)

        aug_col1, aug_col2 = st.columns(2)
        aug_col1.image(denorm(b_aug), caption=f"Before (aug {i+1})", use_column_width=True)
        aug_col2.image(denorm(a_aug), caption=f"After (aug {i+1})", use_column_width=True)

else:
    st.info("Upload one or two images to begin.")
