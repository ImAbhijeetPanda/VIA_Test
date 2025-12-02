import torch
import streamlit as st
import numpy as np
from PIL import Image

from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage
from torchvision.models.efficientnet import EfficientNet
import torch.nn as nn
import torch.serialization


# ============================================================
# PAGE CONFIG / DEVICE
# ============================================================
st.set_page_config(
    page_title="Cervical Cancer Image Classifier",
    layout="wide",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small CSS touch for prediction badge
st.markdown(
    """
    <style>
    .pred-badge-positive {
        background-color:#16a34a;
        color:white;
        padding:0.2rem 0.6rem;
        border-radius:0.5rem;
        font-weight:600;
    }
    .pred-badge-negative {
        background-color:#dc2626;
        color:white;
        padding:0.2rem 0.6rem;
        border-radius:0.5rem;
        font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        transforms.CenterCrop(224),
    ]),
    SafeColorJitter(0.0, 0.0, 0.0),
)

train_transform = PairedTransform(
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ]),
    SafeColorJitter(0.2, 0.2, 0.2),
)

to_pil = ToPILImage()
softmax = torch.nn.Softmax(dim=1)


def denorm(t: torch.Tensor) -> Image.Image:
    t = (t * 0.5) + 0.5
    t = torch.clamp(t, 0, 1)
    return to_pil(t)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    torch.serialization.add_safe_globals([EfficientNet])
    model_path = "best_efficientnet_full.pth"  # file must be in repo root
    model = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError("Loaded object is not a torch.nn.Module. Check the checkpoint file.")
    model = model.to(device).eval()
    return model


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### About this demo")
    st.write(
        "Research prototype for **cervical cancer image classification** "
        "based on paired colposcopy images (before / after application)."
    )
    st.caption(
        "This tool is for **research and demonstration only** and is "
        "not a medical diagnostic device."
    )
    st.caption(f"Running on: `{device}`")


# ============================================================
# MAIN LAYOUT
# ============================================================
st.title("Cervical Cancer Image Classifier")

st.markdown(
    "Upload **before** and **after** cervix images. "
    "If you only upload one image, it will be used as **both** inputs."
)

u_col1, u_col2 = st.columns(2)
before_file = u_col1.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"])
after_file  = u_col2.file_uploader("Upload AFTER image",  type=["jpg", "jpeg", "png"])

# Single-image mode logic
single_mode = False
if before_file and not after_file:
    after_file = before_file
    single_mode = True
elif after_file and not before_file:
    before_file = after_file
    single_mode = True

run_prediction = False
if before_file and after_file:
    # Show predict button only when we have at least one image
    run_prediction = st.button("🔍 Run Prediction", type="primary")
else:
    st.info("Upload one or two images to enable prediction.")

if run_prediction and before_file and after_file:
    before_pil = Image.open(before_file).convert("RGB")
    after_pil  = Image.open(after_file).convert("RGB")

    # ========================================================
    # 1) PREDICTION (FIRST)
    # ========================================================
    b_det, a_det = val_transform(before_pil, after_pil)

    with torch.no_grad():
        inp = torch.cat([b_det, a_det], dim=0).unsqueeze(0).to(device)
        logits = model(inp)
        probs = softmax(logits).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    label = "Positive" if pred == 1 else "Negative"
    badge_class = "pred-badge-positive" if pred == 1 else "pred-badge-negative"

    st.markdown("---")
    st.subheader("Prediction")

    p_col1, p_col2 = st.columns([2, 3])

    with p_col1:
        st.markdown(
            f"#### Predicted Class: "
            f"<span class='{badge_class}'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**Negative Probability:** {probs[0]:.4f}")
        st.write(f"**Positive Probability:** {probs[1]:.4f}")
        if single_mode:
            st.info("Single-image mode: the same image was used as BEFORE and AFTER.")

    with p_col2:
        st.markdown("**Class Probabilities**")
        st.bar_chart({"Negative": [probs[0]], "Positive": [probs[1]]})

    # ========================================================
    # 2) UPLOADED IMAGES (SMALLER THUMBNAILS)
    # ========================================================
    st.markdown("---")
    st.subheader("Uploaded Images")

    img_col1, img_col2 = st.columns(2)
    img_col1.image(before_pil, caption="BEFORE", width=320)
    img_col2.image(after_pil, caption="AFTER", width=320)

    # ========================================================
    # 3) MODEL INPUT (VAL TRANSFORM)
    # ========================================================
    st.markdown("---")
    st.subheader("Model Input (val_transform)")

    vt_col1, vt_col2 = st.columns(2)
    vt_col1.image(denorm(b_det), caption="Before (val-transformed)", width=320)
    vt_col2.image(denorm(a_det), caption="After (val-transformed)", width=320)

    # ========================================================
    # 4) AUGMENTATIONS (EXPANDER)
    # ========================================================
    with st.expander("Show training-time augmentations", expanded=False):
        for i in range(3):
            b_aug, a_aug = train_transform(before_pil, after_pil)
            aug_c1, aug_c2 = st.columns(2)
            aug_c1.image(
                denorm(b_aug),
                caption=f"Before (augmentation {i+1})",
                width=320,
            )
            aug_c2.image(
                denorm(a_aug),
                caption=f"After (augmentation {i+1})",
                width=320,
            )
