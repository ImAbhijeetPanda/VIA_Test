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

# Small CSS for prediction badge
st.markdown(
    """
    <style>
    .pred-badge-positive {
        background-color:#16a34a;
        color:white;
        padding:0.25rem 0.7rem;
        border-radius:0.5rem;
        font-weight:600;
        font-size:0.95rem;
    }
    .pred-badge-negative {
        background-color:#dc2626;
        color:white;
        padding:0.25rem 0.7rem;
        border-radius:0.5rem;
        font-weight:600;
        font-size:0.95rem;
    }
    .small-caption {
        font-size:0.8rem;
        color:#9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SafeColorJitter — Custom color jitter with pixel clamping
# ============================================================
class SafeColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, img):
        img = super().__call__(img)
        img = F.to_tensor(img)
        img = torch.clamp(img, 0, 1)
        img = F.to_pil_image(img)
        return img


# ============================================================
# Paired Transformations for VIA-style Dataset
# ============================================================
spatial_transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),                     # horizontal flip
    transforms.RandomRotation(15),                         # rotate ±15 degrees
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # crop & resize
])

spatial_transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

normalize_transform = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)


class PairedTransform:
    """
    Applies identical geometric transforms (flip, rotation, crop)
    to both before and after images, then applies independent
    SafeColorJitter and normalization.
    """
    def __init__(self, spatial_transform, color_jitter, normalize=True):
        self.spatial_transform = spatial_transform
        self.color_jitter = color_jitter
        self.normalize = normalize

    def __call__(self, before_img, after_img):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        torch.manual_seed(seed)
        before_img = self.spatial_transform(before_img)
        torch.manual_seed(seed)
        after_img = self.spatial_transform(after_img)

        before_img = self.color_jitter(before_img)
        after_img  = self.color_jitter(after_img)

        to_tensor = transforms.ToTensor()
        before_t, after_t = to_tensor(before_img), to_tensor(after_img)
        if self.normalize:
            before_t = normalize_transform(before_t)
            after_t  = normalize_transform(after_t)

        return before_t, after_t


train_transform = PairedTransform(
    spatial_transform_train,
    SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
)
val_transform = PairedTransform(
    spatial_transform_val,
    SafeColorJitter(brightness=0.0, contrast=0.0, saturation=0.0),
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
        "based on paired colposcopy images (before / after)."
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

single_mode = False
if before_file and not after_file:
    after_file = before_file
    single_mode = True
elif after_file and not before_file:
    before_file = after_file
    single_mode = True

run_prediction = False
if before_file and after_file:
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

    # Left: label + text
    with p_col1:
        st.markdown(
            f"#### Predicted Class: "
            f"<span class='{badge_class}'>{label}</span>",
            unsafe_allow_html=True,
        )
        if single_mode:
            st.markdown(
                "<span class='small-caption'>Single-image mode: the same image was used as BEFORE and AFTER.</span>",
                unsafe_allow_html=True,
            )

    # Right: compact gauge bars
    with p_col2:
        st.write("**Class Probabilities**")
        st.write(f"Negative: {probs[0]:.3f}")
        st.progress(float(probs[0]))
        st.write(f"Positive: {probs[1]:.3f}")
        st.progress(float(probs[1]))

    # ========================================================
    # 2) UPLOADED IMAGES
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
    # 4) AUGMENTATIONS (EACH BLOCK = SINGLE TRANSFORM)
    # ========================================================
    with st.expander("Show individual augmentations", expanded=False):
        st.markdown(
            "Each block below shows a **single augmentation type** applied "
            "to the BEFORE and AFTER images."
        )
        st.markdown("---")

        # 4.1 SafeColorJitter only
        st.markdown("### Augmentation 1 – SafeColorJitter")
        cj = SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        b_cj = cj(before_pil.copy())
        a_cj = cj(after_pil.copy())
        col1, col2 = st.columns(2)
        col1.image(b_cj, caption="Before (SafeColorJitter)", width=320)
        col2.image(a_cj, caption="After (SafeColorJitter)", width=320)
        st.markdown("---")

        # 4.2 RandomRotation only (±15°), same angle for both
        st.markdown("### Augmentation 2 – RandomRotation(±15°)")
        angle = float(torch.empty(1).uniform_(-15, 15).item())
        b_rot = F.rotate(before_pil.copy(), angle)
        a_rot = F.rotate(after_pil.copy(), angle)
        col1, col2 = st.columns(2)
        col1.image(b_rot, caption=f"Before (RandomRotation {angle:.1f}°)", width=320)
        col2.image(a_rot, caption=f"After (RandomRotation {angle:.1f}°)", width=320)
        st.markdown("---")

        # 4.3 RandomHorizontalFlip only, same decision for both
        st.markdown("### Augmentation 3 – RandomHorizontalFlip")
        do_flip = bool(torch.rand(1).item() > 0.5)
        if do_flip:
            b_flip = F.hflip(before_pil.copy())
            a_flip = F.hflip(after_pil.copy())
            flip_label = "applied"
        else:
            b_flip = before_pil.copy()
            a_flip = after_pil.copy()
            flip_label = "not applied"
        col1, col2 = st.columns(2)
        col1.image(b_flip, caption=f"Before (RandomHorizontalFlip {flip_label})", width=320)
        col2.image(a_flip, caption=f"After (RandomHorizontalFlip {flip_label})", width=320)
        st.markdown("---")

        # 4.4 RandomResizedCrop only (224, scale=(0.8, 1.0)), same params for both
        st.markdown("### Augmentation 4 – RandomResizedCrop(224, scale=(0.8, 1.0))")
        rrc = transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        # Get params once, apply to both via functional API
        i, j, h, w = transforms.RandomResizedCrop.get_params(before_pil, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        b_rrc = F.resized_crop(before_pil.copy(), i, j, h, w, (224, 224))
        a_rrc = F.resized_crop(after_pil.copy(), i, j, h, w, (224, 224))
        col1, col2 = st.columns(2)
        col1.image(b_rrc, caption="Before (RandomResizedCrop 224, scale=(0.8,1.0))", width=320)
        col2.image(a_rrc, caption="After (RandomResizedCrop 224, scale=(0.8,1.0))", width=320)
