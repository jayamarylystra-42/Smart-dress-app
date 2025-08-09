# app.py - simple Streamlit dress-tryon + skin-tone demo
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Smart Dress Try-On", layout="centered")
st.title("Smart Dress Shopping + Makeup Advisor")

st.write("Upload a full-body photo and a dress PNG (transparent). This demo overlays the dress and gives a simple skin-tone suggestion.")

# Upload files
person_file = st.file_uploader("Upload your photo", type=["jpg","jpeg","png"])
dress_file  = st.file_uploader("Upload dress image (PNG with transparent background)", type=["png"])

if person_file and dress_file:
    # 1) Open images
    person = Image.open(person_file).convert("RGB")
    dress  = Image.open(dress_file).convert("RGBA")

    # Show the original photo
    st.subheader("Your Photo")
    st.image(person, use_column_width=True)

    # 2) Estimate skin tone from a small central/upper crop (better than averaging whole image)
    w, h = person.size
    # sample area roughly where face is expected (center horizontally, upper third vertically)
    cx, cy = w // 2, max(h // 6, 10)
    crop_w, crop_h = max(w // 8, 10), max(h // 12, 10)
    left  = max(cx - crop_w, 0)
    top   = max(cy - crop_h, 0)
    right = min(cx + crop_w, w)
    bottom= min(cy + crop_h, h)
    face_crop = person.crop((left, top, right, bottom))
    face_np = np.array(face_crop)
    avg_rgb = face_np.mean(axis=(0,1))

    st.subheader("Skin tone (approx.)")
    st.write(f"Average RGB (sampled region): {avg_rgb.astype(int)}")

    # 3) Simple rule to suggest warm vs cool palette
    r, g, b = avg_rgb
    if r > b + 10:
        st.success("Detected: Warm undertone — try earthy / coral / gold shades.")
    elif b > r + 10:
        st.success("Detected: Cool undertone — try blues / plums / silvers.")
    else:
        st.success("Detected: Neutral undertone — many colors will suit you (nudes, rose, mauve).")

    # 4) Overlay the dress onto the person (basic placement)
    # resize dress to match person width proportionally (tweak multiplier as needed)
    target_height = int(h * 0.55)  # dress height as fraction of person photo height
    scale = target_height / max(dress.height, 1)
    new_w = int(dress.width * scale)
    new_h = int(dress.height * scale)
    dress_resized = dress.resize((new_w, new_h), Image.ANTIALIAS)

    # position: centered horizontally, top-of-dress approx at 1/3 height of the person image
    x = (w - new_w) // 2
    y = int(h * 0.30)

    # paste using alpha channel
    composed = person.convert("RGBA")
    composed.paste(dress_resized, (x, y), dress_resized)

    st.subheader("Try-On Result")
    st.image(composed, use_column_width=True)

    st.info("Tip: If dress placement is off, try a full-body front photo (head to toe) and/or crop your photo so shoulders and hips are visible.")
else:
    st.write("Please upload both your photo and a transparent dress PNG to see the result.")
    
