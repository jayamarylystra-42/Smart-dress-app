import streamlit as st
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Simple Dress Try-On", layout="centered")
st.title("Simple Dress Try-On + Skin Tone + Size + Chatbot")

# Upload inputs
person_file = st.file_uploader("Upload full-body photo (jpg/png)", type=["jpg", "jpeg", "png"])
dress_file = st.file_uploader("Upload transparent dress PNG", type=["png"])

# Sidebar controls for color adjustment
st.sidebar.header("Dress Color Adjustments")
hue_shift = st.sidebar.slider("Hue shift (-180 to 180)", -180, 180, 0)
saturation = st.sidebar.slider("Saturation multiplier (0.5 to 2.0)", 0.5, 2.0, 1.0)
brightness = st.sidebar.slider("Brightness multiplier (0.5 to 2.0)", 0.5, 2.0, 1.0)

# Helper function: shift HSV (simplified)
def shift_hsv(img, hue_deg=0, sat_mul=1.0, val_mul=1.0):
    img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3]
    rgb = arr[..., :3]

    # Convert RGB to HSV using matplotlib (works well)
    import matplotlib.colors as mc
    rgb_norm = rgb / 255.0
    hsv = np.apply_along_axis(lambda x: mc.rgb_to_hsv(x), 2, rgb_norm)

    # Adjust HSV
    hsv[..., 0] = (hsv[..., 0] + hue_deg / 360.0) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mul, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_mul, 0, 1)

    # Convert back to RGB
    rgb_adj = np.apply_along_axis(lambda x: mc.hsv_to_rgb(x), 2, hsv)
    rgb_adj = (rgb_adj * 255).astype(np.uint8)

    out_arr = np.dstack((rgb_adj, alpha))
    return Image.fromarray(out_arr, mode="RGBA")

# Estimate skin tone
def estimate_skin_tone(img):
    w, h = img.size
    crop_box = (w // 3, h // 8, w * 2 // 3, h // 4)
    crop = img.crop(crop_box).convert("RGB")
    arr = np.array(crop)
    avg_rgb = arr.mean(axis=(0, 1))
    r, g, b = avg_rgb
    if r > b + 15:
        tone = "Warm"
        makeup = "earthy, coral, gold"
    elif b > r + 15:
        tone = "Cool"
        makeup = "pinks, plums, blues"
    else:
        tone = "Neutral"
        makeup = "nude, rose, mauve"
    return tone, makeup

# Estimate size roughly from width of shoulders (fake since no landmarks)
def estimate_size(img):
    w, h = img.size
    # Fake size estimate: use width relative to height (just a dummy heuristic)
    ratio = w / h
    if ratio < 0.3:
        return "XS / S"
    elif ratio < 0.4:
        return "S / M"
    elif ratio < 0.5:
        return "M / L"
    else:
        return "L / XL"

# Simple chatbot (rule-based)
def chatbot_reply(question, context):
    q = question.lower()
    if "foundation" in q or "shade" in q:
        return f"Your undertone is {context['skin_tone']}. Try foundation shades labeled accordingly."
    if "size" in q or "fit" in q:
        return f"Estimated size: {context['size']}. Measure shoulders and hips for best fit."
    if "makeup" in q or "color" in q:
        return f"Suggested makeup colors: {context['makeup']}."
    return "Ask me about foundation, size, or makeup!"

# Compose final image: center dress on person roughly, no complex fitting
def compose_image(person, dress):
    person = person.convert("RGBA")
    dress = shift_hsv(dress, hue_shift, saturation, brightness)

    # Resize dress relative to person width (about 60%)
    w_p, h_p = person.size
    w_d, h_d = dress.size
    scale = (w_p * 0.6) / w_d
    dress_resized = dress.resize((int(w_d * scale), int(h_d * scale)), Image.ANTIALIAS)

    # Paste dress at center horizontally, vertically at 25% from top
    x = (w_p - dress_resized.width) // 2
    y = int(h_p * 0.25)

    composed = person.copy()
    composed.paste(dress_resized, (x, y), dress_resized)
    return composed

# Main UI logic
if person_file and dress_file:
    person_img = Image.open(person_file)
    dress_img = Image.open(dress_file)

    st.subheader("Uploaded Photo")
    st.image(person_img, use_column_width=True)

    st.subheader("Dress Overlay with Color Adjustments")
    final_img = compose_image(person_img, dress_img)
    st.image(final_img, use_column_width=True)

    skin_tone, makeup_suggestion = estimate_skin_tone(person_img)
    size_estimate = estimate_size(person_img)

    st.write(f"**Estimated skin undertone:** {skin_tone}")
    st.write(f"**Makeup suggestions:** {makeup_suggestion}")
    st.write(f"**Estimated dress size:** {size_estimate}")

    # Chatbot
    question = st.text_input("Chat with stylist (ask about size, makeup, foundation, etc.)")
    if question:
        context = {"skin_tone": skin_tone, "makeup": makeup_suggestion, "size": size_estimate}
        answer = chatbot_reply(question, context)
        st.markdown(f"**Stylist:** {answer}")

else:
    st.info("Please upload both a full-body photo and a transparent dress PNG.")

