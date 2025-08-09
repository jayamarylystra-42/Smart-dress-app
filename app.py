# app.py - Smart Dress Try-On + Skin Tone + Chatbot + Color slider + Accessories + Before/After
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import mediapipe as mp
import os

# Optional OpenAI (for better chat). If not available, app uses rule-based fallback.
try:
    import openai
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

st.set_page_config(page_title="Smart Dress Try-On (Hackathon)", layout="centered")
st.title("Smart Dress Try-On & AI Stylist")

st.markdown("""
**Upload** a full-body front photo and a transparent dress PNG.  
Use the controls to auto-fit, change dress color, add accessories, and chat with your stylist.
""")

# --- Upload UI ---
col1, col2 = st.columns(2)
with col1:
    person_file = st.file_uploader("Upload full-body photo (jpg/png)", type=["jpg","jpeg","png"])
with col2:
    dress_file  = st.file_uploader("Upload dress PNG (transparent)", type=["png"])

# accessory uploads (optional)
st.markdown("**Optional accessories (transparent PNGs)**")
acc_col1, acc_col2 = st.columns(2)
with acc_col1:
    acc_neck = st.file_uploader("Upload necklace PNG (optional)", type=["png"], key="neck")
with acc_col2:
    acc_glass = st.file_uploader("Upload glasses PNG (optional)", type=["png"], key="glass")

# color slider & tweak
st.sidebar.header("Styling Controls")
hue_shift = st.sidebar.slider("Dress hue shift (-180..180)", -180, 180, 0)
sat_factor = st.sidebar.slider("Dress saturation factor (0.2–2.0)", 0.2, 2.0, 1.0)
value_factor = st.sidebar.slider("Dress brightness factor (0.5–1.8)", 0.5, 1.8, 1.0)
scale_factor = st.sidebar.slider("Dress size multiplier", 0.9, 1.4, 1.15)
y_offset = st.sidebar.slider("Vertical offset (px)", -200, 200, 0)

show_before_after = st.sidebar.checkbox("Show before / after comparison", value=False)

# Chat UI
st.markdown("---")
st.subheader("Chat with your virtual stylist")
chat_input = st.text_input("Ask about fit, colors, makeup, or size...")

# Mediapipe setup
mp_pose = mp.solutions.pose

# helpers
def resize_max_width(pil_img, max_w=1024):
    w, h = pil_img.size
    if w <= max_w:
        return pil_img
    new_h = int(h * (max_w / w))
    return pil_img.resize((max_w, new_h), Image.ANTIALIAS)

def landmarks_from_pil(pil_img, model_complexity=1):
    img_rgb = np.array(pil_img.convert("RGB"))
    h, w = img_rgb.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=model_complexity) as pose:
        results = pose.process(img_rgb)
    return results, w, h

def to_pixel(landmark, w, h):
    x = min(max(landmark.x, 0.0), 1.0)
    y = min(max(landmark.y, 0.0), 1.0)
    return int(round(x * w)), int(round(y * h))

def compute_metrics(results, w, h):
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark
    L = mp_pose.PoseLandmark
    try:
        left_sh = to_pixel(lm[L.LEFT_SHOULDER.value], w, h)
        right_sh= to_pixel(lm[L.RIGHT_SHOULDER.value], w, h)
        left_hip = to_pixel(lm[L.LEFT_HIP.value], w, h)
        right_hip= to_pixel(lm[L.RIGHT_HIP.value], w, h)
        left_ank = to_pixel(lm[L.LEFT_ANKLE.value], w, h)
        right_ank= to_pixel(lm[L.RIGHT_ANKLE.value], w, h)
        nose     = to_pixel(lm[L.NOSE.value], w, h)
    except Exception:
        return None
    mid_shoulder = ((left_sh[0] + right_sh[0])//2, (left_sh[1] + right_sh[1])//2)
    mid_hip = ((left_hip[0] + right_hip[0])//2, (left_hip[1] + right_hip[1])//2)
    def dist(a,b): return float(np.linalg.norm(np.array(a)-np.array(b)))
    shoulder_width = dist(left_sh, right_sh)
    hip_width = dist(left_hip, right_hip)
    torso_height = dist(mid_shoulder, mid_hip)
    ankle_mid = ((left_ank[0]+right_ank[0])//2, (left_ank[1]+right_ank[1])//2)
    head_to_ankle = dist(nose, ankle_mid)
    return {
        "left_shoulder": left_sh, "right_shoulder": right_sh,
        "left_hip": left_hip, "right_hip": right_hip,
        "mid_shoulder": mid_shoulder, "mid_hip": mid_hip,
        "shoulder_width": shoulder_width, "hip_width": hip_width,
        "torso_height": torso_height, "height_px": head_to_ankle,
        "image_w": w, "image_h": h
    }

def shift_hsv(pil_img, hue_deg=0, sat_mul=1.0, val_mul=1.0):
    # convert to numpy HSV, apply shifts, convert back
    img = np.array(pil_img.convert("RGBA"))
    alpha = img[...,3]
    rgb = img[...,:3]
    hsv = Image.fromarray(rgb).convert("HSV")
    arr = np.array(hsv).astype(np.int16)
    # Hue shift scaled to 0-255 hue channel
    arr[...,0] = (arr[...,0].astype(int) + int(hue_deg/360.0*255)) % 256
    arr[...,1] = np.clip(arr[...,1].astype(float)*sat_mul, 0, 255).astype(np.uint8)
    arr[...,2] = np.clip(arr[...,2].astype(float)*val_mul, 0, 255).astype(np.uint8)
    new_rgb = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")
    out = Image.new("RGBA", pil_img.size)
    out.paste(new_rgb, (0,0))
    out.putalpha(Image.fromarray(alpha))
    return out

def estimate_skin_tone(person_pil):
    w,h = person_pil.size
    cx, cy = w//2, max(h//6, 10)
    cw, ch = max(w//10, 10), max(h//12, 10)
    l,t,r,b = max(cx-cw,0), max(cy-ch,0), min(cx+cw,w), min(cy+ch,h)
    crop = person_pil.crop((l,t,r,b)).convert("RGB")
    arr = np.array(crop)
    avg = arr.mean(axis=(0,1))
    r,g,b = avg
    if r > b + 10: tone="Warm"; suggestion="earthy / coral / gold"
    elif b > r + 10: tone="Cool"; suggestion="pinks / plums / blues"
    else: tone="Neutral"; suggestion="nude / rose / mauve"
    return tone, suggestion, avg.astype(int)

def estimate_size_label(metrics):
    shoulder = metrics["shoulder_width"]
    img_h = metrics["image_h"]
    prop = shoulder / max(1, img_h)
    if prop < 0.18: return "XS / S"
    if prop < 0.22: return "S / M"
    if prop < 0.26: return "M / L"
    return "L / XL"

def fit_and_paste_dress(person_pil, dress_pil, metrics, scale_factor=1.15, y_offset_px=0):
    w = metrics["image_w"]; h = metrics["image_h"]
    base_width = max(metrics["shoulder_width"], metrics["hip_width"])
    target_w = max(1, int(round(base_width * scale_factor)))
    dw, dh = dress_pil.size
    scale = target_w / max(dw,1)
    new_w, new_h = int(round(dw*scale)), int(round(dh*scale))
    dress_resized = dress_pil.resize((new_w, new_h), Image.ANTIALIAS)
    cx = metrics["mid_shoulder"][0]
    x = int(round(cx - new_w/2))
    top_y = int(round(metrics["mid_shoulder"][1] - 0.15*new_h + y_offset_px))
    x = max(-new_w//2, min(x, w - new_w//2))
    top_y = max(-new_h//2, min(top_y, h - new_h//2))
    composed = person_pil.convert("RGBA")
    composed.paste(dress_resized, (x, top_y), dress_resized)
    return composed

def paste_accessory(person_pil, accessory_pil, anchor, scale_px=120, y_offset=0, x_offset=0):
    # anchor is (x,y) pixel pos to center accessory around
    w,h = person_pil.size
    aw, ah = accessory_pil.size
    scale = scale_px / max(1, aw)
    nw, nh = int(aw*scale), int(ah*scale)
    acc_resized = accessory_pil.resize((nw, nh), Image.ANTIALIAS)
    x = anchor[0] - nw//2 + x_offset
    y = anchor[1] - nh//2 + y_offset
    x = max(-nw//2, min(x, w - nw//2))
    y = max(-nh//2, min(y, h - nh//2))
    out = person_pil.convert("RGBA")
    out.paste(acc_resized, (x,y), acc_resized)
    return out

def draw_debug(person_pil, metrics):
    import cv2
    img = np.array(person_pil.convert("RGB")).copy()
    pts = [metrics["left_shoulder"], metrics["right_shoulder"], metrics["left_hip"], metrics["right_hip"], metrics["mid_shoulder"], metrics["mid_hip"]]
    for (x,y) in pts:
        cv2.circle(img, (int(x), int(y)), 6, (0,255,0), -1)
    return Image.fromarray(img)

# --- Main processing flow ---
if person_file and dress_file:
    person = Image.open(person_file).convert("RGB")
    dress  = Image.open(dress_file).convert("RGBA")
    person_proc = resize_max_width(person, max_w=1024)
    st.subheader("Input (resized for processing)")
    st.image(person_proc, use_column_width=True)

    with st.spinner("Detecting body landmarks..."):
        results, w, h = landmarks_from_pil(person_proc, model_complexity=1)
        metrics = compute_metrics(results, w, h)

    if metrics is None:
        st.error("Could not detect body landmarks. Use a clear, full-body front-facing photo, one person only.")
    else:
        st.success("Landmarks detected.")
        st.write("Shoulder px:", int(metrics["shoulder_width"]), " Hip px:", int(metrics["hip_width"]))
        tone, tone_sugg, avg_rgb = estimate_skin_tone(person_proc)
        st.write("Skin undertone:", tone, "| Suggestion:", tone_sugg)
        size_label = estimate_size_label(metrics)
        st.write("Estimated size (rough):", size_label)

        # color transform on dress (apply HSV tweak)
        dress_tinted = shift_hsv(dress, hue_deg=hue_shift, sat_mul=sat_factor, val_mul=value_factor)

        # auto-fit the tinted dress
        composed = fit_and_paste_dress(person_proc, dress_tinted, metrics, scale_factor=scale_factor, y_offset_px=y_offset)

        # add accessories if provided (necklace anchored at mid_hip~neckline; glasses anchored at mid_shoulder ~ nose)
        if acc_neck is not None:
            acc_neck_pil = Image.open(acc_neck).convert("RGBA")
            # approximate anchor: halfway between mid_shoulder and mid_hip
            anchor = ((metrics["mid_shoulder"][0] + metrics["mid_hip"][0])//2, (metrics["mid_shoulder"][1] + metrics["mid_hip"][1])//2)
            composed = paste_accessory(composed, acc_neck_pil, anchor, scale_px=int(metrics["shoulder_width"]*0.8), y_offset= int(metrics["torso_height"]*0.15))

        if acc_glass is not None:
            acc_glass_pil = Image.open(acc_glass).convert("RGBA")
            # anchor near nose point (use mid_shoulder.x, mid_shoulder.y - small amount) because nose available in computation too
            anchor = (metrics["mid_shoulder"][0], max(metrics["mid_shoulder"][1] - int(metrics["torso_height"]*0.45), 0))
            composed = paste_accessory(composed, acc_glass_pil, anchor, scale_px=int(metrics["shoulder_width"]*0.5), y_offset=-int(metrics["torso_height"]*0.35))

        if show_before_after:
            colA, colB = st.columns(2)
            with colA:
                st.subheader("Before")
                st.image(person_proc, use_column_width=True)
            with colB:
                st.subheader("After")
                st.image(composed, use_column_width=True)
        else:
            st.subheader("Auto-fit Try-On")
            st.image(composed, use_column_width=True)

        # debug view
        debug_img = draw_debug(person_proc, metrics)
        st.subheader("Landmarks (debug)")
        st.image(debug_img, use_column_width=True)

        # chat logic (use OpenAI if available)
        analysis_context = {
            "skin_tone": tone,
            "skin_suggestion": tone_sugg,
            "size_label": size_label,
            "shoulder_px": int(metrics["shoulder_width"]),
            "hip_px": int(metrics["hip_width"]),
            "torso_px": int(metrics["torso_height"])
        }

        def rule_reply(q, ctx):
            ql = q.lower()
            if "foundation" in ql or "shade" in ql:
                return f"Your undertone seems {ctx['skin_tone']}. Try foundation shades labeled {('W' if ctx['skin_tone']=='Warm' else 'C' if ctx['skin_tone']=='Cool' else 'N')} or sample in natural light."
            if "size" in ql or "fit" in ql:
                return f"Estimated size: {ctx['size_label']}. This is a rough estimate; measure shoulders/hips for accuracy."
            if "makeup" in ql:
                return f"Makeup tip: {ctx['skin_suggestion']}"
            if "color" in ql or "dress" in ql:
                return f"Recommended colors: {ctx['skin_suggestion']}"
            return "I can suggest matching colors, foundation undertone, or accessory placement. Ask about 'foundation', 'size' or 'color'."

        if chat_input:
            if HAVE_OPENAI and ("OPENAI_API_KEY" in st.secrets or os.environ.get("OPENAI_API_KEY")):
                try:
                    key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                    openai.api_key = key
                    system = "You are a helpful fashion stylist assistant. Use the context to answer user questions concisely."
                    prompt = f"Context: {analysis_context}\nUser question: {chat_input}"
                    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system","content":system},{"role":"user","content":prompt}], max_tokens=250)
                    reply = resp["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    st.error("OpenAI error: " + str(e))
                    reply = rule_reply(chat_input, analysis_context)
            else:
                reply = rule_reply(chat_input, analysis_context)
            st.markdown("**Assistant:** " + reply)
else:
    st.info("Upload both a full-body photo and a transparent dress PNG to try automatic fitting and chat.")
        
