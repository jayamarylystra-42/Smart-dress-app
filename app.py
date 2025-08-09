# app.py - Smart Dress Try-On + Skin Tone + Landmarks + Chatbot + Color slider + Accessories
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

# Optional imports that may fail on some hosts
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Smart Dress Try-On", layout="centered")
st.title("Smart Dress Try-On + AI Stylist (6-feature demo)")

st.markdown("Upload a front-facing full-body photo and a transparent dress PNG. Use the sidebar to tune color/size; chat with the stylist below.")

# --------- Uploads ----------
col1, col2 = st.columns(2)
with col1:
    person_file = st.file_uploader("Upload full-body photo (jpg/png)", type=["jpg","jpeg","png"])
with col2:
    dress_file = st.file_uploader("Upload dress PNG (transparent)", type=["png"])

st.markdown("**Optional Accessory Uploads**")
acc_col1, acc_col2 = st.columns(2)
with acc_col1:
    acc_neck = st.file_uploader("Necklace PNG (optional)", type=["png"], key="neck")
with acc_col2:
    acc_glass = st.file_uploader("Glasses PNG (optional)", type=["png"], key="glass")

# --------- Sidebar controls ----------
st.sidebar.header("Styling Controls")
hue_shift = st.sidebar.slider("Dress hue shift (-180..180)", -180, 180, 0)
sat_factor = st.sidebar.slider("Dress saturation (0.2–2.0)", 0.2, 2.0, 1.0)
val_factor = st.sidebar.slider("Dress brightness (0.5–1.8)", 0.5, 1.8, 1.0)
scale_factor = st.sidebar.slider("Dress size multiplier", 0.9, 1.4, 1.15)
y_offset = st.sidebar.slider("Vertical offset (px)", -200, 200, 0)
show_before_after = st.sidebar.checkbox("Show before/after comparison", value=False)
st.sidebar.markdown("---")
st.sidebar.write("If OpenAI chat is desired, set `OPENAI_API_KEY` as an env var or Streamlit secret.")

# --------- helper functions ----------
def resize_max_width(img, max_w=1024):
    w,h = img.size
    if w <= max_w:
        return img
    new_h = int(h * (max_w / w))
    return img.resize((max_w, new_h), Image.ANTIALIAS)

def landmarks_from_pil(img_pil, model_complexity=1):
    if not MEDIAPIPE_AVAILABLE:
        return None, None, None
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=model_complexity) as pose:
        results = pose.process(img)
    return results, w, h

def to_pixel(landmark, w, h):
    x = min(max(landmark.x, 0.0), 1.0)
    y = min(max(landmark.y, 0.0), 1.0)
    return int(round(x * w)), int(round(y * h))

def compute_metrics(results, w, h):
    if not MEDIAPIPE_AVAILABLE or not results or not results.pose_landmarks:
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
    mid_shoulder = ((left_sh[0]+right_sh[0])//2, (left_sh[1]+right_sh[1])//2)
    mid_hip = ((left_hip[0]+right_hip[0])//2, (left_hip[1]+right_hip[1])//2)
    def dist(a,b): return float(np.linalg.norm(np.array(a)-np.array(b)))
    shoulder_w = dist(left_sh, right_sh)
    hip_w = dist(left_hip, right_hip)
    torso_h = dist(mid_shoulder, mid_hip)
    ankle_mid = ((left_ank[0]+right_ank[0])//2, (left_ank[1]+right_ank[1])//2)
    head_to_ankle = dist(nose, ankle_mid)
    return {
        "left_shoulder": left_sh, "right_shoulder": right_sh,
        "left_hip": left_hip, "right_hip": right_hip,
        "mid_shoulder": mid_shoulder, "mid_hip": mid_hip,
        "shoulder_width": shoulder_w, "hip_width": hip_w,
        "torso_height": torso_h, "height_px": head_to_ankle,
        "image_w": w, "image_h": h
    }

def fit_and_paste_dress(person_pil, dress_pil, metrics, scale_factor=1.15, y_offset_px=0):
    w = metrics["image_w"]; h = metrics["image_h"]
    base_width = max(metrics["shoulder_width"], metrics["hip_width"])
    target_w = max(1, int(round(base_width * scale_factor)))
    dw, dh = dress_pil.size
    scale = target_w / max(dw,1)
    new_w = int(round(dw * scale)); new_h = int(round(dh * scale))
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
    aw, ah = accessory_pil.size
    scale = scale_px / max(1, aw)
    nw, nh = int(aw*scale), int(ah*scale)
    acc = accessory_pil.resize((nw, nh), Image.ANTIALIAS)
    x = anchor[0] - nw//2 + x_offset
    y = anchor[1] - nh//2 + y_offset
    x = max(-nw//2, min(x, person_pil.size[0] - nw//2))
    y = max(-nh//2, min(y, person_pil.size[1] - nh//2))
    out = person_pil.convert("RGBA")
    out.paste(acc, (x, y), acc)
    return out

def shift_hsv(pil_img, hue_deg=0, sat_mul=1.0, val_mul=1.0):
    rgba = pil_img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[...,3]
    rgb = arr[...,:3]
    # convert to HSV using PIL hack
    hsv = Image.fromarray(rgb).convert("HSV")
    arrh = np.array(hsv).astype(np.int16)
    arrh[...,0] = (arrh[...,0] + int(hue_deg/360.0*255)) % 256
    arrh[...,1] = np.clip(arrh[...,1].astype(float)*sat_mul,0,255).astype(np.uint8)
    arrh[...,2] = np.clip(arrh[...,2].astype(float)*val_mul,0,255).astype(np.uint8)
    new_rgb = Image.fromarray(arrh.astype(np.uint8), mode="HSV").convert("RGB")
    out = Image.new("RGBA", pil_img.size)
    out.paste(new_rgb, (0,0))
    out.putalpha(Image.fromarray(alpha))
    return out

def estimate_skin_tone(person_pil):
    w,h = person_pil.size
    cx, cy = w//2, max(h//6, 10)
    cw, ch = max(w//10, 10), max(h//12, 10)
    left = max(cx-cw, 0); top = max(cy-ch, 0); right = min(cx+cw, w); bottom = min(cy+ch, h)
    crop = person_pil.crop((left, top, right, bottom)).convert("RGB")
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

def draw_debug(person_pil, metrics):
    import cv2
    img = np.array(person_pil.convert("RGB")).copy()
    pts = [metrics["left_shoulder"], metrics["right_shoulder"], metrics["left_hip"], metrics["right_hip"], metrics["mid_shoulder"], metrics["mid_hip"]]
    for x,y in pts:
        cv2.circle(img, (int(x), int(y)), 6, (0,255,0), -1)
    return Image.fromarray(img)

# --------- Main ----------
if person_file and dress_file:
    person = Image.open(person_file).convert("RGB")
    dress = Image.open(dress_file).convert("RGBA")
    person_proc = resize_max_width(person, max_w=1024)
    st.subheader("Input (resized for processing)")
    st.image(person_proc, use_column_width=True)

    if not MEDIAPIPE_AVAILABLE:
        st.warning("MediaPipe not available in this environment — auto-fit & landmark detection disabled. Install 'mediapipe' for full features.")
        # simple overlay fallback
        tinted = shift_hsv(dress, hue_deg=hue_shift, sat_mul=sat_factor, val_mul=val_factor)
        w,h = person_proc.size
        # naive paste: center horizontally, vertical at 30%
        new_h = int(person_proc.size[1]*0.55)
        scale = new_h / max(1, tinted.size[1])
        tinted_r = tinted.resize((int(tinted.size[0]*scale), new_h), Image.ANTIALIAS)
        x = (w - tinted_r.size[0])//2
        y = int(person_proc.size[1]*0.30) + y_offset
        comp = person_proc.convert("RGBA")
        comp.paste(tinted_r, (x,y), tinted_r)
        st.image(comp, use_column_width=True)
    else:
        with st.spinner("Detecting body landmarks..."):
            results, w, h = landmarks_from_pil(person_proc, model_complexity=1)
            metrics = compute_metrics(results, w, h)
        if metrics is None:
            st.error("Could not detect landmarks. Use clear, full-body front photo.")
        else:
            st.success("Landmarks detected.")
            tone, tone_sugg, avg_rgb = estimate_skin_tone(person_proc)
            st.write("Skin undertone:", tone, "| suggestion:", tone_sugg)
            size_label = estimate_size_label(metrics)
            st.write("Estimated size (rough):", size_label)
            # color transform and fit
            tinted = shift_hsv(dress, hue_deg=hue_shift, sat_mul=sat_factor, val_mul=val_factor)
            composed = fit_and_paste_dress(person_proc, tinted, metrics, scale_factor=scale_factor, y_offset_px=y_offset)
            # accessories
            if acc_neck:
                accN = Image.open(acc_neck).convert("RGBA")
                anchor = ((metrics["mid_shoulder"][0]+metrics["mid_hip"][0])//2, (metrics["mid_shoulder"][1]+metrics["mid_hip"][1])//2)
                composed = paste_accessory(composed, accN, anchor, scale_px=int(metrics["shoulder_width"]*0.8), y_offset=int(metrics["torso_height"]*0.15))
            if acc_glass:
                accG = Image.open(acc_glass).convert("RGBA")
                anchor = (metrics["mid_shoulder"][0], max(metrics["mid_shoulder"][1] - int(metrics["torso_height"]*0.45), 0))
                composed = paste_accessory(composed, accG, anchor, scale_px=int(metrics["shoulder_width"]*0.5), y_offset=-int(metrics["torso_height"]*0.35))
            if show_before_after:
                a,b = st.columns(2)
                with a:
                    st.subheader("Before")
                    st.image(person_proc, use_column_width=True)
                with b:
                    st.subheader("After")
                    st.image(composed, use_column_width=True)
            else:
                st.subheader("Auto-fit Try-On")
                st.image(composed, use_column_width=True)
            st.subheader("Landmarks (debug view)")
            st.image(draw_debug(person_proc, metrics), use_column_width=True)

            # Chatbot
            analysis = {"skin_tone": tone, "skin_suggestion": tone_sugg, "size_label": size_label}
            q = st.text_input("Chat with stylist (type a question):")
            def rule_reply(q, ctx):
                lq = q.lower()
                if "foundation" in lq or "shade" in lq: return f"Your undertone seems {ctx['skin_tone']}. Try foundation shades labeled {'W' if ctx['skin_tone']=='Warm' else 'C' if ctx['skin_tone']=='Cool' else 'N'}."
                if "size" in lq or "fit" in lq: return f"Estimate: {ctx['size_label']}. This is a quick guide; measure shoulders/hips to confirm."
                if "makeup" in lq: return f"Makeup tip: {ctx['skin_suggestion']}"
                if "color" in lq: return f"Colors: {ctx['skin_suggestion']}"
                return "Ask about foundation, color, size, or makeup."
            if q:
                if OPENAI_AVAILABLE and (st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
                    try:
                        key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
                        openai.api_key = key
                        sys = "You are a friendly fashion stylist assistant. Use the provided context."
                        prompt = f"Context: {analysis}\nUser: {q}"
                        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system","content":sys},{"role":"user","content":prompt}], max_tokens=200)
                        reply = resp["choices"][0]["message"]["content"].strip()
                    except Exception as e:
                        st.error("OpenAI error: " + str(e))
                        reply = rule_reply(q, analysis)
                else:
                    reply = rule_reply(q, analysis)
                st.markdown("**Assistant:** " + reply)
else:
    st.info("Upload both a full-body photo and a transparent dress PNG to try the auto-fit and chat.")
