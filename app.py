# app.py - Streamlit Smart Dress Try-On + Skin Analysis + Chatbot (OpenAI optional)
import streamlit as st
from PIL import Image
import numpy as np
import mediapipe as mp
import io
import math
import os

# Try to import OpenAI; optional
try:
    import openai
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

st.set_page_config(page_title="Smart Dress Try-On + Chatbot", layout="centered")
st.title("Smart Dress Shopping + Makeup & Size Advisor")

st.markdown("""
Upload a full-body photo and a transparent dress PNG.  
This app will detect body landmarks (MediaPipe), estimate skin tone & size, auto-fit the dress, and provide recommendations.  
You can chat with the assistant below (OpenAI optional).
""")

# ---------- helper: OpenAI setup ----------
OPENAI_API_KEY = None
if HAVE_OPENAI:
    # Streamlit secrets: set in Streamlit Cloud or in local env
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY

# ---------- uploads ----------
person_file = st.file_uploader("Upload full-body photo (jpg/png)", type=["jpg","jpeg","png"])
dress_file  = st.file_uploader("Upload dress PNG (transparent background)", type=["png"])

# ---------- mediapipe + image helpers ----------
mp_pose = mp.solutions.pose

def resize_max_width(pil_img, max_w=1024):
    w, h = pil_img.size
    if w <= max_w:
        return pil_img
    new_h = int(h * (max_w / w))
    return pil_img.resize((max_w, new_h), Image.ANTIALIAS)

def landmarks_from_pil(pil_img, model_complexity=1, min_detection_confidence=0.5):
    img_rgb = np.array(pil_img.convert("RGB"))
    h, w = img_rgb.shape[:2]
    with mp_pose.Pose(static_image_mode=True, model_complexity=model_complexity,
                      enable_segmentation=False, min_detection_confidence=min_detection_confidence) as pose:
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

    def dist(a,b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    shoulder_width = dist(left_sh, right_sh)
    hip_width = dist(left_hip, right_hip)
    torso_height = dist(mid_shoulder, mid_hip)
    # estimate height in px as distance from nose to ankle (average ankle)
    ankle_mid = ((left_ank[0]+right_ank[0])//2, (left_ank[1]+right_ank[1])//2)
    head_to_ankle = dist(nose, ankle_mid)

    return {
        "left_shoulder": left_sh,
        "right_shoulder": right_sh,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "mid_shoulder": mid_shoulder,
        "mid_hip": mid_hip,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_height": torso_height,
        "height_px": head_to_ankle,
        "image_w": w,
        "image_h": h
    }

def fit_and_paste_dress(person_pil, dress_pil, metrics, scale_factor=1.15, y_offset_px=0):
    w = metrics["image_w"]
    h = metrics["image_h"]
    base_width = max(metrics["shoulder_width"], metrics["hip_width"])
    target_w = max(1, int(round(base_width * scale_factor)))

    dress_w0, dress_h0 = dress_pil.size
    scale = target_w / max(dress_w0, 1)
    new_w = int(round(dress_w0 * scale))
    new_h = int(round(dress_h0 * scale))
    dress_resized = dress_pil.resize((new_w, new_h), Image.ANTIALIAS)

    center_x = metrics["mid_shoulder"][0]
    x = int(round(center_x - new_w / 2))
    top_y = int(round(metrics["mid_shoulder"][1] - 0.15 * new_h + y_offset_px))

    x = max(-new_w//2, min(x, w - new_w//2))
    top_y = max(-new_h//2, min(top_y, h - new_h//2))

    composed = person_pil.convert("RGBA")
    composed.paste(dress_resized, (x, top_y), dress_resized)
    return composed

def draw_landmarks_on_pil(person_pil, metrics):
    import cv2
    img = np.array(person_pil.convert("RGB")).copy()
    pts = [metrics["left_shoulder"], metrics["right_shoulder"], metrics["left_hip"], metrics["right_hip"], metrics["mid_shoulder"], metrics["mid_hip"]]
    for (x,y) in pts:
        cv2.circle(img, (int(x), int(y)), 6, (0,255,0), -1)
    return Image.fromarray(img)

# ---------- Skin tone / suggestion ----------
def estimate_skin_tone(person_pil):
    w,h = person_pil.size
    # sample a small rectangle near upper center (face region)
    cx, cy = w//2, max(h//6, 10)
    crop_w, crop_h = max(w//10, 10), max(h//12, 10)
    left = max(cx - crop_w, 0)
    top  = max(cy - crop_h, 0)
    right= min(cx + crop_w, w)
    bottom=min(cy + crop_h, h)
    crop = person_pil.crop((left, top, right, bottom)).convert("RGB")
    arr = np.array(crop)
    avg = arr.mean(axis=(0,1))  # R,G,B
    r,g,b = avg
    # heuristic for undertone
    if r > b + 10:
        tone = "Warm"
        suggestion = "Try earthy, coral, bronze, or golden shades."
    elif b > r + 10:
        tone = "Cool"
        suggestion = "Try pinks, plums, blue-based shades, and silver accents."
    else:
        tone = "Neutral"
        suggestion = "Neutral undertone — many shades will suit you: nudes, rose, mauve."
    return tone, suggestion, avg.astype(int)

# ---------- Size estimation ----------
def estimate_size_label(metrics, image_h_px):
    # Basic heuristic mapping px -> clothing size requires calibration by camera/distance.
    # We'll do relative sizing based on shoulder width proportion to image height.
    shoulder = metrics["shoulder_width"]
    prop = shoulder / max(1, image_h_px)  # proportion of height
    # thresholds are arbitrary and may need tuning with your dataset
    if prop < 0.18:
        return "XS / Small"
    elif prop < 0.22:
        return "S / M"
    elif prop < 0.26:
        return "M / L"
    else:
        return "L / XL"

# ---------- Main ----------
if person_file and dress_file:
    # load images
    person = Image.open(person_file).convert("RGB")
    dress  = Image.open(dress_file).convert("RGBA")
    proc_person = resize_max_width(person, max_w=1024)

    st.subheader("Uploaded (resized for processing)")
    st.image(proc_person, use_column_width=True)

    with st.spinner("Detecting pose landmarks..."):
        results, w, h = landmarks_from_pil(proc_person, model_complexity=1, min_detection_confidence=0.5)
        metrics = compute_metrics(results, w, h)

    if metrics is None:
        st.error("Could not detect body landmarks. Please upload a clear, front-facing, full-body photo (one person only).")
    else:
        st.success("Body landmarks detected.")
        st.write("Shoulder width (px):", int(metrics["shoulder_width"]))
        st.write("Hip width (px):", int(metrics["hip_width"]))
        st.write("Torso height (px):", int(metrics["torso_height"]))
        st.write("Estimated total height (px from nose to ankle):", int(metrics["height_px"]))

        # skin tone
        tone, suggestion_text, avg_rgb = estimate_skin_tone(proc_person)
        st.subheader("Skin tone estimate")
        st.write(f"Undertone: **{tone}** — {suggestion_text}")
        st.write("Sampled average RGB:", list(avg_rgb))

        # size label (very rough)
        size_label = estimate_size_label(metrics, metrics["image_h"])
        st.subheader("Estimated size (rough)")
        st.write(size_label)
        st.info("This size is a rough estimate from image proportions — use only for a quick guide.")

        # fit dress
        composed = fit_and_paste_dress(proc_person, dress, metrics, scale_factor=1.18, y_offset_px=0)
        st.subheader("Auto-fit Try-On")
        st.image(composed, use_column_width=True)

        # debug landmarks
        debug = draw_landmarks_on_pil(proc_person, metrics)
        st.subheader("Detected Landmarks (debug view)")
        st.image(debug, use_column_width=True)

        # prepare context string for chat use
        analysis_context = {
            "skin_tone": tone,
            "skin_suggestion": suggestion_text,
            "size_label": size_label,
            "shoulder_px": int(metrics["shoulder_width"]),
            "hip_px": int(metrics["hip_width"]),
            "torso_px": int(metrics["torso_height"])
        }

        st.markdown("---")
        st.subheader("Chat with your virtual stylist")
        chat_input = st.text_input("Ask about matching, makeup, size, or styling tips:")

        # Simple rule-based reply fallback
        def rule_based_reply(user_text, ctx):
            ut = user_text.lower()
            if "foundation" in ut or "shade" in ut:
                return f"For foundation, start with your undertone **{ctx['skin_tone']}**; try shades labelled for warm (W), cool (C), or neutral (N). Sample in natural light."
            if "size" in ut or "fit" in ut:
                return f"Based on the photo, I estimate **{ctx['size_label']}**. For a better fit, check shoulder and hip measurements in cm (calibrate with a reference object)."
            if "makeup" in ut:
                return f"Makeup tip: {ctx['skin_suggestion']}"
            if "color" in ut or "dress" in ut:
                return f"Try these colours: {ctx['skin_suggestion']}"
            return "I recommend trying the dress shown and adjusting size/color based on your comfort. Ask about foundation, colors, or sizing."

        if chat_input:
            if OPENAI_API_KEY and HAVE_OPENAI:
                # use OpenAI ChatCompletion (gpt-3.5-turbo)
                try:
                    prompt_system = (
                        "You are a friendly fashion stylist assistant. Use the provided context about skin tone and estimated size "
                        "to answer user questions helpfully, concisely and politely."
                    )
                    messages = [
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": f"Context: {analysis_context}\n\nUser question: {chat_input}"}
                    ]
                    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=250)
                    reply = resp["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    st.error("OpenAI error: " + str(e))
                    reply = rule_based_reply(chat_input, {"skin_tone": tone, "skin_suggestion": suggestion_text, "size_label": size_label})
            else:
                reply = rule_based_reply(chat_input, {"skin_tone": tone, "skin_suggestion": suggestion_text, "size_label": size_label})
            st.markdown("**Assistant:** " + reply)
else:
    st.info("Please upload both a full-body photo and a transparent dress PNG to try automatic fitting and chat.")
    
