import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math
from PIL import Image

st.title("YOSHIRO 姿勢分析AI")

uploaded_file = st.file_uploader("画像を選択", type=["jpg", "jpeg", "png"])

def angle_from_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return abs(round(math.degrees(math.atan2(dx, dy)), 1))

def percent(value, max_value):
    return min(round(value / max_value * 100, 1), 100)

def judge_lordosis(angle):
    if angle < 10:
        return "正常"
    elif angle < 20:
        return "軽度"
    elif angle < 35:
        return "中等度"
    else:
        return "重度"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)

        if results.pose_landmarks:
            annotated = image_np.copy()

            mp_draw.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            st.image(annotated, caption="MediaPipe自動検出", use_container_width=True)

            landmarks = results.pose_landmarks.landmark

            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]

            shoulder_p = (int(shoulder.x * w), int(shoulder.y * h))
            hip_p = (int(hip.x * w), int(hip.y * h))
            knee_p = (int(knee.x * w), int(knee.y * h))
            ankle_p = (int(ankle.x * w), int(ankle.y * h))
            ear_raw_p = (int(ear.x * w), int(ear.y * h))

            ear_lobe_p = (
                ear_raw_p[0] - 30,
                ear_raw_p[1] + 38
            )

            center_p = (
                int((shoulder_p[0] + hip_p[0]) / 2),
                int((shoulder_p[1] + hip_p[1]) / 2)
            )

            thoracic_angle = angle_from_vertical(shoulder_p, center_p)
            lumbar_angle = angle_from_vertical(center_p, hip_p)
            lordosis_angle = thoracic_angle + lumbar_angle

            st.subheader("解析結果")
            st.write("胸椎角:", thoracic_angle, "度")
            st.write("腰椎角:", lumbar_angle, "度")
            st.write("反り腰角:", lordosis_angle, "度")
            st.write("進行度:", percent(lordosis_angle, 35), "%")
            st.write("判定:", judge_lordosis(lordosis_angle))

            st.subheader("取得座標")
            st.write("耳:", ear_lobe_p)
            st.write("肩峰:", shoulder_p)
            st.write("体の中心点:", center_p)
            st.write("大転子:", hip_p)
            st.write("膝:", knee_p)
            st.write("外果:", ankle_p)

        else:
            st.error("骨格を検出できませんでした")
