import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math
from PIL import Image
from rembg import remove

st.title("YOSHIRO 姿勢分析AI V20")

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

def line_x_at_y(p1, p2, y):
    x1, y1 = p1
    x2, y2 = p2
    if y2 == y1:
        return x1
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

def position_against_line(point, line_start, line_end):
    px, py = point
    base_x = line_x_at_y(line_start, line_end, py)
    diff = px - base_x

    if diff > 8:
        return "前方", round(diff, 1)
    elif diff < -8:
        return "後方", round(diff, 1)
    else:
        return "線上", round(diff, 1)

def find_body_center_from_alpha(alpha, y):
    h, w = alpha.shape
    y1 = max(0, y - 5)
    y2 = min(h, y + 5)

    band = alpha[y1:y2, :]
    mask = band > 20
    xs = np.where(mask)[1]

    if len(xs) == 0:
        return None, None, None

    back_x = int(xs.min())
    front_x = int(xs.max())
    body_width = front_x - back_x

    center_x = int(front_x - body_width * 0.45)
    corrected_back_x = int(front_x - body_width * 0.9)

    return center_x, corrected_back_x, front_x

def to_point(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def choose_back_point(left_p, right_p, facing_right=True):
    if facing_right:
        return left_p if left_p[0] <= right_p[0] else right_p
    else:
        return left_p if left_p[0] >= right_p[0] else right_p

def choose_lower_point(left_p, right_p):
    return left_p if left_p[1] >= right_p[1] else right_p

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    cutout = remove(image).convert("RGBA")
    cutout_np = np.array(cutout)

    image_np = np.array(image)
    h, w, _ = image_np.shape
    alpha = cutout_np[:, :, 3]

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            nose_p = to_point(landmarks[mp_pose.PoseLandmark.NOSE], w, h)

            left_ear_p = to_point(landmarks[mp_pose.PoseLandmark.LEFT_EAR], w, h)
            right_ear_p = to_point(landmarks[mp_pose.PoseLandmark.RIGHT_EAR], w, h)

            left_shoulder_p = to_point(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
            right_shoulder_p = to_point(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)

            left_hip_p = to_point(landmarks[mp_pose.PoseLandmark.LEFT_HIP], w, h)
            right_hip_p = to_point(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], w, h)

            left_knee_p = to_point(landmarks[mp_pose.PoseLandmark.LEFT_KNEE], w, h)
            right_knee_p = to_point(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], w, h)

            left_ankle_p = to_point(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE], w, h)
            right_ankle_p = to_point(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE], w, h)

            ear_raw_p = choose_back_point(left_ear_p, right_ear_p, True)

            # 肩峰・大転子は後方側
            shoulder_p = choose_back_point(left_shoulder_p, right_shoulder_p, True)
            hip_p = choose_back_point(left_hip_p, right_hip_p, True)

            # 膝中心・外果は「下にあるMediaPipe点」
            knee_p = choose_lower_point(left_knee_p, right_knee_p)
            ankle_p = choose_lower_point(left_ankle_p, right_ankle_p)

            # 耳垂補正
            ear_lobe_p = (
                ear_raw_p[0] - 30,
                ear_raw_p[1] + 38
            )

            # 体の中心点
            center_y = int(
                shoulder_p[1] + (hip_p[1] - shoulder_p[1]) * 0.5
            )

            center_x, back_x, front_x = find_body_center_from_alpha(alpha, center_y)

            if center_x is None:
                center_p = (
                    int((shoulder_p[0] + hip_p[0]) / 2),
                    center_y
                )
            else:
                center_p = (center_x, center_y)

            thoracic_angle = angle_from_vertical(shoulder_p, center_p)
            lumbar_angle = angle_from_vertical(center_p, hip_p)
            lordosis_angle = thoracic_angle + lumbar_angle

            hip_pos, hip_diff = position_against_line(
                hip_p,
                shoulder_p,
                knee_p
            )

            shoulder_pos, shoulder_diff = position_against_line(
                shoulder_p,
                ear_lobe_p,
                hip_p
            )

            knee_pos, knee_diff = position_against_line(
                knee_p,
                hip_p,
                ankle_p
            )

            if hip_pos == "前方" and shoulder_pos in ["前方", "線上"]:
                posture_type = "反り腰"
            elif hip_pos == "前方" and shoulder_pos == "後方":
                posture_type = "スウェイバック"
            elif hip_pos == "後方":
                posture_type = "猫背傾向"
            else:
                posture_type = "正常〜軽度"

            annotated = image_np.copy()

            mp_draw.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # 採用点を赤で表示
            cv2.circle(annotated, shoulder_p, 10, (0, 0, 255), -1)
            cv2.circle(annotated, hip_p, 10, (0, 0, 255), -1)
            cv2.circle(annotated, knee_p, 10, (0, 0, 255), -1)
            cv2.circle(annotated, ankle_p, 10, (0, 0, 255), -1)

            # 耳垂・体の中心点
            cv2.circle(annotated, ear_lobe_p, 10, (0, 255, 255), -1)
            cv2.circle(annotated, center_p, 10, (0, 0, 255), -1)

            if back_x is not None and front_x is not None:
                cv2.circle(annotated, (back_x, center_y), 6, (0, 255, 255), -1)
                cv2.circle(annotated, (front_x, center_y), 6, (0, 255, 255), -1)
                cv2.line(annotated, (back_x, center_y), (front_x, center_y), (255, 255, 0), 2)

            cv2.line(annotated, shoulder_p, center_p, (255, 255, 0), 3)
            cv2.line(annotated, center_p, hip_p, (255, 255, 0), 3)

            cv2.line(annotated, shoulder_p, knee_p, (0, 180, 255), 2)
            cv2.line(annotated, ear_lobe_p, hip_p, (0, 255, 180), 2)

            st.image(
                annotated,
                caption="骨格検出結果",
                use_container_width=True
            )

            st.subheader("反り腰角の解析 V20")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("胸椎角", f"{thoracic_angle}°")

            with col2:
                st.metric("腰椎角", f"{lumbar_angle}°")

            with col3:
                st.metric(
                    "反り腰角",
                    f"{round(lordosis_angle, 1)}°",
                    f"{percent(lordosis_angle, 35)}%"
                )
                st.write(judge_lordosis(lordosis_angle))

            st.subheader("総合判定")
            st.success(posture_type)

            st.subheader("判定条件")
            st.write("大転子：肩峰〜膝中心ラインに対して", hip_pos, hip_diff, "px")
            st.write("肩峰：耳垂〜大転子ラインに対して", shoulder_pos, shoulder_diff, "px")
            st.write("膝：大転子〜外果ラインに対して", knee_pos, knee_diff, "px")

            st.subheader("採用ルール")
            st.write("肩峰・大転子：後方側のMediaPipe点")
            st.write("膝中心・外果：下にあるMediaPipe点")
            st.write("耳垂：補正点")
            st.write("体の中心点：人物切り抜きから算出")

            st.subheader("取得座標")
            st.write("左肩:", left_shoulder_p)
            st.write("右肩:", right_shoulder_p)
            st.write("採用肩峰:", shoulder_p)

            st.write("左大転子:", left_hip_p)
            st.write("右大転子:", right_hip_p)
            st.write("採用大転子:", hip_p)

            st.write("左膝:", left_knee_p)
            st.write("右膝:", right_knee_p)
            st.write("採用膝:", knee_p)

            st.write("左外果:", left_ankle_p)
            st.write("右外果:", right_ankle_p)
            st.write("採用外果:", ankle_p)

            st.write("耳垂:", ear_lobe_p)
            st.write("体の中心点:", center_p)

        else:
            st.error("骨格を検出できませんでした")
