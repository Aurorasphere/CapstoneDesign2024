import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# 데이터 경로 설정
data_path = '/home/aurorasphere/Programming/TEST_CAPSTONE/data_tmp'

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드
model = load_model('action_recognition_model.h5')

# 데이터 디렉토리에서 파일 이름 읽기 및 actions 리스트 생성
actions = []
file_names = [f for f in os.listdir(data_path) if f.endswith('.avi')]
for file_name in file_names:
    action = file_name.split('_')[-1].replace('.avi', '')
    if action not in actions:
        actions.append(action)

# LabelEncoder 초기화
label_encoder = LabelEncoder()
label_encoder.fit(actions)

# 랜드마크 추출 함수
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    keypoints = np.concatenate([pose, face, lh, rh])
    return keypoints

# 한글 폰트 설정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 적절한 한글 폰트 경로 설정
font = ImageFont.truetype(font_path, 32)

# 실시간 웹캠 비디오 스트림
cap = cv2.VideoCapture(0)

sequence = []
sentence = []
predictions = []
threshold = 0.5

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지 -> RGB 이미지
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 검출
        results = holistic.process(image)

        # BGR 이미지로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 키포인트 추출
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # 예측
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # 예측 결과가 일정 임계값을 초과하면 텍스트로 변환
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    action_index = np.argmax(res)
                    if action_index < len(label_encoder.classes_):
                        action = label_encoder.inverse_transform([action_index])[0]
                        if len(sentence) > 0:
                            if action != sentence[-1]:
                                sentence.append(action)
                        else:
                            sentence.append(action)

            # 문장 길이 제한
            if len(sentence) > 5:
                sentence = sentence[-5:]

        # 이미지에 텍스트 추가
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        draw.rectangle([(0, 0), (640, 40)], fill=(245, 117, 16))
        draw.text((10, 0), ' '.join(sentence), font=font, fill=(255, 255, 255))
        image = np.array(image_pil)

        # 화면에 이미지 출력
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

