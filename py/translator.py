import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# actions 리스트를 csv 파일에서 불러오는 함수
def load_actions_from_csv(csv_path, encoding='utf-8'):
    df = pd.read_csv(csv_path, encoding=encoding)
    return df.iloc[:, 6].unique().tolist()  # 6번째 열이 action 이름이라고 가정

# csv 파일 경로
csv_path = '~/Programming/TEST_CAPSTONE/data/KETI-2017-SL-Annotation-v2_1.csv'

# actions 리스트를 csv 파일에서 불러오기
actions = load_actions_from_csv(csv_path, encoding='utf-8')  # utf-8 인코딩 사용

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드
model = load_model('action_recognition_model.h5')

# LabelEncoder 로드 (actions 리스트에 맞게 변경)
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

        # 화면에 텍스트 출력
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 화면에 이미지 출력
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

