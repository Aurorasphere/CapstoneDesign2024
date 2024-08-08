import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
import warnings
from multiprocessing import Pool
import mediapipe as mp

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')

# 데이터 경로 설 정
data_path = '/home/aurorasphere/Programming/TEST_CAPSTONE/data/video'
actions = []

# 랜드마크 추출 함수
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    keypoints = np.concatenate([pose, face, lh, rh])
    return keypoints

print("데이터 디렉토리에서 파일 이름 읽기")
# 데이터 디렉토리에서 파일 이름 읽기
file_names = [f for f in os.listdir(data_path) if f.endswith('.avi')]
for file_name in file_names:
    action = file_name.split('_')[-1].replace('.avi', '')
    if action not in actions:
        actions.append(action)

# 라벨 인코더 초기화
label_encoder = LabelEncoder()
label_encoder.fit(actions)

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic

def process_file(file_name):
    print(f"Processing {file_name}")
    action = file_name.split('_')[-1].replace('.avi', '')
    action_label = label_encoder.transform([action])[0]
    
    cap = cv2.VideoCapture(os.path.join(data_path, file_name))
    sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) == 30:
                break
    cap.release()
    
    return sequence, action_label

print("데이터 준비 시작")
# 멀티프로세싱을 사용하여 데이터 준비
with Pool(os.cpu_count()) as pool:
    results = pool.map(process_file, file_names)

sequences, labels = zip(*results)

print("데이터 준비 완료, numpy 배열로 변환 중")
X_data = np.array(sequences)
y_data = to_categorical(labels)

# 데이터 정규화
X_data = X_data / np.max(X_data)

# 데이터 저장
np.save('X_data.npy', X_data)
np.save('y_data.npy', y_data)
print("데이터 저장 완료")

# 모델 정의
input_shape = X_data.shape[2]  # keypoints shape
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(30, input_shape)))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(actions), activation='softmax'))

# 학습률 조정
optimizer = Adam(learning_rate=0.0001)

# 모델 컴파일
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("모델 학습 시작")
# 모델 학습
history = model.fit(X_data, y_data, epochs=500, batch_size=32, validation_split=0.2)
print("모델 학습 완료")

# 학습 결과를 CSV 파일로 저장
import pandas as pd

history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# 모델 저장
model.save('action_recognition_model.h5')
print("모델 저장 완료")

