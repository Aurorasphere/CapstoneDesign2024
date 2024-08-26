import os
import pandas as pd
import shutil

# CSV 파일 경로
csv_file = '/home/aurorasphere/Programming/CapstoneDesign2024/data/KETI-2017-SL-Annotation-v2_1.csv'
# 파일이 위치한 디렉토리 경로
file_dir = '/home/aurorasphere/Programming/CapstoneDesign2024/data/video'
# 정면과 측면 영상이 이동할 디렉토리 경로
front_dir = '/home/aurorasphere/Programming/CapstoneDesign2024/data/video/front'
side_dir = '/home/aurorasphere/Programming/CapstoneDesign2024/data/video/side'

# 필요한 디렉토리 생성
os.makedirs(front_dir, exist_ok=True)
os.makedirs(side_dir, exist_ok=True)

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 각 행을 순회하며 파일 이동
for index, row in df.iterrows():
    base_name = row['파일명'].strip().rsplit('.', 1)[0]  # 파일명에서 확장자를 제거한 부분
    direction = row['방향'].strip()  # '방향' 정보 (정면, 측면 등)
    
    # 실제 파일의 확장자를 확인하여 파일 이동
    for ext in ['.avi', '.MOV', '.MTS']:
        file_path = os.path.join(file_dir, base_name + ext)
        if os.path.exists(file_path):
            # 파일을 이동할 대상 디렉토리 결정
            if direction == '정면':
                target_dir = front_dir
            elif direction == '측면':
                target_dir = side_dir
            else:
                continue  # 방향이 정면이나 측면이 아니면 스킵
            
            target_path = os.path.join(target_dir, os.path.basename(file_path))
            
            # 파일 이동
            try:
                shutil.move(file_path, target_path)
                print(f"Moved {file_path} to {target_path}")
            except Exception as e:
                print(f"Error moving {file_path} to {target_path}: {e}")
            break  # 한 파일을 찾으면 더 이상 다른 확장자를 확인할 필요 없음
        else:
            print(f"File not found: {file_path}")

print("File moving completed.")

