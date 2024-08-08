import pandas as pd
import os

# CSV 파일 경로
csv_file = '/home/aurorasphere/Programming/TEST_CAPSTONE/data/KETI-2017-SL-Annotation-v2_1.csv'
# 파일이 위치한 디렉토리 경로
file_dir = '/home/aurorasphere/Programming/TEST_CAPSTONE/data/video'
# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 각 행을 순회하며 파일명 변경
for index, row in df.iterrows():
    # 기존 파일명 (확장자 무시하고 .avi로 변경)
    old_file_name = row['파일명'].strip().rsplit('.', 1)[0] + '.avi'
    old_file = os.path.join(file_dir, old_file_name)
    
    # 새로운 파일명 생성 (원래 파일명 뒤에 한국어 뜻을 붙임)
    base_name = old_file_name.rsplit('.', 1)[0]
    new_file_name = f"{base_name}_{row['한국어'].strip()}.avi"
    new_file = os.path.join(file_dir, new_file_name)

    # 디버그 출력
    print(f"Old file path: {old_file}")
    print(f"New file path: {new_file}")

    # 파일명 변경
    if os.path.exists(old_file):
        try:
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")
        except Exception as e:
            print(f"Error renaming {old_file} to {new_file}: {e}")
    else:
        print(f"File not found: {old_file}")

print("File renaming completed.")

