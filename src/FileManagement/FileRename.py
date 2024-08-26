import pandas as pd
import os

# CSV 파일 경로
csv_file = '/home/aurorasphere/Programming/CapstoneDesign2024/data/KETI-2017-SL-Annotation-v2_1.csv'
# 파일이 위치한 디렉토리 경로
file_dir = '/home/aurorasphere/Programming/CapstoneDesign2024/data/video/word/front'
# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 각 행을 순회하며 파일명 변경
for index, row in df.iterrows():
    base_name = row['파일명'].strip().rsplit('.', 1)[0]
    # 지원하는 확장자 목록
    extensions = ['.avi', '.MOV', '.MTS']
    
    # 기존 파일명을 찾기 위해 확장자 순회
    for ext in extensions:
        old_file_name = base_name + ext
        old_file = os.path.join(file_dir, old_file_name)
        
        # 파일이 존재할 경우 새로운 이름으로 변경
        if os.path.exists(old_file):
            new_file_name = f"{base_name}_{row['한국어'].strip()}{ext}"
            new_file = os.path.join(file_dir, new_file_name)
            
            # 디버그 출력
            print(f"Old file path: {old_file}")
            print(f"New file path: {new_file}")
            
            try:
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} -> {new_file}")
            except Exception as e:
                print(f"Error renaming {old_file} to {new_file}: {e}")
            
            # 파일을 찾았으므로 다른 확장자는 검사하지 않음
            break
    else:
        print(f"File not found for base name: {base_name}")

print("File renaming completed.")

