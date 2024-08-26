import os

# 파일이 위치한 디렉토리 경로
file_dir = '/home/aurorasphere/Programming/CapstoneDesign2024/data/video'

# 파일 디렉토리의 모든 파일을 순회
for file_name in os.listdir(file_dir):
    # 파일 확장자가 .avi, .MOV, .MTS 중 하나인지 확인
    if file_name.endswith(('.avi', '.MOV', '.MTS')):
        # 한국어 뜻을 제거한 원래 파일명 추출
        base_name = file_name.rsplit('_', 1)[0]
        ext = file_name.rsplit('.', 1)[1]
        original_file_name = f"{base_name}.{ext}"
        old_file = os.path.join(file_dir, file_name)
        new_file = os.path.join(file_dir, original_file_name)
        
        # 디버그 출력
        print(f"Old file path: {old_file}")
        print(f"New file path: {new_file}")
        
        # 파일 이름 변경
        try:
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")
        except Exception as e:
            print(f"Error renaming {old_file} to {new_file}: {e}")

print("File renaming completed.")

