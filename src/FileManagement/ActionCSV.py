import pandas as pd

# CSV 파일 경로
csv_file = '/home/aurorasphere/Programming/CapstoneDesign2024/data/KETI-2017-SL-Annotation-v2_1.csv'

# CSV 파일에서 데이터 읽기
df = pd.read_csv(csv_file, encoding='utf-8')

# '한국어' 열에서 유니크한 값들을 가져와서 actions 리스트로 정의
actions = df['한국어'].unique().tolist()

