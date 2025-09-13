import yfinance as yf
import numpy as np
import stumpy
from dtaidistance import dtw
import matplotlib.pyplot as plt
import pandas as pd

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
# 실행 환경에 맞는 한글 폰트를 지정해주세요.
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지


# --- 1. 데이터 준비 ---
# 분석할 종목과 기간 설정
ticker = "005930.KS"
start_date = "2023-01-01"
end_date = "2025-09-13" # 현재 날짜
window_size = 30
future_days = 5

# yfinance로 데이터 다운로드
stock_data = yf.download(ticker, start=start_date, end=end_date)

# 'Close' 컬럼에서 결측치가 있는 행을 제거 (1차원 Series로 선택)
cleaned_close_data = stock_data['Close'].dropna()

# 최종 데이터(numpy 배열)로 변환
prices = cleaned_close_data.values

# 데이터에 아주 미세한 랜덤 노이즈를 추가하여 상수 구간(standard deviation = 0) 문제를 해결합니다.
noise = np.random.randn(len(prices)) * 1e-7
prices = prices + noise

# 에러 방지를 위한 확인 단계
if len(prices) <= window_size:
    print("\n[문제 발견] 최종 데이터의 길이가 window_size보다 작거나 같습니다.")
    exit()
else:
    print(f"최종 분석용 데이터 '{ticker}' 준비 완료. (길이: {len(prices)}일)")


# --- 2. 매트릭스 프로파일 계산 ---
print("\n매트릭스 프로파일 계산 중... (시간이 소요될 수 있습니다)")
matrix_profile = stumpy.stump(prices.flatten(), m=window_size)


# --- 3. 매트릭스 프로파일 결과 분석 ---
motif_index = np.argmin(matrix_profile[:, 0])
neighbor_index = matrix_profile[motif_index, 1]
print(f"가장 강력한 패턴(Motif) 시작 인덱스: {motif_index}")
print(f"해당 패턴과 가장 유사한 다른 패턴의 시작 인덱스: {neighbor_index}")


# --- 4. DTW 계산 ---
historical_motif = prices[motif_index : motif_index + window_size]
recent_data = prices[-window_size:]

# ❗️❗️❗️ 에러 해결 부분 ❗️❗️❗️
# DTW 함수에 입력하기 전, 데이터를 깨끗한 1D numpy 배열로 명시적 변환
# 이것이 "ambiguous" 에러를 해결하는 핵심입니다.
s1 = np.array(historical_motif, dtype=np.double)
s2 = np.array(recent_data, dtype=np.double)
distance = dtw.distance(s1, s2)

print(f"\n과거 대표 패턴과 최근 데이터 간의 DTW 거리: {distance:.2f}")

# --- 5. 예측 및 시각화 ---
historical_outcome = prices[motif_index + window_size : motif_index + window_size + future_days]

prices_series = cleaned_close_data
fig, axs = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
plt.style.use('seaborn-v0_8-whitegrid')

axs[0].set_title(f"'{ticker}' 전체 주가 및 탐색된 주요 패턴", fontsize=16)
axs[0].plot(prices_series.index, prices, color='gray', alpha=0.7, label='전체 종가 (노이즈 포함)')
axs[0].plot(prices_series.index[motif_index : motif_index + window_size], historical_motif, color='red', linewidth=2.5, label=f'패턴 1 (인덱스: {motif_index})')
axs[0].plot(prices_series.index[neighbor_index : neighbor_index + window_size], prices[neighbor_index : neighbor_index + window_size], color='blue', linewidth=2.5, label=f'패턴 2 (인덱스: {neighbor_index})')
axs[0].legend()

axs[1].set_title(f"과거 패턴 vs 최근 데이터 (DTW 거리: {distance:.2f})", fontsize=16)
axs[1].plot(historical_motif, label='과거 대표 패턴', color='red')
axs[1].plot(recent_data, label='최근 데이터', color='green')
axs[1].legend()

axs[2].set_title(f"과거 패턴 직후 {future_days}일 주가 흐름 (예측)", fontsize=16)
axs[2].plot(historical_outcome, label=f'과거 패턴 이후 {future_days}일', color='purple', marker='o')
axs[2].legend()

plt.tight_layout()
plt.show()