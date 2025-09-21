import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import stumpy
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import threading
import time
import itertools
import sys

# Matplotlib의 기본 설정을 초기화합니다.
plt.rcdefaults()

# 스피너 애니메이션을 위한 클래스
class Spinner:
    def __init__(self, message="Calculating..."):
        self.spinner_cycle = itertools.cycle(['-', '/', '|', '\\'])
        self.message = message
        self.stop_running = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()

    def spin(self):
        while not self.stop_running.is_set():
            sys.stdout.write(f'\r{self.message} {next(self.spinner_cycle)}')
            sys.stdout.flush()
            time.sleep(0.1)

    def stop(self):
        self.stop_running.set()
        self.thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()

# 1. 데이터 준비
csv_filepath = r'E:\2025KMG\chartML\stock_data.csv' # CSV 파일 경로
company_name = 'Samsung Electronics'                # 그래프 제목에 표시될 회사 이름

window_size = 30
future_days = 5

try:
    # CSV 파일을 읽어오면서 'Date' 컬럼을 인덱스로 지정하고 날짜 형식으로 변환합니다.
    stock_data = pd.read_csv(csv_filepath, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print(f"\n[Error] File not found at path: {csv_filepath}")
    exit()

cleaned_close_data = stock_data['Close'].dropna()
prices = cleaned_close_data.values
noise = np.random.randn(len(prices)) * 1e-7
prices = prices + noise

if len(prices) <= window_size:
    print(f"\n[Problem] Data length ({len(prices)}) is shorter than window_size ({window_size}).")
    exit()
else:
    print(f"Data for '{company_name}' loaded successfully. (Length: {len(prices)} days)")

# 2. 매트릭스 프로파일 계산
print("\nCalculating Matrix Profile...")
spinner = Spinner("Matrix Profile")
calculation_result = {}

def calculate_stump():
    mp = stumpy.stump(prices.flatten(), m=window_size, ignore_trivial=True)
    calculation_result['matrix_profile'] = mp

spinner.start()
calculation_thread = threading.Thread(target=calculate_stump)
calculation_thread.start()
calculation_thread.join()
spinner.stop()

matrix_profile = calculation_result['matrix_profile']
print("Matrix Profile calculation complete.")

# 3. 매트릭스 프로파일 결과 분석
motif_index = np.argmin(matrix_profile[:, 0])
neighbor_index = int(matrix_profile[motif_index, 1])
print(f"\nMotif found at index: {motif_index}")
print(f"Nearest neighbor found at index: {neighbor_index}")

# 4. DTW 계산
historical_motif = prices[motif_index : motif_index + window_size]
recent_data = prices[-window_size:]

s1 = np.array(historical_motif, dtype=np.double)
s2 = np.array(recent_data, dtype=np.double)
distance, path = fastdtw(s1, s2, dist=euclidean)

print(f"\nDTW distance between historical pattern and recent data: {distance:,.2f}")

# 5. 예측 결과 생성 및 터미널 출력
historical_outcome = prices[motif_index + window_size : motif_index + window_size + future_days]

print(f"\n--- Prediction for Next {future_days} Days (based on historical pattern) ---")
for i, price in enumerate(historical_outcome):
    print(f"Day {i+1}: {price:,.2f}")


# 6. 시각화
prices_series = cleaned_close_data
fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1, 1]}, constrained_layout=True)
plt.style.use('seaborn-v0_8-whitegrid')

# 그래프 1
axs[0].set_title(f"'{company_name}' Price and Detected Patterns", fontsize=16)
axs[0].plot(prices_series.index, prices, color='gray', alpha=0.7, label='Close Price (with noise)')
axs[0].plot(prices_series.index[motif_index : motif_index + window_size], historical_motif, color='red', linewidth=2.5, label=f'Pattern 1 (Index: {motif_index})')

if neighbor_index < len(prices) - window_size:
    axs[0].plot(prices_series.index[neighbor_index : neighbor_index + window_size], prices[neighbor_index : neighbor_index + window_size], color='blue', linewidth=2.5, label=f'Pattern 2 (Index: {neighbor_index})')

handles, labels = axs[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axs[0].legend(by_label.values(), by_label.keys())

# 그래프 2
axs[1].set_title(f"Historical Pattern vs. Recent Data (DTW Distance: {distance:,.2f})", fontsize=16)
axs[1].plot(historical_motif, label='Historical Pattern', color='red')
axs[1].plot(recent_data, label='Recent Data', color='green')

handles, labels = axs[1].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axs[1].legend(by_label.values(), by_label.keys())

# 그래프 3
axs[2].set_title(f"Prediction: Price Movement for Next {future_days} Days", fontsize=16)
axs[2].plot(historical_outcome, label=f'Movement after Pattern ({future_days} days)', color='purple', marker='o')

handles, labels = axs[2].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axs[2].legend(by_label.values(), by_label.keys())


plt.show()