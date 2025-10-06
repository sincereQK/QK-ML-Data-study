import matplotlib
matplotlib.use('Qt5Agg') # 리사이즈 가능한 창 사용 설정

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import matplotlib.pyplot as plt

# 1. 데이터 준비
csv_filepath = r'data\Samsung Electronics.csv' # 분석할 CSV 파일 경로
company_name = 'Samsung Electronics'              # 그래프 제목 등에 표시될 회사 이름

window_size = 60      # 과거 60일 데이터를 보고
prediction_horizon = 5  # 향후 5일을 예측

try:
    stock_data = pd.read_csv(csv_filepath, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print(f"\n[Error] File not found at path: {csv_filepath}")
    exit()

# 'Close' 컬럼만 선택하여 2D 배열 형태로 변환
close_prices = stock_data['Close'].values.reshape(-1, 1)

print(f"Data for '{company_name}' loaded successfully. (Length: {len(close_prices)} days)")

# 2. LSTM을 위한 데이터 전처리
# 데이터 정규화 (0과 1 사이의 값으로 변환)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 학습 데이터셋 생성 (Sliding Window)
X_train, y_train = [], []
for i in range(window_size, len(scaled_data) - prediction_horizon + 1):
    X_train.append(scaled_data[i - window_size:i, 0])
    y_train.append(scaled_data[i:i + prediction_horizon, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# LSTM 입력 형태로 변환 (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(f"\nTraining data created: X_train shape {X_train.shape}, y_train shape {y_train.shape}")

# 3. LSTM 모델 구축
print("Building LSTM model...")
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=prediction_horizon))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# 4. 모델 학습
print("\nTraining model...")
model.fit(X_train, y_train, epochs=25, batch_size=32)
print("Model training complete.")


# 5. 향후 5일 예측
print("\nPredicting next 5 days...")

last_window = scaled_data[-window_size:]
last_window_reshaped = last_window.reshape((1, window_size, 1))

predicted_prices_scaled = model.predict(last_window_reshaped)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)


# 6. 예측 결과 출력
print(f"\n--- Prediction for Next {prediction_horizon} Days ---")
for i, price in enumerate(predicted_prices[0]):
    print(f"Day {i+1}: {price:,.2f}")


# 7. 예측 결과 시각화
print("\nDisplaying prediction graph...")

# 예측 기간에 대한 날짜 인덱스 생성 (주말 제외 영업일 기준)
last_date = stock_data.index[-1]
future_dates = pd.bdate_range(start=last_date, periods=prediction_horizon + 1, freq='B')[1:]

# 그래프 설정
plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# 과거 데이터 표시 (최근 100일)
plt.plot(stock_data.index[-100:], close_prices[-100:], color='blue', label='Historical Prices')

# 예측 데이터 표시
plt.plot(future_dates, predicted_prices[0], color='red', marker='o', linestyle='--', label='Predicted Prices')

# 그래프 제목 및 라벨 설정
plt.title(f'{company_name} - Price Prediction', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# 그래프 표시
plt.show()