import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

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

# 첫 번째 LSTM 레이어 (return_sequences=True로 설정하여 다음 레이어에 시퀀스 전달)
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2)) # 과적합 방지를 위한 Dropout

# 두 번째 LSTM 레이어
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# 출력 레이어 (예측할 기간만큼의 노드 수 설정)
model.add(Dense(units=prediction_horizon))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# 4. 모델 학습
print("\nTraining model...")
# epochs: 전체 데이터를 몇 번 반복 학습할지, batch_size: 한 번에 몇 개의 샘플을 학습할지
model.fit(X_train, y_train, epochs=25, batch_size=32)
print("Model training complete.")


# 5. 향후 5일 예측
print("\nPredicting next 5 days...")

# 예측을 위한 마지막 60일 데이터 추출
last_window = scaled_data[-window_size:]
# 모델 입력에 맞게 3차원 형태로 변환 (1, 60, 1)
last_window_reshaped = last_window.reshape((1, window_size, 1))

# 주가 예측
predicted_prices_scaled = model.predict(last_window_reshaped)

# 예측된 값을 원래 주가 단위로 변환
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)


# 6. 예측 결과 출력
print(f"\n--- Prediction for Next {prediction_horizon} Days ---")
for i, price in enumerate(predicted_prices[0]):
    print(f"Day {i+1}: {price:,.2f}")