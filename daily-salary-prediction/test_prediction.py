# run_prediction.py

# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Input
import os
import glob
from datetime import datetime, timedelta
import sys
import json
import warnings

# TensorFlow 및 기타 경고 메시지를 숨겨서 출력을 깔끔하게 만듭니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def run_analysis(company_name_to_find):
    try:
        # CSV 파일 동적 검색 및 불러오기
        current_file_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_file_path)
        folder_path = os.path.join(current_script_dir, '..', '..', 'CSV')
        
        # 테스트를 위해 고정된 날짜를 사용
        today_str = '20250610'
        found_file = None
        
        # CSV 폴더 존재 여부를 확인
        if not os.path.isdir(folder_path):
            raise FileNotFoundError("CSV 폴더를 찾을 수 없습니다.")

        for filename in os.listdir(folder_path):
            if today_str in filename and filename.endswith(f'{company_name_to_find}.csv'):
                found_file = os.path.join(folder_path, filename)
                break
        
        if found_file is None:
            raise FileNotFoundError(f"CSV 폴더에서 날짜({today_str})와 '{company_name_to_find}' 이름이 포함된 CSV 파일을 찾을 수 없습니다.")
        
        df = pd.read_csv(found_file)

        # 데이터 전처리 및 기술적 지표 계산
        df['일자'] = pd.to_datetime(df['일자'], format='%Y%m%d')
        df = df.sort_values('일자').reset_index(drop=True)
        df['sma20'] = ta.sma(df['현재가'], length=20)
        df['rsi14'] = ta.rsi(df['현재가'], length=14)
        df.dropna(subset=['sma20', 'rsi14'], inplace=True)
        df = df.reset_index(drop=True)

        features = ['시가', '고가', '저가', '현재가', '거래량', 'sma20', 'rsi14']
        data = df[features].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # 모델 파일 관리 및 학습/불러오기 로직
        company_name = df['종목명'].iloc[0]
        model_directory = 'model'
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)

        # model_filename = os.path.join(model_directory, f'model_{company_name}_{today_str}.keras')
        # 테스트용 코드 - 구분을 위해서 오늘 날짜를 사용
        model_filename = os.path.join(model_directory, f'model_{company_name}_{datetime.now().strftime("%Y%m%d")}.keras')
        should_train_new_model = True

        search_pattern = os.path.join(model_directory, f'model_{company_name}_*.keras')
        existing_models = glob.glob(search_pattern)

        if existing_models:
            latest_model_file = max(existing_models, key=os.path.getctime)
            model_date_str = latest_model_file.split('_')[-1].replace('.keras', '')
            model_date = datetime.strptime(model_date_str, '%Y%m%d')
            
            if datetime.now() - model_date < timedelta(days=7):
                model = load_model(latest_model_file, compile=False)
                should_train_new_model = False

        if should_train_new_model:
            train_data_len = int(np.ceil(len(scaled_data) * 0.8))
            train_data = scaled_data[0:train_data_len, :]
            x_train, y_train = [], []
            look_back = 60
            for i in range(look_back, len(train_data)):
                x_train.append(train_data[i-look_back:i, :])
                y_train.append(train_data[i, 3])
            x_train, y_train = np.array(x_train), np.array(y_train)

            model = Sequential([
                Input(shape=(x_train.shape[1], x_train.shape[2])),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            model.save(model_filename)
            
        # 다음 날 주가 예측
        look_back = 60
        last_look_back_days_scaled = scaled_data[-look_back:]
        x_predict = np.reshape(last_look_back_days_scaled, (1, look_back, len(features)))
        predicted_price_scaled = model.predict(x_predict, verbose=0)

        dummy_predict = np.zeros((1, len(features)))
        dummy_predict[:, 3] = predicted_price_scaled
        predicted_price = scaler.inverse_transform(dummy_predict)[:, 3][0]

        # 매매 신호 판단
        latest_data = df.iloc[-1]
        current_price = latest_data['현재가']
        current_sma20 = latest_data['sma20']
        current_rsi14 = latest_data['rsi14']

        is_buy = (predicted_price > current_price * 1.02 and current_price > current_sma20 and current_rsi14 < 70)
        is_sell = (predicted_price < current_price * 0.98)
        signal = 1 if is_buy else 0 if is_sell else None
        
        # 성공 결과 반환
        return {
            "status": "success",
            "company_name": company_name,
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "sma20": float(current_sma20),
            "rsi14": round(float(current_rsi14), 2),
            "signal": signal
        }

    except Exception as e:
        # 오류 발생 시 에러 메시지 반환
        return {
            "status": "error",
            "company_name": company_name_to_find,
            "message": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        company_to_process = sys.argv[1]
        result = run_analysis(company_to_process)
        print(json.dumps(result, ensure_ascii=False))
    else:
        error_result = {
            "status": "error",
            "company_name": "N/A",
            "message": "분석할 회사 이름이 전달되지 않았습니다."
        }
        print(json.dumps(error_result, ensure_ascii=False))