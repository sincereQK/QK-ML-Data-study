# main.py

import subprocess
import json
import sys
import os # os 라이브러리 추가

company_to_analyze = "SK하이닉스" # 분석할 회사 이름을 여기에 입력

print(f"'{company_to_analyze}'에 대한 주가 예측 분석을 시작합니다...")

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    script_to_run_path = os.path.join(current_script_dir, "run_prediction.py")
    # 테스트용 코드
    # script_to_run_path = os.path.join(current_script_dir, "test_prediction.py")

    # subprocess.run을 사용해 run_prediction.py 스크립트를 실행합니다.
    completed_process = subprocess.run(
        [sys.executable, script_to_run_path, company_to_analyze],
        capture_output=True,
        text=True,
        check=True,
        encoding='cp949'
    )
    
    # run_prediction.py가 출력한 문자열 결과를 가져옵니다.
    output = completed_process.stdout
    
    # 가져온 JSON 형식의 문자열을 파이썬 딕셔너리 형태로 변환합니다.
    result = json.loads(output)
    
    # 결과 처리
    print("\n--- 분석 결과 ---")
    if result.get("status") == "success":
        print(f" 분석 성공: {result['company_name']}")
        print(f" - 현재가: {result['current_price']:,.0f}원")
        print(f" - 예측가: {result['predicted_price']:,.0f}원")
        print(f" - 20일 이평선: {result['sma20']:,.0f}원")
        print(f" - RSI14: {result['rsi14']:.2f}")
        
        signal = result['signal']
        if signal == 1:
            print(" - 최종 신호: 매수 (1)")
        elif signal == 0:
            print(" - 최종 신호: 매도 (0)")
        else:
            print(" - 최종 신호: 관망 (None)")

    else: # status가 'error'일 경우
        print(f" 분석 실패: {result.get('company_name', 'N/A')}")
        print(f" - 오류 메시지: {result.get('message', '알 수 없는 오류')}")

except subprocess.CalledProcessError as e:
    # run_prediction.py 스크립트 자체가 실행되지 않았거나, 실행 중 비정상 종료된 경우
    print("\n--- 스크립트 실행 오류 ---")
    print(f" '{company_to_analyze}' 분석 스크립트 실행에 실패했습니다.")
    print(f" - 에러 출력: {e.stderr}")
except json.JSONDecodeError:
    # run_prediction.py가 JSON이 아닌 다른 형태의 문자열을 출력한 경우
    print("\n--- 결과 처리 오류 ---")
    print(" 분석 스크립트로부터 받은 결과를 처리할 수 없습니다 (JSON 형식 오류).")
