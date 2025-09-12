# main_message.py

import subprocess
import json
import sys
import os

# --- 분석을 시작할 회사 이름을 여기에 지정합니다 ---
company_to_analyze = "SK하이닉스"

print(f"'{company_to_analyze}'에 대한 주가 예측 분석을 시작합니다...")
print("-" * 40) # 구분선 추가

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    script_to_run_path = os.path.join(current_script_dir, "run_prediction.py")

    process = subprocess.Popen(
        [sys.executable, script_to_run_path, company_to_analyze],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='cp949',
        bufsize=1 # 버퍼 없이 바로바로 출력하도록 설정
    )

    final_result_json = None
    
    for line in iter(process.stdout.readline, ''):
        line = line.strip() # 양 끝의 공백 제거
        if not line:
            continue

        # 최종 결과인지, 상태 메시지인지 확인
        if line.startswith("RESULT:"):
            # 'RESULT:' 접두어를 제거하고 JSON 문자열만 추출
            final_result_json = line.replace("RESULT:", "", 1)
        elif line.startswith("STATUS:"):
            # 'STATUS:' 접두어를 제거하고 상태 메시지만 출력
            status_message = line.replace("STATUS:", "", 1)
            print(f"[진행 중] {status_message}")

    # 프로세스가 끝날 때까지 기다리고, 에러가 있었는지 확인
    process.wait()
    if process.returncode != 0:
        # 에러가 있었다면 에러 메시지를 stderr에서 읽음
        error_output = process.stderr.read()
        raise subprocess.CalledProcessError(process.returncode, process.args, stderr=error_output)

    # 최종 결과 JSON을 파이썬 딕셔너리로 변환
    result = json.loads(final_result_json)
    
    # --- 결과 처리 ---
    print("-" * 40)
    print("\n--- 분석 결과 ---")
    if result.get("status") == "success":
        print(f"✅ 분석 성공: {result['company_name']}")
        print(f"  - 현재가: {result['current_price']:,.0f}원")
        print(f"  - 예측가: {result['predicted_price']:,.0f}원")
        print(f"  - 20일 이평선: {result['sma20']:,.0f}원")
        print(f"  - RSI14: {result['rsi14']:.2f}")
        
        signal = result['signal']
        if signal == 1:
            print("  - 최종 신호: 매수 (1)")
        elif signal == 0:
            print("  - 최종 신호: 매도 (0)")
        else:
            print("  - 최종 신호: 관망 (None)")
    else:
        print(f" 분석 실패: {result.get('company_name', 'N/A')}")
        print(f"  - 오류 메시지: {result.get('message', '알 수 없는 오류')}")

except subprocess.CalledProcessError as e:
    print("\n--- 스크립트 실행 오류 ---")
    print(f" '{company_to_analyze}' 분석 스크립트 실행에 실패했습니다.")
    print(f"  - 에러 출력: {e.stderr}")
except Exception as e:
    print("\n--- 처리 중 알 수 없는 오류 발생 ---")
    print(f" 오류 내용: {e}")