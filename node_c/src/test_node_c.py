import sys
import os
import numpy as np
from node_c_prototype import NodeC

def run_tests():
    print("========================================")
    print("🚀 Node C 자동화 테스트 스크립트 실행")
    print("========================================")
    
    try:
        # NodeC 객체 생성 시 enable_konlpy, enable(Sentiment), enable_db 등이 
        # 모두 True로 작동하며, 실패 시 에러를 뿜거나 안전하게 넘어갑니다.
        node_c = NodeC()
    except Exception as e:
        print(f"초기화 중 에러 발생: {e}")
        return

    # --- 테스트 1: Sarcasm (모순/감정 숨김) ---
    print("\n[Test 1] 거짓 웃음 (Sarcasm/Suppression)")
    text1 = "오늘 진행한 프로젝트 발표 완전히 망쳤어."
    face_v1, face_a1 = 0.8, -0.2  # 표정: 긍정적이고 차분함 (억지 웃음)
    face_vector1 = np.array([face_v1, face_a1], dtype=np.float32)
    
    result1 = node_c.process_data(text1, face_va=face_vector1)
    
    print(f" - 입력 텍스트: {text1}")
    print(f" - 추출된 키워드: {node_c.text_processor.extract_keywords(text1)}")
    print(f" - 입력 표정 (V, A): {face_v1}, {face_a1}")
    print(" - 반환 결과:")
    for k, v in result1.items():
        print(f"   * {k}: {v}")

    # --- 테스트 2: 정직한 감정 (Alignment 일치) ---
    print("\n[Test 2] 솔직한 슬픔 (Consistent)")
    text2 = "시험 성적이 떨어져서 너무 우울해."
    face_v2, face_a2 = -0.8, 0.3  # 표정: 부정적이고 스트레스 받음 (우울함)
    face_vector2 = np.array([face_v2, face_a2], dtype=np.float32)
    
    result2 = node_c.process_data(text2, face_va=face_vector2)

    # --- 테스트 3: 표정 + 음성 모두 있는 경우 (3-모달 융합) ---
    print("\n[Test 3] 3-모달 융합 (표정 + 음성 톤 + 텍스트 동시 입력)")
    text3 = "요즘 너무 피곤해서 힘들어."
    face_vector3 = np.array([-0.5, 0.1], dtype=np.float32)  # 표정: 약간 부정적, 차분
    voice_vector3 = np.array([-0.7, 0.6], dtype=np.float32)  # 음성: 부정적, 높은 각성
    # 먼저 face_va로 상태를 세팅하고, voice_va도 함께 전달
    result3 = node_c.process_data(text3, face_va=face_vector3, voice_va=voice_vector3)

    print(f" - 입력 텍스트: {text3}")
    print(f" - 표정 (V, A): {face_vector3.tolist()}")
    print(f" - 음성 톤 (V, A): {voice_vector3.tolist()}")
    print(" - 반환 결과:")
    for k, v in result3.items():
        print(f"   * {k}: {v}")
    
    print(f" - 입력 텍스트: {text2}")
    print(f" - 추출된 키워드: {node_c.text_processor.extract_keywords(text2)}")
    print(f" - 입력 표정 (V, A): {face_v2}, {face_a2}")
    print(" - 반환 결과:")
    for k, v in result2.items():
        print(f"   * {k}: {v}")

    print("\n✅ 모든 테스트가 완료되었습니다.")

if __name__ == "__main__":
    run_tests()
