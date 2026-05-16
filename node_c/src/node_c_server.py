import grpc
from concurrent import futures
import sys
import os
import time
import math

sys.path.append(os.path.dirname(__file__))
import aura_pb2
import aura_pb2_grpc
from bridge_sender_integrated import process_and_build_request, send_with_safety


class AuraPerceptionServicer(aura_pb2_grpc.AuraPerceptionServicer):
    def __init__(self):
        # ================================================
        # 감정 변화 임계치 필터링 (Emotion Change Throttling)
        # ================================================
        # 표정은 카메라 프레임 단위로 들어오므로, 의미 있는 변화가 없으면
        # Node B 호출을 건너뛰어 불필요한 LLM 추론을 방지한다.
        self.last_valence = 0.0
        self.last_arousal = 0.0
        self.last_sent_time = 0.0
        self.threshold = 0.12    # 유클리드 거리 기준 최소 변화량
        self.min_interval = 1.0  # 최소 전송 간격 (초)

    def is_significant_change(self, v: float, a: float) -> bool:
        """
        이전 상태 대비 감정 변화가 충분히 크고, 최소 전송 간격이 지났는지 확인.
        Returns: True이면 Node B 호출 진행, False이면 스킵
        """
        now = time.time()
        # 최소 간격 미충족
        if now - self.last_sent_time < self.min_interval:
            return False
        # 유클리드 거리로 V-A 변화량 계산
        diff = math.sqrt((v - self.last_valence) ** 2 + (a - self.last_arousal) ** 2)
        return diff >= self.threshold

    def SendFacePerception(self, request, context):
        print(f"\n[Node C Server] 수신된 표정 데이터: {request.emotion_label} "
              f"(V:{request.valence:.2f}, A:{request.arousal:.2f})")

        # 감정 변화가 미미하면 Node B 호출 없이 즉시 반환
        if not self.is_significant_change(request.valence, request.arousal):
            print(f"  -> 변화량 미달 (threshold={self.threshold}), Node B 호출 스킵")
            return aura_pb2.EmpathyResponse(
                session_id="session_live",
                text="상태 유지 (변화량 적음)",
                strategy="skip"
            )

        # 상태 업데이트
        self.last_valence = request.valence
        self.last_arousal = request.arousal
        self.last_sent_time = time.time()

        user_text = request.text if request.text else ""
        if user_text:
            print(f"  -> STT 텍스트 수신: '{user_text}'")
        else:
            print(f"  -> STT 텍스트 없음 (표정 데이터만 처리)")

        fused_emotion = aura_pb2.FusedEmotionState(
            primary_emotion=request.emotion_label,
            confidence=request.confidence,
            valence=request.valence,
            arousal=request.arousal,
            description=f"표정 기반 분석: {request.emotion_label}"
        )

        print("  -> Node C 분석 시작 및 Node B로 프롬프트 전송...")
        prompt_request = process_and_build_request(
            session_id="session_live",
            user_text=user_text,
            candidates=[request],
            fused_emotion=fused_emotion,
            face_va=[request.valence, request.arousal],  # 표정 V-A
            voice_va=None                                 # 음성은 별도 호출로 수신
        )

        response = send_with_safety(prompt_request)
        print(f"  -> Node B 응답 수신 완료!")
        return response

    def SendVoicePerception(self, request, context):
        """
        음성 인식 결과 수신.
        음성 톤(V, A)과 STT 텍스트를 함께 처리하며,
        표정 데이터와는 별도로 voice_va로 전달하여 3-모달 융합에 참여한다.
        """
        print(f"\n[Node C Server] 수신된 음성 데이터: '{request.text}' "
              f"(V:{request.valence:.2f}, A:{request.arousal:.2f})")

        user_text = request.text if request.text else ""

        fused_emotion = aura_pb2.FusedEmotionState(
            primary_emotion="voice_emotion",
            confidence=request.confidence,
            valence=request.valence,
            arousal=request.arousal,
            description=f"음성 톤 분석: {request.emotion_label}"
        )

        print(f"  -> Node C 분석 시작 (텍스트: '{user_text}') 및 Node B로 전송...")
        prompt_request = process_and_build_request(
            session_id="session_voice",
            user_text=user_text,
            candidates=[request],
            fused_emotion=fused_emotion,
            face_va=None,                                  # 표정은 별도 호출로 수신
            voice_va=[request.valence, request.arousal]   # 음성 톤 V-A
        )

        response = send_with_safety(prompt_request)
        print(f"  -> Node B 응답 수신 완료!")
        return response


def serve():
    port = 5052
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aura_pb2_grpc.add_AuraPerceptionServicer_to_server(AuraPerceptionServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print("======================================================")
    print(f"[Node C] AuraPerception Server가 {port}번 포트에서 열렸습니다!")
    print("   Node A 코드에서 target_address를 'localhost:5052'로 변경해주세요.")
    print("   EmotionCandidate.text 필드에 STT 결과를 담아서 보내주세요.")
    print("======================================================")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()