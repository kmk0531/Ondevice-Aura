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
    def SendFacePerception(self, request, context):
        print(f"\n[Node C Server] 수신된 표정 데이터: {request.emotion_label} (V:{request.valence:.2f}, A:{request.arousal:.2f})")

        # EmotionCandidate.text 필드에서 STT 텍스트 수신
        # Node A가 text 필드에 STT 결과를 담아서 보내줘야 함
        user_text = request.text if request.text else ""

        if user_text:
            print(f"  -> STT 텍스트 수신: '{user_text}'")
        else:
            print(f"  -> STT 텍스트 없음 (표정 데이터만 처리)")

        # Fused Emotion 상태 생성
        fused_emotion = aura_pb2.FusedEmotionState(
            primary_emotion = request.emotion_label,
            confidence      = request.confidence,
            valence         = request.valence,
            arousal         = request.arousal
        )

        # Node C 파이프라인 가동 및 Node B로 프롬프트 전송
        print("  -> Node C 분석 시작 및 Node B로 프롬프트 전송...")
        prompt_request = process_and_build_request(
            session_id       = "session_live",
            user_text        = user_text,
            nonverbal_vector = [request.valence, request.arousal],
            candidates       = [request],
            fused_emotion    = fused_emotion
        )

        response = send_with_safety(prompt_request)
        print(f"  -> Node B 응답 수신 완료!")
        return response

    def SendVoicePerception(self, request, context):
        """
        음성 인식 결과 수신 (SendFacePerception과 동일한 처리)
        Node A가 STT 결과를 별도로 보낼 때 사용
        """
        print(f"\n[Node C Server] 수신된 음성 데이터: '{request.text}' (V:{request.valence:.2f}, A:{request.arousal:.2f})")
        return self.SendFacePerception(request, context)


def serve():
    port = 5052
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aura_pb2_grpc.add_AuraPerceptionServicer_to_server(AuraPerceptionServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print("======================================================")
    print(f"🚀 [Node C] AuraPerception Server가 {port}번 포트에서 열렸습니다!")
    print("   Node A 코드에서 target_address를 'localhost:50052'로 변경해주세요.")
    print("   EmotionCandidate.text 필드에 STT 결과를 담아서 보내주세요.")
    print("======================================================")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()