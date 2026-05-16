from typing import List, Dict, Optional
from neo4j import GraphDatabase
from deep_translator import GoogleTranslator
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 공감 관련 ATOMIC Relation 설정
# ==========================================
EMPATHY_RELATIONS = ['xReact', 'oReact', 'xWant', 'oWant', 'xEffect', 'oEffect', 'xIntent', 'xAttr', 'xNeed']

RELATION_WEIGHTS = {
    'xReact': 1.0,   # 가장 중요 — PersonX 감정 반응
    'oReact': 0.9,   # 타인의 감정 반응
    'xWant':  0.7,   # PersonX의 욕구
    'xEffect': 0.7,  # PersonX에 미치는 영향
    'xIntent': 0.6,  # PersonX의 의도
    'xNeed':  0.6,   # PersonX에게 필요한 것
    'xAttr':  0.5,   # PersonX의 성격/속성
    'oWant':  0.5,   # 타인의 욕구
    'oEffect': 0.5,  # 타인에게 미치는 영향
}

RELATION_LABELS_KO = {
    'xReact':  '예상 감정 반응',
    'oReact':  '주변인의 감정 반응',
    'xWant':   '예상 욕구/바람',
    'oWant':   '주변인의 욕구/바람',
    'xEffect': '예상 결과/영향',
    'oEffect': '주변인에 대한 영향',
    'xIntent': '추정 의도',
    'xAttr':   '성격/속성',
    'xNeed':   '필요한 것',
}

# ==========================================
# 1. Neo4j 연동 모듈 (Knowledge Graph)
# ==========================================
class KGSearcher:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password123", enable_db=True):
        self.enable_db = enable_db
        if self.enable_db:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
                self.translator = GoogleTranslator(source='ko', target='en')
                self.translation_cache = {} # 번역 결과 캐싱 (속도 향상)
                print("Connected to Neo4j Database & Translator Ready (Optimized).")
            except Exception as e:
                print(f"Neo4j Connection Failed: {e}")
                print("Falling back to Dummy Data mode.")
                self.enable_db = False

    def close(self):
        if self.enable_db and hasattr(self, 'driver'):
            self.driver.close()

    def search_context(self, keywords_ko: List[str]) -> List[Dict]:
        """KG 검색 결과를 구조화된 딕셔너리 리스트로 반환합니다.
        Returns: [{source, relation, target, weight}, ...]
        """
        contexts = []
        if not keywords_ko:
            return contexts

        if self.enable_db:
            # 실시간 번역: 한국어 키워드 -> 영어 키워드 변환 (캐시 및 배치 최적화)
            try:
                # 1. 캐시에 없는 단어들만 골라내기
                to_translate = [kw for kw in keywords_ko if kw not in self.translation_cache]
                
                if to_translate:
                    # 2. 여러 단어를 한 번에 번역 (네트워크 호출 1회로 단축)
                    results = self.translator.translate_batch(to_translate)
                    for ko, en in zip(to_translate, results):
                        self.translation_cache[ko] = en.lower()
                
                # 3. 전체 번역 결과 조립
                translated = [self.translation_cache[kw] for kw in keywords_ko]
                
                # 검색 후보에 한국어와 번역된 영어를 모두 포함
                search_keywords = list(set(keywords_ko + translated))
            except Exception as e:
                print(f"Translation Error during KG Search: {e}")
                search_keywords = keywords_ko

            try:
                # 개별 관계 레이블(xReact, oReact 등)로 검색하여 인덱스 활용
                query = """
                MATCH (n:Concept)-[r]->(m:Concept)
                WHERE type(r) IN $relations
                AND any(kw IN $keywords WHERE 
                    toLower(n.name) CONTAINS (' ' + kw + ' ') OR 
                    toLower(n.name) STARTS WITH (kw + ' ') OR 
                    toLower(n.name) ENDS WITH (' ' + kw) OR 
                    toLower(n.name) = kw
                )
                RETURN n.name AS source, type(r) AS relation, m.name AS target
                LIMIT 10
                """
                
                with self.driver.session() as session:
                    result = session.run(query, keywords=search_keywords, relations=EMPATHY_RELATIONS)
                    for record in result:
                        rel = record['relation']
                        contexts.append({
                            'source': record['source'],
                            'relation': rel,
                            'target': record['target'],
                            'weight': RELATION_WEIGHTS.get(rel, 0.5),
                        })
            except Exception as e:
                print(f"Neo4j Query Error: {e}")

        # 공감 관련도(weight) 기준으로 정렬 — 감정 반응이 최우선
        contexts.sort(key=lambda x: x['weight'], reverse=True)
        return contexts[:7]

# ==========================================
# 2. 텍스트 키워드 추출 (Entity Linking)
# ==========================================
class TextProcessor:
    def __init__(self, enable_mecab=True):
        self.enable_mecab = enable_mecab
        if self.enable_mecab:
            try:
                from konlpy.tag import Mecab
                # Jetson/Ubuntu 설치 경로를 명시적으로 지정합니다.
                self.tagger = Mecab('/usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ko-dic')
                print("Loading Mecab (C++) with mecab-ko-dic...")
            except Exception as e:
                print(f"Mecab 로드 실패 (설치 확인 필요): {e}")
                self.enable_mecab = False

    def extract_keywords(self, text: str) -> List[str]:
        if not self.enable_mecab:
            return []
            
        nouns = self.tagger.nouns(text)
        keywords = [noun for noun in nouns if len(noun) >= 2]
        return list(set(keywords))

# ==========================================
# 3. 텍스트 감성 분석 (On-Device 최적화)
# ==========================================
class TextSentimentAnalyzer:
    def __init__(self, enable=False):
        self.enable = enable
        if self.enable:
            try:
                from transformers import pipeline
                import torch
                print("Loading lightweight sentiment model for Jetson...")
                self.pipe = pipeline(
                    "text-classification", 
                    model="matthewburke/korean_sentiment", 
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                print(f"transformers 모델 로드 실패 (에러: {e})\nDummy 모드로 작동합니다.")
                self.enable = False

    def get_sentiment(self, text: str) -> np.ndarray:
        """
        텍스트를 분석하여 Russell의 2차원 (Valence, Arousal) 벡터로 매핑합니다.
        """
        if not self.enable:
            # 모델이 꺼져있을 경우 중립 수치 반환 (가짜 우울증 삽입 금지)
            return np.array([0.0, 0.0], dtype=np.float32)
        
        result = self.pipe(text)[0]
        label = result['label']
        score = result['score']
        
        if label == "LABEL_1" or "pos" in label.lower(): 
            valence = score * 1.0
            arousal = 0.0
        else:
            valence = -score * 1.0
            arousal = score * 0.5
            
        return np.array([valence, arousal], dtype=np.float32)

# ==========================================
# 4. 정렬도 검사 (Sentiment Alignment Check)
# ==========================================
class AlignmentChecker:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def check_alignment(self, text_sentiment_prob: np.ndarray, face_emotion_prob: np.ndarray) -> Dict:
        """
        언어(텍스트)의 감정과 비언어(표정/목소리)의 감정이 일치하는지 코사인 유사도로 검사합니다.
        """
        dot_product = np.dot(text_sentiment_prob, face_emotion_prob)
        norm_text = np.linalg.norm(text_sentiment_prob)
        norm_face = np.linalg.norm(face_emotion_prob)
        
        if norm_text == 0 or norm_face == 0:
            score = 0.0
        else:
            score = dot_product / (norm_text * norm_face)
            
        is_consistent = bool(score >= self.threshold)
        
        return {
            "score": round(float(score), 4),
            "is_consistent": is_consistent,
            "detected_anomaly": "Sarcasm/Suppression" if not is_consistent else "None"
        }

# ==========================================
# 4.5. 3-모달 감정 융합 레이어 (Modal Fusion)
# ==========================================
class ModalFusionLayer:
    """
    표정(face), 음성 톤(voice), 텍스트 감성(text) 세 신호를
    가중 평균으로 융합하여 단일 감정 벡터를 생성합니다.

    가중치 근거:
    - 비언어 신호(표정+음성)가 언어보다 감정 표현에 직접적이므로 0.4씩 부여
    - 텍스트는 사회적 필터링이 개입될 수 있으므로 0.2 부여
    - 신호가 일부 없을 경우 나머지 신호의 가중치를 비례적으로 재정규화
    """
    WEIGHTS = {'face': 0.4, 'voice': 0.4, 'text': 0.2}

    def fuse(self,
             face_va: Optional[np.ndarray] = None,
             voice_va: Optional[np.ndarray] = None,
             text_va: Optional[np.ndarray] = None) -> np.ndarray:
        """제공된 신호만 사용해 가중 평균 계산. 없는 신호는 자동 제외 후 가중치 재정규화."""
        signals, weights = [], []
        if face_va is not None:
            signals.append(np.asarray(face_va, dtype=np.float32))
            weights.append(self.WEIGHTS['face'])
        if voice_va is not None:
            signals.append(np.asarray(voice_va, dtype=np.float32))
            weights.append(self.WEIGHTS['voice'])
        if text_va is not None:
            signals.append(np.asarray(text_va, dtype=np.float32))
            weights.append(self.WEIGHTS['text'])

        if not signals:
            return np.array([0.0, 0.0], dtype=np.float32)

        total = sum(weights)
        return sum((w / total) * s for w, s in zip(weights, signals))

# ==========================================
# 4.5 KG 결과를 구조화된 프롬프트로 변환
# ==========================================
def format_kg_for_prompt(kg_results: List[Dict]) -> str:
    """KG 검색 결과를 관계 타입별로 그룹핑하여 LLM이 이해하기 쉬운 형태로 변환"""
    if not kg_results:
        return "관련 배경지식 없음."
    
    from collections import defaultdict
    groups = defaultdict(list)
    
    for item in kg_results:
        rel = item['relation']
        category = RELATION_LABELS_KO.get(rel, rel)
        target = item['target']
        # 중복 제거
        if target not in groups[category]:
            groups[category].append(target)
    
    # 카테고리 순서 고정 (감정 > 욕구 > 영향 > 기타)
    order = ['예상 감정 반응', '주변인의 감정 반응', '예상 욕구/바람', '주변인의 욕구/바람',
             '예상 결과/영향', '주변인에 대한 영향', '추정 의도', '필요한 것', '성격/속성']
    
    sections = []
    for category in order:
        if category in groups:
            lines = [f"  - {t}" for t in groups[category]]
            sections.append(f"[{category}]\n" + "\n".join(lines))
    
    return "\n\n".join(sections)

# ==========================================
# 5. Node C 메인 파이프라인 (팀 구조에 맞춰 PyTorch Fusion 제거)
# ==========================================
class NodeC:
    def __init__(self):
        self.kg_searcher = KGSearcher(enable_db=True)
        self.text_processor = TextProcessor(enable_mecab=True)
        self.sentiment_analyzer = TextSentimentAnalyzer(enable=True)
        self.alignment_checker = AlignmentChecker(threshold=0.5)
        self.modal_fusion = ModalFusionLayer()
        # 세션 간 비언어 신호 상태 유지 (표정/음성이 별도 호출로 들어오므로)
        self.last_face_va: Optional[np.ndarray] = None
        self.last_voice_va: Optional[np.ndarray] = None

    def process_data(self,
                     text: str,
                     face_va: Optional[np.ndarray] = None,
                     voice_va: Optional[np.ndarray] = None) -> Dict:
        """
        text       : STT 또는 사용자 입력 텍스트
        face_va    : 표정 분석 결과 [valence, arousal] (없으면 이전 값 재사용)
        voice_va   : 음성 톤 분석 결과 [valence, arousal] (없으면 이전 값 재사용)
        """
        start_time = time.time()

        # 수신된 신호로 세션 상태 업데이트
        if face_va is not None:
            self.last_face_va = np.asarray(face_va, dtype=np.float32)
        if voice_va is not None:
            self.last_voice_va = np.asarray(voice_va, dtype=np.float32)

        # 텍스트가 없는 경우 무거운 분석 스킵
        if not text or text.strip() == "":
            return {
                "kg_context": [],
                "kg_context_formatted": "관련 배경지식 없음.",
                "alignment": {"score": 1.0, "is_consistent": True, "detected_anomaly": "None"},
                "priority": 1,
                "latency_ms": 0.0
            }

        # 1. Entity Linking & KG Search
        print(f"    [Step 1] 키워드 추출 및 KG 검색 중...", end=" ", flush=True)
        keywords = self.text_processor.extract_keywords(text)
        kg_context = self.kg_searcher.search_context(keywords)
        kg_context_formatted = format_kg_for_prompt(kg_context)
        print(f"완료 ({len(kg_context)}건)")

        # 2. 텍스트 감성 추출 (V-A)
        print(f"    [Step 2] 텍스트 감정 분석 중...", end=" ", flush=True)
        text_sentiment = self.sentiment_analyzer.get_sentiment(text)
        print(f"완료 (V:{text_sentiment[0]:.2f}, A:{text_sentiment[1]:.2f})")

        # 3. 3-모달 융합: 표정 + 음성 톤 → 비언어 융합 벡터
        fused_nonverbal = self.modal_fusion.fuse(
            face_va=self.last_face_va,
            voice_va=self.last_voice_va
        )
        available = []
        if self.last_face_va is not None: available.append('face')
        if self.last_voice_va is not None: available.append('voice')
        print(f"    [Step 3] 비언어 융합 ({'+'.join(available) if available else 'none'}): "
              f"V:{fused_nonverbal[0]:.2f}, A:{fused_nonverbal[1]:.2f}")

        # 4. Alignment Check: 텍스트 감성 vs 융합된 비언어 벡터
        alignment_result = self.alignment_checker.check_alignment(text_sentiment, fused_nonverbal)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "kg_context": kg_context,
            "kg_context_formatted": kg_context_formatted,
            "alignment": alignment_result,
            "fused_nonverbal": fused_nonverbal.tolist(),
            "priority": 2 if not alignment_result["is_consistent"] else 1,
            "latency_ms": round(latency_ms, 2)
        }
