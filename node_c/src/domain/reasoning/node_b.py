class EmpathyService:
    def __init__(self, repo, gemma_model):
        self.repo = repo
        self.gemma = gemma_model

    def run_mock_test(self, scenario, keyword=None):
        print(f"\n[Input Scenario]: {scenario}")
        
        # 1. 키워드 추출 (Gemma에게 요청)
        if keyword is None:
            extract_prompt = f"다음 문장에서 가장 핵심적인 감정/상태 키워드 하나만 단어로 추출해줘: {scenario}"
            keyword = self.gemma.generate(extract_prompt).strip().replace(".", "")
        
        print(f"[Target Keyword]: {keyword}")
        
        # 2. 지식 검색 (Neo4j)
        knowledge = self.repo.get_emotional_context(keyword)
        if not knowledge:
            knowledge_str = "(관련된 지식을 찾지 못했습니다)"
        else:
            knowledge_str = "\n".join(knowledge)
            
        print(f"[Retrieved Knowledge]:\n{knowledge_str}")

        # 3. Gemma 프롬프트 생성 (RAG)
        prompt = f"""
        당신은 공감 능력이 뛰어난 AI 어시스턴트 '아우라(Aura)'입니다.
        아래의 '참고 지식'을 바탕으로 사용자의 상황에 깊이 공감하고 위로를 건네주세요.

        [사용자 상황]: {scenario}
        [참고 지식 (ConceptNet/VAD)]:
        {knowledge_str}

        [아우라의 공감 답변]:"""
        
        # 4. Gemma 생성
        response = self.gemma.generate(prompt)
        print(f"[Aura's Response]: {response}\n" + "-"*50)
