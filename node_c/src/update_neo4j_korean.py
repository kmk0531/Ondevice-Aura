from neo4j import GraphDatabase
from deep_translator import GoogleTranslator

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password123"

def translate_and_update_db():
    print("========================================")
    print("🌐 Neo4j 다국어(한국어) 노드 자동 번역 업데이트 스크립트")
    print("========================================")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("✅ DB 연결 성공")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return

    translator = GoogleTranslator(source='en', target='ko')

    with driver.session() as session:
        # name 속성이 있고, 아직 name_ko(한국어)가 없는 노드만 찾기 (재개 가능)
        result = session.run("MATCH (n) WHERE n.name IS NOT NULL AND n.name_ko IS NULL RETURN id(n) AS id, n.name AS name")
        nodes = [{"id": record["id"], "name": record["name"]} for record in result]
        
        print(f"🔍 총 {len(nodes)}개의 노드를 발견했습니다. 번역을 시작합니다...")
        
        success_count = 0
        for node in nodes:
            try:
                name_en = node["name"]
                # 소문자 변환 후 번역 (더 정확한 결과를 위해)
                name_ko = translator.translate(name_en.lower())
                
                # 번역 결과 교정 (예: project -> 프로젝트, exam -> 시험 등)
                # 구글 번역기가 이상하게 번역할 수 있는 주요 IT/학교 용어 보정
                corrections = {
                    "project": "프로젝트",
                    "presentation": "발표",
                    "exam": "시험",
                    "grade": "성적",
                    "score": "점수",
                    "assignment": "과제",
                    "stress": "스트레스",
                    "progress": "진행"
                }
                if name_en.lower() in corrections:
                    name_ko = corrections[name_en.lower()]
                
                # DB 노드에 name_ko 속성 추가/업데이트
                session.run("MATCH (n) WHERE id(n) = $id SET n.name_ko = $name_ko", 
                            id=node["id"], name_ko=name_ko)
                
                print(f"  [업데이트 완료] {name_en} -> {name_ko}")
                success_count += 1
            except Exception as e:
                print(f"  [에러] '{node['name']}' 번역 중 실패: {e}")
                
    driver.close()
    print(f"\n 업데이트 완료! (성공: {success_count}/{len(nodes)})")

if __name__ == "__main__":
    translate_and_update_db()
