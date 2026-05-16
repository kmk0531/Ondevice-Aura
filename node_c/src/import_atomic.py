import csv
from collections import defaultdict
from neo4j import GraphDatabase
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 공감 추론에 핵심적인 ATOMIC relation만 선별
# ==========================================
EMPATHY_RELATIONS = {
    'xReact', 'oReact',     # 감정 반응
    'xWant', 'oWant',       # 욕구/바람
    'xEffect', 'oEffect',   # 영향
    'xIntent',              # 의도
    'xAttr',                # 성격/속성
    'xNeed',                # 필요
}

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password123"

def clear_db(session):
    print("🧨 기존 노이즈 데이터 삭제 중... (잠시만 기다려주세요)")
    try:
        # 최신 Neo4j 버전의 트랜잭션 분할 삭제 방식
        session.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
    except:
        # 구버전 대비용
        session.run("MATCH (n) DETACH DELETE n")
    print("✅ 기존 데이터 삭제 완료!")

def create_indexes(session):
    print("🛠 검색 속도 향상을 위해 인덱스 생성 중...")
    try:
        session.run("CREATE INDEX node_name IF NOT EXISTS FOR (n:Concept) ON (n.name)")
    except Exception as e:
        pass

def _flush_batch(driver, batch):
    """relation 타입별로 Neo4j에 배치 저장 (개별 관계 레이블 사용)"""
    with driver.session() as session:
        for relation, items in batch.items():
            if not items:
                continue
            # 관계 이름은 ATOMIC의 고정된 집합에서만 나오므로 f-string 사용이 안전합니다.
            query = f"""
            UNWIND $batch AS row
            MERGE (head:Concept {{name: row.head}})
            MERGE (tail:Concept {{name: row.tail}})
            MERGE (head)-[:{relation}]->(tail)
            """
            session.run(query, batch=items)

def import_data(driver, tsv_path):
    print(f"🚀 ATOMIC 데이터셋 고속 로드 시작: {tsv_path}")
    print(f"📋 공감 관련 관계만 필터링: {', '.join(sorted(EMPATHY_RELATIONS))}")
    
    batch = defaultdict(list)
    batch_size = 5000
    total_count = 0
    filtered_count = 0
    
    start_time = time.time()
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 3:
                continue
            head, relation, tail = row[0], row[1], row[2]
            
            # 공감 관련 relation만 필터링
            if relation not in EMPATHY_RELATIONS:
                filtered_count += 1
                continue
            
            # 의미 없는 'none' 응답은 공감에 방해되므로 제외
            if tail.strip().lower() == "none":
                continue
                
            batch[relation].append({
                "head": head.lower(),
                "tail": tail.lower()
            })
            
            total_items = sum(len(v) for v in batch.values())
            if total_items >= batch_size:
                _flush_batch(driver, batch)
                total_count += total_items
                batch = defaultdict(list)
                elapsed = time.time() - start_time
                if total_count % 50000 == 0:
                    print(f"  ... {total_count}개 엣지 저장 완료 ({elapsed:.1f}초 경과)")
                
        # 남은 배치 처리
        remaining = sum(len(v) for v in batch.values())
        if remaining:
            _flush_batch(driver, batch)
            total_count += remaining
            
    print(f"🎉 총 {total_count}개의 고품질 공감 지식이 DB에 저장되었습니다! (소요 시간: {time.time() - start_time:.1f}초)")
    print(f"🗑️  필터링된 비공감 트리플: {filtered_count}건")

if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        clear_db(session)
        create_indexes(session)
    import_data(driver, "/home/ys9072/EmpathyModel/train.tsv")
    driver.close()
