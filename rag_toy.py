import math
from collections import Counter

def fake_embed(text):
    """玩具版"嵌入":统计每个字出现的次数,返回字典"""
    return Counter(text)

def cosine_similarity(vec1, vec2):
    """计算两个"字典向量"的余弦相似度"""
    # 找出两个向量里出现过的所有字
    all_keys = set(vec1.keys()) | set(vec2.keys())
    
    # 点积:对应位置相乘再相加
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
    
    # 各自的"长度"(L2 范数)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


# 用户提问
query = "苹果好吃吗"

# 知识库(三段"文档")
docs = [
    "苹果是一种水果,口感清脆甘甜",
    "今天股市大跌,投资者损失惨重",
    "梨子和苹果都很好吃",
]

# 把 query 和每个 doc 都转成"向量"
query_vec = fake_embed(query)

print(f"问题:{query}\n")
print("各文档与问题的相似度:")
for doc in docs:
    doc_vec = fake_embed(doc)
    sim = cosine_similarity(query_vec, doc_vec)
    print(f"  相似度 {sim:.3f}  |  {doc}")  