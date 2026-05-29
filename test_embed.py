from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 加载模型（第一次会下载，之后直接读缓存）
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 2. 把文本变成向量
sentences = [
    "苹果好吃吗",
    "苹果是一种水果，口感清脆甘甜",
    "今天股市大跌，投资者损失惨重",
    "这个水果非常美味",          # ← 和第一句没有字重叠，但语义近
]

embeddings = model.encode(sentences)

print(f"向量维度：{embeddings.shape}")  # (4, 512)

# 3. 手动算余弦相似度（和你的 rag_toy.py 逻辑一样，只是换成 numpy）
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query_vec = embeddings[0]  # "苹果好吃吗"

print("\n各句子与问题的相似度：")
for i, sent in enumerate(sentences[1:], 1):
    sim = cosine_sim(query_vec, embeddings[i])
    print(f"  {sim:.3f}  |  {sent}")