import chromadb
from sentence_transformers import SentenceTransformer

# 1. 初始化嵌入模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 2. 创建 Chroma 客户端（persistent = 存到磁盘，重启不丢失）
client = chromadb.PersistentClient(path="./chroma_db")

# 3. 创建或获取一个"集合"（类比：数据库里的一张表）
collection = client.get_or_create_collection(name="my_docs")

# 4. 准备文档
docs = [
    "苹果是一种水果，口感清脆甘甜",
    "今天股市大跌，投资者损失惨重",
    "梨子和苹果都很好吃",
    "这个水果非常美味",
]

# 5. 把文档变成向量，存入 Chroma
embeddings = model.encode(docs).tolist()  # Chroma 需要 list，不是 numpy array

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(docs))],  # 每条必须有唯一 ID
)

print(f"已存入 {collection.count()} 条文档\n")

# 6. 用问题搜索最相关的 2 条
query = "苹果好吃吗"
query_vec = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_vec],
    n_results=2,
)

print(f"问题：{query}\n")
print("最相关的文档：")
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"  距离 {distance:.3f}  |  {doc}")