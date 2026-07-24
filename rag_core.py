"""
rag_core.py

把embedding模型和Chroma向量库的初始化逻辑单独抽出来，
main.py（负责/chat查询）和tasks.py（负责Celery后台处理上传文档）共用这份逻辑。

【更新】Chroma改成HttpClient连接独立的chroma服务容器，
不再用PersistentClient直接打开本地文件夹。
原因：api容器和worker容器是两个独立进程，如果都用PersistentClient
打开同一份本地chroma_db文件，会出现"其中一个在写入时，另一个的连接
把文件锁住"的问题，导致写入操作卡死、既不报错也不完成。
改成HttpClient访问独立的Chroma服务器后，
真正的并发读写控制交给Chroma服务器自己处理，天然支持多客户端。
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

# === 全局：嵌入模型 ===
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# === 全局：Chroma客户端，改为连接独立的chroma服务 ===
# CHROMA_HOST/CHROMA_PORT 从环境变量读取，默认值方便本地不用Docker时也能跑（连localhost）
chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=int(os.getenv("CHROMA_PORT", "8000")),
)
collection = chroma_client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"},
)

# BGE 模型的查询指令前缀（仅查询用，文档不加）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
