"""
tasks.py

定义Celery应用和异步任务。
这里的process_document_task就是原来main.py里upload_document接口
"解析文档 -> 切块 -> embedding -> 存入Chroma"这部分耗时逻辑，
搬到这里作为一个独立的后台任务，由Celery worker进程执行，
不再占用FastAPI主进程的请求处理时间。
"""

import os
import uuid

from celery import Celery

from rag_core import embed_model, collection
from doc_utils import parse_document, split_text

# === 创建Celery应用 ===
# broker: 任务队列本身（谁负责传递"有个新任务"这个消息）
# backend: 任务结果存储的地方（任务跑完了，结果存哪，方便之后查询）
# 这里broker和backend都用同一个Redis，是最简单常见的配置方式
celery_app = Celery(
    "fastapi_learn",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)


@celery_app.task(bind=True, name="process_document")
def process_document_task(self, saved_path: str, filename: str):
    """
    后台异步任务：解析文档 -> 切块 -> embedding -> 存入Chroma

    参数：
        saved_path: 文件在共享卷里的实际存储路径
        filename:   原始文件名，用于存进metadata里做溯源
    """
    try:
        # 1. 解析文档内容
        text = parse_document(saved_path)

        # 2. 切分成小块
        chunks = split_text(text)

        # 3. 向量化 + 入库（这是最耗时的一步，现在跑在后台，不阻塞用户请求）
        doc_id = str(uuid.uuid4())[:8]
        embeddings = embed_model.encode(chunks, normalize_embeddings=True).tolist()
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            metadatas=[
                {"source": filename, "chunk_index": i} for i in range(len(chunks))
            ],
        )

        return {
            "status": "done",
            "filename": filename,
            "doc_id": doc_id,
            "chunks_count": len(chunks),
            "chars_total": len(text),
        }

    finally:
        # 处理完（不管成功还是失败）都清理掉临时保存的文件
        if os.path.exists(saved_path):
            os.unlink(saved_path)
