from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import time
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from database import engine, get_db
from database import Base
from conversation import Conversation

# embedding模型 + Chroma向量库，改成从共享模块导入（原本是直接写在这个文件里的）
from rag_core import embed_model, chroma_client, collection, QUERY_INSTRUCTION

# Celery应用 + 异步任务，用于文档上传的后台处理
from tasks import celery_app, process_document_task

# 启动时：自动创建数据库表
Base.metadata.create_all(bind=engine)

# === 应用启动时加载 .env ===
load_dotenv()

# === 创建一个全局的 DeepSeek 客户端 ===
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

app = FastAPI()

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 性能监控中间件
@app.middleware("http")
async def log_request_time(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"⏱️  {request.method} {request.url.path} - {duration:.3f}s")
    response.headers["X-Process-Time"] = f"{duration:.3f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理的异常"""
    print(f"❌ 未处理异常: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "error_type": type(exc).__name__,
        }
    )


# === 请求和响应的 Pydantic 模型 ===
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"   # 会话 ID，不传则用 default
    use_rag: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    session_id: str = ""


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # 1. 读取该会话的历史消息（最近10条）
    history = (
        db.query(Conversation)
        .filter(Conversation.session_id == req.session_id)
        .order_by(Conversation.id.desc())
        .limit(10)
        .all()[::-1]   # 反转回正序
    )
    history_messages = [
        {"role": h.role, "content": h.content} for h in history
    ]

    # 2. RAG 检索
    sources = []
    context_prompt = ""
    if req.use_rag and collection.count() > 0:
        query_vec = embed_model.encode(
            QUERY_INSTRUCTION + req.question,
            normalize_embeddings=True,
        ).tolist()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        chunks = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        filtered = [(c, m) for c, m, d in zip(chunks, metas, distances) if d < 0.5]
        if filtered:
            chunks_f, metas_f = zip(*filtered)
            context_prompt = "\n\n".join(
                f"[片段{i+1}] {c}" for i, c in enumerate(chunks_f)
            )
            sources = list(set(m["source"] for m in metas_f))

    # 3. 拼完整 messages（system + 历史 + 当前问题）
    system_content = "你是一个专业的问答助手，回答简洁准确。"
    if context_prompt:
        system_content += (
            "\n\n请严格根据以下背景资料回答，如果资料中没有相关信息，"
            f"请明确告知用户「根据现有资料无法回答这个问题」，不要编造：\n{context_prompt}"
        )
    else:
        system_content += (
            "\n\n当前没有检索到相关背景资料，请先明确告知用户"
            "「未检索到相关文档内容，以下是基于通用知识的回答」，再回答。"
        )

    messages = (
        [{"role": "system", "content": system_content}]
        + history_messages
        + [{"role": "user", "content": req.question}]
    )

    # 4. 调用 DeepSeek
    resp = deepseek_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
    )
    answer = resp.choices[0].message.content

    # 5. 存入对话历史
    db.add(Conversation(session_id=req.session_id, role="user", content=req.question))
    db.add(Conversation(session_id=req.session_id, role="assistant", content=answer))
    db.commit()

    return ChatResponse(answer=answer, sources=sources, session_id=req.session_id)


# 查看对话历史接口
@app.get("/history/{session_id}")
def get_history(session_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.id)
        .all()
    )
    return [{"role": r.role, "content": r.content, "time": r.created_at} for r in records]


# 清空某会话历史
@app.delete("/history/{session_id}")
def clear_history(session_id: str, db: Session = Depends(get_db)):
    db.query(Conversation).filter(Conversation.session_id == session_id).delete()
    db.commit()
    return {"message": f"会话 {session_id} 历史已清空"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    改造后的上传接口：
    只负责【校验格式】+【保存文件到共享卷】+【丢进Celery队列】，
    立刻返回task_id，不再让用户等待"解析+切块+embedding+入库"这个耗时过程。
    """
    # 1. 检查格式
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".txt", ".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="只支持 .txt / .pdf / .docx")

    # 2. 保存到共享卷目录（api和worker两个容器都挂载了同一个volume，都能访问到这个路径）
    upload_dir = "/app/shared_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}{suffix}")
    with open(saved_path, "wb") as f:
        f.write(await file.read())

    # 3. 丢进Celery异步任务队列，立刻返回，不等处理完成
    task = process_document_task.delay(saved_path, file.filename)

    return {
        "message": "文档已接收，正在后台处理",
        "task_id": task.id,
        "filename": file.filename,
    }


@app.get("/documents/status/{task_id}")
def get_upload_status(task_id: str):
    """查询某次文档上传处理任务的进度/结果"""
    task_result = celery_app.AsyncResult(task_id)
    response = {"task_id": task_id, "state": task_result.state}
    if task_result.state == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.state == "FAILURE":
        response["error"] = str(task_result.info)
    return response


@app.get("/")
def read_root():
    return {"message": "RAG 文档问答系统 API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
