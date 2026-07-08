from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import time
import os
import tempfile
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer

from database import engine, get_db
from models import Base
from conversation import Conversation
from doc_utils import parse_document, split_text

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

# === 全局：嵌入模型 + Chroma ===
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"},
)

# BGE 模型的查询指令前缀（仅查询用，文档不加）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

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
    # 1. 检查格式
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".txt", ".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="只支持 .txt / .pdf / .docx")

    # 2. 把上传文件存到临时目录，再解析
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = parse_document(tmp_path)
    finally:
        os.unlink(tmp_path)   # 解析完立即删除临时文件

    # 3. 切分
    chunks = split_text(text)

    # 4. 向量化 + 入库
    doc_id = str(uuid.uuid4())[:8]   # 用文件的短 ID 做前缀
    embeddings = embed_model.encode(chunks, normalize_embeddings=True).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"source": file.filename, "chunk_index": i}
                   for i in range(len(chunks))],
    )

    return {
        "message": "上传成功",
        "filename": file.filename,
        "doc_id": doc_id,
        "chunks_count": len(chunks),
        "chars_total": len(text),
    }


@app.get("/")
def read_root():
    return {"message": "RAG 文档问答系统 API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)