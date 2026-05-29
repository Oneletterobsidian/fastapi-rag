from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
import httpx
from fastapi import FastAPI, Depends, HTTPException     # 确保 HTTPException 已导入
from database import engine, get_db
from models import Base
import schemas
import crud
from fastapi.security import OAuth2PasswordRequestForm
import auth
from fastapi.middleware.cors import CORSMiddleware
import time
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks

import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from fastapi import UploadFile, File
import tempfile, os, uuid
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from conversation import Conversation

from doc_utils import parse_document, split_text

# 启动时:自动创建数据库表
Base.metadata.create_all(bind=engine)

# === 应用启动时加载 .env(放在 app = FastAPI() 附近)===
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

class RegisterRequest(BaseModel):
    username: str
    password: str

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

    # 2. RAG 检索（同上）
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
        chunks   = results["documents"][0]
        metas    = results["metadatas"][0]
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
        system_content += f"\n\n请优先根据以下背景资料回答，资料中没有的内容可结合自身知识：\n{context_prompt}"

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
    db.add(Conversation(session_id=req.session_id, role="user",      content=req.question))
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
    return {"message": "FastAPI + SQLite 商品 API"}

@app.get("/products", response_model=schemas.ProductListResponse)
def list_products(
    keyword: str = "",
    category: str = "",
    min_price: float | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False,
    limit: int = 100,
    skip: int = 0,
    db: Session = Depends(get_db)
):
    total, products = crud.list_products(
        db, keyword, category, min_price, max_price, in_stock_only, limit, skip
    )
    return {"total": total, "products": products}


@app.get("/products/{product_id}", response_model=schemas.ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    return crud.get_product_by_id(db, product_id)


@app.post("/products", status_code=201, response_model=schemas.ProductResponse)
def create_product(product: schemas.ProductCreate, db: Session = Depends(get_db)):
    return crud.create_product(db, product)


@app.put("/products/{product_id}", response_model=schemas.ProductResponse)
def update_product(
    product_id: int,
    product_update: schemas.ProductUpdate,
    db: Session = Depends(get_db)
):
    return crud.update_product(db, product_id, product_update)


@app.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    crud.delete_product(db, product_id)

@app.patch("/products/{product_id}", response_model=schemas.ProductResponse)
def patch_product(
    product_id: int,
    product_patch: schemas.ProductPatch,
    db: Session = Depends(get_db)
):
    return crud.patch_product(db, product_id, product_patch)

# 加在 main.py 末尾
@app.get("/github/{username}")
async def get_github_user(username: str):
    """异步获取 GitHub 用户信息"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.github.com/users/{username}")
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"用户 {username} 不存在")
        data = response.json()
        return {
            "username": data["login"],
            "name": data.get("name"),
            "bio": data.get("bio"),
            "followers": data["followers"],
            "public_repos": data["public_repos"],
        }
    
# ============================================
# 假装这是用户数据库(暂时用字典)
# ============================================
fake_users_db = {
    "alice": {
        "username": "alice",
        "hashed_password": auth.hash_password("123456"),  # 启动时哈希
    }
}

# ============================================
# 接口 1: 注册
# ============================================
@app.post("/register")
def register(req: RegisterRequest):
    if req.username in fake_users_db:
        raise HTTPException(status_code=400, detail="用户名已存在")
    fake_users_db[req.username] = {
        "username": req.username,
        "hashed_password": auth.hash_password(req.password),
    }
    return {"message": f"用户 {req.username} 注册成功"}

# ============================================
# 接口 2: 登录(返回 token)
# ============================================
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    token = auth.create_access_token(form_data.username)
    return {"access_token": token, "token_type": "bearer"}

# ============================================
# 接口 3: 看自己的信息(受保护,需要 token)
# ============================================
@app.get("/me")
def get_me(username: str = Depends(auth.get_current_user)):
    return {"username": username, "message": "Hello!"}

@app.get("/demo/exception-handler")
def demo_exception_handler():
    """演示 FastAPI 的全局异常处理 —— 故意触发 ZeroDivisionError。
    用于学习阶段验证 @app.exception_handler 是否正确兜底。"""
    return {"result": 10 / 0}

@app.post("/register-v2")
def register_v2(req: RegisterRequest, background_tasks: BackgroundTasks):
    if req.username in fake_users_db:
        raise HTTPException(status_code=400, detail="用户名已存在")
    fake_users_db[req.username] = {
        "username": req.username,
        "hashed_password": auth.hash_password(req.password),
    }
    background_tasks.add_task(send_welcome_email, req.username)
    return {"message": "注册成功,邮件正在发送"}


def send_welcome_email(username: str):
    """模拟发邮件"""
    print(f"📧 开始给 {username} 发邮件...")
    time.sleep(3)
    print(f"📧 邮件已发给 {username}")