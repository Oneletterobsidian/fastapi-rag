import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── 初始化三个组件 ──────────────────────────────────────
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_docs")

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── 知识库（模拟几段"文档"）────────────────────────────
knowledge = [
    "苹果富含维生素C，口感清脆甘甜，是最受欢迎的水果之一。",
    "香蕉含有丰富的钾元素，有助于缓解肌肉疲劳。",
    "今天A股市场大跌，沪指下跌2.3%，投资者损失惨重。",
    "西瓜含水量高达92%，是夏季消暑的理想水果。",
    "梨子性凉，有润肺止咳的功效，适合秋冬季节食用。",
]

# 存入 Chroma（如果已经存过就跳过，避免重复）
if collection.count() == 0:
    embeddings = embed_model.encode(knowledge).tolist()
    collection.add(
        documents=knowledge,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(knowledge))],
    )
    print(f"知识库已建立，共 {len(knowledge)} 条\n")
else:
    print(f"知识库已存在，共 {collection.count()} 条\n")


# ── 核心函数：RAG 问答 ──────────────────────────────────
def rag_chat(question: str, n_results: int = 2) -> str:
    # 第一步：检索
    query_vec = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=n_results)
    retrieved_docs = results["documents"][0]

    print(f"检索到的相关片段：")
    for doc in retrieved_docs:
        print(f"  · {doc}")
    print()

    # 第二步：拼 Prompt
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = f"""你是一个问答助手。请根据下面提供的背景资料回答用户的问题。
如果背景资料中没有相关信息，请直接说"我不知道"，不要编造。

背景资料：
{context}

用户问题：{question}
"""

    # 第三步：生成
    response = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ── 测试 ────────────────────────────────────────────────
questions = [
    "苹果有什么营养价值？",
    "夏天吃什么水果好？",
    "今天股市怎么样？",       # 知识库里有
    "榴莲的热量高吗？",       # 知识库里没有，应该回答"我不知道"
]

for q in questions:
    print(f"{'='*50}")
    print(f"问题：{q}")
    answer = rag_chat(q)
    print(f"回答：{answer}\n")