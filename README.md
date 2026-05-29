# FastAPI 学习项目

我学习 FastAPI 的完整代码记录，涵盖从基础 Web 开发到 RAG 应用的完整路径。最终阶段实现了一个工业级的检索增强生成（RAG）系统作为核心成果。

## 学习路径与对应接口

| 阶段 | 学习主题 | 接口 |
|------|---------|------|
| 1 | 基础 CRUD + SQLAlchemy | `/products/*` |
| 2 | 异步外部 API 调用 | `/github/{username}` |
| 3 | JWT 鉴权 | `/register` `/login` `/me` |
| 4 | 后台任务 | `/register-v2` |
| 5 | **RAG 应用（核心成果）** | `/documents/upload` `/chat` `/history/*` |
| - | 异常处理演示 | `/demo/exception-handler` |

## 核心成果：RAG 系统

### 技术栈

| 模块 | 技术选型 |
|------|---------|
| Web 框架 | FastAPI |
| 关系数据库 | SQLite + SQLAlchemy |
| 向量数据库 | Chroma（cosine 距离）|
| 嵌入模型 | BGE-small-zh-v1.5（512 维，CPU 可跑）|
| 大语言模型 | DeepSeek V4-Flash |
| 文档解析 | pypdf + python-docx |
| 认证 | JWT |

### RAG 数据流

```
用户上传文档
    ↓
文档解析（doc_utils.py）
    ↓
切分成 chunk（默认 300 字符 + 50 字符 overlap）
    ↓
BGE 嵌入 + 归一化 → 存入 Chroma（cosine 索引）
    ↓
用户提问
    ↓
查询编码（加 BGE 指令前缀 + 归一化）
    ↓
向量检索（cosine 距离 < 0.5 过滤）
    ↓
拼接 Prompt（背景资料 + 历史 + 当前问题）
    ↓
DeepSeek 生成回答 → 写入对话历史
```

### 关键设计决策

- **BGE 非对称编码**：查询加指令前缀（`"为这个句子生成表示以用于检索相关文章："`），文档不加 —— 提升短查询召回精度
- **嵌入归一化 + cosine 距离**：避免向量长度对相似度判断的污染
- **距离阈值数据驱动**：0.5 经实测校准 —— 相关查询距离约 0.30，无关查询距离约 0.70，区分度 0.4
- **临时文件处理**：上传文件用 `tempfile` 写盘后立即解析，`try/finally` 保证清理
- **持久化存储**：Chroma 数据存 `./chroma_db`，重启不丢

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 5. 启动服务

```bash
uvicorn main:app --reload
```

打开 http://localhost:8000/docs 查看 API 文档。

## 测试 RAG 功能

启动后用 Swagger UI 测试：

1. **上传文档**：`POST /documents/upload`，上传一份 `.txt` / `.pdf` / `.docx`
2. **基于文档提问**：`POST /chat`，`use_rag: true`
   - 相关问题 → `sources` 返回文档名
   - 无关问题 → `sources` 为空，模型用自身知识回答

## 项目结构

```
fastapi-learn/
├── main.py             # FastAPI 主入口（所有路由）
├── doc_utils.py        # 文档解析与切分
├── database.py         # SQLAlchemy 引擎
├── models.py           # ORM 模型
├── conversation.py     # 对话历史表
├── schemas.py          # Pydantic schema（products 用）
├── crud.py             # 业务数据库操作（products 用）
├── auth.py             # JWT 认证
├── requirements.txt
├── .env.example
└── README.md
```

## 学习笔记

本项目伴随完整的 Obsidian 学习笔记，记录了每个阶段的关键设计决策与踩坑过程。笔记没有放进 GitHub（属于个人学习资料），需要的话另行索取。

## License

MIT
