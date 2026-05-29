from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ============================================
# 1. 数据库连接 URL
# ============================================
# SQLite 数据库,文件叫 fastapi_learn.db,放在项目根目录
DATABASE_URL = "sqlite:///./fastapi_learn.db"

# ============================================
# 2. 创建数据库引擎
# ============================================
# engine 是 SQLAlchemy 与数据库通信的"总管"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}    # SQLite 专用配置
)

# ============================================
# 3. 创建会话工厂
# ============================================
# SessionLocal 是个"会话生产线",每次调用 SessionLocal() 就得到一个新会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ============================================
# 4. 创建模型基类
# ============================================
# 所有的 SQLAlchemy 模型都要继承这个 Base
Base = declarative_base()

# ============================================
# 5. 数据库会话依赖
# ============================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()