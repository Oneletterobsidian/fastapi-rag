"""
conftest.py

pytest会自动加载这个文件，这里定义的东西对tests/目录下所有测试文件都生效。

这个文件做的核心事情：
1. 让main.py使用的数据库指向一个独立的测试专用SQLite文件，不碰你正式的fastapi_learn.db
2. 在main.py真正导入rag_core之前，先把rag_core"偷梁换柱"成一个假模块，
   这样测试时不会真的去加载几百MB的embedding模型、也不会真的去连Chroma服务器
3. 提供几个测试文件都会用到的公共工具（fixture）：client、db_session等
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# ============================================
# 第一步：在导入任何项目代码之前，先设置好环境变量
# ============================================
# 数据库指向独立的测试文件（不设置的话main.py默认用fastapi_learn.db，会污染真实数据）
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_fastapi_learn.db")

# DeepSeek的API Key：main.py在import时会创建OpenAI客户端对象，
# 如果环境变量完全没有这个key，某些版本的OpenAI SDK会在创建对象时就报错
# 这里给一个假的占位值，避免这个问题（反正测试里也不会真的发请求）
os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy-key-not-real")


# ============================================
# 第二步：把rag_core模块"偷梁换柱"成假的
# ============================================
# main.py和tasks.py都有一行 `from rag_core import embed_model, collection, ...`
# 这里的关键技巧：Python的模块导入是全局缓存的（sys.modules），
# 只要我们在main.py真正被import之前，把sys.modules["rag_core"]
# 设置成一个假模块，之后任何地方 `from rag_core import xxx` 拿到的都是这个假的
_fake_rag_core = MagicMock()
_fake_rag_core.embed_model = MagicMock(name="embed_model")
_fake_rag_core.chroma_client = MagicMock(name="chroma_client")
_fake_rag_core.collection = MagicMock(name="collection")
_fake_rag_core.QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
sys.modules["rag_core"] = _fake_rag_core

# 注意：tasks.py里的Celery应用本身不需要mock——
# 创建Celery(broker=...)这个对象并不会真的去连Redis，
# 只有真正调用.delay()/.apply_async()发起异步任务时才需要连接。
# 我们测试时要么直接调用任务函数本身（不走Celery），
# 要么在测试/chat和/documents接口时用monkeypatch替换掉.delay这个方法，
# 所以tasks.py可以正常真实导入，不需要额外造假模块。


# ============================================
# 第三步：现在才导入项目代码（这时候rag_core已经是假的了）
# ============================================
from fastapi.testclient import TestClient
from database import Base, engine
import main


# ============================================
# Fixture们
# ============================================

@pytest.fixture(scope="session", autouse=True)
def _setup_test_database():
    """
    整个测试会话（session）级别，只跑一次。
    autouse=True 表示不需要每个测试函数显式声明依赖它，自动生效。
    """
    # main.py导入时其实已经建过表了（Base.metadata.create_all那行），
    # 这里再调用一次是保险起见，SQLAlchemy的create_all本身是幂等的（表已存在就跳过）
    Base.metadata.create_all(bind=engine)

    yield

    # 所有测试跑完之后清理：先释放连接池（Windows上如果不这么做，
    # 连接还占用着文件句柄，下面删除文件时会报PermissionError），再删表、删文件
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    test_db_file = "test_fastapi_learn.db"
    if os.path.exists(test_db_file):
        os.remove(test_db_file)


@pytest.fixture(autouse=True)
def _reset_shared_mocks():
    """
    每个测试跑之前自动重置一次embed_model和collection这两个mock的调用记录/返回值配置。
    没有这一步的话，前一个测试里设置的返回值会"泄漏"到下一个测试，互相干扰。
    """
    _fake_rag_core.embed_model.reset_mock(return_value=True, side_effect=True)
    _fake_rag_core.collection.reset_mock(return_value=True, side_effect=True)
    yield


@pytest.fixture
def client():
    """FastAPI提供的测试客户端，可以直接模拟发HTTP请求给app，不需要真的启动uvicorn服务器"""
    return TestClient(main.app)


@pytest.fixture
def db_session():
    """
    需要直接操作数据库（不通过HTTP接口）时用这个，比如测试Conversation模型本身。
    测试跑完自动清理掉这次测试插入的数据，保持测试之间互不干扰。
    """
    from database import SessionLocal
    from conversation import Conversation

    session = SessionLocal()
    yield session
    session.query(Conversation).delete()
    session.commit()
    session.close()


@pytest.fixture
def mock_embed_model():
    """需要在测试里配置embedding模型的假返回值时，用这个拿到那个mock对象"""
    return _fake_rag_core.embed_model


@pytest.fixture
def mock_collection():
    """需要在测试里配置Chroma查询结果的假返回值时，用这个拿到那个mock对象"""
    return _fake_rag_core.collection
