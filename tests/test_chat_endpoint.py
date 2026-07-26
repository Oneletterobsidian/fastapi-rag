"""
test_chat_endpoint.py

测试 POST /chat 这个核心接口。

这里会大量用到mock，原因：
- embed_model（几百MB的模型）和collection（Chroma查询）已经在conftest.py里
  被整体换成了假对象，这里只需要"配置"它们这次调用应该返回什么
- deepseek_client是真实的OpenAI客户端对象（没有被conftest换掉），
  用monkeypatch单独替换它的.chat.completions.create方法，
  避免测试的时候真的发网络请求去调用DeepSeek的API（慢、花钱、还需要真实API Key）

一个需要注意的点：main.py注册了全局异常处理器(global_exception_handler)，
会把所有未捕获的异常都转换成500响应返回，而不是让pytest看到原始的Python报错堆栈。
所以如果某个测试意外地返回500，多半是mock没配置对，可以看看响应体里的error_type字段找线索。
"""

from unittest.mock import MagicMock

import main


def _mock_deepseek_reply(monkeypatch, answer_text: str):
    """小工具函数：让deepseek_client.chat.completions.create返回一个指定的回答文本"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=answer_text))]
    monkeypatch.setattr(
        main.deepseek_client.chat.completions,
        "create",
        MagicMock(return_value=mock_response),
    )


def test_chat_without_rag_when_knowledge_base_empty(client, monkeypatch, mock_collection):
    """向量库里还没有任何文档时（collection.count()==0），
    应该走"通用知识回答"这条分支，不做检索，sources应该是空列表"""
    mock_collection.count.return_value = 0
    _mock_deepseek_reply(monkeypatch, "这是mock的通用回答")

    resp = client.post(
        "/chat",
        json={"question": "你好", "session_id": "chat-test-1", "use_rag": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "这是mock的通用回答"
    assert data["sources"] == []
    assert data["session_id"] == "chat-test-1"
    # 向量库是空的，不应该真的去调用collection.query
    mock_collection.query.assert_not_called()


def test_chat_with_relevant_document_found(
    client, monkeypatch, mock_collection, mock_embed_model
):
    """向量库里有文档、且检索结果足够相关（distance < 0.5）时，
    应该把检索到的内容作为背景资料，sources里要包含来源文件名"""
    mock_collection.count.return_value = 5
    mock_embed_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_collection.query.return_value = {
        "documents": [["这是关于产品定价的说明文字"]],
        "metadatas": [[{"source": "产品手册.txt"}]],
        "distances": [[0.1]],  # 小于0.5的阈值，会被采纳
    }
    _mock_deepseek_reply(monkeypatch, "根据产品手册，价格是xxx")

    resp = client.post(
        "/chat",
        json={"question": "产品定价是多少", "session_id": "chat-test-2"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "根据产品手册，价格是xxx"
    assert data["sources"] == ["产品手册.txt"]


def test_chat_filters_out_low_relevance_results(
    client, monkeypatch, mock_collection, mock_embed_model
):
    """核心业务逻辑验证：distance >= 0.5 的检索结果被判定为"不够相关"，
    应该被过滤掉，不作为背景资料塞给大模型，sources也应该是空的"""
    mock_collection.count.return_value = 5
    mock_embed_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_collection.query.return_value = {
        "documents": [["完全不相关的内容"]],
        "metadatas": [[{"source": "无关文档.txt"}]],
        "distances": [[0.9]],  # 超过阈值0.5，应该被过滤掉
    }
    _mock_deepseek_reply(monkeypatch, "未检索到相关文档内容，以下是基于通用知识的回答...")

    resp = client.post(
        "/chat",
        json={"question": "随便问问", "session_id": "chat-test-3"},
    )

    assert resp.status_code == 200
    assert resp.json()["sources"] == []


def test_chat_saves_conversation_history(client, monkeypatch, mock_collection, db_session):
    """验证/chat接口调用之后，用户提问和AI回答都被正确存进了数据库"""
    from conversation import Conversation

    mock_collection.count.return_value = 0
    _mock_deepseek_reply(monkeypatch, "这是回答内容")

    client.post(
        "/chat",
        json={"question": "这是我的问题", "session_id": "chat-test-4"},
    )

    records = (
        db_session.query(Conversation)
        .filter_by(session_id="chat-test-4")
        .order_by(Conversation.id)
        .all()
    )
    assert len(records) == 2
    assert records[0].role == "user"
    assert records[0].content == "这是我的问题"
    assert records[1].role == "assistant"
    assert records[1].content == "这是回答内容"


def test_chat_uses_default_session_id_when_not_provided(client, monkeypatch, mock_collection):
    """session_id是可选参数，不传的话main.py里定义的默认值应该是"default" """
    mock_collection.count.return_value = 0
    _mock_deepseek_reply(monkeypatch, "回答")

    resp = client.post("/chat", json={"question": "问题"})

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "default"
