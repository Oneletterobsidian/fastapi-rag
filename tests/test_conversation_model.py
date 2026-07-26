"""
test_conversation_model.py

测试Conversation这个SQLAlchemy模型本身能不能正常存取数据。
用db_session这个fixture直接操作数据库（跳过HTTP接口那一层），
专注验证"数据模型本身对不对"，跟"接口逻辑对不对"分开测试。
"""

from conversation import Conversation


def test_create_and_query_conversation(db_session):
    """最基本的：能存进去，也能按session_id查出来"""
    convo = Conversation(session_id="test-session-1", role="user", content="你好")
    db_session.add(convo)
    db_session.commit()

    result = (
        db_session.query(Conversation)
        .filter_by(session_id="test-session-1")
        .first()
    )

    assert result is not None
    assert result.role == "user"
    assert result.content == "你好"
    # created_at有默认值（func.now()），插入之后应该自动填上，不应该是None
    assert result.created_at is not None


def test_multiple_messages_in_same_session_ordered_by_id(db_session):
    """验证同一个会话里存多条消息，按id顺序取出来的顺序是对的
    （main.py里/chat接口拿历史记录时依赖这个顺序）"""
    db_session.add(Conversation(session_id="test-session-2", role="user", content="第一句"))
    db_session.add(Conversation(session_id="test-session-2", role="assistant", content="第二句"))
    db_session.add(Conversation(session_id="test-session-2", role="user", content="第三句"))
    db_session.commit()

    results = (
        db_session.query(Conversation)
        .filter_by(session_id="test-session-2")
        .order_by(Conversation.id)
        .all()
    )

    assert len(results) == 3
    assert [r.content for r in results] == ["第一句", "第二句", "第三句"]


def test_different_sessions_are_isolated(db_session):
    """不同session_id之间的消息不应该互相串"""
    db_session.add(Conversation(session_id="session-a", role="user", content="属于A"))
    db_session.add(Conversation(session_id="session-b", role="user", content="属于B"))
    db_session.commit()

    results_a = db_session.query(Conversation).filter_by(session_id="session-a").all()

    assert len(results_a) == 1
    assert results_a[0].content == "属于A"
