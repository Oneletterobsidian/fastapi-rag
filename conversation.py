# conversation.py
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from database import Base

class Conversation(Base):
    __tablename__ = "conversations"
    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)   # 会话 ID，区分不同对话
    role       = Column(String)               # "user" 或 "assistant"
    content    = Column(Text)                 # 消息内容
    created_at = Column(DateTime, default=func.now())