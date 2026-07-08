# demo.py
import streamlit as st
import requests

st.set_page_config(page_title="RAG 文档问答系统", layout="wide")
st.title("📄 RAG 文档问答系统")
st.caption("基于 FastAPI + 向量检索 | fastapi-rag")

# 侧边栏：展示你的技术亮点，面试官会看到
with st.sidebar:
    st.markdown("**技术栈**")
    st.markdown("- FastAPI 后端\n- 向量检索优化\n- 检索距离分离度：0.30 → 0.70")

query = st.text_input("请输入你的问题", placeholder="例如：这份文档讲了什么？")

if st.button("提问", type="primary"):
    if query:
        with st.spinner("检索中..."):
            response = requests.post("http://localhost:8000/chat", json={"question": query, "session_id": "default"})
            result = response.json()
        st.markdown("### 回答")
    result = response.json()
    st.write(result["answer"])
    if result["sources"]:
        st.caption(f"参考来源：{', '.join(result['sources'])}")