import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 加载 .env 文件里的环境变量
load_dotenv()

# 2. 从环境变量读 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

# 3. 创建客户端,base_url 指向 DeepSeek
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# 4. 发起一次对话
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好,用一句话介绍你自己"}
    ],
)

# 5. 取出回答
answer = response.choices[0].message.content
print("DeepSeek 回答:", answer)