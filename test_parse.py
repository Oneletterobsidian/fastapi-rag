"""测试 doc_utils 的解析和切分功能"""
from doc_utils import parse_document, split_text

# 测试（先用一个 txt 文件验证）
text = parse_document("test.txt")
print(f"读取到 {len(text)} 个字符")
print(text[:200])

# 测试切分
chunks = split_text(text)
print(f"共切出 {len(chunks)} 个块")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} ({len(chunk)} 字符) ---")
    print(chunk)