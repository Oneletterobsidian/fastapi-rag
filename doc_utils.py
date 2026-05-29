def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def parse_document(path: str) -> str:
    """根据文件后缀自动选择解析方式"""
    if path.endswith(".txt"):
        return read_txt(path)
    elif path.endswith(".pdf"):
        return read_pdf(path)
    elif path.endswith(".docx"):
        return read_docx(path)
    else:
        raise ValueError(f"不支持的文件格式：{path}")

def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    把长文本切成固定大小的块，相邻块之间有重叠
    chunk_size: 每块最大字符数
    overlap:    相邻块重叠的字符数（避免在句子中间切断导致语义丢失）
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks