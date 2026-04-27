# src/create_db.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 设置 HuggingFace 镜像（加速模型下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'

# ==================== 路径设置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # src 文件夹
BASE_DIR = os.path.dirname(SCRIPT_DIR)                    # 项目根目录 My_RAG

DOC_PATH = os.path.join(BASE_DIR, "data", "python_notes.md")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

print(f"📁 项目根目录: {BASE_DIR}")
print(f"📄 正在加载文档: {DOC_PATH}")

# 检查文件是否存在（重要！避免报错）
if not os.path.exists(DOC_PATH):
    print("❌ 错误：找不到 python_notes.md 文件！")
    print(f"   当前查找路径: {DOC_PATH}")
    print(f"   请确认文件位于: {os.path.join(BASE_DIR, 'data')}")
    # 显示 data 文件夹里实际有什么文件
    data_dir = os.path.join(BASE_DIR, "data")
    if os.path.exists(data_dir):
        print(f"   data 文件夹里的文件: {os.listdir(data_dir)}")
    raise FileNotFoundError(f"文件不存在: {DOC_PATH}")

print("✅ 文件路径检查通过，开始创建向量数据库...")

# 1. 加载文档
loader = TextLoader(file_path=DOC_PATH, encoding="utf-8")
docs = loader.load()
print(f"✅ 文件加载成功！共加载 {len(docs)} 个文档")

# 2. 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"✅ 文档切分完成！共切分成 {len(chunks)} 个片段")

# 3. 嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. 创建 Chroma 向量数据库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("🎉 Chroma向量数据库创建成功！")
print(f"💾 数据库保存位置: {DB_PATH}")
print(f"   共存储了 {len(chunks)} 个向量片段")