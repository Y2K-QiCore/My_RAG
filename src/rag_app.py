# src/rag_app.py
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
import gradio as gr

# ==================== 路径设置（和 create_db.py 保持一致） ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '1'

print(f"📁 项目根目录: {BASE_DIR}")
print(f"💾 正在加载向量数据库: {DB_PATH}")

# 1. 加载向量数据库
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",   # 也可以试 "mmr"
    search_kwargs={"k": 6}      # 每次检索返回 6 个最相关的片段
)

print(f"✅ 向量数据库加载成功！当前片段数量: {vectorstore._collection.count()}")

# 2. 构建 Prompt（你可以继续优化这个提示词）
template = """你是一个友好、专业、耐心且幽默的 Python 编程助手。
请严格基于以下上下文内容来回答用户的问题。
如果上下文无法回答，请诚实地说“我在笔记中没有找到相关内容”，不要编造答案。

上下文：
{context}

问题：{question}

回答（用中文，清晰易懂）："""

prompt = ChatPromptTemplate.from_template(template)

# 3. 初始化 DeepSeek LLM
llm = ChatDeepSeek(
    model="deepseek-chat",      # 或者 "deepseek-reasoner" 如果你想用推理模型
    temperature=0.7,
    api_key="sk-9c5be891476c4431ab0206e357200370"
    # 如果你有 DeepSeek API Key，可以在这里设置或通过环境变量 DEEPSEEK_API_KEY
)

# 4. 构建 RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Gradio 聊天函数
def answer_question(message, history):
    try:
        response = rag_chain.invoke(message)
        return response
    except Exception as e:
        return f"抱歉，发生错误：{str(e)}"

# 6. 启动 Gradio 界面
demo = gr.ChatInterface(
    fn=answer_question,
    title="🐍 Python 助手",
    description="基于你的 python_notes.md 笔记回答 Python 相关问题（使用 DeepSeek + Chroma）",
    #theme="soft",
    examples=[
        "Python 中列表和元组有什么区别？",
        "如何用装饰器实现一个简单的计时器？",
        "解释一下 Python 的 GIL 是什么？",
        "给我一个异步爬虫的简单例子"
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    print("🚀 启动 Gradio 界面...")
    demo.launch(
        server_name="127.0.0.1",   # 局域网可访问
        server_port=7860,        # 默认端口
        share=False              # 改为 True 可以生成公网链接（临时）
    )