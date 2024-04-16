import os
# import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate

os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-sVWmwYZ3Gc7sLBXy6MWanG5z4MooxpgW81wLKso75S5XpHjb"

# # 提取网页信息
# url = "https://myzhengyuan.com/post/93853.html"
# res = requests.get(url)
# with open("93853.html","w") as f:
#     f.write(res.text)
#
# loader = TextLoader('./93853.html')
# documents = loader.load()

# 提取文档txt信息
documents = TextLoader("./senticnet_zh/senticnet1.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
# chunk_size：表示文本分割的块大小, chunk_overlap：表示分割出的文本块之间重叠的部分大小。
splits = text_splitter.split_documents(documents)

# 对分块内容进行嵌入并存储块
embeddings = OpenAIEmbeddings()
# 这里可以用的存储方式有很多eg：FAISS，Chroma，weaviate等
db = Chroma.from_documents(splits, embeddings)

# 检索
retriever = db.as_retriever()

# 构建模板
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
Context:{context}
Question: {question}
Helpful Answer:"""

# # 构建初始 Messages 列表  用于对话大模型
# messages = [
#     SystemMessagePromptTemplate.from_template(template),
#     HumanMessagePromptTemplate.from_template('{question}')
# ]

# 使用ChatPromptTemplate来创建一个提示模板
prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=2048)
# max_tokens参数指定了模型生成文本时允许的最大token数

# 构建一个RAG流程链条，将检索器、提示模板和LLM连接起来
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)
query = "adj被信以为真_想象是个积极的词语还是消极的呢？为什么？你所依赖的情感词典是我给你的上下文片段吗？"

print(rag_chain.invoke(query))
