import csv
import os

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-qc9P7FUiYiDswqkU3z0F353HUIhP7MghopcGxfknEkd2f7el"

# 处理数据
data_by_dialogue_id = {}
with open('./test_sent_emo_10.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)

    for row in csv_reader:
        dialogue_id = row[5]
        utterance = row[1]
        speaker = row[2]
        emotion = row[3]
        sentiment = row[4]

        if dialogue_id in data_by_dialogue_id:
            data_by_dialogue_id[dialogue_id].append({
                'speaker': speaker,
                'utterance': utterance,
                'emotion': emotion,
                'sentiment': sentiment
            })
        else:
            data_by_dialogue_id[dialogue_id] = [{
                'speaker': speaker,
                'utterance': utterance,
                'emotion': emotion,
                'sentiment': sentiment
            }]

# 提取文档csv信息
documents = CSVLoader(file_path='./train_sent_emo_100_random.csv').load()
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
splits = text_splitter.split_documents(documents)
# 对分块内容进行嵌入并存储块
embeddings = OpenAIEmbeddings()

# 创建带有说话人信息的 Chroma 向量存储
db = Chroma.from_documents(splits, embeddings)

# 检索器
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=50)

# 构建模板
Template = """
Now you're an expert at emotion analysis. Complete the following tasks I have given you.
chat_history: {chat_history}
question: {human_input}
Chatbot: """
Input1 = """
Select an emotion label as the answer for the utterance based on the following conversation content and context information.
Pay attention to the context of the conversation and the mood of the punctuation at the end of the sentence.
Conversation is historical conversation information. Context is the mood of the conversation alike to my utterance. You can choose whether to refer to context or not.
conversation: {conversation}
context: {context}
utterance: {utterance}
Emotion label selection: < joy, surprise, sadness, neutral, anger, disgust, fear>
Only the emotion label needs to be included in the answer.
answer:
"""
Input2 = """
preUtterance: {preUtterance}
This is the last line of utterance where I want to identify emotions. 
Please help me analyze the possible intentions of the speaker and the possible reactions of the listener based on preUtterance. Just give a short answer
"""
input3 = """
Please explain what you think is the reason for this emotion.You can refer to intentions, reactions, conservation and context.
Just give a short answer
"""
# 初始化 Prompt 对象
output_parser = StrOutputParser()

# 初始化情绪标签的统计字典
emotion_stats = {'joy': {'correct': 0, 'total': 0, 'predicted': 0},
                 'surprise': {'correct': 0, 'total': 0, 'predicted': 0},
                 'sadness': {'correct': 0, 'total': 0, 'predicted': 0},
                 'neutral': {'correct': 0, 'total': 0, 'predicted': 0},
                 'anger': {'correct': 0, 'total': 0, 'predicted': 0},
                 'disgust': {'correct': 0, 'total': 0, 'predicted': 0},
                 'fear': {'correct': 0, 'total': 0, 'predicted': 0}}
emotion_labels = ['joy', 'sadness', 'surprise', 'neutral', 'anger', 'disgust', 'fear']

# 把得到的提示保存到tips_data中
tips_data = []

# 开始对每个对话进行提问
for dialogue_id, data in data_by_dialogue_id.items():
    print(f"Dialogue_ID: {dialogue_id}")
    conversation = ""
    pre_utterance = ""
    for item in data:
        conversation += item['speaker'] + ": " + item['utterance'] + "\n"
        utterance = item['speaker'] + ": " + item['utterance']
        docs = retriever.get_relevant_documents(utterance)
        target_emotion = "Emotion: " + item['emotion']
        # 遍历 docs 列表
        for doc in docs:
            if target_emotion in doc.page_content:
                docs[0] = doc
                break
        # 取docs[0]:如果存在这个emotion的话，docs[0]即是这个emotion中最相似的语句；如果没找到这个emotion的话，docs[0]是所有中最相似的语句
        alike_utterance = docs[0].page_content.replace("\n", ". ")
        prompt = PromptTemplate(template=Template,
                                input_variables=["chat_history", "human_input"],
                                output_parser=output_parser)
        memory = ConversationBufferMemory(memory_key="chat_history")
        # llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, memory=memory)
        llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
        # 第一个问题
        input1 = Input1.format(conversation=conversation, context=alike_utterance, utterance=utterance)
        answer1 = llm_chain.predict(human_input=input1)
        # 第二个问题
        input2 = Input2.format(preUtterance=pre_utterance)
        answer2 = llm_chain.predict(human_input=input2)
        # 第三个问题
        answer3 = llm_chain.predict(human_input=input3)
        # 保存上一句对话
        pre_utterance = utterance
        # print("answer1:", answer1)
        # print("answer2:", answer2)
        # print("answer3:", answer3)
        # 把提示存到tips_data中
        tips_data.append({
            'Utterance': item['utterance'],
            'Speaker': item['speaker'],
            'Emotion': item['emotion'],
            'Sentiment': item['sentiment'],
            'Dialogue_ID': dialogue_id,
            'tips': answer3
        })
        if len(answer1) > 10:
            continue
        if answer1 == item['emotion']:
            emotion_stats[item['emotion']]['correct'] += 1
        emotion_stats[item['emotion']]['total'] += 1
        if any(label in answer1 for label in emotion_labels):
            emotion_stats[answer1]['predicted'] += 1
    total_correct = 0
    total_total = 0
    for emotion, stats in emotion_stats.items():
        total_correct += stats['correct']
        total_total += stats['total']
    average_accuracy = total_correct / total_total if total_total > 0 else 0
    print(f"Average accuracy: {average_accuracy}")

# 写入CSV文件
csv_file_path = "./output1.csv"  # 保存的CSV文件路径
csv_columns = ['Utterance', 'Speaker', 'Emotion', 'utterance', 'Sentiment', 'Dialogue_ID', 'tips']  # CSV文件的列名

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()  # 写入列名
    for data in tips_data:
        writer.writerow(data)  # 逐行写入数据
# # 计算每个情绪标签的准确率、召回率和 F1 分数
# total_correct = 0
# total_total = 0
# total_predicted = 0
# for emotion, stats in emotion_stats.items():
#     precision = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0
#     recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#     print(f"Emotion: {emotion}\n, Precision: {precision}\n, Recall: {recall}\n, F1 Score: {f1}\n")
#     total_correct += stats['correct']
#     total_total += stats['total']
#     total_predicted += stats['predicted']
#
# average_accuracy = total_correct / total_total if total_total > 0 else 0
# print(f"Average accuracy: {average_accuracy}")
