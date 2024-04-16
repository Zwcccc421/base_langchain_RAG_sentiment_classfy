import csv
import os

from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-qc9P7FUiYiDswqkU3z0F353HUIhP7MghopcGxfknEkd2f7el"
# 处理数据
data_by_dialogue_id = {}
with open('./test_sent_emo_50.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    headers = next(csv_reader)

    for row in csv_reader:
        dialogue_id = row[5]
        utterance = row[1]
        speaker = row[2]
        emotion = row[3]

        if dialogue_id in data_by_dialogue_id:
            data_by_dialogue_id[dialogue_id].append({
                'speaker': speaker,
                'utterance': utterance,
                'emotion': emotion
            })
        else:
            data_by_dialogue_id[dialogue_id] = [{
                'speaker': speaker,
                'utterance': utterance,
                'emotion': emotion
            }]

# 提取文档csv信息
documents = CSVLoader(file_path='./train_sent_emo_100_random.csv').load()
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
splits = text_splitter.split_documents(documents)
# 对分块内容进行嵌入并存储块
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(splits, embeddings)

# 检索  由于检索函数只能input string所以使用这个检索器不能实现优先同一说话者
retriever = db.as_retriever()

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=2048)

# 构建模板
template = """
Now you're an expert at emotion analysis. Select an emotion label as the answer for the question based on the following conversation content and context information.
Pay attention to the context of the conversation and the mood of the punctuation at the end of the sentence.
Context is the mood of the conversation similar to my question. You can choose whether to refer to context or not.
conversation: {conversation}
context: {context}
question: {question}
Emotion label selection: < joy, surprise, sadness, neutral, anger, disgust, fear>
Only the label needs to be included in the answer.
answer: """
# 初始化情绪标签的统计字典
emotion_stats = {'joy': {'correct': 0, 'total': 0, 'predicted': 0},
                 'surprise': {'correct': 0, 'total': 0, 'predicted': 0},
                 'sadness': {'correct': 0, 'total': 0, 'predicted': 0},
                 'neutral': {'correct': 0, 'total': 0, 'predicted': 0},
                 'anger': {'correct': 0, 'total': 0, 'predicted': 0},
                 'disgust': {'correct': 0, 'total': 0, 'predicted': 0},
                 'fear': {'correct': 0, 'total': 0, 'predicted': 0}}
emotion_labels = ['joy', 'sadness', 'surprise', 'neutral', 'anger', 'disgust', 'fear']
for dialogue_id, data in data_by_dialogue_id.items():
    print(f"Dialogue_ID: {dialogue_id}")
    conversation = ""
    for item in data:
        conversation += item['speaker'] + ": " + item['utterance'] + "\n"
        template = template.format(conversation=conversation, context="{context}", question="{question}")
        prompt = ChatPromptTemplate.from_template(template)
        # 构建一个RAG流程链条，将检索器、提示模板和LLM连接起来
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        question = item['speaker'] + ": " + item['utterance']
        answer = rag_chain.invoke(question)
        if answer == item['emotion']:
            emotion_stats[item['emotion']]['correct'] += 1
        emotion_stats[item['emotion']]['total'] += 1
        if any(label in answer for label in emotion_labels):
            emotion_stats[answer]['predicted'] += 1

    # for item in data:
    #     question = item['speaker'] + ": " + item['utterance']
    #     answer = rag_chain.invoke(question)
    #     # print(question + " " + item['emotion'] + " " + answer)
    #     if answer == item['emotion']:
    #         emotion_stats[item['emotion']]['correct'] += 1
    #     emotion_stats[item['emotion']]['total'] += 1
    #     if any(label in answer for label in emotion_labels):
    #         emotion_stats[answer]['predicted'] += 1

# 计算每个情绪标签的准确率、召回率和 F1 分数
total_correct = 0
total_total = 0
total_predicted = 0
for emotion, stats in emotion_stats.items():
    precision = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0
    recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print(f"Emotion: {emotion}\n, Precision: {precision}\n, Recall: {recall}\n, F1 Score: {f1}\n")
    total_correct += stats['correct']
    total_total += stats['total']
    total_predicted += stats['predicted']

average_accuracy = total_correct / total_total if total_total > 0 else 0
# average_precision = total_correct / total_predicted if total_predicted > 0 else 0
# average_recall = total_correct / total_total if total_total > 0 else 0
# if average_precision + average_recall > 0:
#     average_f1 = 2 * (average_precision * average_recall) / (average_precision + average_recall)
# else:
#     average_f1 = 0
print(f"Average accuracy: {average_accuracy}")
# print(f"Average Precision: {average_precision}")
# print(f"Average Recall: {average_recall}")
# print(f"Average F1 Score: {average_f1}")
