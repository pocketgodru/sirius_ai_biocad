from mistralai import Mistral
from langchain_community.tools import TavilySearchResults , JinaSearch
import concurrent.futures
import json
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer
from utils.search_article import *
import os

tokenizer = AutoTokenizer.from_pretrained("mistral-community/pixtral-12b")

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = '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'

# Инициализация нескольких клиентов Mistral
#api_key_1 = 'eLES5HrVqduOE1OSWG6C5XyEUeR7qpXQ'
#client_1 = Mistral(api_key=api_key_1)

api_key_2 = 'VPqG8sCy3JX5zFkpdiZ7bRSnTLKwngFJ'
client_2 = Mistral(api_key=api_key_2)

api_key_3 = 'cvyu5Rdk2lS026epqL4VB6BMPUcUMSgt'
client_3 = Mistral(api_key=api_key_3)

api_key_4 = 'lCZWDjyQSEc5gJsATEcKjP9cCjWsB7lg'
client_4 = ChatMistralAI(api_key=api_key_4, model="pixtral-12b-2409")


def count_tokens_in_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
    return len(tokens["input_ids"][0])

# Настройка Tavily Search
def setup_search(question):
    try:
        tavily_tool = TavilySearchResults(max_results=20)
        results = tavily_tool.invoke({"query": f"{question}"})
        if isinstance(results, list):  # Проверка на корректный тип
            return results, 'tavily_tool'
        else:
            print("Unexpected format from TavilySearchResults:", results)
    except Exception as e:
        print("Error with TavilySearchResults:", e)
        
    # Попробовать JinaSearch, если TavilySearchResults не сработал
    try:
        jina_tool = JinaSearch()
        results = json.loads(str(jina_tool.invoke({"query": f"{question}"})))
        if isinstance(results, list):  # Проверка на корректный тип
            return results, 'jina_tool'
        else:
            print("Unexpected format from JinaSearch:", results)
    except Exception as e:
        print("Error with JinaSearch:", e)
        
    return [], ''  # Возвращаем пустой список, если оба метода не сработали

#Какие ещё есть модели на основе Transformer?
def ask_question_to_mistral(text, question, context , images=[]):
    prompt = f"Answer the following question without mentioning it or repeating the original text on which the question is asked in style markdown.IN RUSSIAN:\nQuestion: {question}\n\nText:\n{text}"
    
    message_content = [{"type": "text", "text": prompt}] + images
    
    response = client_2.chat.complete(
        model="pixtral-12b-2409",
        messages=[{"role": "user", "content": f'{message_content}\n\nAdditional Context from Web Search:\n{context}' }],
    )
    
    return response.choices[0].message.content

def process_article_for_summary(text, preferences , images=[], compression_percentage=30):
    prompt = f"""
    You are a commentator.
    # article:
    {text}

    # Instructions:
    ## Summarize:
    In clear and concise language, summarize the key points and themes presented in the article by cutting it by {compression_percentage} percent in the markdown format.
    Taking into account the user's wishes "{preferences}"
    """
    message_content = [{"type": "text", "text": prompt}] + images
    
    response = client_3.chat.complete(
        model="pixtral-12b-2409",
        messages=[{"role": "user", "content": message_content}]
    ).choices[0].message.content

    return response

def process_large_article_for_summary(text, preferences , images=[], compression_percentage=30):
    # Map prompt template
    map_template = f"""Следующий текст состоит из текста и изображений:
        Текст: {{docs}}
        Изображения: {{images}}

        На основе приведенного материала, выполните сжатие текста, выделяя основные темы и важные моменты. 
        Уровень сжатия: {compression_percentage}%. 
        С учетом пожеланий пользователя "{preferences}"
        Полезный ответ:"""

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=client_4, prompt=map_prompt)

    # Reduce prompt template
    reduce_template = f"""Следующий текст состоит из нескольких кратких итогов:
        {{docs}}

        На основе этих кратких итогов, выполните финальное сжатие текста, объединяя основные темы и ключевые моменты. 
        Уровень сжатия: {compression_percentage}%. 
        С учетом пожеланий пользователя "{preferences}"
        Результат предоставьте на русском языке в формате Markdown.

        Полезный ответ:"""

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=client_4, prompt=reduce_prompt)

    # Combine documents chain for Reduce step
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # ReduceDocumentsChain configuration
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=128000,
    )

    # MapReduceDocumentsChain combining Map and Reduce
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    # Text splitter configuration
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=100000,
        chunk_overlap=14000,
    )

    # Split the text into documents
    split_docs = text_splitter.create_documents([text])

    # Include image descriptions
    image_descriptions = "\n".join(
        [f"Изображение {i+1}: {img['image_url']}" for i, img in enumerate(images)]
    )

    summary = map_reduce_chain.run({"input_documents": split_docs, "images": image_descriptions})
    
    return summary

def ask_question_to_mistral_with_large_text(text, question, context , images=[]):
    # Prompts for QA
    map_template = f"""Следующий текст содержит статью/произведение:
    Текст: {{docs}}
    Изображения: {{images}}
    На основе приведенного текста, ответьте на следующий вопрос:

    Вопрос: {question}

    Ответ должен быть точным. Пожалуйста, ответьте на русском языке в формате Markdown.

    Информация из интернета:
    
    {context}

    Полезный ответ:"""

    reduce_template = """Следующий текст содержит несколько кратких ответов на вопрос:
    {{docs}}

    Объедините их в финальный ответ. Ответ предоставьте на русском языке в формате Markdown.

    Полезный ответ:"""

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=client_4, prompt=map_prompt)

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=client_4, prompt=reduce_prompt)

    # Combine documents chain for Reduce step
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # ReduceDocumentsChain configuration
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=128000,
    )

    # MapReduceDocumentsChain combining Map and Reduce
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    # Text splitter configuration
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=100000,
        chunk_overlap=14000,
    )

    # Split the text into documents
    split_docs = text_splitter.create_documents([text])

    # Include image descriptions
    image_descriptions = "\n".join(
        [f"Изображение {i+1}: {img['image_url']}" for i, img in enumerate(images)]
    )    

    answer = map_reduce_chain.run({"input_documents": split_docs, "question": question , "images": image_descriptions})
    
    return answer

def init_summ(text, preferences , images=[], compression_percentage=30):

    if len(images) >= 8 : 
        images = images[:7]

    if count_tokens_in_text(text=text) < 128_000:
        return process_article_for_summary(text, preferences , images, compression_percentage)
    else:
        return process_large_article_for_summary(text, preferences, images, compression_percentage)
    
def init_qa(text, question, images=[]):

    if len(images) >= 8 : 
        images = images[:7]

    search_tool, tool = setup_search(question)
    context = ''
    if search_tool:  # Проверка на то, что список результатов не пуст
        if tool == 'tavily_tool':
            for result in search_tool:
                context += f"{result.get('url', 'N/A')} : {result.get('content', 'No content')} \n"
        elif tool == 'jina_tool':
            for result in search_tool:
                context += f"{result.get('link', 'N/A')} : {result.get('snippet', 'No snippet')} : {result.get('content', 'No content')} \n"

    if count_tokens_in_text(text=(text + context)) < 128_000:
        return ask_question_to_mistral(text, question, context ,images)
    else:
        return ask_question_to_mistral_with_large_text(text, question, context , images)


'''
from gradio_client import Client

client = Client("Belemort/test_biocad")


def init_qa(text, question, images=[]):
    images_base64 = ",".join(images)  # Convert images list to a comma-separated string

    # Making a request to the Belemort/test_biocad API for Question Answering
    result = client.predict(
        text_input=text,
        images_base64=images_base64,
        task="Question Answering",
        question=question,
        api_name="/gradio_interface"
    )

    return result['Answer'], result['Topics'] , result['Articles']

def init_summ(text, preferences , images=[], compression_percentage=30):
    images_base64 = ",".join(images)  # Convert images list to a comma-separated string

    # Making a request to the Belemort/test_biocad API for Summarization
    result = client.predict(
        text_input=text,
        images_base64=images_base64,
        task="Summarization",
        question='',
        compression_percentage=compression_percentage,
        api_name="/gradio_interface"
    )
    
    return result['Summary'], result['Topics'] , result['Articles']
'''