from mistralai import Mistral
from langchain_community.tools import TavilySearchResults , JinaSearch
import concurrent.futures
import json
from utils.search_article import *
import os
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = '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'

api_key_2 = 'VPqG8sCy3JX5zFkpdiZ7bRSnTLKwngFJ'
client_2 = Mistral(api_key=api_key_2)

api_key_3 = 'cvyu5Rdk2lS026epqL4VB6BMPUcUMSgt'
client_3 = Mistral(api_key=api_key_3)

# Настройка Tavily Search
def setup_search(question):
    tavily_tool = TavilySearchResults(max_results=20,)
    return tavily_tool.invoke({"query": f"{question}"})
    #tool = JinaSearch()
    #return json.loads(str(tool.invoke({"query": f"{question}"})))

def ask_question_to_mistral(text, question, images=[]):
    prompt = f"Answer the following question without mentioning it or repeating the original text on which the question is asked in style markdown:\nQuestion: {question}\n\nText:\n{text}"
    
    message_content = [{"type": "text", "text": prompt}] + images

    search_tool = setup_search(question)
    context = ''
    for result in search_tool:
        context += f"{result['url']} : {result['content']} \n"
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        extract_future = executor.submit(init, text, images)
        response = client_2.chat.complete(
            model="pixtral-12b-2409",
            messages=[{"role": "user", "content": f'{message_content}\n\nAdditional Context from Web Search:\n{context}' }],
        )
        key_topics , result_article_json = extract_future.result() 
    
    return response.choices[0].message.content, key_topics, result_article_json

def process_article_for_summary(text, images=[], compression_percentage=30):
    prompt = f"""
    You are a commentator.
    # article:
    {text}

    # Instructions:
    ## Summarize:
    In clear and concise language, summarize the key points and themes presented in the article by cutting it by {compression_percentage} percent in the markdown format.

    """
    message_content = [{"type": "text", "text": prompt}] + images
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        extract_future = executor.submit(init, text, images)
        response = client_3.chat.complete(
            model="pixtral-12b-2409",
            messages=[{"role": "user", "content": message_content}]
        )
        key_topics , result_article_json = extract_future.result() 
    return response.choices[0].message.content, key_topics, result_article_json
