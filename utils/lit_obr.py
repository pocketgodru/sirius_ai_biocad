from mistralai import Mistral
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from transformers import AutoTokenizer
import json
from json_repair import repair_json


tokenizer = AutoTokenizer.from_pretrained("mistral-community/pixtral-12b")


api_key = 'vjOgcQPigpidK7njWV5jPidP69CHg5Yg'
model = "pixtral-12b-2409"
client = Mistral(api_key=api_key)
client_4 = ChatMistralAI(api_key=api_key, model=model)
def count_tokens_in_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
    return len(tokens["input_ids"][0])
prom = """
#####
# Выведи итог строго в формате JSON. Убедись, что:
# - JSON является валидным и имеет правильную вложенность.
# - Все строки (ключи и значения) заключены в двойные кавычки.
# - Нет лишних запятых.
# - Используй формат структуры, приведённой ниже.
#####
{
  "comparison_table": {
    "markdown": "| article title | criterion name 1 | criterion name 2 | criterion name 3 |\n|---------------|------------------|------------------|------------------|\n| article title 1 | result | result | result |\n| article title 2 | result | result | result |\n| article title 3 | result | result | result |"
  },
  "quotes": {
    "criterion name 1": {
      "article title 1": "citation",
      "article title 2": "citation",
      "article title 3": "citation"
    },
    "criterion name 2": {
      "article title 1": "citation",
      "article title 2": "citation",
      "article title 3": "citation"
    },
    "criterion name 3": {
      "article title 1": "citation",
      "article title 2": "citation",
      "article title 3": "citation"
    }
  },
  "conclusion": "result"
}
#####
# Убедись, что:
# - Поле "comparison_table.markdown" содержит корректно отформатированную таблицу с заголовками и данными.
# - Поля "quotes" содержат цитаты по указанным критериям для каждой статьи.
# - Поле "conclusion" включает краткое заключение о сравнении статей.
# 
# Если есть неуверенность, уточни формат или структуру перед генерацией.
#####
"""
def process_scientific_articles_for_analysis_1(text, criter_prompts=""):
    promt = f"""Анализируйте научные статьи на основе критериев, предоставленных пользователем. Извлеките соответствующие данные из текста и представьте краткий сравнительный обзор.
        Предоставьте краткий обзор литературы в следующем формате в виде таблицы, включая названия статей а не их индексы в строке сравнения.

        Представим сравнения в виде таблицы, где:

        В первом вертикальном столбце указаны названия статей в сокращенном виде, не теряя смысла, и ни в каком другом формате.
        Последующие столбцы представляют параметры, представленные ниже.
        Строки содержат краткие цитаты, извлеченные из текста.
        Дополнительно под таблицей приведите прямые цитаты из текста без каких-либо обобщений и изменений, подтверждающие данные, представленные в таблице. Эти цитаты должны состоять только из предложений из текста, без учета дат публикации и имен авторов. Если данных нет, укажите «Данные отсутствуют». Каждую цитату размещайте в отдельной строке под соответствующим критерием в таблице, группируйте цитаты по статьям и указывайте названия статей не индексы. Напишите цитаты на том языке, на котором они встречаются в тексте.

        Обязательно предоставь полезный и четкий вывод. 

        Нумерацию статей начните с первой исключая ноль. Не включать в цитаты авторов и даты публикации статей, не нумеровать каждую строку цитаты, а представлять каждую цитату с новой строки:
        {criter_prompts}

        
        Статьи:
        {text}

        Приведи краткий обзор литературы в следующем формате:
        {prom}
        """


    chat_response = client.chat.complete(
        model=model,
        messages= [{ "role": "user", "content":  [{ "type": "text", "text": promt}] }]
    )

    return chat_response.choices[0].message.content
def process_scientific_articles_for_analysis_2(text, images=[], criter_prompts=""):
    map_template = f"""
        {{docs}}
        Analyze the scientific articles based on the criteria provided by the user. Extract the relevant data from the text and present a concise comparative review.
        Provide a summary literature review in the following format as a table, including the article titles (not their indices) in the comparison row.

        Present the comparisons in the form of a table where:

        The first vertical column lists the titles of the articles, shortened without losing their meaning, and in no other format.
        Subsequent columns represent the parameters provided below.
        Rows contain concise quotes extracted from the text.
        Additionally, below the table, provide direct quotes from the text without any summarization or changes that confirm the data presented in the table. These quotes must consist only of sentences from the text, excluding publication dates and author names. If no data is available, state "No data available." Present each quote on a separate line under the corresponding criterion in the table, group the quotes by article, and include the article titles (not indices). Write the quotes in the language in which they appear in the text.

        Start numbering the articles from the first (excluding zero). Do not include the authors or publication dates of the articles in the quotes, do not number each quote line, but present each quote on a new line:
        {criter_prompts}
        
        Give a brief literature review in the following format:
        {prom}
    """

    map_template = map_template + '\n' + criter_prompts

    reduce_template = f"""Следующий текст состоит из нескольких кратких итогов:
        {{docs}}

        На основе этих кратких итогов, проведи анализ научных статей по введенным критериям, объединяя основные данные и выводя обобщающий литературный обзор.
        Выведи результат в следующем формате:

        1. Таблица, где:
        - Первая колонка по вертикали — это названия статей (сокращенные без потери смысла).
        - Последующие колонки — это критерии анализа.
        - Строки содержат краткие данные по тексту каждой статьи, соответствующие критериям.

        2. Под таблицей укажи прямые цитаты из текста, подтверждающие данные в таблице. Каждую цитату:
        - Группируй по статьям.
        - Пиши на языке оригинала текста.
        - Не включай авторов и даты написания статьи.
        - Если данных нет, укажи "Данных нет".

        Обязательно предоставь полезный и четкий вывод. 
        Результат:

        Приведи краткий обзор литературы в следующем формате:
        {prom}
    """


    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=client_4, prompt=map_prompt)

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=client_4, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=128000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=100000,
        chunk_overlap=14000,
    )

    split_docs = text_splitter.create_documents([text])

    image_descriptions = "\n".join(
        [f"Изображение {i+1}: {img['image_url']}" for i, img in enumerate(images)]
    )

    result = map_reduce_chain.run({"input_documents": split_docs, "images": image_descriptions})

    return result
def init(text_data, criter):

    criter_prompt = '\n'.join([f'{i+1}. {criter[i]}' for i in range(len(criter))])

    if count_tokens_in_text(text_data) < 128000:
        rezult = process_scientific_articles_for_analysis_1(text_data, criter_prompt)
    else:
        rezult = process_scientific_articles_for_analysis_2(text_data, criter_prompts = criter_prompt)

    return json.loads(repair_json(rezult[7:-4]))  #repair_json(rezult[7:-4])
    
