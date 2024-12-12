import json
import concurrent.futures
from mistralai import Mistral
import arxiv  # Using arxiv for article search
from transformers import AutoTokenizer
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_mistralai import ChatMistralAI

api_key_1 = 'eLES5HrVqduOE1OSWG6C5XyEUeR7qpXQ'
client_1 = Mistral(api_key=api_key_1)

llm = ChatMistralAI(api_key=api_key_1, model="pixtral-12b-2409")

tokenizer = AutoTokenizer.from_pretrained("mistral-community/pixtral-12b")
def count_tokens_in_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
    return len(tokens["input_ids"][0])

# Function to analyze text and extract key topics
def extract_key_topics(content, images=[]):
    prompt = f"""
    Extract the primary themes from the text below. List each theme in as few words as possible, focusing on essential concepts only. Format as a concise, unordered list with no extraneous words.

    ```{content}```

    LIST IN ENGLISH:
    - 
    """
    message_content = [{"type": "text", "text": prompt}] + images

    response = client_1.chat.complete(
        model="pixtral-12b-2409",
        messages=[{"role": "user", "content": message_content}]
    )
    return response.choices[0].message.content


def extract_key_topics_with_large_text(content, images=[]):
    # Map prompt template for extracting key themes
    map_template = f"""
        Текст: {{docs}}
        Изображения: {{images}}

        Extract the primary themes from the text below. List each theme in as few words as possible, focusing on essential concepts only. Format as a concise, unordered list with no extraneous words.
        LIST IN ENGLISH:
        - 

        :"""

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce prompt template to further refine and extract key themes
    reduce_template = f"""Следующий текст состоит из нескольких кратких итогов:
        {{docs}}

        Extract the primary themes from the text below. List each theme in as few words as possible, focusing on essential concepts only. Format as a concise, unordered list with no extraneous words.
        LIST IN ENGLISH:
        - 

        :"""

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

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
    split_docs = text_splitter.create_documents([content])

    # Include image descriptions (optional, if required by the prompt)
    image_descriptions = "\n".join(
        [f"Изображение {i+1}: {img['image_url']}" for i, img in enumerate(images)]
    )

    # Run the summarization chain to extract key themes
    key_topics = map_reduce_chain.run({"input_documents": split_docs, "images": image_descriptions})
    return key_topics

# Function to search relevant articles with arxiv.py
def search_relevant_articles_arxiv(key_topics, max_articles=10):
    articles_by_topic = {}
    final_topics = []

    def fetch_articles_for_topic(topic):
        topic_articles = []
        try:
            # Fetch articles using arxiv.py based on the topic
            search = arxiv.Search(
                query=topic,
                max_results=max_articles,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in search.results():
                article_data = {
                    "title": result.title,
                    "doi": result.doi,
                    "summary": result.summary,
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url
                }
                topic_articles.append(article_data)
            final_topics.append(topic)
        except Exception as e:
            print(f"Error fetching articles for topic '{topic}': {e}")

        return topic, topic_articles

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Use threads to fetch articles for each topic
        futures = {executor.submit(fetch_articles_for_topic, topic): topic for topic in key_topics}
        for future in concurrent.futures.as_completed(futures):
            topic, articles = future.result()
            if articles:
                articles_by_topic[topic] = articles

    return articles_by_topic, list(set(final_topics))

# Initialization function
def init(content, images=[]):

    if len(images) >= 8 : 
        images = images[:7]

    if count_tokens_in_text(content) < 128000:
        key_topics = extract_key_topics(content, images) 
        key_topics = [topic.strip("- ") for topic in key_topics.split("\n") if topic]

        # 2. Find relevant articles based on extracted topics using arxiv.py
        articles_by_topic, final_topics = search_relevant_articles_arxiv(key_topics)

        # 3. Output the result as JSON
        result_json = json.dumps(articles_by_topic, indent=4)

        return final_topics, result_json
    else:
        key_topics = extract_key_topics_with_large_text(content, images) 
        key_topics = [topic.strip("- ") for topic in key_topics.split("\n") if topic]

        # 2. Find relevant articles based on extracted topics using arxiv.py
        articles_by_topic, final_topics = search_relevant_articles_arxiv(key_topics)

        # 3. Output the result as JSON
        result_json = json.dumps(articles_by_topic, indent=4)

        return final_topics, result_json



# Example usage:
# topics, json_output = init(text)
# print("Topics:", topics)
# print("JSON output:", len(json_output))
