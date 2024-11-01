import json
import concurrent.futures
from mistralai import Mistral
import arxiv  # Using arxiv.py for article search

api_key_1 = 'eLES5HrVqduOE1OSWG6C5XyEUeR7qpXQ'
client_1 = Mistral(api_key=api_key_1)

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

# Function to search relevant articles with arxiv.py
def search_relevant_articles_arxiv(key_topics, max_articles=100):
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
    # 1. Extract key topics from text
    key_topics = extract_key_topics(content, images)
    key_topics = [topic.strip("- ") for topic in key_topics.split("\n") if topic]

    # 2. Find relevant articles based on extracted topics using arxiv.py
    articles_by_topic, final_topics = search_relevant_articles_arxiv(key_topics)

    # 3. Output the result as JSON
    result_json = json.dumps(articles_by_topic, indent=4)

    return final_topics, result_json

# Example usage:
# topics, json_output = init("Your text content here")
# print("Topics:", topics)
# print("JSON output:", json_output)