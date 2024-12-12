from gradio_client import Client

def summarize_text(text_input, preferences, images_base64="", compression_percentage=30):
    client = Client("Belemort/test_biocad")
    result = client.predict(
		text_input=text_input + preferences,
		images_base64=images_base64,
		task="Summarization",
		question="",
		crit="",
		compression_percentage=compression_percentage,
		api_name="/gradio_interface"
)

    return result

def answer_question(text_input, question, images_base64=""):
    client = Client("Belemort/test_biocad")
    result = client.predict(
		text_input=text_input,
		images_base64=images_base64,
		task="Question Answering",
		question=question,
		crit="",
		compression_percentage=30,
		api_name="/gradio_interface"
    )
    print("Question Answering Result:", result)
    return result

def literature_review(text_data, criteria):
    client = Client("Belemort/test_biocad")

    criter_prompt = '\n'.join([f'{i+1}. {criteria[i]}' for i in range(len(criteria))])

    result = client.predict(
		text_input=text_data,
		images_base64=" ",
		task="Lit Obzor",
		question="",
		crit=criter_prompt,
		compression_percentage=30,
		api_name="/gradio_interface"
)
    return result

def search_article(text_data):
    client = Client("Belemort/test_biocad")
    result = client.predict(
            text_input=text_data,
            images_base64="",
            task="Search Article",
            question="",
            crit="",
            compression_percentage=0,
            api_name="/gradio_interface"
    )
    
    return result
