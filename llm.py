from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch


STYLE = "Provide a clear and concise answer to the question based on the content of the text."
PROMPT_TRIGGER = "ANSWER THE QUESTION"
OUTPUT_LANGUAGE = "English"

'''
# Clear GPU memory if applicable
torch.cuda.empty_cache()

# Constants for the model and chunking
MODEL_CONTEXT_WINDOW = 32768
CHUNK_SIZE = 3000  # Size of each chunk in characters
CHUNK_OVERLAP = 500

# Style and language configurations for answering questions
STYLE = "Provide a clear and concise answer to the question based on the content of the text."
PROMPT_TRIGGER = "ANSWER THE QUESTION"


OUTPUT_LANGUAGE = "English"
VERBOSE = True

# Templates for answering questions prompts
combine_prompt_template = """
Based on the text provided, answer the following question.
{style}

Question: {quastion}



{content}


{trigger} in {language}:
"""

map_prompt_template = """
Based on the following portion of the text, answer the question:

Question: {quastion}

{text}

ANSWER THE QUESTION in {language}:
"""

# Function to answer a question for content that fits in the model's context window
def answer_question_base(llm, quastion, content: str) -> str:
    """Answer a question using the whole content at once that fits in model's context window."""
    prompt = PromptTemplate.from_template(combine_prompt_template)
    chain = prompt | llm
    return chain.invoke({"content": content , "quastion": quastion ,"style": STYLE , "trigger": PROMPT_TRIGGER , "language": OUTPUT_LANGUAGE})

# Function to answer a question using the map-reduce approach for larger content
def answer_question_map_reduce(llm, QUASTION, content: str) -> str:
    """Answer a question using content larger than model's context window with map-reduce approach."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.create_documents([content])

    # Define prompts
    map_prompt = PromptTemplate.from_template(map_prompt_template).partial(
        quastion=QUASTION,
        language=OUTPUT_LANGUAGE
    )
    combine_prompt = PromptTemplate.from_template(combine_prompt_template).partial(
        style=STYLE,
        trigger=PROMPT_TRIGGER,
        quastion=QUASTION,
        language=OUTPUT_LANGUAGE,
    )

    # Create and run chain using map-reduce
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        combine_document_variable_name="content",
    )

    return chain.run(split_docs)


def init(hf_pipeline , content, quastion):
    if content:
        content_tokens = hf_pipeline.get_num_tokens(content)
        print(f"Content length: {len(content)} chars, {content_tokens} tokens.")

        
        # Determine whether to use base method or map-reduce based on token count
        if content_tokens < 0.75 * MODEL_CONTEXT_WINDOW:
            print("Using question-answering: base")
            answer = answer_question_base(hf_pipeline, quastion, content)
        else:
            print("Using question-answering: map-reduce")
            answer = answer_question_map_reduce(hf_pipeline, quastion, content)

        print(f"Answer length: {len(answer)} chars, {hf_pipeline.get_num_tokens(answer) - content_tokens} tokens.")
        answer = answer.split(f'{PROMPT_TRIGGER} in {OUTPUT_LANGUAGE}:')
        print("Answer:\n" + answer[len(answer)-1] + "\n")
        return answer
    else:
        print("Failed to load the PDF file.")
'''

def init(client, content, question = None):
    if content:
        
        PROMT1 = f"""Based on the text provided, answer the following question.
            {STYLE}

            Question: {question}



            {content}


            {PROMPT_TRIGGER} in {OUTPUT_LANGUAGE}:
            """
        
        PROMT2 = f"""Write a summary of the following text delimited by triple backquotes.
                ```{content}```

                NUMBERED LIST SUMMARY in English:"""
        API_NAME_1="/model_chat_1"
        API_NAME_2="/model_chat"
        
        prompt = ""
        api_name_ = ""
        
        if question: 
            prompt = PROMT1
            api_name_ = API_NAME_1
        else: 
            prompt = PROMT2
            api_name_ = API_NAME_2
            
        result = client.predict(
            query=prompt,
            history=[],
            radio='72B',
            system="You are an example of Quick Insight for working with scientific literature.",
            api_name=api_name_
        )

        return result[1][0][1]['text']
    else:
        print("Failed to load the PDF file.")