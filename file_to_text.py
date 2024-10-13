from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
import os

def load_text(file_name: str) -> str:
    """
    Load content from various types of documents.

    Args:
        file_name (str): The path to the file.

    Returns:
        str: The combined text content of the file.
    """
    # Check the file extension to determine the loader
    ext = os.path.splitext(file_name)[1].lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_name)
        elif ext == '.txt':
            loader = TextLoader(file_name)
        elif ext == '.csv':
            loader = CSVLoader(file_name)
        elif ext in ['.doc', '.docx']:
            loader = PyPDFLoader(file_name)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        documents = loader.load()
        print(f"Loaded {len(documents)} page{'' if len(documents) == 1 else 's'} from {file_name}.")
        return "\n".join(doc.page_content for doc in documents)
    
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return ""
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return ""