import ollama
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

def get_context(question):
    search_results = search(question, num=3, stop=3, pause=2)
    top_links = list(search_results)
    scraped_texts = []
    for link in top_links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        scraped_texts.append(text)

    all_scraped_text = '\n'.join(scraped_texts)
    
    print(all_scraped_text)
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_text(all_scraped_text)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    db = Chroma.from_texts(documents, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    context = retriever.get_relevant_documents(question)

    return context

def generate_output(question,context):
    # Prepare the prompt with the input text
    # Define a prompt template
    prompt = f"""Use the following pieces of context to answer the question at the end.  
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer only factual information based on the context.
    Context: {context}.\n
    Question: {question}
    Helpful Answer:"""
    
    response = ollama.chat(model='qwen:0.5b', messages=[
        
        {
            'role': 'system',
            'content': 'You are a useful AI assistant, answer only based on the information from the user prompt and nothing else.',
            
            'role': 'user',
            'content': f'{prompt}',
            },
        ])
    
    output = response['message']['content']
    
    # Return the generated text
    return output

# Main function to take user input and save the output
def main():
    # Take user input
    #question = input("Enter your text: ")
    
    question = "Who did Tucker Carlson interviewed?"
    
    context = get_context(question)
    
    # Generate the output using the LLM model
    output = generate_output(question,context)
    
    print(output)

if __name__ == "__main__":
    main()
