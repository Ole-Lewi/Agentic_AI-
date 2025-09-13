# Setting up the knowledge base with FAISS
import os
import torch
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up embedding model
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)


# Load research publications
def load_research_publications(documents_path):
    """Load research publications from .txt files and return as documents"""
    documents = []
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# Chunking
def chunk_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(documents)


# Store chunks into FAISS
def build_faiss_index(chunks, embeddings, index_path="faiss_index"):
    """Build and save FAISS index"""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore


# Intelligent retrieval
def search_research_db(query, vectorstore, top_k=5):
    """Search FAISS index for relevant chunks"""
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    relevant_chunks = []
    for doc, score in results:
        relevant_chunks.append({
            "content": doc.page_content,
            "title": doc.metadata.get("source", "Unknown"),
            "similarity": 1 - score  # convert distance to similarity-ish
        })
    return relevant_chunks


# Generating knowledge-based answers
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def answer_research_question(query, vectorstore, llm):
    """Generate an answer based on retrieved research"""
    relevant_chunks = search_research_db(query, vectorstore, top_k=3)

    context = "\n\n".join([   # Formatting context for clarity
        f"From {chunk['title']}:\n{chunk['content']}"   
        for chunk in relevant_chunks
    ])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
"""
    )

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)

    return response.content, relevant_chunks


# ----------------- MAIN -----------------
# Load docs -> chunk -> build index
docs = load_research_publications(".")  # path to your txt files
chunks = chunk_documents(docs)
vectorstore = build_faiss_index(chunks, embeddings)

# Init Groq LLM
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
)

# Ask a research question
answer, sources = answer_research_question(
    "What are the applications of machine learning?",
    vectorstore,
    llm
)

print("AI Answer:", answer)
print("\nBased on sources:")
for source in sources:
    print(f"- {source['title']}")
