"""
Tools for PDF processing, text splitting, and Pinecone vector store creation
with OpenAI embeddings.
"""

import os
from pathlib import Path
from typing import List
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone


def load_pdfs_from_directory(pdf_directory: str) -> List[Document]:
    """
    Load multiple PDFs from a directory and convert them to Document objects.
    
    Args:
        pdf_directory (str): Path to the directory containing PDF files
        
    Returns:
        List[Document]: List of langchain Document objects with PDF content
    """
    documents = []
    pdf_path = Path(pdf_directory)
    
    if not pdf_path.exists():
        raise ValueError(f"Directory {pdf_directory} does not exist")
    
    # Get all PDF files in the directory
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return documents
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_file)
            text_content = ""
            
            # Extract text from all pages
            for page_num, page in enumerate(doc):
                text = page.get_text()
                text_content += f"\n--- Page {page_num + 1} ---\n{text}"
            
            # Create Document object
            doc_obj = Document(
                page_content=text_content,
                metadata={
                    "source": pdf_file.name,
                    "file_path": str(pdf_file)
                }
            )
            documents.append(doc_obj)
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents


def split_documents_recursively(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks using recursive character text splitter.
    
    Args:
        documents (List[Document]): List of Document objects to split
        chunk_size (int): Maximum size of each chunk (characters)
        chunk_overlap (int): Overlap between chunks (characters)
        
    Returns:
        List[Document]: List of split Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    print(f"Splitting {len(documents)} documents...")
    split_docs = splitter.split_documents(documents)
    print(f"Created {len(split_docs)} text chunks")
    
    return split_docs


def create_pinecone_vector_store(
    documents: List[Document],
    index_name: str,
    namespace: str = "",
    batch_size: int = 100
) -> PineconeVectorStore:
    """
    Create a Pinecone vector store with OpenAI embeddings.
    
    Args:
        documents (List[Document]): List of split Document objects
        index_name (str): Name of the Pinecone index
        namespace (str): Optional namespace in Pinecone index
        batch_size (int): Number of documents to process in each batch
        
    Returns:
        PineconeVectorStore: Pinecone vector store object
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"Creating vector store in Pinecone index '{index_name}'...")
    
    # Create vector store from documents
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        batch_size=batch_size
    )
    
    print(f"Vector store created with {len(documents)} documents")
    return vector_store


def get_pinecone_vector_store(
    index_name: str,
    namespace: str = ""
) -> PineconeVectorStore:
    """
    Get an existing Pinecone vector store.
    
    Args:
        index_name (str): Name of the Pinecone index
        namespace (str): Optional namespace in Pinecone index
        
    Returns:
        PineconeVectorStore: Pinecone vector store object
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
    return vector_store


def process_pdfs_to_pinecone(
    pdf_directory: str,
    index_name: str,
    namespace: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> PineconeVectorStore:
    """
    End-to-end pipeline: Load PDFs -> Split -> Create Pinecone vector store.
    
    Args:
        pdf_directory (str): Path to directory containing PDFs
        index_name (str): Name of the Pinecone index
        namespace (str): Optional namespace in Pinecone index
        chunk_size (int): Maximum size of each chunk (characters)
        chunk_overlap (int): Overlap between chunks (characters)
        
    Returns:
        PineconeVectorStore: Pinecone vector store object
    """
    print("=== Starting PDF to Pinecone Pipeline ===\n")
    
    # Step 1: Load PDFs
    print("Step 1: Loading PDFs...")
    documents = load_pdfs_from_directory(pdf_directory)
    
    if not documents:
        raise ValueError("No documents loaded from PDFs")
    
    print(f"Loaded {len(documents)} documents\n")
    
    # Step 2: Split documents recursively
    print("Step 2: Splitting documents...")
    split_docs = split_documents_recursively(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"Created {len(split_docs)} chunks\n")
    
    # Step 3: Create Pinecone vector store
    print("Step 3: Creating Pinecone vector store...")
    vector_store = create_pinecone_vector_store(
        split_docs,
        index_name=index_name,
        namespace=namespace
    )
    
    print("\n=== Upload Complete ===\n")
    return vector_store