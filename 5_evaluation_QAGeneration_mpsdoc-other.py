import logging
import sys
from dotenv import load_dotenv
import os
from pathlib import Path
import json

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    ServiceContext,
    VectorStoreIndex,
    Response,
)

import pandas as pd
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
 
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
from pinecone import Pinecone,ServerlessSpec

# Initialize logging
logging.basicConfig(level=logging.INFO)

print("Going to create example questions...")

# Load and split documents
dir_reader = SimpleDirectoryReader(input_dir="./mps-other/")
documents = dir_reader.load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)


llm2 = OpenAI(model="gpt-4", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

# Create the Pinecone index
index_name = "temp-mpsdoc-helper"
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)
# Check if the index exists, and create it if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Set this based on your embeddings
        metric="cosine",  # Or use 'euclidean', 'dotproduct'
        spec=ServerlessSpec(
            cloud="aws",  # Adjust cloud provider if needed
            region="us-east-1"  # Replace with your preferred region
        )
    )


pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
documents=documents,
storage_context=storage_context,
llm=llm2,
embed_model=embed_model,
node_parser=node_parser, 
show_progress=True,
)
print("finished ingesting...")

# Initialize the query engine
query_engine = index.as_query_engine()

# Set static node ids for reproducibility
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

# Initialize the LLM
llm1 = OpenAI(model="gpt-4o")

# Generate question-context pairs
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)

qa_dataset = generate_question_context_pairs(nodes, llm=llm1, num_questions_per_chunk=2)
queries = qa_dataset.queries.values()

# Generate answers


print("Generating answers for the questions...")
answers = []
query_engine = index.as_query_engine()
for query in queries:
    response = query_engine.query(query)  
    answer_text = response.response.strip() if hasattr(response, 'response') else str(response) 
    answers.append({
        "question": query,
        "answer": answer_text
    })


# Save question-answer pairs to JSON
output_data = {"answers": answers}
output_path = "5_QA_dataset.json"
with open(output_path, "w") as outfile:
    json.dump(output_data, outfile, indent=4)

print(f"Saved answers to {output_path}")
