from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, Document, StorageContext, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices.composability import ComposableGraph
from llama_index.core.query_engine import ComposableGraphQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import CSVReader
from llama_index.readers.file import PDFReader
from llama_index.core.indices.keyword_table import GPTSimpleKeywordTableIndex

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path
from llama_index.readers.file.unstructured import UnstructuredReader


import pandas as pd
import asyncio
import sys
import json
import ragas

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator
)

from llama_index.core.node_parser import SentenceSplitter

from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_utilization
    )

from ragas.metrics.critique import harmfulness    
from ragas.integrations.llama_index import evaluate
from datasets import Dataset  # Import Dataset to create the required structure

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Set up OpenAI models for embeddings and LLM
llm_model = OpenAI(model="gpt-4", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

# Pinecone index name and settings
index_name = "5-5-mpsdoc-with-other-composable-helper"

# Check if the Pinecone index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimension size should match your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Define paths for CSV and PDF files

#mpsdoc_file_path = r"C:\Users\USER\ProjectsLLM\Documentation-helper_updated_libraries\mps-docs"
#othermpsdoc_file_path = r"C:\Users\USER\ProjectsLLM\Documentation-helper_updated_libraries\mps-other"


dir_reader = SimpleDirectoryReader(
    input_dir="./mps-docs",
    file_extractor={".html": UnstructuredReader()},
)

mpsdoc_documents = dir_reader.load_data()

mpsdoc_node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
mpsdoc_nodes = mpsdoc_node_parser.get_nodes_from_documents(mpsdoc_documents)

llm2 = OpenAI(model="gpt-4", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)

mpdsdoc_index = VectorStoreIndex.from_documents(
documents=mpsdoc_documents,
storage_context=storage_context,
llm=llm2,
embed_model=embed_model,
node_parser=mpsdoc_node_parser, 
show_progress=True,
)



dir_reader = SimpleDirectoryReader(
    input_dir="./mps-other",
    file_extractor={".html": UnstructuredReader()},
)

mpsother_documents = dir_reader.load_data()

mpsother_node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
mpsother_nodes = mpsother_node_parser.get_nodes_from_documents(mpsdoc_documents)

llm2 = OpenAI(model="gpt-4", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)



mpdsother_index = VectorStoreIndex.from_documents(
documents=mpsother_documents,
storage_context=storage_context,
llm=llm2,
embed_model=embed_model,
node_parser=mpsother_node_parser, 
show_progress=True,
)

print("finished ingesting...")



# Create a Composable Graph with the root index class specified
# Using GPTSimpleKeywordTableIndex as the root index class in this example
composable_graph = ComposableGraph.from_indices(
    root_index_cls=GPTSimpleKeywordTableIndex,
    children_indices=[mpdsdoc_index, mpdsother_index],    
    index_summaries=[
        "Use this for general questions about MPS.",
        "Use this for questions related to MPS generator. MPS Generators, Tests and testing, Preprocessing, Error Handling, LOOP, SWITCH,IF, Template Switch, Configuration Pattern, Copy with Trace, Generator Priorities, Logical Checkpoints some of the keywords."
    ]
)

query_engine = ComposableGraphQueryEngine(graph=composable_graph)


# Load the JSON data
with open('5_QA_dataset_manually_corrected.json') as file:
    data = json.load(file)

# Initialize lists for eval_questions and eval_answers
eval_questions = []
eval_answers = []


# Populate the lists with all questions and answers
for item in data['answers']:
    eval_questions.append(item['question'])
    eval_answers.append(item['answer'])



#eval_answers = [[a] for a in eval_answers]
# Create a Dataset in the format expected by RAGAS
data = {
    "question": eval_questions,
    "answer": eval_answers
}
ragas_dataset = Dataset.from_dict(data)


metrics = [
    faithfulness,
    answer_relevancy,
    context_utilization,
    harmfulness,
]



# Correctly call the evaluate function
result = evaluate(
    query_engine=query_engine,  # The LlamaIndex Query Engine to evaluate
    metrics=metrics,            # List of metrics to use for evaluation
    dataset=ragas_dataset,      # The dataset to evaluate against
)


# Convert to a DataFrame and display
evaluationResults=result.to_pandas().to_csv('5_5_RAGAs_mpsdoc_other_composable_auto_keywords.csv', sep=',')
print(evaluationResults)

exit()

