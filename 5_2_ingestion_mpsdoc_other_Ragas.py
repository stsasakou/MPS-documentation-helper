from dotenv import load_dotenv
import os
import llama_index.core
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
 
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings
from pinecone import Pinecone,ServerlessSpec
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



load_dotenv()
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)
if __name__ == "__main__":
    print("Going to ingest pinecone documentation...")
    

    dir_reader = SimpleDirectoryReader(
        input_dir="./mps-docs-and-other",
        file_extractor={".html": UnstructuredReader()},
    )
    
    documents = dir_reader.load_data()
    
    node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(documents)

    llm2 = OpenAI(model="gpt-4", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    
    # Create the Pinecone index
    index_name = "5-2-mpsdoc-and-other-helper"
    
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
    evaluationResults=result.to_pandas().to_csv('5_2_RAGAs_mpsdoc_and_other.csv', sep=',')
    print(evaluationResults)

    exit()


