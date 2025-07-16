import logging
import sys
from dotenv import load_dotenv
import os
from pathlib import Path


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


print("Going to create example questions...")


dir_reader = SimpleDirectoryReader(input_dir="./mps-docs/")


documents = dir_reader.load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)

# by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"


llm = OpenAI(model="gpt-3.5-turbo")
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)

qa_dataset = generate_question_context_pairs(nodes, llm=llm, num_questions_per_chunk=2)
queries = qa_dataset.queries.values()
# print(list(queries)[2])

# [optional] save
qa_dataset.save_json("a_mps_eval_qa_dataset.json")


print("Finished evaluation dataset...")
