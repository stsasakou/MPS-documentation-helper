from llama_index.core import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
import json
import time

# Load the JSON file and extract questions
file_path = "./5_mps_other_eval_qa_dataset.json"
with open(file_path, 'r') as file:
    data = json.load(file)
    eval_questions = list(data['queries'].values())

# Initialize an empty list for answers
eval_answers = []

# Load documents into LlamaIndex (you may need to adjust this path)
index_path = "./path_to_your_index"  # Path to where your index is saved

# Load or create the index
try:
    index = GPTSimpleVectorIndex.load_from_disk(index_path)
except FileNotFoundError:
    documents = SimpleDirectoryReader("./documents_directory").load_data()
    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk(index_path)

# Function to get answers from LlamaIndex
def get_answer_from_llamaindex(question):
    try:
        response = index.query(question)
        answer = response.response.strip()
        return answer
    except Exception as e:
        print(f"Error generating answer for question: {question}\nError: {e}")
        return "Error: Could not generate answer."

# Process each question and get answers
for question in eval_questions:
    print(f"Processing question: {question}")
    answer = get_answer_from_llamaindex(question)
    eval_answers.append(answer)
    time.sleep(1)  # Optional delay if needed

# Print formatted questions and answers
print("eval_questions =", eval_questions)
print("eval_answers =", eval_answers)

# Save output to a JSON file
output = {
    "eval_questions": eval_questions,
    "eval_answers": eval_answers
}
with open("5_Ragas_evaluation_output_llamaindex.json", "w") as outfile:
    json.dump(output, outfile, indent=4)

print("Evaluation questions and answers saved to '5_Ragas_evaluation_output_llamaindex.json'")
