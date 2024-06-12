from functions import RAG
import os
from pprint import pprint

# embedding_models = [
#     'mistral-embed',
#     'text-embedding-3-large',
#     'gte-large-en-v1.5',
#     'all-MiniLM-L6-v2',
#     'voyage-large-2-instruct'
# ]
# RAG_mistral_one_doc = RAG(model_name=embedding_models[4], data_type='multi-doc')
# RAG_mistral_one_doc.create_csv_pipe()

main_folder = 'processed_csv'

# Initialize a dictionary to store the results
results = {
    'multi-doc': [],
    'one-doc': []
}

RAG_mistral_one_doc = RAG(model_name='mistral-embed', data_type='multi-doc')

# Iterate through the subfolders and files
for subfolder in ['multi-doc', 'one-doc']:
    subfolder_path = os.path.join(main_folder, subfolder)
    for file_name in os.listdir(subfolder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(subfolder_path, file_name)
            accuracy = RAG_mistral_one_doc.get_accuracy(file_path)
            results[subfolder].append((file_name, accuracy))
pprint(results)
