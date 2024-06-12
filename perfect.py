from functions import RAG
import pandas as pd
def update_file():
    directory = 'C:/Users/Lenna/OneDrive/Skrivebord/Technische Bestuurskunde/BEP/BEP_RAG/mini-wikipedia'
    file = ['S08_set1_a2_pefect_chunk.txt']
    csv_file = ['Labeled_S08_set1_a2.csv']
    df_name = 'S08_set1_a2_pefect_chunk.csv'
    rag_perfect = RAG(directory, csv_list = csv_file)
    chunks, total_words, filenames = rag_perfect.create_chunks_from_perfect_chunk(file)
    rag_perfect.add_to_collection(chunks)
    answer_accuracy, chunk_accuracy =rag_perfect.get_accuracy(df_name)
    print('answer accuracy', answer_accuracy)
    print('retriever accuracy', chunk_accuracy)

update_file()