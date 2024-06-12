import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def create_chuncks_from_directory(directory):
    document_chunks = []
    total_words = 0
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                filenames.append(filename.replace('.txt', ''))
                text = file.read()
                # Count the number of words in the text
                words = text.split()
                num_words = len(words)
                total_words += num_words

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
                all_splits = text_splitter.create_documents([text])
                document_chunks.extend(all_splits)
    return document_chunks, total_words, filenames

def filter():
    directory = 'C:/Users/Lenna/OneDrive/Skrivebord/Technische Bestuurskunde/BEP/BEP_RAG/mini-wikipedia/context'

    wiki_chunks, total_words, filenames = create_chuncks_from_directory(directory)
    df_processed = pd.read_csv('processed_Data.csv')
    print("Number of rows before filtering: ", len(df_processed))
    df_unique = df_processed.drop_duplicates(subset='Question')
    print("Number of rows after filtering: ", len(df_unique))
    # create a boolean mask to filter the DataFrame
    mask = df_unique['ArticleFile'].isin(filenames)

    # filter the DataFrame using the mask
    df_masked = df_unique[mask]

    # reset the index of the filtered DataFrame
    df_masked = df_masked.reset_index(drop=True)
    print("Number of rows after filtering: ", len(df_masked))
    df_masked.head()
    return df_masked

def label_loop():
    df = filter()
    file_name = 'S08_set1_a4'
    subset = df[df['ArticleFile'] == file_name]  # create subset of data
    i = 0  # initialize index counter

    Answers = []
    Chunk_fragment = []

    while True:
        print(subset['Question'].iloc[i])  # print current row
        if subset['Answer'].iloc[i]:
            print(f"\nThe existing answer is: {subset['Answer'].iloc[i]}")
            
        user_answer = input("Enter the answer. 'a' for keep exsisting answer and 'q' to quit:")  # get user input
        if user_answer == 'a':
            user_answer = subset['Answer'].iloc[i]
        if user_answer == 'q':  # if user input is 'q'
            break
        Answers.append(user_answer)
        user_chunk = input("Enter the chunk with the answer ")  # get user input
        Chunk_fragment.append(user_chunk)
        print('\n\n')
        i += 1  # increment index counter
        if i == len(subset):  # check if we've reached the end of the subset
            print("You've reached the end of the subset.")
            break  # exit the loop
    indices = df[df['ArticleFile'] == file_name].index

    df.loc[indices[:len(Answers)], 'Answer'] = Answers
    df.loc[indices[:len(Chunk_fragment)], 'Annotated Chunk'] = Chunk_fragment
    subset_2 = df[df['ArticleFile'] == file_name]
    csv_name = 'Labeled_' + file_name + '.csv'
    subset_2.to_csv(csv_name, index=False)


label_loop()