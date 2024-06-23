import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import umap.umap_ as umap
import os
from mistralai.client import MistralClient
import voyageai
from dotenv import load_dotenv
load_dotenv()

class RAG:
    def __init__(self, model_name, data_type):
        self.directory = 'C:/Users/Lenna/OneDrive/Skrivebord/Technische Bestuurskunde/BEP/BEP_RAG/mini-wikipedia'
        self.model_name = model_name
        self.data_type = data_type
        self.model = self.get_model(model_name)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=f"{model_name}_{data_type}")
        self.df = self.set_data(data_type)
        self.openai_api_key = os.getenv("openai_api_key")

    def create_csv_pipe(self):
        document_chunks, total_words, filenames = self.create_chunks_from_directory()
        print(f'---------Processing {self.model_name}_{self.data_type} ----------')
        print(f'chunks: {len(document_chunks)} total words: {total_words}')
        self.add_to_collection(document_chunks)
        self.update_columns()

    def chunk_and_create_collection(self):
        document_chunks, total_words, filenames = self.create_chunks_from_directory()
        print(f'---------Processing {self.model_name}_{self.data_type} ----------')
        print(f'chunks: {len(document_chunks)} total words: {total_words}')
        self.add_to_collection(document_chunks)

    def get_collection(self):
        return self.collection

    def get_model(self, model_name):
        if model_name in ['mistral-embed', 'text-embedding-3-large', 'voyage-large-2-instruct']:
            return model_name
        elif model_name == 'gte-large-en-v1.5':
            return SentenceTransformer(f'Alibaba-NLP/{model_name}', trust_remote_code=True)
        else:
            return SentenceTransformer(model_name, trust_remote_code=True)
        
    def set_data(self, data_type):
        if data_type == 'one-doc':
            self.txt_file = ['S08_set1_a2.txt']
            return self.read_multibletiple(['Labeled_S08_set1_a2.csv'])
        elif data_type == 'multi-doc':
            self.txt_file = ['S08_set1_a2.txt','S08_set1_a3.txt', 'S08_set1_a4.txt', 'S08_set1_a8.txt']
            csv_list = ['Labeled_S08_set1_a2.csv', 'Labeled_S08_set1_a3.csv', 'Labeled_S08_set1_a4.csv', 'Labeled_S08_set1_a8.csv']
            return self.read_multibletiple(csv_list)
        elif data_type == 'perfect-chunk-one-doc':
            pass
        elif data_type == 'perfect-chunk-multi-doc':
            pass
        else:
            return 'ERROR'

    def read_multibletiple(self, csv_list):
        dfs = []
        for file in csv_list:
            df = pd.read_csv(file)
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.dropna(subset=['Annotated Chunk'])
        return self.df

    def create_chunks_from_directory(self):
        direct = self.directory+'/context'
        document_chunks = []
        total_words = 0
        filenames = []
        for filename in os.listdir(direct):
            if filename.endswith(".txt"):
                if filename in self.txt_file:
                    with open(os.path.join(direct, filename), 'r', encoding='utf-8') as file:
                        filenames.append(filename.replace('.txt', ''))
                        text = file.read()
                        # Count the number of words in the text
                        words = text.split()
                        num_words = len(words)
                        total_words += num_words

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                        all_splits = text_splitter.create_documents([text])
                        document_chunks.extend(all_splits)
        document_chunks = [p.page_content for p in document_chunks]
        return document_chunks, total_words, filenames
    
    def create_chunks_from_perfect_chunk(self, file_list:list[str]):
        direct = self.directory+'/perfectly_chunked'
        document_chunks = []
        total_words = 0
        for filename in os.listdir(direct):
            if filename.endswith(".txt"):
                if filename in file_list:
                    print('')
                    print(filename)
                    with open(os.path.join(direct, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        num_words = len(text.split())
                        total_words += num_words
                        # Split the text by '--' and append each chunk to document_chunks
                        chunks = text.split('--')
                        for chunk in chunks:
                            if chunk.strip():  # Ignore empty chunks
                                document_chunks.append(chunk.strip())
        filenames = []
        return document_chunks, total_words, filenames

    def embed(self, chunk):
        if self.model_name == 'mistral-embed':
            return self.get_mistral_embed(chunk)
        elif self.model_name == 'text-embedding-3-large':
            return self.get_text_embedding_large(chunk)
        elif self.model_name == 'voyage-large-2-instruct':
            return self.get_voyage(chunk)
        else:
            return self.model.encode(chunk).tolist()

    def add_to_collection(self, document_chunks):
        embeddings_all = []
        documents = []
        for d in document_chunks:
            embeddings_all.append(self.embed(d))
            documents.append(d)
        ids = [str(i) for i in range(len(embeddings_all))]
        self.collection.add(
            documents=documents,
            embeddings = embeddings_all, 
            ids = ids)

    def query(self, query):
        query_embed = self.embed(query)
        results = self.collection.query(
            query_embeddings = query_embed,
            n_results = 5,
            )
        return results['ids'][0]

    def get_chunk_by_id(self, ids):
        results = self.collection.get(
            ids = ids
            )
        return results['documents'][0]
    
    def apply_query(self, row):
        question = row['Question']
        new_ids = self.query(question)
        row['top_n_chunk_id'] = new_ids  # Use this instead of row.set_value
        answer = self.gpt_answer(question, new_ids)
        row['answer llm'] = answer  # Use this instead of row.set_value
        
        # ------------ annotating the chunk by the ground truth annotation ---------------
        annotated_chunk = row['Annotated Chunk']
        chunk_annotation = False

        for chunk_id in new_ids:
            chunk = self.get_chunk_by_id(chunk_id)
            if annotated_chunk in chunk:
                chunk_annotation = True
                break
        row['chunk_annotation'] = chunk_annotation

        # --------------- Hand annotation process for answer validation -------------
        print('QUESTION:', question)
        print('ANNOTATED ANSWER:', row['Answer'])
        print('LLM ANSWER:', answer)
        label = input('Label: 1 - correct, 0 - incorrect: ')
        row['answer_annotation'] = label

        return row

    def update_columns(self):
        self.df = self.df.apply(self.apply_query, axis=1)

        os.makedirs('processed_csv', exist_ok=True)

        csv_name = f'processed_csv/{self.model_name}_{self.data_type}.csv'
        self.df.to_csv(csv_name, index=False)

    def gpt_answer(self, question, chunk_ids):
        openai_client = OpenAI(
            api_key=self.openai_api_key,
        )
        prompt= f"""
        You will receive a questions and a few chunks of text. Please only answer the question based on the provided chunks. 
        Please keep all answers as short as possible. Yes/no questions should be answered with a single word. Yes or no. If the answer is not provided in the chunks then answer: 'not specified'.
        Question: {question}
        Chunks:
        """
        for id in chunk_ids:
            chunk = self.get_chunk_by_id(id)
            prompt += f"{chunk}\n"

        completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )
        return completion.choices[0].message.content

    def get_text_embedding_large(self, text):
        openai_client = OpenAI(
            api_key=self.openai_api_key,
        )
        model = 'text-embedding-3-large'
        text = text.replace("\n", " ")
        return openai_client.embeddings.create(input = [text], model=model).data[0].embedding

    def get_voyage(self, text):
        voyage_api_key = os.getenv("voyage_api_key")

        vo = voyageai.Client(
            api_key=voyage_api_key
        )
        text = text.replace("\n", " ")

        documents_embeddings = vo.embed(
            text, model="voyage-large-2-instruct", input_type="document"
        ).embeddings[0]

        return documents_embeddings
        
    def get_mistral_embed(self, text):
        client = MistralClient(api_key=os.getenv("mistral_api_key"))

        text = text.replace("\n", " ")
        embeddings_batch_response = client.embeddings(
            model="mistral-embed",
            input=[text],
        )
        return embeddings_batch_response.data[0].embedding
            
    def check_answers(self, file_name):
        df = pd.read_csv('updated_file.csv')
        count = 0

        # Create a new column 'answer_annotation' and initialize with NaN
        df['answer_annotation'] = pd.NaT

        for index, row in df.iterrows():
            annotated_answer = row['Answer']
            llm_answer = row['answer llm']
            print('QUESTION', row['Question'])
            print('ANNOTATED ANSWER')
            print(annotated_answer)
            print('\n')
            print('LLM ANSWER')
            print(llm_answer)
            label = input('Label: 1 - correct, 0 - incorrect: ')

            # Save the label in the 'answer_annotation' column
            df.at[index, 'answer_annotation'] = label

            if label == '1':
                count += 1

        # Save the updated DataFrame to the csv file
        file_name = file_name + '.csv'
        df.to_csv(file_name, index=False)
        
    def check_string_in_chunks(self):
        count = 0
        df = pd.read_csv('updated_file.csv')

        for index, row in df.iterrows():
            top_n_chunk_ids_str  = row['top_n_chunk_id']
            annotated_chunk = row['Annotated Chunk']
            
            top_n_chunk_ids = ast.literal_eval(top_n_chunk_ids_str)
            boolean = False
            c = ''
            for chunk_id in top_n_chunk_ids:
                chunk = self.get_chunk_by_id(chunk_id)
                c += chunk + '\n'
                if annotated_chunk in chunk:
                    count += 1
                    boolean = True
                    break

            df.at[index, 'chunk_annotation'] = boolean
        df.to_csv('updated_file.csv', index=False)
        return count/len(df)
    
    def get_accuracy(self, df_name):
        df = pd.read_csv(df_name)
        answer_count = 0
        chunk_count = 0
        for index, row in df.iterrows():
            if str(row['answer_annotation']) == str(1):
                answer_count += 1
            
            if row['chunk_annotation'] == True:
                chunk_count += 1
        
        return answer_count/len(df), chunk_count/len(df)
    

class UMAP():
    def __init__(self, model_name, data_type, collection, df_path, RAG_client):
        self.model_name = model_name
        self.data_type = data_type
        self.collection = collection
        self.df = pd.read_csv(df_path)
        self.df_path = df_path
        self.RAG_client = RAG_client

    def get_all_vectors(self):
        ids = [str(i) for i in range(self.collection.count())]
        response = self.collection.get(
            ids=ids,
            include=['embeddings']
        )
        return response['embeddings']
    
    def check_document_type_match(self, row_id, question_id):
        row_type = self.get_row_type(row_id)
        question_type = self.get_question_type(question_id)
        return row_type == question_type

    def get_row_type(self, id):
        if 0 <= id <= 67:
            return 'leopard'
        elif 68 <= id <= 132:
            return 'penguin'
        elif 133 <= id <= 213:
            return 'polar bear'
        elif 214 <= id <= 279:
            return 'beetle'
        else:
            return 'unknown'

    def get_question_type(self, id):
        if 0 <= id <= 31:
            return 'leopard'
        elif 32 <= id <= 62:
            return 'penguin'
        elif 63 <= id <= 92:
            return 'polar bear'
        elif 93 <= id <= 122:
            return 'beetle'
        else:
            return 'unknown'

    def get_correct_chunk_id(self, annotated_chunk):
        ids = [str(i) for i in range(self.collection.count())]
        response = self.collection.get(
            ids=ids,
            include=['documents']
        )
        ids = [i for i in response['ids']]
        docs = [i for i in response['documents']]
    
        for id, d in list(zip(ids, docs)):
            if annotated_chunk in d:
                return int(id)
        return 'Something fishy'
    
    def get_chunk_by_id(self, ids):
        results = self.collection.get(
            ids = ids
            )
        return results['documents']
    
    def target_document_accuracy(self):
        document_types = ['leopard', 'penguin', 'polar bear', 'beetle', 'unknown']
        accuracy_scores = {doc_type: {'correct': 0, 'total': 0} for doc_type in document_types}

        for i, row in self.df.iterrows():
            question_id = i
            doc_type = self.get_question_type(question_id)

            top_n_chunk_ids_str = row['top_n_chunk_id']
            document_ids = [int(i) for i in ast.literal_eval(top_n_chunk_ids_str)]

            for document_id in document_ids:
                is_match = self.check_document_type_match(document_id, question_id)

                accuracy_scores[doc_type]['total'] += 1
                if is_match:
                    accuracy_scores[doc_type]['correct'] += 1

        # Calculate accuracy percentages
        for doc_type in accuracy_scores:
            total = accuracy_scores[doc_type]['total']
            correct = accuracy_scores[doc_type]['correct']
            accuracy = (correct / total) * 100 if total > 0 else 0
            accuracy_scores[doc_type]['accuracy'] = round(accuracy, 2)

        return accuracy_scores
    
    def cluster(self, questions_id_list):
        
        # Find optimal n_neighbors
        #optimal_params = self.optimal_params(self.get_all_vectors())
        # print(optimal_params)
        optimal_n_neighbors = 4
        print(f'OPTIMAL neighbors for {self.df_path} optimal n neighbors: {optimal_n_neighbors}')

        # Apply UMAP with optimal n_neighbors
        vectors = self.get_all_vectors()
  
        reducer = umap.UMAP(n_neighbors=optimal_n_neighbors, random_state=42)
        umap_embeds = reducer.fit_transform(vectors)

        fig, axs = plt.subplots(3, 1, figsize=(4.5, 10))  # Increased the figure size
        # Add the title to the main plot
        s = ', '.join(map(str, questions_id_list))
        title = f'{self.model_name} {self.data_type}, questions_ids: {s}'
        fig.suptitle(title, fontsize=16)

        # Document type colors
        colors = {
            'leopard': 'yellow',
            'penguin': 'black',
            'polar bear': 'grey',
            'beetle': 'orange'
        }

        # Create a color map for the entire dataset
        color_map = []
        for i in range(len(vectors)):
            doc_type = self.id_document_type(i)
            color_map.append(colors[doc_type])

        # Plot the provided questions
        for i, question_id in enumerate(questions_id_list):
            row = self.df.iloc[question_id]
            question = row['Question']
            top_n_chunk_ids_str = row['top_n_chunk_id']
            questions_ids = [int(i) for i in ast.literal_eval(top_n_chunk_ids_str)]
            correct_id = self.get_correct_chunk_id(row['Annotated Chunk'])
            ax = axs[i]

            # Plot all points with colors based on document type
            scatter = ax.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=color_map, s=3)

            # Plot top_n_chunks in blue
            for id in questions_ids:
                ax.scatter(umap_embeds[id, 0], umap_embeds[id, 1], c='blue', s=50)

            # Plot correct_id in red or green
            if correct_id in questions_ids:
                ax.scatter(umap_embeds[correct_id, 0], umap_embeds[correct_id, 1], c='green', s=100)
            else:
                ax.scatter(umap_embeds[correct_id, 0], umap_embeds[correct_id, 1], c='red', s=100)

            # Set the title of the subplot to the question with line breaks
            ax.set_title(question)

        # Create a legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Leopard'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Penguin'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Polar Bear'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Beetle'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Top N Chunks'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct Chunk (In Top N)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Correct Chunk (Not In Top N)')
        ]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        if not os.path.exists('umaps'):
            os.makedirs('umaps')
        name = f'umaps/{self.model_name}_{self.data_type}_color'
        plt.savefig(name)

        # Show the plot
        plt.tight_layout()  # Adjust the spacing between subplots
        plt.show()

    def optimal_params(self, collection, max_n_neighbors=100, max_n_clusters=10):
        # Try a range of n_neighbors values
        n_neighbors_range = np.arange(4, max_n_neighbors, 10)
        # Try a range of n_clusters values
        n_clusters_range = np.arange(2, max_n_clusters + 1)
        scores = []

        for n_neighbors in n_neighbors_range:
            for n_clusters in n_clusters_range:
                # Apply UMAP
                reducer = umap.UMAP(n_neighbors=n_neighbors)
                umap_embeds = reducer.fit_transform(collection)

                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(umap_embeds)

                # Calculate the silhouette score
                score = silhouette_score(umap_embeds, kmeans.labels_)
                scores.append((score, n_neighbors, n_clusters))

        # The optimal parameters are the ones that give the highest score
        optimal_params = max(scores, key=lambda x: x[0])

        return optimal_params
