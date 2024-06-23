from functions import RAG, UMAP

# Initialize the function

RAG_mini_one_doc = RAG(model_name='all-MiniLM-L6-v2', data_type='one-doc')
RAG_mini_multi_doc = RAG(model_name='all-MiniLM-L6-v2', data_type='multi-doc')

#RAG_mini_one_doc.chunk_and_create_collection()
RAG_mini_multi_doc.chunk_and_create_collection()

#Creating UMAP functions

# df_path_one_doc = 'processed_csv//one-doc//all-MiniLM-L6-v2_one-doc.csv'
# umapper_mistral_one_doc = UMAP(
#     collection=RAG_mini_one_doc.collection, 
#     df_path=df_path_one_doc,
#     model_name=RAG_mini_one_doc.model_name,
#     data_type=RAG_mini_one_doc.data_type,
#     RAG_client=RAG_mini_one_doc
# )

df_path_multi_doc = 'processed_csv//multi-doc//all-MiniLM-L6-v2_multi-doc.csv'

umapper_mistral_multi_doc = UMAP(
    collection=RAG_mini_multi_doc.collection, 
    df_path=df_path_multi_doc,
    model_name=RAG_mini_multi_doc.model_name,
    data_type=RAG_mini_multi_doc.data_type,
    RAG_client=RAG_mini_multi_doc
)
question_id_list = [3, 19, 20]

z = umapper_mistral_multi_doc.target_document_accuracy()
print(z)