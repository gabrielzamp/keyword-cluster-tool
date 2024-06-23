import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import openai
import time

# Streamlit page setup
st.title('Keyword Research Cluster Analysis Tool')
st.subheader('Leverage OpenAI to cluster similar keywords into groups from your keyword list.')
st.write("Created By: Brandon Lazovic")
st.text("")
st.markdown("""
Welcome to the Keyword Analysis Tool! This interactive web application helps SEO professionals and digital marketers analyze and cluster keywords based on their semantic similarity. By leveraging advanced AI embeddings, the tool identifies closely related keywords and suggests optimal groupings for SEO, or to identify the primary keyword to target in a common keyword cluster.

### How to Use:
1. **Upload a CSV File**: Your CSV should contain a list of keywords, each with associated search volume and cost-per-click (CPC) data.
2. **Enter Your OpenAI API Key**: The API key is required to generate semantic embeddings via the OpenAI platform.

### Requirements:
- The CSV file must include the following columns and column names: `Keywords`, `Search Volume`, and `CPC`.
- Ensure that your OpenAI API key is valid to process embeddings.

Enter your .csv file and OpenAI API key below to get started. This tool will create a table that displays your keyword clusters and .csv to download your results.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Check for required columns
    required_columns = {'Keywords', 'Search Volume', 'CPC'}
    if not required_columns.issubset(data.columns):
        st.error("CSV file must include the following columns: Keywords, Search Volume, CPC")
    else:
        st.write('Loaded ', len(data), ' rows from the spreadsheet.')
        # Proceed with processing...

# API Key input
api_key = st.text_input("Enter your OpenAI API key", type="password")
if api_key:
    openai.api_key = api_key

# Function to generate embeddings
def get_embedding(text, model="text-embedding-ada-002", max_retries=3):
    text = text.replace("\n", " ")
    retries = 0
    while retries <= max_retries:
        try:
            return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        except Exception as e:
            st.error(f"Error while generating embedding for text: {text}. Error: {e}")
            retries += 1
            if retries <= max_retries:
                st.info(f"Retrying {retries}/{max_retries}...")
                time.sleep(2 ** retries)
            else:
                st.error(f"Max retries reached. Skipping keyword: {text}")
                return None

# Function to choose the best keyword
def choose_best_keyword(keyword1, keyword2):
    prompt = f"Identify which keyword users are more likely to search on Google for SEO: '{keyword1}' or '{keyword2}'. Only include the keyword in the response. If both keywords are similar, select the first one. You must choose a keyword based on which one has the best grammar, spelling, or natural language."
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=20
    )
    best_keyword_reason = response['choices'][0]['text'].strip()

    if keyword1.lower() in best_keyword_reason.lower():
        return keyword1
    elif keyword2.lower() in best_keyword_reason.lower():
        return keyword2
    else:
        st.warning(f"Unexpected response from GPT-3.5 Turbo: {best_keyword_reason}")
        return keyword1  # Fallback to the first keyword

# Function to identify primary variants
def identify_primary_variants(cluster_data):
    primary_variant_df = pd.DataFrame(columns=['Cluster ID', 'Keywords', 'Is Primary', 'Primary Keyword', 'GPT-3.5 Reason'])
    new_rows = []
    
    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        best_keyword_reason = None  # Reset for each cluster

        if len(keywords) == 2:
            primary = choose_best_keyword(keywords[0], keywords[1])
            best_keyword_reason = primary
        else:
            embeddings = [get_embedding(keyword) for keyword in keywords]
            similarity_matrix = cosine_similarity(np.array(embeddings))
            avg_similarity = np.mean(similarity_matrix, axis=1)
            primary_idx = np.argmax(avg_similarity)
            primary = keywords[primary_idx]

        for idx, keyword in enumerate(keywords):
            is_primary = 'Yes' if keyword == primary else 'No'
            gpt_reason = best_keyword_reason if len(keywords) == 2 else None
            new_row = {
                'Cluster ID': cluster_id,
                'Keywords': keyword,
                'Is Primary': is_primary,
                'Primary Keyword': primary,
                'GPT-3.5 Reason': gpt_reason
            }
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)

# Process the data if both file and API key are provided
if uploaded_file and api_key:
    # Assuming columns are named 'Keywords', 'Search Volume', and 'CPC'
    keywords = data['Keywords'].tolist()

    # Generate embeddings
    st.write("Generating embeddings for the keywords...")
    embeddings = [get_embedding(keyword) for keyword in keywords if get_embedding(keyword) is not None]

    # Calculate similarity and cluster
    if embeddings:
        st.write("Calculating similarity...")
        similarity_matrix = cosine_similarity(embeddings)
        st.write("Clustering keywords...")
        clusters = fcluster(linkage(1 - similarity_matrix, method='average'), t=0.2, criterion='distance')
        data['Cluster ID'] = clusters

        # Check for same CPC and Search Volume within each cluster
        st.write("Filtering clusters based on CPC and Search Volume...")
        valid_clusters = []
        invalid_keywords = []
        for cluster_id, group in data.groupby('Cluster ID'):
            if group['Search Volume'].nunique() == 1 and group['CPC'].nunique() == 1:
                valid_clusters.append(cluster_id)
            else:
                invalid_keywords.extend(group['Keywords'].tolist())
        
        filtered_data = data[data['Cluster ID'].isin(valid_clusters)]
        
        # Assign unique cluster IDs to invalid keywords
        invalid_data = data[data['Keywords'].isin(invalid_keywords)].copy()
        invalid_data['Cluster ID'] = range(filtered_data['Cluster ID'].max() + 1, 
                                           filtered_data['Cluster ID'].max() + 1 + len(invalid_data))

        # Combine valid and invalid data
        combined_data = pd.concat([filtered_data, invalid_data], ignore_index=True)

        st.write("Identifying primary keywords within clusters...")
        cluster_data = combined_data[['Cluster ID', 'Keywords']]
        primary_variant_df = identify_primary_variants(cluster_data)
        combined_data = pd.merge(combined_data, primary_variant_df, on=['Cluster ID', 'Keywords'], how='left')

        # Drop unnecessary columns and rename if needed
        combined_data = combined_data.drop(columns=['Unnamed: 3'], errors='ignore')

        # Output results
        st.write("Analysis complete. Review the clusters below:")
        st.dataframe(combined_data)

        # Download link
        st.download_button('Download Analysis Results', combined_data.to_csv(index=False).encode('utf-8'), 'analysis_results.csv', 'text/csv', key='download-csv')
    else:
        st.error("Failed to generate embeddings for all keywords.")
