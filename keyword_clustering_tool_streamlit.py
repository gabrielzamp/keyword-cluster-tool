import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from openai import OpenAI
import time
import io

# Streamlit page setup
st.title('Keyword Research Cluster Analysis Tool')
st.subheader('Leverage OpenAI to cluster similar keywords into groups from your keyword list.')

# Instructions
st.markdown("""
## How to Use This Tool

1. **Prepare Your Data**: 
   - Create a CSV file with the following columns: 'Keywords', 'Search Volume', and 'CPC'.
   - Ensure your data is clean and formatted correctly.

2. **Get Your OpenAI API Key**:
   - If you don't have an OpenAI API key, sign up at [OpenAI](https://openai.com).
   - Ensure you have access to the GPT-4 model.

3. **Upload Your File**:
   - Use the file uploader below to upload your CSV file.

4. **Enter Your API Key**:
   - Input your OpenAI API key in the text box provided.

5. **Run the Analysis**:
   - The tool will automatically process your data once both the file and API key are provided.

6. **Review Results**:
   - Examine the clustered keywords and primary variants in the displayed table.

7. **Download Results**:
   - Use the download button to get a CSV file of your analysis results.

## Sample CSV Template

You can download a sample CSV template to see the required format:
""")

# Sample CSV template
sample_data = pd.DataFrame({
    'Keywords': ['buy shoes online', 'purchase shoes online', 'best running shoes', 'comfortable walking shoes'],
    'Search Volume': [1000, 800, 1500, 1200],
    'CPC': [0.5, 0.4, 0.7, 0.6]
})

# Create a download button for the sample CSV
csv_buffer = io.StringIO()
sample_data.to_csv(csv_buffer, index=False)
csv_str = csv_buffer.getvalue()

st.download_button(
    label="Download Sample CSV Template",
    data=csv_str,
    file_name="sample_keyword_data.csv",
    mime="text/csv"
)

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    required_columns = {'Keywords', 'Search Volume', 'CPC'}
    if not required_columns.issubset(data.columns):
        st.error("CSV file must include the following columns: Keywords, Search Volume, CPC")
    else:
        st.write('Loaded ', len(data), ' rows from the spreadsheet.')

# API Key input
api_key = st.text_input("Enter your OpenAI API key", type="password")
if api_key:
    client = OpenAI(api_key=api_key)

# Function to generate embeddings
def get_embedding(text, model="text-embedding-ada-002", max_retries=3):
    text = text.replace("\n", " ")
    retries = 0
    while retries <= max_retries:
        try:
            response = client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error while generating embedding for text: {text}. Error: {e}")
            retries += 1
            if retries <= max_retries:
                time.sleep(2 ** retries)
            else:
                st.error(f"Max retries reached. Skipping keyword: {text}")
                return None

# Function to choose the best keyword
def choose_best_keyword(keyword1, keyword2):
    prompt = f"Identify which keyword users are more likely to search on Google for SEO: '{keyword1}' or '{keyword2}'. Only include the keyword in the response. If both keywords are similar, select the first one. You must choose a keyword based on which one has the best grammar, spelling, or natural language."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an SEO expert tasked with selecting the best keyword for search optimization."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=20
    )
    best_keyword_reason = response.choices[0].message.content.strip()

    if keyword1.lower() in best_keyword_reason.lower():
        return keyword1
    elif keyword2.lower() in best_keyword_reason.lower():
        return keyword2
    else:
        st.warning(f"Unexpected response from GPT-4: {best_keyword_reason}")
        return keyword1  # Fallback to the first keyword

# Function to identify primary variants
def identify_primary_variants(cluster_data):
    primary_variant_df = pd.DataFrame(columns=['Cluster ID', 'Keywords', 'Is Primary', 'Primary Keyword', 'GPT-4 Reason'])
    new_rows = []
    
    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        best_keyword_reason = None  # Reset for each cluster
        primary = None  # Initialize primary keyword

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
                'GPT-4 Reason': gpt_reason
            }
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)

# Process the data if both file and API key are provided
if uploaded_file and api_key:
    # Assuming columns are named 'Keywords', 'Search Volume', and 'CPC'
    keywords = data['Keywords'].tolist()
    search_volumes = data['Search Volume'].tolist()
    cpcs = data['CPC'].tolist()

    st.write("Generating embeddings for the keywords...")
    embeddings = [get_embedding(keyword) for keyword in keywords if get_embedding(keyword) is not None]

    if embeddings:
        st.write("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)

        # Incorporate Search Volume and CPC into the similarity matrix
        for i in range(len(keywords)):
            for j in range(len(keywords)):
                if search_volumes[i] != search_volumes[j] or cpcs[i] != cpcs[j]:
                    similarity_matrix[i][j] = 0

        st.write("Clustering keywords...")
        clusters = fcluster(linkage(1 - similarity_matrix, method='average'), t=0.2, criterion='distance')
        data['Cluster ID'] = clusters

        st.write("Identifying primary keywords within clusters...")
        cluster_data = data[['Cluster ID', 'Keywords']]
        primary_variant_df = identify_primary_variants(cluster_data)
        combined_data = pd.merge(data, primary_variant_df, on=['Cluster ID', 'Keywords'], how='left')

        # Output results
        st.write("Analysis complete. Review the clusters below:")
        st.dataframe(combined_data)

        # Download link
        st.download_button('Download Analysis Results', combined_data.to_csv(index=False).encode('utf-8'), 'analysis_results.csv', 'text/csv', key='download-csv')
    else:
        st.error("Failed to generate embeddings for all keywords.")
