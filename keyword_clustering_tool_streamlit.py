import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import io

st.title('Keyword Research Cluster Analysis Tool')
st.subheader('Leverage OpenAI to cluster similar keywords into groups from your keyword list.')

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

sample_data = pd.DataFrame({
    'Keywords': ['buy shoes online', 'purchase shoes online', 'best running shoes', 'comfortable walking shoes'],
    'Search Volume': [1000, 800, 1500, 1200],
    'CPC': [0.5, 0.4, 0.7, 0.6]
})

csv_buffer = io.StringIO()
sample_data.to_csv(csv_buffer, index=False)
csv_str = csv_buffer.getvalue()

st.download_button(
    label="Download Sample CSV Template",
    data=csv_str,
    file_name="sample_keyword_data.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    required_columns = {'Keywords', 'Search Volume', 'CPC'}
    if not required_columns.issubset(data.columns):
        st.error("CSV file must include the following columns: Keywords, Search Volume, CPC")

api_key = st.text_input("Enter your OpenAI API key", type="password")

async def fetch_embedding(session, text, model="text-embedding-ada-002", max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": text, "model": model},
                timeout=10
            ) as response:
                result = await response.json()
                return result['data'][0]['embedding']
        except asyncio.TimeoutError:
            retries += 1
        except Exception:
            retries += 1
    return None

async def generate_embeddings(keywords):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, keyword) for keyword in keywords]
        results = await asyncio.gather(*tasks)

    embeddings = [res for res in results if res is not None]
    valid_keywords = [kw for kw, res in zip(keywords, results) if res is not None]

    return embeddings, valid_keywords

async def choose_best_keyword(session, keyword1, keyword2):
    prompt = f"Identify which keyword users are more likely to search on Google for SEO: '{keyword1}' or '{keyword2}'. Only include the keyword in the response. If both keywords are similar, select the first one."
    try:
        response = await session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are an SEO expert tasked with selecting the best keyword for search optimization."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20
            }
        )
        result = await response.json()
        best_keyword = result['choices'][0]['message']['content'].strip()
        if keyword1.lower() in best_keyword.lower():
            return keyword1
        elif keyword2.lower() in best_keyword.lower():
            return keyword2
        else:
            return keyword1
    except Exception:
        return keyword1

async def identify_primary_variants(session, cluster_data):
    primary_variant_df = pd.DataFrame(columns=['Cluster ID', 'Keywords', 'Is Primary', 'Primary Keyword'])
    new_rows = []

    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        
        if len(keywords) == 2:
            primary = await choose_best_keyword(session, keywords[0], keywords[1])
        elif len(keywords) >= 3:
            embeddings, valid_keywords = await generate_embeddings(keywords)
            if not embeddings:
                continue
            similarity_matrix = cosine_similarity(np.array(embeddings))
            avg_similarity = np.mean(similarity_matrix, axis=1)
            primary_idx = np.argmax(avg_similarity)
            primary = valid_keywords[primary_idx]
        else:
            continue  # Skip clusters with only one keyword

        for keyword in keywords:
            is_primary = 'Yes' if keyword == primary else 'No'
            new_row = {
                'Cluster ID': cluster_id,
                'Keywords': keyword,
                'Is Primary': is_primary,
                'Primary Keyword': primary
            }
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)

async def process_data(keywords, search_volumes, cpcs):
    progress_bar = st.progress(0)
    
    # Create a DataFrame with all the data
    df = pd.DataFrame({
        'Keywords': keywords,
        'Search Volume': search_volumes,
        'CPC': cpcs
    })

    # Group by Search Volume and CPC to ensure only keywords with matching values are clustered
    grouped = df.groupby(['Search Volume', 'CPC'])

    clusters = [-1] * len(df)  # Initialize all cluster IDs as -1
    cluster_id = 0

    for name, group in grouped:
        group_indices = group.index.tolist()  # Get indices of the current group
        if len(group) > 1:  # Only process groups with more than one keyword
            keywords_group = group['Keywords'].tolist()
            embeddings, valid_keywords = await generate_embeddings(keywords_group)
            
            if embeddings:
                similarity_matrix = cosine_similarity(np.array(embeddings))
                group_clusters = fcluster(linkage(1 - similarity_matrix, method='average'), t=0.2, criterion='distance')
                
                for idx, gc in zip(group_indices, group_clusters):
                    clusters[idx] = cluster_id + gc
                cluster_id += max(group_clusters)  # Increment cluster ID correctly

    # Assign clusters to the DataFrame
    df['Cluster ID'] = clusters

    # Filter out keywords not in any cluster
    df_clustered = df[df['Cluster ID'] != -1].copy()

    # Extract unique keywords that were not clustered
    unique_keywords_df = df[df['Cluster ID'] == -1][['Keywords', 'Search Volume', 'CPC']]

    progress_bar.progress(0.5)

    if not df_clustered.empty:
        async with aiohttp.ClientSession() as session:
            primary_variant_df = await identify_primary_variants(session, df_clustered[['Cluster ID', 'Keywords']])
            combined_data = pd.merge(df_clustered, primary_variant_df, on=['Cluster ID', 'Keywords'], how='left')

        st.write("### Clustered Keywords")
        st.dataframe(combined_data)

        st.download_button(
            label='Download Clustered Analysis Results',
            data=combined_data.to_csv(index=False).encode('utf-8'),
            file_name='clustered_analysis_results.csv',
            mime='text/csv'
        )

        # Display unique keywords that were not clustered
        st.write("### Unique Keywords (Not Clustered)")
        st.dataframe(unique_keywords_df)

        # Download button for unique keywords
        st.download_button(
            label='Download Unique Keywords',
            data=unique_keywords_df.to_csv(index=False).encode('utf-8'),
            file_name='unique_keywords.csv',
            mime='text/csv'
        )

        progress_bar.progress(1.0)
    else:
        st.error("No clusters were formed due to unique CPC and Search Volume values.")

if uploaded_file is not None and api_key:
    keywords = data['Keywords'].tolist()
    search_volumes = data['Search Volume'].tolist()
    cpcs = data['CPC'].tolist()
    asyncio.run(process_data(keywords, search_volumes, cpcs))
elif uploaded_file is None and api_key:
    st.warning("Please upload a CSV file to proceed.")
elif uploaded_file is not None and not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
else:
    st.info("Please upload a CSV file and enter your OpenAI API key to start the analysis.")
