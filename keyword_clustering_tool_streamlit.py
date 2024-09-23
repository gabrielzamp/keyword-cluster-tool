import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
import io
from openai import OpenAI

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
    except Exception as e:
        st.error(f"Error choosing best keyword: {e}")
        return keyword1

async def identify_primary_variants(session, cluster_data):
    primary_variant_df = pd.DataFrame(columns=['Cluster ID', 'Keyword', 'Is Primary', 'Primary Keyword'])
    new_rows = []

    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keyword'].tolist()
        
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
            continue  # Skip clusters with only one keyword (shouldn't happen, but just in case)

        for keyword in keywords:
            is_primary = 'Yes' if keyword == primary else 'No'
            new_row = {
                'Cluster ID': cluster_id,
                'Keyword': keyword,
                'Is Primary': is_primary,
                'Primary Keyword': primary
            }
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)

async def process_data(keywords, search_volumes, cpcs):
    progress_bar = st.progress(0)
    
    with suppress_stdout():
        st.write("Generating embeddings for the keywords...")
        embeddings, valid_keywords = await generate_embeddings(keywords)
        progress_bar.progress(0.25)

        if embeddings:
            st.write("Clustering keywords...")
            
            # Create a DataFrame with all the data
            df = pd.DataFrame({
                'Keyword': valid_keywords,
                'Search Volume': search_volumes,
                'CPC': cpcs
            })
            df['Embedding'] = embeddings

            # Group by Search Volume and CPC
            grouped = df.groupby(['Search Volume', 'CPC'])

            clusters = []
            cluster_id = 0

            for (sv, cpc), group in grouped:
                if len(group) > 1:  # Only process groups with more than one keyword
                    group_embeddings = np.array(group['Embedding'].tolist())
                    similarity_matrix = cosine_similarity(group_embeddings)
                    
                    # Cluster within this SV and CPC group
                    group_clusters = fcluster(linkage(1 - similarity_matrix, method='average'), t=0.2, criterion='distance')
                    
                    # Assign cluster IDs
                    for gc in group_clusters:
                        clusters.append(cluster_id + gc)
                    cluster_id = max(clusters)
                else:
                    # Assign -1 to indicate this keyword is not in any cluster
                    clusters.extend([-1])

            df['Cluster ID'] = clusters

            # Filter out keywords not in any cluster
            df_clustered = df[df['Cluster ID'] != -1].copy()

            progress_bar.progress(0.75)

            st.write("Identifying primary keywords within clusters...")
            async with aiohttp.ClientSession() as session:
                primary_variant_df = await identify_primary_variants(session, df_clustered[['Cluster ID', 'Keyword']])
                combined_data = pd.merge(df_clustered, primary_variant_df, on=['Cluster ID', 'Keyword'], how='left')

            st.write("Analysis complete. Review the clusters below:")
            st.dataframe(combined_data)

            st.download_button(
                label='Download Analysis Results',
                data=combined_data.to_csv(index=False).encode('utf-8'),
                file_name='analysis_results.csv',
                mime='text/csv',
                key='download-csv'
            )
            progress_bar.progress(1.0)
        else:
            st.error("Failed to generate embeddings for all keywords.")

# Make sure to update the main part of the script to pass search_volumes and cpcs
if uploaded_file is not None and api_key:
    keywords = data['Keywords'].tolist()
    search_volumes = data['Search Volume'].tolist()
    cpcs = data['CPC'].tolist()
    with suppress_stdout():
        asyncio.run(process_data(keywords, search_volumes, cpcs))
