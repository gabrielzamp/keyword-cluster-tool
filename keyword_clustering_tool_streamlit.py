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

# Asynchronous function to get embeddings in parallel
async def fetch_embedding(session, text, model="text-embedding-ada-002"):
    try:
        response = await session.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": text, "model": model}
        )
        result = await response.json()
        return result['data'][0]['embedding']
    except Exception as e:
        st.error(f"Error fetching embedding for '{text}': {e}")
        return None

# Function to generate embeddings asynchronously
async def generate_embeddings(keywords):
    embeddings = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, keyword) for keyword in keywords]
        results = await asyncio.gather(*tasks)
        embeddings = [res for res in results if res is not None]
    return embeddings

# Function to choose the best keyword between two options using GPT-4
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
            st.warning(f"Unexpected response from GPT-4: {best_keyword}")
            return keyword1  # Fallback to the first keyword
    except Exception as e:
        st.error(f"Error choosing best keyword: {e}")
        return keyword1  # Fallback to the first keyword

# Function to identify primary variants
async def identify_primary_variants(session, cluster_data):
    primary_variant_df = pd.DataFrame(columns=['Cluster ID', 'Keywords', 'Is Primary', 'Primary Keyword', 'GPT-4 Reason'])
    new_rows = []
    
    for cluster_id, group in cluster_data.groupby('Cluster ID'):
        keywords = group['Keywords'].tolist()
        primary = None  # Initialize primary keyword

        if len(keywords) == 2:
            primary = await choose_best_keyword(session, keywords[0], keywords[1])
        else:
            embeddings = await generate_embeddings(keywords)
            similarity_matrix = cosine_similarity(np.array(embeddings))
            avg_similarity = np.mean(similarity_matrix, axis=1)
            primary_idx = np.argmax(avg_similarity)
            primary = keywords[primary_idx]

        for keyword in keywords:
            is_primary = 'Yes' if keyword == primary else 'No'
            new_row = {
                'Cluster ID': cluster_id,
                'Keywords': keyword,
                'Is Primary': is_primary,
                'Primary Keyword': primary,
                'GPT-4 Reason': primary  # Reason can be expanded if necessary
            }
            new_rows.append(new_row)

    return pd.DataFrame(new_rows)

# Function to process the data and run the analysis
async def process_data(keywords, search_volumes, cpcs):
    st.write("Generating embeddings for the keywords...")
    embeddings = await generate_embeddings(keywords)

    if embeddings:
        st.write("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)

        # Clustering keywords
        st.write("Clustering keywords...")
        clusters = fcluster(linkage(1 - similarity_matrix, method='average'), t=0.2, criterion='distance')
        data['Cluster ID'] = clusters

        # Identifying primary keywords
        st.write("Identifying primary keywords within clusters...")
        async with aiohttp.ClientSession() as session:
            primary_variant_df = await identify_primary_variants(session, data[['Cluster ID', 'Keywords']])
            combined_data = pd.merge(data, primary_variant_df, on=['Cluster ID', 'Keywords'], how='left')

        # Output results
        st.write("Analysis complete. Review the clusters below:")
        st.dataframe(combined_data)

        # Download link
        st.download_button('Download Analysis Results', combined_data.to_csv(index=False).encode('utf-8'), 'analysis_results.csv', 'text/csv', key='download-csv')
    else:
        st.error("Failed to generate embeddings for all keywords.")

# Run the analysis if both file and API key are provided
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
