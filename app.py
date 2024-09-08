import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import io

# Title of the app
st.title('301 Redirect Recommender')

# OpenAI API key from secrets
openai.api_key = st.secrets["api_key"]

# Function to fetch embeddings from OpenAI API
def get_embedding(text):
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

# File uploader
st.header("Upload Your Crawl Data CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Extract the filename for output
    original_filename = uploaded_file.name.split(".")[0]

    # Map columns
    st.header("Map Columns")
    url_column = st.selectbox("Select the URL Column", df.columns)
    status_code_column = st.selectbox("Select the Status Code Column", df.columns)

    if st.button("Generate 301 Recommendations"):
        # Filter 4xx/5xx URLs (error pages)
        df_errors = df[df[status_code_column].astype(str).str.startswith(('4', '5'))]
        # Filter 200 URLs (working pages)
        df_working = df[df[status_code_column] == 200]

        if df_errors.empty or df_working.empty:
            st.write("No 4xx/5xx errors or no 200 working URLs found.")
        else:
            st.write(f"Found {len(df_errors)} error pages and {len(df_working)} working pages.")
            
            # Initialize empty embeddings
            embeddings_errors = []
            embeddings_working = []
            
            try:
                st.write("Fetching embeddings for error URLs...")
                embeddings_errors = [get_embedding(url) for url in df_errors[url_column]]
                st.write("Fetching embeddings for working URLs...")
                embeddings_working = [get_embedding(url) for url in df_working[url_column]]
            except Exception as e:
                st.error(f"Error fetching embeddings: {e}")
                st.stop()

            # Create DataFrame for output and initialize recommendation columns
            df_output = df_errors[[url_column, status_code_column]].copy()
            df_output['Recommendation 1'] = ''
            df_output['Recommendation 2'] = ''
            df_output['Recommendation 3'] = ''

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(embeddings_errors, embeddings_working)
            
            # Debugging: Show the shape of the similarity matrix
            st.write(f"Similarity matrix shape: {similarity_matrix.shape}")
            
            # Loop through error URLs to get top 3 working URLs
            for idx, row in df_output.iterrows():
                # Get top 3 similar working URLs based on cosine similarity
                top_3_indices = np.argsort(similarity_matrix[idx])[-3:][::-1]
                
                # Debugging: Output the top 3 indices for each error URL
                st.write(f"Top 3 indices for {row[url_column]}: {top_3_indices}")
                
                # Get the corresponding URLs from df_working
                top_3_urls = df_working.iloc[top_3_indices][url_column].values
                
                # Assign top 3 recommendations to the output DataFrame
                df_output.at[idx, 'Recommendation 1'] = top_3_urls[0] if len(top_3_urls) > 0 else ''
                df_output.at[idx, 'Recommendation 2'] = top_3_urls[1] if len(top_3_urls) > 1 else ''
                df_output.at[idx, 'Recommendation 3'] = top_3_urls[2] if len(top_3_urls) > 2 else ''

            # Display the final output table
            st.write("301 Redirect Recommendations")
            st.dataframe(df_output)

            # Create downloadable CSV
            output_filename = f"{original_filename} - 301 Recommendations.csv"
            csv_data = df_output.to_csv(index=False)

            # Provide download button
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv"
            )
