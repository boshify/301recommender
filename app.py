import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import io

# Title of the app
st.title('301 Recommender')

# OpenAI API key from secrets
openai.api_key = st.secrets["api_key"]

# Function to get embeddings from OpenAI
def get_embedding(text):
    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding

# Step 1: Upload CSV file
st.header("Upload Crawl Data CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Initialize the dataframe
df = None

if uploaded_file is not None:
    # Step 2: Load and display the uploaded data
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded data:")
    st.dataframe(df.head())

    # Extract original file name for the download file name
    original_filename = uploaded_file.name.split(".")[0]

    # Step 3: Map the URL and Status Code columns
    st.header("Map URL and Status Code Columns")
    url_column = st.selectbox("Select the URL Column", df.columns)
    status_code_column = st.selectbox("Select the Status Code Column", df.columns)

    # Step 4: Button to trigger recommendation
    if st.button("Recommend 301s"):
        # Filter rows where the status code is 4xx or 5xx
        def filter_errors(status_code):
            return str(status_code).startswith('4') or str(status_code).startswith('5')
        
        # Filter the dataframe for 4xx and 5xx status codes
        df_errors = df[df[status_code_column].apply(filter_errors)]

        # Filter the dataframe for working URLs (status code 200)
        df_working = df[df[status_code_column] == 200]

        # Check if there are any errors to process
        if df_errors.empty or df_working.empty:
            st.write("No 4xx/5xx errors found or no 200 working URLs found.")
        else:
            # Add progress bar and message
            progress_bar = st.progress(0)
            st.write(f"Processing {len(df_errors)} error rows...")

            # Step 5: Get embeddings for error URLs and working URLs
            embeddings_errors = []
            embeddings_working = []
            try:
                # Fetch embeddings for the error URLs
                for url in df_errors[url_column]:
                    embeddings_errors.append(get_embedding(url))
                
                # Fetch embeddings for the working URLs
                for url in df_working[url_column]:
                    embeddings_working.append(get_embedding(url))
                
                # Check if embeddings are successfully generated
                st.write(f"Generated {len(embeddings_errors)} embeddings for error URLs.")
                st.write(f"Generated {len(embeddings_working)} embeddings for working URLs.")
            
            except Exception as e:
                st.write(f"Error fetching embeddings: {e}")
                st.stop()

            # Step 6: Calculate recommendations using cosine similarity
            df_output = df_errors[[url_column, status_code_column]].copy()
            
            # Debugging: Print the size of the output DataFrame and working DataFrame
            st.write(f"Number of error URLs: {len(df_output)}")
            st.write(f"Number of working URLs: {len(df_working)}")
            
            df_output['Recommendation 1'] = ''
            df_output['Recommendation 2'] = ''
            df_output['Recommendation 3'] = ''

            # Calculate cosine similarity and recommend top 3 URLs
            similarity_matrix = cosine_similarity(embeddings_errors, embeddings_working)
            st.write(f"Similarity matrix shape: {similarity_matrix.shape}")
            
            # Loop through error URLs by index
            for idx in range(len(df_output)):
                # Get top 3 most similar working URLs
                top_3_indices = np.argsort(similarity_matrix[idx])[-3:][::-1]
                
                # Debugging: Output the top 3 indices
                st.write(f"Top 3 indices for URL {df_output.iloc[idx][url_column]}: {top_3_indices}")
                
                # Extract the URLs from the working DataFrame
                top_3_urls = df_working.iloc[top_3_indices][url_column].values
                
                # Debugging: Print the actual recommended URLs to ensure correct extraction
                st.write(f"Top 3 recommended URLs for {df_output.iloc[idx][url_column]}: {top_3_urls}")
                
                # Assign the recommended URLs to the output DataFrame
                if len(top_3_urls) > 0:
                    df_output.at[idx, 'Recommendation 1'] = top_3_urls[0]
                if len(top_3_urls) > 1:
                    df_output.at[idx, 'Recommendation 2'] = top_3_urls[1]
                if len(top_3_urls) > 2:
                    df_output.at[idx, 'Recommendation 3'] = top_3_urls[2]

            # Display the final output table
            st.write("Processing complete!")
            st.header("301 Redirect Recommendations")
            st.dataframe(df_output)

            # Step 7: Create a downloadable CSV file
            output_filename = f"{original_filename} - 301 Recommendations.csv"
            csv_data = df_output.to_csv(index=False)

            # Create an in-memory buffer to hold the CSV
            buffer = io.StringIO(csv_data)

            # Provide the download button
            st.download_button(
                label="Download CSV",
                data=buffer.getvalue(),
                file_name=output_filename,
                mime="text/csv"
            )
