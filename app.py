import streamlit as st
import pandas as pd
from openai import OpenAI
import io

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["api_key"])

# Title of the app
st.title('301 Redirect Recommender')

# Function to generate prompt for OpenAI using the updated API
def get_redirect_suggestion(broken_url, working_urls):
    prompt = (f"This URL slug is broken and does not serve a page: {broken_url}. "
              f"Recommend the best URL to redirect it to using semantic context from this list. "
              f"Only output the URL slug and no additional text:\n{working_urls}")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for recommending redirects."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error fetching recommendations from OpenAI: {e}")
        return None

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

    # Generate 301 recommendations
    if st.button("Generate 301 Recommendations"):
        # Filter 4xx/5xx URLs (error pages)
        df_errors = df[df[status_code_column].astype(str).str.startswith(('4', '5'))]
        # Filter 200 URLs (working pages)
        df_working = df[df[status_code_column] == 200]

        if df_errors.empty or df_working.empty:
            st.write("No 4xx/5xx errors or no 200 working URLs found.")
        else:
            st.write(f"Found {len(df_errors)} error pages and {len(df_working)} working pages.")
            
            # Get list of working URLs
            working_urls_list = "\n".join(df_working[url_column].tolist())
            
            # Create DataFrame for output and initialize recommendation columns
            df_output = df_errors[[url_column, status_code_column]].copy()
            df_output['Recommendation 1'] = ''
            df_output['Recommendation 2'] = ''

            # Loop through error URLs to get top 2 redirect recommendations
            for idx, row in df_output.iterrows():
                broken_url = row[url_column]
                
                # Get the first recommendation
                recommendation_1 = get_redirect_suggestion(broken_url, working_urls_list)
                
                if recommendation_1:
                    # Get the second recommendation by removing the first recommendation from the list
                    working_urls_list_filtered = "\n".join([url for url in df_working[url_column].tolist() if url != recommendation_1])
                    recommendation_2 = get_redirect_suggestion(broken_url, working_urls_list_filtered)
                else:
                    recommendation_2 = None

                # Assign recommendations to the output DataFrame
                df_output.at[idx, 'Recommendation 1'] = recommendation_1 if recommendation_1 else ''
                df_output.at[idx, 'Recommendation 2'] = recommendation_2 if recommendation_2 else ''

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
