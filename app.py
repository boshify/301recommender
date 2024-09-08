import streamlit as st
import pandas as pd

# Title of the app
st.title('301 Recommender')

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
        
        # Check if there are any errors to process
        if df_errors.empty:
            st.write("No 4xx or 5xx errors found.")
        else:
            # Add progress bar and message
            progress_bar = st.progress(0)
            st.write(f"Processing {len(df_errors)} rows...")

            # Simulate row processing
            for i in range(1, 101):
                progress_bar.progress(i)

            # Step 5: Generate output table (empty recommendations for now)
            st.write("Processing complete!")

            # Create a new DataFrame for the final output with empty recommendation columns
            df_output = df_errors[[url_column, status_code_column]].copy()
            df_output['Recommendation 1'] = ''
            df_output['Recommendation 2'] = ''
            df_output['Recommendation 3'] = ''

            # Display the final output table
            st.header("301 Redirect Recommendations")
            st.dataframe(df_output)

# For future steps: Logic to fill in Recommendation columns
