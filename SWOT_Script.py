#!/usr/bin/env python
# coding: utf-8

# OBJECTIVE: The objective is to develop a functional AI Agent, leveraging Streamlit and LangChain, capable of autonomously conducting a SWOT analysis from provided organizational data. This agent must deliver a comprehensive analysis, including a visual representation of key findings, demonstrating a practical application of AI in strategic business analysis.

# ## Import Library and dependencies

# In[19]:


# Importing Libraries
import os
import streamlit as st
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import matplotlib.pyplot as plt
import pypdf
import tiktoken


# ## Print Library versions

# In[21]:


# Printing Library Versions
print(f"streamlit version: {st.__version__}")
print(f"langchain version: {langchain.__version__}")
print(f"matplotlib version: {plt.matplotlib.__version__}")
print(f"tiktoken version: {tiktoken.__version__}")
print(f"pypdf version: {pypdf.__version__}")

try:
    import google.generativeai as genai
    print(f"google-generativeai version: {genai.__version__}")
except ImportError:
    print("google-generativeai not installed or version not accessible")


# ## Setting up API

# In[23]:


# Load API Key from environment variable
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize tiktoken for token counting
encoder = tiktoken.get_encoding("cl100k_base")

# Function to count tokens
def count_tokens(text):
    return len(encoder.encode(text))

# Configure Gemini model via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.7)


# ## Prompt for SWOT

# In[25]:


# Define the Prompt Template for SWOT Analysis
swot_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Perform a detailed SWOT analysis on the following text:
    {text}

    Provide the response in this structured format and give maximum of 4 pointers for each component of SWOT, be factual and
    data-driven whenever possible. Refer to provided input text thoroughly:
    
    **Strengths:**
    - Example Strength 1
    - Example Strength 2

    **Weaknesses:**
    - Example Weakness 1
    - Example Weakness 2

    **Opportunities:**
    - Example Opportunity 1
    - Example Opportunity 2

    **Threats:**
    - Example Threat 1
    - Example Threat 2
    """
)


# ## Initialize LLM Chain

# In[27]:


# Create LLM Chain
swot_chain = LLMChain(llm=llm, prompt=swot_prompt)

def parse_swot(text):
    sections = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
    current_section = None

    for line in text.split("\n"):
        line = line.strip()

        # Handle different section markers
        if "Strengths" in line:
            current_section = "Strengths"
        elif "Weaknesses" in line:
            current_section = "Weaknesses"
        elif "Opportunities" in line:
            current_section = "Opportunities"
        elif "Threats" in line:
            current_section = "Threats"
        elif line.startswith("-") and current_section:
            sections[current_section].append(line[1:].strip())

    return sections


# ## Display SWOT in quadrant format

# In[29]:


# Function to display SWOT in quadrant format
def display_swot_quadrant(swot_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    
    colors = {"Strengths": "#add8e6", "Weaknesses": "#f4a582", "Opportunities": "#b3e2a9", "Threats": "#f4c2c2"}
    positions = {"Strengths": (0, 1), "Weaknesses": (1, 1), "Opportunities": (0, 0), "Threats": (1, 0)}
    
    for key, (x, y) in positions.items():
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=colors[key], alpha=0.5))
        ax.text(x + 0.5, y + 0.8, key, fontsize=12, fontweight='bold', ha='center')
        text = "\n".join(swot_data[key])
        ax.text(x + 0.5, y + 0.4, text, fontsize=10, ha='center', va='center')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    st.pyplot(fig)


# ## Set up WebApp page

# In[31]:


# Set the page title
st.set_page_config(page_title="Smart SWOT Analysis")

# Display the title
st.title("Smart SWOT Analysis")


# Display Objective
st.write("## OBJECTIVE:")
st.write("Let's unlock insights from your business extract with a quick SWOT analysis.")

# Provide instructions
st.write("Upload a PDF or paste a business extract below, and the AI will generate a SWOT analysis:")


# File upload
uploaded_file = st.file_uploader("Choose an upload file", type="pdf")

if uploaded_file is not None:
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        input_text = text  # Set input_text to the extracted PDF text
        st.write("PDF text extracted.")

    except pypdf.errors.PdfReadError:
        st.error("Error: Unable to read PDF. Please ensure it is not corrupted.")
        input_text = ""

else:
    # Text input
    input_text = st.text_area("Or, enter text here:")


# ## Generate SWOT and display on WebApp

# In[35]:


# Initialize session state for token tracking
if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0


# Generate and display the SWOT analysis

if st.button("Generate SWOT Analysis"):
    if input_text:
        with st.spinner('Generating SWOT analysis...'):
            try:
                swot_report = swot_chain.run(text=input_text)
                swot_data = parse_swot(swot_report)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a PDF or enter text before generating the SWOT analysis.")

    # Display SWOT analysis
    if any(swot_data.values()):
        col_s, col_w = st.columns(2)
        with col_s:
            st.subheader("🔵 Strengths")
            for item in swot_data["Strengths"]:
                st.write(f"- {item}")
        with col_w:
            st.subheader("🟠 Weaknesses")
            for item in swot_data["Weaknesses"]:
                st.write(f"- {item}")
        col_o, col_t = st.columns(2)
        with col_o:
            st.subheader("🟢 Opportunities")
            for item in swot_data["Opportunities"]:
                st.write(f"- {item}")
        with col_t:
            st.subheader("🔴 Threats")
            for item in swot_data["Threats"]:
                st.write(f"- {item}")
    else:
        st.error("Parsing failed! Check the LLM output formatting.")


    # Prepare data for download
    report_bytes = swot_report.encode("utf-8")  # Encode string to bytes
    file_name = "swot_report.txt"  # Set filename
    mime_type = "text/plain" #set mime type

    # Download button
    st.download_button(
        label="Download SWOT Report",
        data=report_bytes,
        file_name=file_name,
        mime=mime_type,
    )


    # Calculate token counts
    query_tokens = count_tokens(input_text)
    response_tokens = count_tokens(swot_report)

    # Update session state
    st.session_state.query_tokens += query_tokens
    st.session_state.response_tokens += response_tokens
    st.session_state.tokens_consumed += (query_tokens + response_tokens)


# ## Display token usage in sidebar

# In[43]:


# Display token usage in sidebar
st.sidebar.write(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")
st.sidebar.write(f"Query Tokens: {st.session_state.query_tokens}")
st.sidebar.write(f"Response Tokens: {st.session_state.response_tokens}")

query_tokens = 0  # Initialize query_tokens
response_tokens = 0 # Initialize response_tokens

print("Tokens consumed in this transaction...")
print("Query token = ", query_tokens)
print("Response tokens = ", response_tokens)

# Reset session state for token counts
st.session_state.tokens_consumed = 0
st.session_state.query_tokens = 0
st.session_state.response_tokens = 0


# In[ ]:





# In[ ]:




