import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import streamlit as st

# Ensure necessary downloads
nltk.download("stopwords")

# Function to process the uploaded text
def read_article(input_text):
    article = input_text.split(". ")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))

    if sentences and not sentences[-1]:  # Remove empty last sentence if any
        sentences.pop()

    return sentences

# Function to calculate sentence similarity
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

# Function to build the similarity matrix
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

# Function to generate the summary
def generate_summary(input_text, top_n=5):
    stop_words = stopwords.words('english')
    sentences = read_article(input_text)

    if len(sentences) == 0:
        return "No valid sentences found in the input."

    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n = min(top_n, len(ranked_sentence))  # Ensure we don't exceed available sentences

    summarize_text = []
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    return ". ".join(summarize_text)

# Streamlit app setup
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown(
    """
    <style>
    html, body, .main {
        background-color: transparent !important;
        color: inherit !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: inherit !important;
    }
    .stAlert, .stTextArea, .stMarkdown {
        color: inherit !important;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white !important;
        border-radius: 8px;
        padding: 8px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Fix yellow warning box in dark mode */
    [data-testid="stAlert"] {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #555 !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Title and description
st.title("📄 Text Summarizer Using NLP")
st.markdown("A simple yet powerful tool to summarize lengthy text documents into concise and meaningful summaries.")

# File uploader
st.sidebar.header("Upload Your Text File")
uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

# Input section
st.sidebar.header("Summary Settings")
top_n = st.sidebar.slider("Select Number of Sentences for the Summary", min_value=1, max_value=10, value=5)

st.sidebar.markdown(
    """
    Instructions:
    1. Upload a text file using the uploader.
    2. Adjust the slider to set the desired number of sentences.
    3. Click 'Generate Summary' to get the summarized text.
    """
)

# Processing the uploaded file
if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")
    st.subheader("📜 Original Text")
    st.text_area("Uploaded Text", input_text, height=250)

    if st.button("Generate Summary"):
        summary = generate_summary(input_text, top_n)
        st.subheader("📝 Generated Summary")
        st.success(summary)
else:
    st.warning("Please upload a text file to generate a summary.")

# Footer
st.markdown(
    """
    ---
    About  
    Built using 🐍 Python, 🖤 Streamlit, and Natural Language Processing (NLP) techniques.
    """
)