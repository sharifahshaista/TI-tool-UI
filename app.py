# type: ignore # type: ignore
"""
Streamlit Research Agent Web Application

This app provides a web interface for:
1. Research Pipeline - Topic research with clarification, SERP generation, and web search
2. Learning Extraction - Extract structured learnings from search results
"""

import streamlit as st
import asyncio
import os
import json
import time
import tempfile
import pandas as pd
import subprocess
import platform
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Optional: Gensim for keyword expansion
try:
    import gensim.downloader as api
    HAS_GENSIM = True
except ImportError:
    api = None  # type: ignore
    HAS_GENSIM = False

# Load environment variables at the module level
load_dotenv()

# Import existing modules
from agents.clarification import get_clarifications
from agents.serp import get_serp_queries
from config.searxng_tools import searxng_web_tool, searxng_client
from config.model_config import get_model
from schemas.datamodel import SearchResultsCollection
from embeddings import CSVEmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research.log'),
        logging.StreamHandler()
    ]
)

# Set page configuration for wide layout
st.set_page_config(
    page_title="TI tool",
    page_icon="🔬",
    layout="wide",  # Use full width of the page
    initial_sidebar_state="expanded"
)

# Custom CSS to change red colours to royal blue
st.markdown("""
    <style>
    /* Change primary button colour from red to royal blue */
    .stButton > button[kind="primary"] {
        background-color: #0E7490 !important;
        border-color: #4169E1 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #0E7490 !important;
        border-color: #1E40AF !important;
    }
    
    /* Change error messages from red to royal blue */
    .stAlert[data-baseweb="notification"][kind="error"] {
        background-color: rgba(65, 105, 225, 0.1) !important;
        border-left-color: #4169E1 !important;
    }
    
    /* Change warning colours to royal blue tones */
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background-color: rgba(65, 105, 225, 0.1) !important;
        border-left-color: #4169E1 !important;
    }
    
    /* Change progress bar colour to royal blue */
    .stProgress > div > div > div > div {
        background-color: #4169E1 !important;
    }
    
    /* Change download button hover to royal blue */
    .stDownloadButton > button:hover {
        border-color: #4169E1 !important;
        color: #4169E1 !important;
    }
    
    /* Change radio button selected state to royal blue */
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #4169E1 !important;
    }
    
    /* Change checkbox selected state to royal blue */
    .stCheckbox > label > div[data-baseweb="checkbox"] > div {
        border-color: #4169E1 !important;
        background-color: #4169E1 !important;
    }
    
    /* Change slider to royal blue */
    .stSlider > div > div > div > div {
        background-color: #4169E1 !important;
    }
    
    /* Change number input focus border to royal blue */
    .stNumberInput > div > div > input:focus {
        border-color: #4169E1 !important;
        box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
    }
    
    /* Change text input focus border to royal blue */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4169E1 !important;
        box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
    }
    
    /* Change selectbox focus to royal blue */
    .stSelectbox > div > div > div:focus-within {
        border-color: #4169E1 !important;
        box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
    }
    
    /* Change metric delta positive color to royal blue */
    [data-testid="stMetricDelta"] svg {
        fill: #4169E1 !important;
    }
    
    /* Change links to royal blue */
    a {
        color: #4169E1 !important;
    }
    
    a:hover {
        color: #1E40AF !important;
    }
    
    /* Change spinner to royal blue */
    .stSpinner > div {
        border-top-color: #4169E1 !important;
    }
    
    /* Change multiselect selected items to royal blue */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #4169E1 !important;
        color: white !important;
    }
    
    /* Change date input focus to royal blue */
    .stDateInput > div > div > input:focus {
        border-color: #4169E1 !important;
        box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
    }
    
    /* Change expander header to royal blue accent */
    .streamlit-expanderHeader {
        background-color: rgba(65, 105, 225, 0.05) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(65, 105, 225, 0.1) !important;
    }
    
    /* Make sidebar divider line black */
    section[data-testid="stSidebar"] > div {
        border-right: 2px solid #000000 !important;
    }
    
    /* Sidebar background colour */
    section[data-testid="stSidebar"] {
        background-color: #C8D3DF !important;
    }
    
    /* Sidebar buttons - complementary colours for turquoise background */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #0E7490 !important;  /* Cyan-700 - darker teal */
        color: white !important;
        border: 1px solid #0E7490 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #155E75 !important;  /* Cyan-800 - even darker on hover */
        border-color: #155E75 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: #000000 !important;  /* Black - clearly indicates current section */
        border-color: #000000 !important;
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: #1A1A1A !important;  /* Very dark gray on hover */
        border-color: #1A1A1A !important;
    }
    
    /* Sidebar input fields - white background for contrast */
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input,
    section[data-testid="stSidebar"] .stTextArea > div > div > textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #0E7490 !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput > div > div > input:focus,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input:focus,
    section[data-testid="stSidebar"] .stTextArea > div > div > textarea:focus {
        border-color: #075985 !important;
        box-shadow: 0 0 0 0.2rem rgba(7, 89, 133, 0.25) !important;
    }
    
    /* Sidebar selectbox and multiselect - white background */
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #0E7490 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
    section[data-testid="stSidebar"] .stMultiSelect > div > div:focus-within {
        border-color: #075985 !important;
        box-shadow: 0 0 0 0.2rem rgba(7, 89, 133, 0.25) !important;
    }
    
    /* Sidebar slider - complementary colour */
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #0E7490 !important;
    }
    
    /* Sidebar metric boxes - white with border */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #22D3EE !important;  /* Cyan-400 - bright accent */
        border-radius: 0.5rem !important;
        padding: 0.25rem 0.5rem !important;  /* Reduced padding */
    }
    
    /* Reduce font size for sidebar metric labels */
    section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        font-size: 11px !important;
    }
    
    /* Reduce font size for sidebar metric values */
    section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 16px !important;
    }
    
    /* Reduce overall metric container height */
    section[data-testid="stSidebar"] [data-testid="stMetric"] > div {
        gap: 2px !important;
    }
    
    /* Sidebar success/info/warning messages */
    section[data-testid="stSidebar"] .stAlert {
        background-color: #FFFFFF !important;
        border-left: 4px solid #0E7490 !important;
    }
    
    /* Sidebar expander */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #A5F3FC !important;  /* Cyan-200 - light turquoise */
        border: 1px solid #22D3EE !important;
        border-radius: 0.25rem !important;
    }
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: #67E8F9 !important;  /* Cyan-300 - brighter on hover */
    }
    
    /* Sidebar radio buttons */
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #0E7490 !important;
    }
    
    /* Alternative selector for sidebar border */
    .css-1d391kg, .st-emotion-cache-1d391kg {
        border-right: 2px solid #000000 !important;
    }
    
    /* Reduce font size for multiselect widget (embeddings selector) */
    .stMultiSelect [data-baseweb="tag"] {
        font-size: 11px !important;
    }
    
    .stMultiSelect [data-baseweb="select"] span {
        font-size: 12px !important;
    }
    
    /* Reduce font size in multiselect dropdown options */
    [data-baseweb="menu"] li {
        font-size: 12px !important;
    }
    
    /* Make multiselect widget wider to accommodate longer filenames */
    .stMultiSelect {
        width: 100% !important;
    }
    
    .stMultiSelect > div {
        width: 100% !important;
    }
    
    .stMultiSelect [data-baseweb="select"] {
        width: 100% !important;
        min-width: 300px !important;
    }
    
    /* Allow tags to wrap to multiple lines if needed */
    .stMultiSelect [data-baseweb="tag"] {
        white-space: normal !important;
        word-break: break-word !important;
    }
    
    /* Increase font size for sidebar markdown text (feature names) */
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 16px !important;
        color: #0D0D0D !important; 
    }
    
    /* Increase font size for sidebar markdown strong text (bold feature names) */
    section[data-testid="stSidebar"] .stMarkdown strong {
        font-size: 16px !important;
        color: #0D0D0D !important; 
    }
    
    /* Increase font size for sidebar captions */
    section[data-testid="stSidebar"] .stCaptionContainer p {
        font-size: 14px !important;
        color: #000000 !important;
    }
    
    /* Increase font size for sidebar header (Features) */
    section[data-testid="stSidebar"] h1 {
        font-size: 32px !important;
    }
    
    /* Change chat input box background to light grey */
    .stChatInput > div > div > textarea {
        background-color: #F5F5F5 !important;  /* Light grey */
    }
    
    /* Chat input container background */
    .stChatInput > div {
        background-color: #F5F5F5 !important;  /* Light grey */
    }
    
    /* Change expander header text to red (for "Steps to use the Chatbot") */
    .streamlit-expanderHeader p {
        color: #FF0000 !important;  /* Red */
    }
    
    /* Change expander content text to red */
    .streamlit-expanderContent p {
        color: #FF0000 !important;  /* Red */
    }

    
    </style>
    """, unsafe_allow_html=True)

# Helper function to format time
def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# Custom logging handler to capture logs for Streamlit display
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that stores logs in session state."""
    
    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if 'crawl_logs' not in st.session_state:
                st.session_state.crawl_logs = []
            st.session_state.crawl_logs.append(msg)
            # Keep only last 200 log entries to avoid memory issues
            if len(st.session_state.crawl_logs) > 200:
                st.session_state.crawl_logs = st.session_state.crawl_logs[-200:]
        except Exception:
            self.handleError(record)


# Initialize session state
if 'crawl_logs' not in st.session_state:
    st.session_state.crawl_logs = []
if 'csv_processing' not in st.session_state:
    st.session_state.csv_processing = False
if 'csv_processed_df' not in st.session_state:
    st.session_state.csv_processed_df = None
if 'csv_metadata' not in st.session_state:
    st.session_state.csv_metadata = None
if 'csv_progress' not in st.session_state:
    st.session_state.csv_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
st.session_state.crawl_results = None
st.session_state.crawling_in_progress = False
st.session_state.crawl_cancel_requested = False
st.session_state.crawl_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0, 'start_time': None}
if 'strategy_confirmed' not in st.session_state:
    st.session_state.strategy_confirmed = False
if 'clarifications' not in st.session_state:
    st.session_state.clarifications = None
if 'serp_queries' not in st.session_state:
    st.session_state.serp_queries = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None
if 'learnings' not in st.session_state:
    st.session_state.learnings = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
if 'rag_store' not in st.session_state:
    st.session_state.rag_store = None
if 'rag_chat' not in st.session_state:
    st.session_state.rag_chat = []
if 'selected_model_config' not in st.session_state:
    st.session_state.selected_model_config = {'provider': 'azure', 'model_name': None}


def reset_session_state():
    """Reset all session state variables to default values"""
    st.session_state.crawl_logs = []
    st.session_state.csv_processing = False
    st.session_state.csv_processed_df = None
    st.session_state.csv_metadata = None
    st.session_state.csv_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
    st.session_state.crawl_results = None
    st.session_state.crawling_in_progress = False
    st.session_state.crawl_cancel_requested = False
    st.session_state.crawl_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0, 'start_time': None}
    st.session_state.strategy_confirmed = False
    st.session_state.clarifications = None
    st.session_state.serp_queries = None
    st.session_state.search_results = None
    st.session_state.current_stage = None
    st.session_state.learnings = None
    st.session_state.detection_results = None
    st.session_state.processing_in_progress = False
    st.session_state.processing_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
    st.session_state.rag_store = None
    st.session_state.rag_chat = []
    st.session_state.selected_model_config = {'provider': 'azure', 'model_name': None}


def standardize_url(url: str) -> str:
    """Ensure URL has a scheme (https://"""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


# Helper function to run async code in Streamlit
def run_async(coro):
    """Run async coroutine in Streamlit-compatible way"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
        return result
    finally:
        loop.close()


def run_clarification_stage(topic: str):
    """Stage 1: Get clarifications (Streamlit version - no input())"""
    async def _async_clarification():
        from agents.clarification import create_clarification_agent
        from schemas.datamodel import ClarificationResponse, ClarificationQuestion

        clarification_agent = create_clarification_agent()
        response = ClarificationResponse(original_query=topic)

        try:
            # Get clarification questions from agent (no user input)
            result = await clarification_agent.run(topic)
            questions = result.output

            # Parse questions and add them WITHOUT answers
            if questions:
                for i, question in enumerate(questions.split('\n'), 1):
                    if question.strip():
                        qa = ClarificationQuestion(
                            question=question.strip(),
                            answer=""  # Empty - will be filled by Streamlit form
                        )
                        response.questions_and_answers.append(qa)
        except Exception as e:
            logging.error(f"Error getting clarifications: {e}")

        logging.info(f"Clarifications received: {len(response.questions_and_answers)} questions")
        return response

    with st.spinner("Generating clarification questions..."):
        clarifications = run_async(_async_clarification())
        st.session_state.clarifications = clarifications
    return clarifications


def run_serp_generation_stage(topic: str, clarifications):
    """Stage 2: Generate SERP queries"""
    async def _async_serp():
        serp_queries = await get_serp_queries(topic, clarifications)
        logging.info(f"Generated {len(serp_queries)} SERP queries")
        return serp_queries

    with st.spinner("Generating SERP queries..."):
        serp_queries = run_async(_async_serp())
        st.session_state.serp_queries = serp_queries
    return serp_queries


def run_search_stage(serp_queries):
    """Stage 3: Execute web searches"""
    async def _async_search():
        results_collection = SearchResultsCollection()
        total_queries = len(serp_queries)

        for idx, query in enumerate(serp_queries, 1):
            status_text.text(f"Searching [{idx}/{total_queries}]: {query}")
            logging.info(f"Searching [{idx}/{total_queries}]: {query}")

            try:
                results = await searxng_client._search(query)
                results_collection.add_result(query, results)
                logging.info(f"Search successful: {query}")
            except Exception as e:
                st.warning(f"Search failed for '{query}': {e}")
                logging.error(f"Search failed for '{query}': {e}")
                continue

            progress_bar.progress(idx / total_queries)

        return results_collection

    progress_bar = st.progress(0)
    status_text = st.empty()

    results_collection = run_async(_async_search())

    status_text.text(f"Search complete! {results_collection.total_queries} queries executed.")
    st.session_state.search_results = results_collection

    return results_collection


def run_learning_extraction_stage(results_collection):
    """Stage 4: Extract learnings"""
    async def _async_learning():
        learnings_dict = {}
        results_list = list(results_collection.results.items())
        total_queries = len(results_list)

        for idx, (query, search_result) in enumerate(results_list, 1):
            status_text.text(f"Extracting learnings [{idx}/{total_queries}]: {query[:50]}...")
            logging.info(f"Extracting learnings [{idx}/{total_queries}]: {query}")

            try:
                learnings = await get_learning_structured(query, search_result.results)
                learnings_dict[query] = learnings
                logging.info(f"Learning extraction successful: {query}")
            except Exception as e:
                st.warning(f"Learning extraction failed for '{query}': {e}")
                logging.error(f"Learning extraction failed for '{query}': {e}")
                continue

            progress_bar.progress(idx / total_queries)

        return learnings_dict

    progress_bar = st.progress(0)
    status_text = st.empty()

    learnings_dict = run_async(_async_learning())

    status_text.text(f"Learning extraction complete! {len(learnings_dict)} learnings extracted.")
    st.session_state.learnings = learnings_dict

    return learnings_dict


def save_search_results(results_collection, filename):
    """Save search results to JSON file and upload to S3"""
    path = Path("data")
    path.mkdir(parents=True, exist_ok=True)

    output_path = path / f"{filename}.json"
    results_collection.to_file(output_path)
    
    # Upload to S3 if configured
    s3_uploaded = False
    s3_key = None
    s3_error = None
    
    try:
        from aws_storage import get_storage
        s3_storage = get_storage()
        s3_key = f"research_results/{filename}.json"
        s3_storage.upload_file(str(output_path), s3_key)
        s3_uploaded = True
        logging.info(f"✓ Uploaded search results to S3: {s3_key}")
    except Exception as e:
        s3_error = str(e)
        logging.warning(f"⚠️ S3 upload skipped: {e}")

    return output_path, s3_uploaded, s3_key, s3_error


def save_learnings(learnings_dict, filename):
    """Save learnings to markdown file"""
    path = Path("data")
    path.mkdir(parents=True, exist_ok=True)

    output_path = path / f"{filename}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        for query, learnings_obj in learnings_dict.items():
            f.write(f"## {query}\n\n")
            # Extract the learnings text from the QueryLearnings object
            learnings_text = learnings_obj.learnings if hasattr(learnings_obj, 'learnings') else str(learnings_obj)
            f.write(learnings_text)
            f.write("\n\n---\n\n")
    
    # Upload to S3 if configured
    try:
        from aws_storage import get_storage
        s3_storage = get_storage()
        s3_key = f"research_results/{filename}.md"
        s3_storage.upload_file(str(output_path), s3_key)
        logging.info(f"✓ Uploaded learnings to S3: {s3_key}")
    except Exception as e:
        logging.warning(f"⚠️ S3 upload skipped: {e}")

    return output_path


def load_search_results(file_path):
    """Load search results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    collection = SearchResultsCollection()
    for query, results in data.items():
        collection.add_result(query, results)

    return collection


# Sidebar
# Helper function to convert image to base64
import base64

def get_base64_image(image_path):
    """Convert image to base64 for inline HTML display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR / "assets"

# Get tool icon for sidebar title
tool_icon = get_base64_image(ASSETS_DIR / "tool_icon.png")

# Display title with icon
if tool_icon:
    st.sidebar.markdown(f'''
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <img src="data:image/png;base64,{tool_icon}" width="50" height="50">
            <h1 style="margin: 0; font-size: 36px;">Technology Intelligence Tool</h1>
        </div>
    ''', unsafe_allow_html=True)
else:
    st.sidebar.title("Technology Intelligence Tool")

st.sidebar.markdown("AI-powered tool equipped with discovery of sources, a combined database and chatbot")

st.sidebar.markdown("<em>Click the buttons below!</em>", unsafe_allow_html=True)

# Get icon for Web Search
web_search_icon = get_base64_image(ASSETS_DIR / "websearch_icon.png") 
folder_icon = get_base64_image(ASSETS_DIR / "folder_icon.png") 
chatbot_icon = get_base64_image(ASSETS_DIR / "chatbot_icon.png") 
about_icon = get_base64_image(ASSETS_DIR / "about_icon.png")
linkedin_icon = get_base64_image(ASSETS_DIR / "linkedin_icon.png")

# Initialize selected page in session state
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Web Search"

# Define features with icons and descriptions
features = [
    {
        "name": "Web Search",
        "icon": web_search_icon,
        "description": "Find sources relevant to your research questions"
    },
    {
        "name": "Database",
        "icon": folder_icon,
        "description": "Explore and filter combined database content containing crawled and summarised content"
    },
    {
        "name": "Chatbot",
        "icon": chatbot_icon,
        "description": "Chat with your technology intelligence database using AI"
    },
    {
        "name": "LinkedIn Home Feed Monitor",
        "icon": linkedin_icon,
        "description": "Monitor and analyse LinkedIn home feed content"
    },
    {
        "name": "About",
        "icon": about_icon,
        "description": "Learn about the Technology Intelligence Tool"
    }
]

# Create clickable feature cards
for feature in features:
    is_selected = st.session_state.selected_page == feature["name"]
    
    # Create a custom HTML button with icon and description side-by-side
    if feature["icon"]:
        # Use columns for layout: icon on left, content on right - tighter spacing
        col1, col2 = st.sidebar.columns([0.8, 5])
        
        with col1:
            # Display icon
            st.markdown(f'''
                <img src="data:image/png;base64,{feature["icon"]}" width="35" height="35" style="margin-top: 8px;">
            ''', unsafe_allow_html=True)
        
        with col2:
            # Button with title
            if st.button(
                feature["name"],
                key=f"btn_{feature['name']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_page = feature["name"]
                st.rerun()
            
            # Description below button in same column
            st.markdown(f'<em style="font-size: 14px; color: #000000; display: block; margin-top: -15px; margin-bottom: 5px;">{feature["description"]}</em>', unsafe_allow_html=True)
    else:
        # Fallback without icon
        if st.sidebar.button(
            feature["name"],
            key=f"btn_{feature['name']}",
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.selected_page = feature["name"]
            st.rerun()
        
        st.sidebar.markdown(f'<em style="font-size: 12px; color: #000000; margin-bottom: 16px; display: block;">{feature["description"]}</em>', unsafe_allow_html=True)

# Set page to the selected page
page = st.session_state.selected_page

# Web Search Page
if page == "Web Search":
    st.header("Web Search")
    st.markdown("AI-powered research with clarification, SERP generation, and web search")
    
    st.subheader("Research Pipeline")
    st.markdown("Conduct comprehensive research with automated clarification and web search")

    # Input Section
    st.subheader("1. Research Topic")
    topic = st.text_area(
        "Enter your research topic:",
        placeholder="e.g., What is the expected growth rate for Singapore companies that sell AI solutions?",
        height=100
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_research = st.button("Start Research", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("Reset", use_container_width=True)

    if reset_btn:
        reset_session_state()
        st.rerun()

    if start_research and topic:
        st.session_state.current_stage = 'clarification'

    # Clarification Stage
    if st.session_state.current_stage == 'clarification' and topic:
        st.subheader("2. Clarification Questions")

        if st.session_state.clarifications is None:
            # Generate clarifications
            clarifications = run_clarification_stage(topic)

            if clarifications.questions_and_answers:
                st.success(f"Generated {len(clarifications.questions_and_answers)} clarification questions")
                st.session_state.current_stage = 'answer_clarifications'
                st.rerun()
            else:
                st.info("No clarification questions needed. Proceeding to search...")
                st.session_state.current_stage = 'search'
                st.rerun()

    # Answer Clarifications
    if st.session_state.current_stage == 'answer_clarifications':
        st.subheader("2. Answer Clarification Questions")

        clarifications = st.session_state.clarifications

        with st.form("clarification_form"):
            st.markdown("Please answer the following questions to refine the research:")

            answers = []
            for idx, qa in enumerate(clarifications.questions_and_answers):
                st.markdown(f"**Question {idx + 1}:** {qa.question}")
                answer = st.text_input(
                    f"Your answer:",
                    key=f"answer_{idx}",
                    label_visibility="collapsed"
                )
                answers.append(answer)

            submitted = st.form_submit_button("Submit Answers", type="primary")

            if submitted:
                # Update clarifications with answers
                for idx, answer in enumerate(answers):
                    clarifications.questions_and_answers[idx].answer = answer

                st.session_state.clarifications = clarifications
                st.session_state.current_stage = 'search'
                st.rerun()

    # Search Stage
    if st.session_state.current_stage == 'search' and topic:
        st.subheader("3. Web Search")

        if st.session_state.serp_queries is None:
            # Generate SERP queries
            serp_queries = run_serp_generation_stage(topic, st.session_state.clarifications)

            if serp_queries:
                st.success(f"Generated {len(serp_queries)} SERP queries")

                with st.expander("View SERP Queries"):
                    for idx, query in enumerate(serp_queries, 1):
                        st.text(f"{idx}. {query}")
            else:
                st.error("No SERP queries generated. Please try again.")
                st.stop()

        if st.session_state.search_results is None:
            if st.button("Execute Web Search", type="primary"):
                results_collection = run_search_stage(st.session_state.serp_queries)

                if results_collection.results:
                    st.success(f"Search complete! Collected {results_collection.total_queries} results.")
                    st.session_state.current_stage = 'save_results'
                    st.rerun()
                else:
                    st.error("No search results collected.")
        else:
            st.success(f"Search complete! Collected {st.session_state.search_results.total_queries} results.")
            st.session_state.current_stage = 'save_results'

    # Save Results Stage
    if st.session_state.current_stage == 'save_results':
        st.subheader("4. Save Results")

        results_collection = st.session_state.search_results

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Queries Executed", results_collection.total_queries)
        with col2:
            st.metric("Timestamp", results_collection.timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        # Show ALL search results
        st.subheader("Search Results")

        # Add a search/filter box
        search_filter = st.text_input("Filter queries:", placeholder="Type to filter results...")

        # Filter results if search term provided
        filtered_results = results_collection.results.items()
        if search_filter:
            filtered_results = [(q, r) for q, r in results_collection.results.items()
                               if search_filter.lower() in q.lower()]

        # Display results count
        st.info(f"Showing {len(filtered_results) if search_filter else len(results_collection.results)} results")

        # Display all results in expandable sections
        for idx, (query, result) in enumerate(filtered_results if search_filter else results_collection.results.items(), 1):
            with st.expander(f"**{idx}. {query}**", expanded=False):
                # Show snippet and full content
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.caption(f"Length: {len(result.results)} chars")
                    st.caption(f"Timestamp: {result.timestamp.strftime('%H:%M:%S')}")

                # Display the full results
                st.text_area(
                    "Search Results:",
                    value=result.results,
                    height=300,
                    key=f"search_result_{idx}",
                    label_visibility="collapsed"
                )

        filename = st.text_input(
            "Enter filename to save results (without extension):",
            value=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        if st.button("Save Results", type="primary"):
            try:
                output_path, s3_uploaded, s3_key, s3_error = save_search_results(results_collection, filename)
                st.success(f"✅ Results saved locally to {output_path}")
                
                # Show S3 upload status
                if s3_uploaded:
                    st.success(f"☁️ Successfully uploaded to S3: `{s3_key}`")
                else:
                    if s3_error:
                        st.warning(f"⚠️ S3 upload failed: {s3_error}")
                    else:
                        st.info("ℹ️ S3 upload skipped (not configured)")

                # Provide download button
                with open(output_path, 'r', encoding='utf-8') as f:
                    json_data = f.read()

                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error saving results: {e}")

# Database View Page
elif page == "Database":
    st.header("📊 Database")
    st.markdown("Consolidated view of all summarised and processed crawled data with advanced search and filtering")
    
    # Initialize reload trigger in session state
    if 'reload_database' not in st.session_state:
        st.session_state.reload_database = False
    
    # Add reload button
    col_header1, col_header2 = st.columns([6, 1])
    with col_header2:
        if st.button("🔄 Reload from S3", use_container_width=True, help="Download latest files from S3 and refresh the database"):
            # Set reload flag
            st.session_state.reload_database = True
            
            # Clear local CSV files
            summarised_dir = Path("summarised_content")
            if summarised_dir.exists():
                for csv_file in summarised_dir.glob("*.csv"):
                    try:
                        csv_file.unlink()
                    except Exception as e:
                        st.warning(f"Could not delete {csv_file.name}: {e}")
            
            # Download from S3
            try:
                from aws_storage import get_storage
                s3_storage = get_storage()
                
                with st.spinner("📥 Downloading latest files from S3..."):
                    summarised_dir.mkdir(parents=True, exist_ok=True)
                    
                    # List all CSV files in S3 summarised_content prefix
                    s3_csv_files = s3_storage.list_files(prefix="summarised_content/", suffix=".csv")
                    
                    if s3_csv_files:
                        downloaded_count = 0
                        for s3_key in s3_csv_files:
                            file_name = s3_key.split('/')[-1]
                            local_path = summarised_dir / file_name
                            
                            if s3_storage.download_file(s3_key, str(local_path)):
                                downloaded_count += 1
                        
                        if downloaded_count > 0:
                            st.success(f"✅ Downloaded {downloaded_count} file(s) from S3")
                            # Clear the flag and rerun
                            st.session_state.reload_database = False
                            st.rerun()
                        else:
                            st.warning("⚠️ No files were downloaded")
                    else:
                        st.info("No CSV files found in S3 summarised_content folder")
            except Exception as e:
                st.error(f"❌ Error reloading from S3: {str(e)}")
    
    # Add instructions
    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown("""
        **Features:**
        - **Reload from S3**: Click the 🔄 button to download the latest files from S3
        - **Multi-file Selection**: Select multiple source files and dates to display simultaneously
        - **Row Selection**: Use checkboxes to select multiple rows to view details
        - **Search & Filter**: Use the search box and column filters to find specific entries
        - **Date Range Filter**: Filter articles by publication date
        - **Export**: Download filtered or complete database as CSV, JSON, or Excel
        
        **Tips:**
        - Use the reload button when new files are added to S3
        - Use the date range filter to find articles from specific time periods
        - Combine filters for more precise results
        - Export your filtered results in multiple formats
        """)
    
    # Import AgGrid
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
    
    # Load all CSV files
    summarised_dir = Path("summarised_content")
    summarised_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files (excluding history.json)
    csv_files = list(summarised_dir.glob("*.csv"))
    
    # If local folder is empty, try to retrieve from S3
    if not csv_files:
        try:
            from aws_storage import get_storage
            s3_storage = get_storage()
            
            with st.spinner("📥 Checking S3 for summarised data..."):
                # List all CSV files in S3 summarised_content prefix
                s3_csv_files = s3_storage.list_files(prefix="summarised_content/", suffix=".csv")
                
                if s3_csv_files:
                    st.info(f"Found {len(s3_csv_files)} CSV file(s) in S3. Downloading...")
                    
                    # Download each file
                    for s3_key in s3_csv_files:
                        file_name = s3_key.split('/')[-1]
                        local_path = summarised_dir / file_name
                        
                        if s3_storage.download_file(s3_key, str(local_path)):
                            st.success(f"✓ Downloaded: {file_name}")
                        else:
                            st.warning(f"⚠️ Failed to download: {file_name}")
                    
                    # Refresh the csv_files list
                    csv_files = list(summarised_dir.glob("*.csv"))
                    
                    if csv_files:
                        st.success(f"✅ Successfully retrieved {len(csv_files)} file(s) from S3")
                else:
                    st.info("No CSV files found in S3 summarised_content folder")
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve from S3: {str(e)}")
    
    if not csv_files:
        st.info("No CSV files found in summarised_content folder. Process some files in Summarisation first!")
        st.stop()
    
    # Load and combine all CSVs
    @st.cache_data
    def load_all_csvs(file_list, _reload_time=None):
        """Load and combine all CSV files
        
        Args:
            file_list: List of CSV file paths to load
            _reload_time: Timestamp to force cache invalidation (prefixed with _ to exclude from hashing)
        """
        all_data = []
        total_rows = 0
        
        for csv_file in file_list:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.stem  # Add source file name
                # Extract date from filename
                # Format could be: name_summarized_YYYYMMDD_HHMMSS or name_com_YYYYMMDD
                parts = csv_file.stem.split('_')
                
                # Try to find date parts (YYYYMMDD format)
                processed_date_raw = None
                processed_date_display = None
                
                # Look for YYYYMMDD pattern in the parts
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():
                        # Found date part (YYYYMMDD)
                        date_part = part
                        # Check if there's a time part after it
                        if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                            # Has time part (HHMMSS)
                            time_part = parts[i + 1]
                            processed_date_raw = f"{date_part}_{time_part}"
                            # Format: DD-MM-YYYY HH:MM:SS
                            processed_date_display = f"{date_part[6:8]}-{date_part[4:6]}-{date_part[0:4]} {time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
                        else:
                            # Only date, no time
                            processed_date_raw = date_part
                            # Format: DD-MM-YYYY
                            processed_date_display = f"{date_part[6:8]}-{date_part[4:6]}-{date_part[0:4]}"
                        break
                
                if processed_date_raw:
                    df['processed_date'] = processed_date_raw
                    df['processed_date_display'] = processed_date_display
                else:
                    df['processed_date'] = 'unknown'
                    df['processed_date_display'] = 'unknown'
                all_data.append(df)
                total_rows += len(df)
            except Exception as e:
                st.warning(f"Could not load {csv_file.name}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Merge date, pubDate, and publication_date columns into a single 'date' column
            if 'publication_date' in combined_df.columns:
                # Start with publication_date
                combined_df['date'] = combined_df['publication_date']
                # Fill in missing values from pubDate if it exists
                if 'pubDate' in combined_df.columns:
                    combined_df['date'] = combined_df['date'].fillna(combined_df['pubDate'])
                    combined_df = combined_df.drop(columns=['pubDate'])
                combined_df = combined_df.drop(columns=['publication_date'])
            elif 'pubDate' in combined_df.columns and 'date' in combined_df.columns:
                # Prefer pubDate, fallback to date
                combined_df['date'] = combined_df['pubDate'].fillna(combined_df['date'])
                combined_df = combined_df.drop(columns=['pubDate'])
            elif 'pubDate' in combined_df.columns:
                # Rename pubDate to date
                combined_df = combined_df.rename(columns={'pubDate': 'date'})
            
            # Standardize date format to DD MMM YYYY
            if 'date' in combined_df.columns:
                def format_date(date_val):
                    if pd.isna(date_val):
                        return ''
                    try:
                        # Try to parse the date
                        parsed_date = pd.to_datetime(date_val, errors='coerce')
                        if pd.notna(parsed_date):
                            # Format as DD MMM YYYY (e.g., 13 Oct 2025)
                            return parsed_date.strftime('%d %b %Y')
                        return str(date_val)  # Return original if parsing fails
                    except:
                        return str(date_val)
                
                combined_df['date'] = combined_df['date'].apply(format_date)
            
            # Merge tags and classification columns into a single 'categories' column
            if 'tags' in combined_df.columns and 'classification' in combined_df.columns:
                # Combine tags and classification, removing duplicates
                def merge_categories(row):
                    tags = str(row.get('tags', '')).strip() if pd.notna(row.get('tags')) else ''
                    classification = str(row.get('classification', '')).strip() if pd.notna(row.get('classification')) else ''
                    
                    # Split by semicolon and clean
                    all_cats = []
                    if tags:
                        all_cats.extend([t.strip() for t in tags.split(';') if t.strip()])
                    if classification:
                        all_cats.extend([c.strip() for c in classification.split(';') if c.strip()])
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_cats = []
                    for cat in all_cats:
                        if cat.lower() not in seen:
                            seen.add(cat.lower())
                            unique_cats.append(cat)
                    
                    return '; '.join(unique_cats) if unique_cats else ''
                
                combined_df['categories'] = combined_df.apply(merge_categories, axis=1)
                # Drop original columns
                combined_df = combined_df.drop(columns=['tags', 'classification'])
            elif 'tags' in combined_df.columns:
                # Rename tags to categories if only tags exist
                combined_df = combined_df.rename(columns={'tags': 'categories'})
            elif 'classification' in combined_df.columns:
                # Rename classification to categories if only classification exists
                combined_df = combined_df.rename(columns={'classification': 'categories'})
            
            # Sort categories alphabetically within each cell
            if 'categories' in combined_df.columns:
                def sort_categories(cat_string):
                    # Handle NaN, None, or empty values
                    if cat_string is None or (isinstance(cat_string, float) and pd.isna(cat_string)):
                        return ''
                    cat_str = str(cat_string).strip()
                    if not cat_str:
                        return ''
                    # Split by semicolon, strip whitespace, sort alphabetically, rejoin
                    cats = [c.strip() for c in cat_str.split(';') if c.strip()]
                    sorted_cats = sorted(cats, key=lambda x: x.lower())
                    return '; '.join(sorted_cats)
                
                combined_df['categories'] = combined_df['categories'].apply(sort_categories)
            
            # Merge url and link columns into a single 'url' column
            if 'url' in combined_df.columns and 'link' in combined_df.columns:
                # Prefer url, fallback to link
                combined_df['url'] = combined_df['url'].fillna(combined_df['link'])
                combined_df = combined_df.drop(columns=['link'])
            elif 'link' in combined_df.columns:
                # Rename link to url if only link exists
                combined_df = combined_df.rename(columns={'link': 'url'})
            
            # Fill empty 'source' column with filename-based source
            if 'source' in combined_df.columns and 'source_file' in combined_df.columns:
                def extract_source_from_filename(row):
                    # If source already has a value, keep it
                    if pd.notna(row.get('source')) and str(row.get('source')).strip():
                        return row.get('source')
                    
                    # Extract source from filename
                    source_file = str(row.get('source_file', ''))
                    if source_file:
                        # Remove '_summarized_YYYYMMDD_HHMMSS' part
                        parts = source_file.split('_summarized_')
                        if len(parts) > 0:
                            source_name = parts[0]
                            # Remove domain extensions like '_com', '_org', etc.
                            # This preserves multi-word sources like "canary_media" -> "Canary Media"
                            source_name = source_name.split('_com')[0].split('_org')[0].split('_net')[0]
                            # Convert underscores to spaces and title case
                            # e.g., "canary_media" -> "Canary Media", "techcrunch" -> "Techcrunch"
                            formatted_source = source_name.replace('_', ' ').title()
                            return formatted_source
                    
                    return ''
                
                combined_df['source'] = combined_df.apply(extract_source_from_filename, axis=1)
            elif 'source_file' in combined_df.columns:
                # Create source column from filename if it doesn't exist
                def create_source_from_filename(source_file):
                    if pd.isna(source_file):
                        return ''
                    source_file = str(source_file)
                    # Remove '_summarized_YYYYMMDD_HHMMSS' part
                    parts = source_file.split('_summarized_')
                    if len(parts) > 0:
                        source_name = parts[0]
                        # Remove domain extensions like '_com', '_org', etc.
                        # This preserves multi-word sources like "canary_media" -> "Canary Media"
                        source_name = source_name.split('_com')[0].split('_org')[0].split('_net')[0]
                        # Convert underscores to spaces and title case
                        # e.g., "canary_media" -> "Canary Media", "techcrunch" -> "Techcrunch"
                        formatted_source = source_name.replace('_', ' ').title()
                        return formatted_source
                    return ''
                
                combined_df['source'] = combined_df['source_file'].apply(create_source_from_filename)
            
            # Handle Start-up column naming - standardize to 'URL to start-up(s)'
            # CSV files may have: 'Start-up', 'URL to start-ups', or 'URL to start-up(s)'
            startup_columns = [col for col in combined_df.columns if col in ['Start-up', 'URL to start-ups', 'URL to start-up(s)']]
            
            if startup_columns:
                # Create the standardized column by merging all variations
                # This handles the case where different CSVs have different column names
                if 'URL to start-up(s)' not in combined_df.columns:
                    combined_df['URL to start-up(s)'] = ''
                
                # Merge data from all startup column variations into the standardized column
                for col in startup_columns:
                    if col != 'URL to start-up(s)':
                        # Fill in values from this column where the standardized column is empty/NaN
                        combined_df['URL to start-up(s)'] = combined_df['URL to start-up(s)'].fillna(combined_df[col])
                        # Also handle empty strings
                        mask = (combined_df['URL to start-up(s)'] == '') | (combined_df['URL to start-up(s)'].isna())
                        combined_df.loc[mask, 'URL to start-up(s)'] = combined_df.loc[mask, col]
                
                # Now drop all the old variations
                to_drop = [col for col in startup_columns if col != 'URL to start-up(s)']
                if to_drop:
                    combined_df = combined_df.drop(columns=to_drop)
            
            return combined_df, total_rows
        return None, 0
    
    # Pass timestamp to force reload when button is pressed
    reload_timestamp = datetime.now().timestamp() if st.session_state.reload_database else None
    
    with st.spinner("Loading all CSV files..."):
        combined_df, total_rows = load_all_csvs(csv_files, _reload_time=reload_timestamp)
    
    # Reset reload flag after loading
    if st.session_state.reload_database:
        st.session_state.reload_database = False
    
    if combined_df is None:
        st.error("Could not load any CSV files!")
        st.stop()
    
    # Deduplicate based on URL
    rows_before_dedup = len(combined_df)
    if 'url' in combined_df.columns:
        # Keep first occurrence, drop duplicates based on URL
        combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
        rows_after_dedup = len(combined_df)
        duplicates_removed = rows_before_dedup - rows_after_dedup
        
        if duplicates_removed > 0:
            st.info(f"ℹ️ Removed {duplicates_removed} duplicate entries based on URL")
    
    # Reindex starting from 1
    combined_df.index = range(1, len(combined_df) + 1)
    
    st.divider()
    
    # Filters and Search
    st.subheader("🔍 Filters & Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Source file filter - changed to multiselect
        # Create mapping from source name to source_file for display
        source_mapping = combined_df[['source', 'source_file']].drop_duplicates()
        source_display_to_file = dict(zip(source_mapping['source'], source_mapping['source_file']))
        source_display_options = sorted(source_display_to_file.keys())
        
        selected_source_names = st.multiselect(
            "Source Files (select one or more)",
            options=source_display_options,
            default=source_display_options,  # All selected by default
            help="Select one or more source files to display"
        )
        
        # Convert selected source names back to source_file values for filtering
        selected_sources = [source_display_to_file[name] for name in selected_source_names]
    
    with col2:
        # Date range - display formatted dates but filter by raw dates
        if 'processed_date' in combined_df.columns:
            # Create mapping from display date to raw date for filtering
            date_mapping = combined_df[['processed_date_display', 'processed_date']].drop_duplicates()
            date_display_to_raw = dict(zip(date_mapping['processed_date_display'], date_mapping['processed_date']))
            date_display_options = sorted(date_display_to_raw.keys())
            
            if len(date_display_options) > 1:
                selected_date_displays = st.multiselect(
                    "Processed Dates (select one or more)",
                    options=date_display_options,
                    default=date_display_options,  # All selected by default
                    help="Select one or more dates to display"
                )
                # Convert selected display dates back to raw dates for filtering
                selected_dates = [date_display_to_raw[display_date] for display_date in selected_date_displays]
            else:
                selected_dates = combined_df['processed_date'].unique().tolist()
        else:
            selected_dates = []
    
    # Publication Date Range Filter
    if 'date' in combined_df.columns:
        st.markdown("**📅 Publication Date Range**")
        
        # Parse dates for filtering (use a temporary Series, don't modify combined_df)
        parsed_dates = pd.to_datetime(combined_df['date'], format='%d %b %Y', errors='coerce')
        
        # Get min and max dates (exclude NaT values)
        valid_dates = parsed_dates.dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "From Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the start date for filtering articles"
                )
            with col_date2:
                end_date = st.date_input(
                    "To Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the end date for filtering articles"
                )
            
            # Validate date range
            if start_date > end_date:
                st.warning("⚠️ Start date must be before or equal to end date")
        else:
            start_date = None
            end_date = None
            st.info("ℹ️ No valid publication dates found in the data")
    else:
        start_date = None
        end_date = None
    
    # Text search with multiple keywords support
    search_query = st.text_input(
        "🔎 Search in database", 
        placeholder="Enter keywords (separate with commas)...",
        help="Enter one or more keywords. Separate multiple keywords with commas. Any keyword will match (OR logic)."
    )
    
    # Apply filters
    filtered_df = combined_df.copy()
    
    # Filter by selected source files
    if selected_sources:
        filtered_df = filtered_df[filtered_df['source_file'].isin(selected_sources)]
    else:
        # If nothing selected, show nothing
        filtered_df = filtered_df.iloc[0:0]
    
    # Filter by selected dates
    if 'processed_date' in combined_df.columns and selected_dates:
        filtered_df = filtered_df[filtered_df['processed_date'].isin(selected_dates)]
    elif 'processed_date' in combined_df.columns and not selected_dates:
        # If nothing selected, show nothing
        filtered_df = filtered_df.iloc[0:0]
    
    # Filter by publication date range
    if 'date' in combined_df.columns and start_date and end_date:
        # Parse dates from the filtered dataframe for comparison
        filtered_parsed_dates = pd.to_datetime(filtered_df['date'], format='%d %b %Y', errors='coerce')
        
        # Convert start_date and end_date to datetime for comparison
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include entire end date
        
        # Filter rows where parsed date is within range
        date_mask = (filtered_parsed_dates >= start_datetime) & (filtered_parsed_dates <= end_datetime)
        filtered_df = filtered_df[date_mask]
    
    # Determine which columns will be displayed (exclude hidden columns)
    exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                   'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author']
    display_columns = [col for col in filtered_df.columns if col not in exclude_cols]
    
    # Multi-keyword search (OR logic)
    if search_query:
        # Parse multiple keywords - split by comma only
        keywords = [k.strip() for k in search_query.split(',') if k.strip()]
        
        if not keywords:
            keywords = [search_query.strip()]
        
        # Apply OR logic - any keyword may match
        final_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)

        for keyword in keywords:
            # Create a mask for this keyword across all display columns
            keyword_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            for col in display_columns:
                keyword_mask |= filtered_df[col].astype(str).str.contains(keyword, case=False, na=False, regex=False)
            # OR with the final mask (any keyword matches)
            final_mask |= keyword_mask

        filtered_df = filtered_df[final_mask]

        # Show which keywords are being searched
        if len(keywords) > 1:
            st.caption(f"🔍 Searching for any of: {', '.join([f'**{k}**' for k in keywords])} (any match)")
        else:
            st.caption(f"🔍 Searching for: **{keywords[0]}**")
    
    st.info(f"Showing {len(filtered_df)} of {len(combined_df)} entries")
    
    st.divider()
    
    # Display results with AgGrid
    st.subheader("📋 Results")
    
    if len(filtered_df) == 0:
        st.warning("No entries match your filters.")
    else:
        # Prepare dataframe for display - exclude specified columns
        exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                       'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author']
        display_df = filtered_df.drop(columns=[col for col in exclude_cols if col in filtered_df.columns])
        
        # Define desired column order
        desired_order = [
            'title',
            'Indicator',  # This is the summary/indicator column
            'date',  # publication date (renamed from publication_date)
            'categories',
            'Dimension',  # matches CSV column name
            'Tech',  # matches CSV column name
            'TRL',  # matches CSV column name
            'URL to start-up(s)',  # matches CSV column name
            'url'  # URL as the last column
        ]
        
        # Reorder columns according to desired order - only include columns that exist
        ordered_cols = [col for col in desired_order if col in display_df.columns]
        
        # Reorder the dataframe (don't include remaining columns - strict ordering)
        display_df = display_df[ordered_cols]
        
        # Fill NaN values with empty strings BEFORE converting to string
        display_df = display_df.fillna('')
        
        # Convert all object-type columns to string to avoid pyarrow conversion errors
        # This handles cases like TRL ranges ('6-7') and mixed-type columns
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        
        # Replace various representations of missing values with empty strings
        display_df = display_df.replace(['nan', 'None', 'NaN', 'NA', 'N/A', '<NA>'], '')
        
        # Reset index to show row numbers starting from 1
        display_df = display_df.reset_index(drop=True)
        
        # Convert categories from semicolon-separated to comma-separated for better display
        if 'categories' in display_df.columns:
            display_df['categories'] = display_df['categories'].apply(
                lambda x: str(x).replace(';', ',') if x and x != '' else ''
            )
        
        st.info(f"📊 Showing {len(display_df)} entries | Use search and filters in the table below")        
        
        # Configure AgGrid options
        gb = GridOptionsBuilder.from_dataframe(display_df)
        
        # Enable features
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        gb.configure_side_bar(filters_panel=True, columns_panel=True)
        gb.configure_default_column(
            filterable=True,
            sortable=True,
            resizable=True,
            wrapText=True,
            autoHeight=True,
            enableCellTextSelection=True,
            editable=False  # Disable editing for all columns
        )
        # Configure specific columns
        if 'url' in display_df.columns:
            gb.configure_column(
                'url',
                headerName='URL',
                width=250,  # Wide enough for full URLs
                wrapText=True,
                autoHeight=True,
                cellStyle={'word-break': 'break-all', 'white-space': 'normal'},
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'Indicator' in display_df.columns:
            gb.configure_column(
                'Indicator', 
                headerName='Summary', 
                width=400,  # Large width for main content
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'title' in display_df.columns:
            gb.configure_column(
                'title', 
                headerName='Title', 
                width=300,  # Wide enough for longer titles
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'date' in display_df.columns:
            gb.configure_column(
                'date', 
                headerName='Date', 
                width=120,  # Adequate for date format
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'categories' in display_df.columns:
            gb.configure_column(
                'categories', 
                headerName='Categories', 
                width=200,  # Enough space for multiple categories
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'Dimension' in display_df.columns:
            gb.configure_column(
                'Dimension', 
                headerName='Dimension', 
                width=120,  # Moderate width
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'Tech' in display_df.columns:
            gb.configure_column(
                'Tech', 
                headerName='Technology', 
                width=150,  # Space for tech names
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'TRL' in display_df.columns:
            gb.configure_column(
                'TRL', 
                headerName='Trl level', 
                width=80,  # Small column for TRL numbers
                type=['textColumn'],  # Explicitly set as text to handle ranges
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'URL to start-up(s)' in display_df.columns:
            gb.configure_column(
                'URL to start-up(s)', 
                headerName='URL to start-up(s)', 
                width=200,  # Wide enough for startup URLs
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        # Enable text selection (read-only mode)
        gb.configure_grid_options(
            enableCellTextSelection=True, 
            ensureDomOrder=True,
            suppressRowClickSelection=True  # Disable row selection on click
        )
        
        gridOptions = gb.build()
        
        # Configure default column options (read-only)
        gridOptions['defaultColDef']['wrapText'] = True
        gridOptions['defaultColDef']['autoHeight'] = True
        gridOptions['defaultColDef']['enableCellTextSelection'] = True
        gridOptions['defaultColDef']['editable'] = False
        
        # Display AgGrid
        grid_response = AgGrid(
            display_df,
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.NO_UPDATE,  # Read-only mode
            fit_columns_on_grid_load=False,
            theme='streamlit',
            width=800,
            height=800,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            reload_data=False
        )
    
    st.divider()
    
    # Export options
    st.subheader("📥 Export Database")
    
    # Define column order for exports (same as display)
    desired_order = [
        'title',
        'Indicator',  # summary/indicator column
        'publication_date',  # publication date (matches CSV column name)
        'categories',
        'Dimension',  # matches CSV column name
        'Tech',  # matches CSV column name
        'TRL',  # matches CSV column name
        'URL to start-up(s)',  # matches CSV column name
        'url'  # URL as the last column
    ]
    
    # Helper function to reorder columns
    def reorder_export_columns(df):
        """Reorder dataframe columns according to desired order, excluding unwanted columns"""
        # Exclude columns that shouldn't be in exports
        exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                       'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author']
        df_clean = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        
        # Get columns that exist in desired order - strict ordering, no remaining columns
        ordered_cols = [col for col in desired_order if col in df_clean.columns]
        return df_clean[ordered_cols]
    
    # CSV & JSON exports
    st.markdown("**CSV & JSON Formats**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Export filtered results as CSV
        if len(filtered_df) > 0:
            filtered_export = reorder_export_columns(filtered_df.copy())
            csv_export = filtered_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Filtered CSV",
                data=csv_export,
                file_name=f"filtered_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        # Export filtered results as JSON
        if len(filtered_df) > 0:
            filtered_export = reorder_export_columns(filtered_df.copy())
            # Convert DataFrame to JSON (orient='records' creates array of objects)
            json_export = filtered_export.to_json(orient='records', indent=2, force_ascii=False).encode('utf-8')
            st.download_button(
                label="📋 Filtered JSON",
                data=json_export,
                file_name=f"filtered_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col3:
        # Export all data as CSV
        all_export = reorder_export_columns(combined_df.copy())
        all_csv = all_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Complete CSV",
            data=all_csv,
            file_name=f"complete_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col4:
        # Export all data as JSON
        all_export = reorder_export_columns(combined_df.copy())
        all_json = all_export.to_json(orient='records', indent=2, force_ascii=False).encode('utf-8')
        st.download_button(
            label="📋 Complete JSON",
            data=all_json,
            file_name=f"complete_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Excel export (separate row)
    st.markdown("**Excel Format**")
    col_excel = st.columns(1)[0]
    
    # Export to Excel (if openpyxl is installed)
    try:
        from io import BytesIO
        output = BytesIO()
        
        # Prepare dataframes for export with reordered columns
        export_all_df = reorder_export_columns(combined_df.copy())
        if 'categories' in export_all_df.columns:
            export_all_df['categories'] = export_all_df['categories'].apply(
                lambda x: '; '.join(x) if isinstance(x, list) else x
            )
        
        # Convert TRL column to string to avoid conversion errors
        if 'TRL' in export_all_df.columns:
            export_all_df['TRL'] = export_all_df['TRL'].astype(str)
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_all_df.to_excel(writer, index=False, sheet_name='All Data')
            if len(filtered_df) > 0 and len(filtered_df) < len(combined_df):
                export_filtered_df = reorder_export_columns(filtered_df.copy())
                if 'categories' in export_filtered_df.columns:
                    export_filtered_df['categories'] = export_filtered_df['categories'].apply(
                        lambda x: '; '.join(x) if isinstance(x, list) else x
                    )
                # Convert TRL column to string for filtered data too
                if 'TRL' in export_filtered_df.columns:
                    export_filtered_df['TRL'] = export_filtered_df['TRL'].astype(str)
                export_filtered_df.to_excel(writer, index=False, sheet_name='Filtered')
        
        st.download_button(
            label="📊 Download as Excel (All Sheets)",
            data=output.getvalue(),
            file_name=f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    except ImportError:
        st.caption("Install openpyxl for Excel export")

elif page == "Chatbot":
    # Steps to use chatbot in an expandable section
    with st.expander("ℹ️ **Steps to use the Chatbot**", expanded=True):
        st.markdown("""
        1. **Load Data**: Use the sidebar to load embeddings from S3 (select sources and click 'Load from S3')
        2. **Configure Settings**: Adjust the number of documents to retrieve in the sidebar
        3. **Start Chatting**: Type your question in the chat input box below
        4. **View Sources**: Click on the expandable sections to see source documents used for answers
        """)
    
    # Initialize session state for chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_processor' not in st.session_state:
        st.session_state.chat_processor = None
    if 'embedding_indexes' not in st.session_state:
        st.session_state.embedding_indexes = {}
    if 'embeddings_built' not in st.session_state:
        st.session_state.embeddings_built = False
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🗄️ Database Status")
        
        # Check if embeddings exist in S3
        s3_available_indexes = []
        try:
            from aws_storage import get_storage
            s3_storage = get_storage()
            s3_files = s3_storage.list_files(prefix="rag_embeddings/", suffix=".pkl")
            
            # Extract index names from S3 files
            for s3_file in s3_files:
                # Extract index name from path: "rag_embeddings/embeddings_name.pkl"
                index_name = s3_file.replace("rag_embeddings/", "").replace(".pkl", "")
                s3_available_indexes.append(index_name)
            
            embeddings_in_s3 = len(s3_available_indexes) > 0
            
            # Debug info
            if embeddings_in_s3:
                st.success(f"☁️ Embeddings found in S3 ({len(s3_available_indexes)} file(s))")
            
            else:
                st.info("☁️ No embeddings found in S3")
        except Exception as e:
            embeddings_in_s3 = False
            s3_error = str(e)
            st.warning(f"⚠️ S3 connection issue: {s3_error}")
        
        if st.session_state.embeddings_built:
            st.success(f"✅ Data indexed")
            
            # Show document count for loaded sources
            if st.session_state.embedding_indexes:
                loaded_sources = list(st.session_state.embedding_indexes.keys())
                total_docs = sum(idx.num_documents for idx in st.session_state.embedding_indexes.values())
                total_chunks = sum(idx.num_chunks for idx in st.session_state.embedding_indexes.values())
                st.metric("Loaded Sources", len(loaded_sources))
                st.metric("Documents", total_docs)
                st.metric("Chunks", total_chunks)
                
                # Show which sources are loaded with date
                with st.expander("📋 Loaded Sources", expanded=False):
                    for source in loaded_sources:
                        # Format: Source_embeddings_YYYYMMDD -> Source (YYYY-MM-DD)
                        display_name = source
                        # Extract date if present (YYYYMMDD pattern at the end)
                        import re
                        date_match = re.search(r'_(\d{8})$', display_name)
                        if date_match:
                            date_str = date_match.group(1)
                            source_part = display_name[:date_match.start()]
                            # Remove _embeddings suffix if present
                            source_part = source_part.replace('_embeddings', '')
                            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                            display_name = f"{source_part} ({formatted_date})"
                        else:
                            # No date found, just remove _embeddings
                            display_name = display_name.replace('_embeddings', '')
                        st.write(f"• {display_name}")
            elif st.session_state.chat_processor:
                try:
                    count = len(st.session_state.chat_processor.documents)
                    st.metric("Documents", count)
                except:
                    pass

        st.divider()
        
        # Load from S3 button with file selection
        if embeddings_in_s3 and not st.session_state.embeddings_built:
            st.markdown("### 📥 Load from S3")
            
            # Format index names for display
            formatted_options = []
            index_display_map = {}
            
            for idx_name in s3_available_indexes:
                # Format: Source_embeddings_YYYYMMDD -> Source (YYYY-MM-DD)
                display_name = idx_name
                import re
                date_match = re.search(r'_(\d{8})$', display_name)
                if date_match:
                    date_str = date_match.group(1)
                    source_part = display_name[:date_match.start()]
                    source_part = source_part.replace('_embeddings', '')
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    display_name = f"{source_part} ({formatted_date})"
                else:
                    display_name = display_name.replace('_embeddings', '')
                
                formatted_options.append(display_name)
                index_display_map[display_name] = idx_name
            
            # Multi-select widget for choosing which indexes to load
            selected_display_names = st.multiselect(
                "Select embeddings to load:",
                options=formatted_options,
                default=formatted_options,  # No files selected by default - user must choose
                help="Choose which embedding indexes to load from S3",
                key="s3_embeddings_select"
            )
            
            # Map back to original index names
            selected_indexes = [index_display_map[name] for name in selected_display_names]
            
            if not selected_indexes:
                st.warning("Please select at least one embedding index to load.")
            elif st.button("☁️ Load from S3", use_container_width=True, type="secondary", disabled=len(selected_indexes) == 0):
                with st.spinner("Downloading embeddings from S3..."):
                    try:
                        from embeddings_processor import JSONEmbeddingProcessor
                        processor = JSONEmbeddingProcessor()
                        
                        embedding_indexes = {}
                        loaded_count = 0
                        
                        # Download only selected indexes from S3
                        for index_name in selected_indexes:
                            try:
                                st.info(f"Downloading {index_name}...")
                                embedding_index = processor.download_index_from_s3(index_name)
                                if embedding_index:
                                    embedding_indexes[index_name] = embedding_index
                                    loaded_count += 1
                                    st.success(f"✓ Loaded {index_name}: {embedding_index.num_documents} docs, {embedding_index.num_chunks} chunks")
                            except Exception as e:
                                st.warning(f"Failed to download {index_name}: {e}")
                        
                        if embedding_indexes:
                            st.session_state.chat_processor = processor
                            st.session_state.embedding_indexes = embedding_indexes
                            st.session_state.embeddings_built = True
                            total_docs = sum(idx.num_documents for idx in embedding_indexes.values())
                            total_chunks = sum(idx.num_chunks for idx in embedding_indexes.values())
                            st.success(f"✅ Loaded {loaded_count} index(es) from S3! {total_docs} documents ({total_chunks} chunks)")
                            st.rerun()
                        else:
                            st.error("Failed to load any embeddings from S3")
                    except Exception as e:
                        st.error(f"Error loading from S3: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        
        # Retrieval settings
        st.markdown("### ⚙️ Retrieval Settings")
        num_results = st.slider(
            "Number of documents to retrieve",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Adjust how many relevant documents are retrieved and used to generate the response"
        )
        
        st.divider()
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.embeddings_built:
        st.stop()
    
    # Load processor if not in session state
    if st.session_state.chat_processor is None or not st.session_state.embedding_indexes:
        try:
            processor = JSONEmbeddingProcessor()
            # Load all indexes from local storage
            local_indexes = processor.list_local_indexes()
            embedding_indexes = {}
            for index_name in local_indexes:
                try:
                    embedding_index = processor.load_index_locally(index_name)
                    source_name = index_name.replace('_embeddings', '')
                    embedding_indexes[source_name] = embedding_index
                except Exception as e:
                    st.warning(f"Failed to load local index {index_name}: {e}")
            
            if embedding_indexes:
                st.session_state.chat_processor = processor
                st.session_state.embedding_indexes = embedding_indexes
            else:
                st.error("No embedding indexes found. Please build embeddings first.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        # Use URL as title if title is None or empty
                        title = source.get('title', 'Untitled')
                        if not title or title.lower() in ['none', 'untitled', 'n/a']:
                            title = source.get('url', 'Untitled')
                        
                        st.markdown(f"**{i}. [{title}]({source['url']})**")
                        st.caption(f"📅 {source.get('publication_date', 'N/A')} | 🏷️ {source.get('categories', 'N/A')}")
                        if source.get('source'):
                            st.caption(f"📺 Source: {source['source']}")
                        if source.get('tech'):
                            st.caption(f"💡 Tech: {source['tech']} | TRL: {source.get('trl', 'N/A')} | Dim: {source.get('dimension', 'N/A')}")
                        
                        # Display Indicator/Summary
                        indicator = source.get('indicator', '')
                        if indicator and indicator != 'N/A':
                            st.markdown(f"**📝 Summary:** {indicator}")
                        
                        # Display URL to start-up(s)
                        startup = source.get('startup', '')
                        if startup and startup != 'N/A' and startup:
                            st.caption(f"🚀 URL to start-up(s): {startup}")
                        
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Chat with the database using AI-powered semantic search"):
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    processor = st.session_state.chat_processor
                    
                    # Parse query for field-based filtering
                    def parse_query_for_filters(query):
                        """Parse natural language query to extract metadata filters and date ranges"""
                        filters = {}
                        date_filters = {}
                        query_lower = query.lower()
                        
                        # Import datetime for date parsing
                        from datetime import datetime, timedelta
                        import re
                        
                        # Source filters (e.g., "from TechCrunch", "TechCrunch articles")
                        sources = ["techcrunch", "carbonherald", "hydrogen-central", "interestingengineering", "canarymedia", "bioenergy-news", "pv-magazine"]
                        for source in sources:
                            if source in query_lower or f"from {source}" in query_lower:
                                # Keep the original source name (preserve hyphens) for matching
                                filters["source"] = source
                                break
                        
                        # TRL filters (Technology Readiness Level)
                        # First check for research stage keywords that map to TRL ranges
                        trl_stage_patterns = {
                            r'\b(basic|fundamental)\s+research\b': '1-3',
                            r'\bproof\s+of\s+concept\b': '1-3',
                            r'\blaboratory\s+(research|validation|testing|demonstration)\b': '1-5',
                            r'\blab\s+(testing|validation|scale)\b': '4-5',
                            r'\bprototype\s+(testing|development)\b': '6-7',
                            r'\bpilot\s+(project|testing|scale)\b': '6-7',
                            r'\b(commercial|deployed|operational)\s+(system|deployment|scale)\b': '8-9',
                            r'\bmarket\s+ready\b': '8-9'
                        }
                        
                        for pattern, trl_range in trl_stage_patterns.items():
                            if re.search(pattern, query_lower):
                                filters["trl_range"] = trl_range
                                break
                        
                        # Then check for explicit TRL numbers (only if no stage pattern matched)
                        if "trl_range" not in filters and "trl" in query_lower:
                            for i in range(1, 10):
                                if f"trl {i}" in query_lower or f"trl{i}" in query_lower:
                                    filters["trl"] = str(i)
                                    break
                        
                        # Enhanced date parsing
                        current_year = datetime.now().year
                        current_month = datetime.now().month
                        
                        # Specific month/year patterns (e.g., "Nov 2025", "November 2025", "in 2025")
                        month_patterns = {
                            r'(january|jan)\s+(\d{4})': 1,
                            r'(february|feb)\s+(\d{4})': 2,
                            r'(march|mar)\s+(\d{4})': 3,
                            r'(april|apr)\s+(\d{4})': 4,
                            r'(may)\s+(\d{4})': 5,
                            r'(june|jun)\s+(\d{4})': 6,
                            r'(july|jul)\s+(\d{4})': 7,
                            r'(august|aug)\s+(\d{4})': 8,
                            r'(september|sept|sep)\s+(\d{4})': 9,
                            r'(october|oct)\s+(\d{4})': 10,
                            r'(november|nov)\s+(\d{4})': 11,
                            r'(december|dec)\s+(\d{4})': 12
                        }
                        
                        for pattern, month_num in month_patterns.items():
                            match = re.search(pattern, query_lower)
                            if match:
                                year = int(match.group(2))
                                # Date range for that month
                                date_filters['year'] = year
                                date_filters['month'] = month_num
                                date_filters['date_after'] = f"{year}-{month_num:02d}-01"
                                # Calculate last day of month
                                if month_num == 12:
                                    next_month = 1
                                    next_year = year + 1
                                else:
                                    next_month = month_num + 1
                                    next_year = year
                                date_filters['date_before'] = f"{next_year}-{next_month:02d}-01"
                                break
                        
                        # Year only patterns (e.g., "in 2025", "from 2025")
                        if not date_filters:
                            year_match = re.search(r'\b(in|from|during)\s+(\d{4})\b', query_lower)
                            if year_match:
                                year = int(year_match.group(2))
                                date_filters['year'] = year
                                date_filters['date_after'] = f"{year}-01-01"
                                date_filters['date_before'] = f"{year + 1}-01-01"
                        
                        # Relative date filters
                        if not date_filters:
                            # Pattern for "last X months/weeks/days"
                            last_period_match = re.search(r'last\s+(\d+)\s+(month|week|day)s?', query_lower)
                            if last_period_match:
                                num = int(last_period_match.group(1))
                                period = last_period_match.group(2)
                                
                                if period == "month":
                                    days_ago = num * 30
                                elif period == "week":
                                    days_ago = num * 7
                                else:  # day
                                    days_ago = num
                                
                                date_after = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                                date_filters['date_after'] = date_after
                            elif "latest" in query_lower or "recent" in query_lower or "new" in query_lower:
                                # Get recent articles (last 2 months / 60 days)
                                two_months_ago = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                                date_filters['date_after'] = two_months_ago
                            elif "this month" in query_lower:
                                date_filters['date_after'] = f"{current_year}-{current_month:02d}-01"
                            elif "last month" in query_lower:
                                last_month = current_month - 1 if current_month > 1 else 12
                                last_year = current_year if current_month > 1 else current_year - 1
                                date_filters['date_after'] = f"{last_year}-{last_month:02d}-01"
                                date_filters['date_before'] = f"{current_year}-{current_month:02d}-01"
                        
                        return filters, date_filters
                    
                    # Parse query for metadata filters
                    metadata_filters, date_filters = parse_query_for_filters(prompt)
                    
                    # Extract source filter separately (since it's not in embeddings metadata)
                    source_filter = metadata_filters.pop("source", None) if metadata_filters else None
                    
                    # Show debug info about detected filters
                    if metadata_filters or date_filters or source_filter:
                        filter_info = []
                        if source_filter:
                            filter_info.append(f"source={source_filter}")
                        if metadata_filters:
                            # Handle TRL range separately for better display
                            display_filters = {}
                            for k, v in metadata_filters.items():
                                if k == 'trl_range':
                                    filter_info.append(f"TRL range: {v}")
                                else:
                                    display_filters[k] = v
                            
                            if display_filters:
                                filter_info.append(f"{', '.join([f'{k}={v}' for k, v in display_filters.items()])}")
                        if date_filters:
                            if 'year' in date_filters and 'month' in date_filters:
                                month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                                filter_info.append(f"Date filter: {month_names[date_filters['month']]} {date_filters['year']}")
                            elif 'year' in date_filters:
                                filter_info.append(f"Date filter: Year {date_filters['year']}")
                            elif 'date_after' in date_filters:
                                filter_info.append(f"Date filter: After {date_filters['date_after']}")
                        
                        st.info("🔍 Metadata filters: " + " | ".join(filter_info))
                    
                    # Query across all embedding indexes
                    all_results = []
                    processor = st.session_state.chat_processor
                    embedding_indexes = st.session_state.embedding_indexes
                    
                    # Get more results initially if we need to filter by date or source
                    # Use num_results * 2 for filtering scenarios to ensure we have enough after filtering
                    initial_top_k = num_results * 2 if (date_filters or source_filter) else num_results
                    
                    # Query all sources (we'll filter by URL domain later)
                    sources_to_query = list(embedding_indexes.items())
                    
                    for source_name, embedding_index in sources_to_query:
                        try:
                            results = processor.query_similar(
                                embedding_index,
                                query_text=prompt,
                                top_k=initial_top_k,
                                filter_metadata=metadata_filters if metadata_filters else None
                            )
                            
                            # Add source information to results
                            for result in results:
                                result['metadata']['source'] = source_name
                            
                            all_results.extend(results)
                        except Exception as e:
                            st.warning(f"Error querying {source_name}: {e}")
                            continue
                    
                    # Apply source filtering by URL domain if specified
                    if source_filter and all_results:
                        filtered_results = []
                        debug_info = {"before_filter": len(all_results), "after_filter": 0, "filtered_out": 0}
                        
                        for result in all_results:
                            url = result['metadata'].get('url', '').lower()
                            # Check if the source filter appears in the URL domain
                            if source_filter in url:
                                filtered_results.append(result)
                            else:
                                debug_info["filtered_out"] += 1
                        
                        debug_info["after_filter"] = len(filtered_results)
                        all_results = filtered_results
                        
                        # Show debug info
                        if debug_info["after_filter"] > 0:
                            st.info(f"🔍 Source filter applied: {debug_info['before_filter']} results → {debug_info['after_filter']} results from '{source_filter}' domain (removed {debug_info['filtered_out']} from other sources)")
                        else:
                            st.warning(f"⚠️ No results found with '{source_filter}' in the URL domain.")
                    
                    # Apply date filtering on results if specified
                    if date_filters and all_results:
                        from datetime import datetime
                        
                        filtered_results = []
                        debug_info = {"before_filter": len(all_results), "after_filter": 0, "filtered_out": 0}
                        
                        for result in all_results:
                            pub_date = result['metadata'].get('publication_date', '')
                            if not pub_date:
                                continue
                            
                            try:
                                # Parse publication date (handle various formats)
                                if 'T' in pub_date:
                                    pub_date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                else:
                                    pub_date_obj = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                                
                                # Check date filters
                                if 'date_after' in date_filters:
                                    after_date = datetime.strptime(date_filters['date_after'], '%Y-%m-%d')
                                    if pub_date_obj < after_date:
                                        debug_info["filtered_out"] += 1
                                        continue
                                
                                if 'date_before' in date_filters:
                                    before_date = datetime.strptime(date_filters['date_before'], '%Y-%m-%d')
                                    if pub_date_obj >= before_date:
                                        debug_info["filtered_out"] += 1
                                        continue
                                
                                filtered_results.append(result)
                                
                            except (ValueError, AttributeError) as e:
                                # Skip results with invalid dates
                                continue
                        
                        debug_info["after_filter"] = len(filtered_results)
                        all_results = filtered_results
                        
                        # Show debug info
                        if 'date_after' in date_filters:
                            st.info(f"📅 Date filter applied: {debug_info['before_filter']} results → {debug_info['after_filter']} results after filtering (removed {debug_info['filtered_out']} old articles)")
                    
                    # Apply TRL range filtering if specified
                    if metadata_filters and 'trl_range' in metadata_filters and all_results:
                        trl_range = metadata_filters.pop('trl_range')
                        filtered_results = []
                        debug_info = {"before_filter": len(all_results), "after_filter": 0, "filtered_out": 0}
                        
                        # Parse TRL range (e.g., "1-3", "4-5", "6-7", "8-9")
                        trl_min, trl_max = map(int, trl_range.split('-'))
                        
                        for result in all_results:
                            trl_str = result['metadata'].get('trl', '')
                            if not trl_str:
                                continue
                            
                            try:
                                # Parse TRL value (handle both int and float)
                                trl_value = int(float(str(trl_str).strip()))
                                
                                # Check if TRL is within range
                                if trl_min <= trl_value <= trl_max:
                                    filtered_results.append(result)
                                else:
                                    debug_info["filtered_out"] += 1
                                    
                            except (ValueError, AttributeError):
                                # Skip results with invalid TRL values
                                continue
                        
                        debug_info["after_filter"] = len(filtered_results)
                        all_results = filtered_results
                        
                        # Show debug info
                        st.info(f"🔬 TRL filter applied (TRL {trl_range}): {debug_info['before_filter']} results → {debug_info['after_filter']} results after filtering (removed {debug_info['filtered_out']} articles)")
                    
                    # Sort all results by score and take top N (based on slider)
                    all_results.sort(key=lambda x: x['score'], reverse=True)
                    results = all_results[:num_results]
                    
                    # Debug: Show how many results were found
                    if source_filter or date_filters:
                        st.caption(f"📊 Found {len(all_results)} matching articles, showing top {len(results)}")
                    
                    # If no results found, try relaxing filters automatically
                    if not results and (metadata_filters.get('trl_range') or date_filters):
                        st.warning("⚠️ No results found with current filters. Attempting to relax filters...")
                        
                        # Strategy: Remove TRL range filter (most restrictive for research queries)
                        relaxed = False
                        
                        # Try removing TRL range first
                        if 'trl_range' in metadata_filters:
                            st.info("🔄 Retrying without TRL range filter...")
                            metadata_filters.pop('trl_range')
                            # Re-run the query without TRL filtering
                            all_results_backup = []
                            for source_name, embedding_index in sources_to_query:
                                try:
                                    results_retry = processor.query_similar(
                                        embedding_index,
                                        query_text=prompt,
                                        top_k=num_results * 2,  # Get more results for filtering
                                        filter_metadata=metadata_filters if metadata_filters else None
                                    )
                                    for result in results_retry:
                                        result['metadata']['source'] = source_name
                                    all_results_backup.extend(results_retry)
                                except Exception:
                                    continue
                            
                            # Re-apply date filters only (no tech filter)
                            if date_filters:
                                from datetime import datetime
                                filtered = []
                                for result in all_results_backup:
                                    pub_date = result['metadata'].get('publication_date', '')
                                    if pub_date:
                                        try:
                                            if 'T' in pub_date:
                                                pub_date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                            else:
                                                pub_date_obj = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                                            
                                            if 'date_after' in date_filters:
                                                after_date = datetime.strptime(date_filters['date_after'], '%Y-%m-%d')
                                                if pub_date_obj < after_date:
                                                    continue
                                            if 'date_before' in date_filters:
                                                before_date = datetime.strptime(date_filters['date_before'], '%Y-%m-%d')
                                                if pub_date_obj >= before_date:
                                                    continue
                                            filtered.append(result)
                                        except:
                                            continue
                                all_results_backup = filtered
                            
                            if all_results_backup:
                                all_results_backup.sort(key=lambda x: x['score'], reverse=True)
                                results = all_results_backup[:num_results]
                                st.success(f"✅ Found {len(results)} results after relaxing TRL filter")
                                relaxed = True
                    
                    # Extract relevant information
                    sources = []
                    context_texts = []
                    
                    if not results:
                        # Semantic fallback: retrieve based on pure relevance without metadata filters
                        st.warning("⚠️ No articles found with metadata filters. Falling back to semantic search...")
                        st.info("🔍 Searching for semantically relevant content in the knowledge base...")
                        
                        # Perform semantic search without any filters
                        fallback_results = []
                        for source_name, embedding_index in sources_to_query:
                            try:
                                semantic_results = processor.query_similar(
                                    embedding_index,
                                    query_text=prompt,
                                    top_k=num_results * 2,  # Get more results for better coverage
                                    filter_metadata=None  # No metadata filtering
                                )
                                for result in semantic_results:
                                    result['metadata']['source'] = source_name
                                fallback_results.extend(semantic_results)
                            except Exception as e:
                                logging.warning(f"Error in semantic fallback for {source_name}: {e}")
                                continue
                        
                        if fallback_results:
                            # Sort by relevance score and take top N (based on slider)
                            fallback_results.sort(key=lambda x: x['score'], reverse=True)
                            results = fallback_results[:num_results]
                            
                            st.success(f"✅ Found {len(results)} semantically relevant articles (metadata filters removed)")
                            st.caption("💡 Results are based on semantic relevance to your query, not metadata filters")
                        else:
                            # Absolute fallback - no results at all
                            error_msg = "No articles found in the knowledge base."
                            if source_filter:
                                error_msg += f" (Source filter: {source_filter})"
                            st.error(error_msg)
                            st.info("💡 Try:\n- Checking if data from that source is loaded\n- Using different keywords\n- Rebuilding embeddings if the knowledge base was recently updated")
                            st.stop()
                    
                    if results:
                        for result in results:
                            metadata = result['metadata']
                            
                            # Add to sources with ALL available fields
                            sources.append({
                                "url": metadata.get("url", ""),
                                "title": metadata.get("title", "Untitled"),
                                "publication_date": metadata.get("publication_date", ""),
                                "categories": metadata.get("categories", ""),
                                "indicator": metadata.get("full_indicator", ""),
                                "dimension": metadata.get("dimension", ""),
                                "tech": metadata.get("tech", ""),
                                "trl": metadata.get("trl", ""),
                                "startup": metadata.get("startup", ""),
                                "source": metadata.get("source", "")
                            })
                            
                            # Add to context (use the chunk text)
                            context_texts.append(result['text'])
                    
                    # Generate response using OpenAI
                    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
                    
                    # Create system prompt with context
                    current_date = datetime.now().strftime("%B %d, %Y")  # e.g., "December 01, 2025"
                    
                    # Add date filter info to help LLM understand the query context
                    date_context = ""
                    if date_filters:
                        if 'year' in date_filters and 'month' in date_filters:
                            month_names = ["", "January", "February", "March", "April", "May", "June", 
                                         "July", "August", "September", "October", "November", "December"]
                            month_name = month_names[date_filters['month']]
                            date_context = f"\n\nIMPORTANT: The user is asking specifically about articles from {month_name} {date_filters['year']}. Only consider the provided sources which have been filtered to this date range."
                        elif 'year' in date_filters:
                            date_context = f"\n\nIMPORTANT: The user is asking specifically about articles from {date_filters['year']}. Only consider the provided sources which have been filtered to this date range."
                        elif 'date_after' in date_filters:
                            date_context = f"\n\nIMPORTANT: The user is asking about recent/latest articles. Only consider the provided sources which have been filtered to recent publications."
                    
                    system_prompt = f"""You are an AI research assistant specialized in technology intelligence and innovation analysis. 
You have access to a knowledge base of articles about emerging technologies, startups, carbon markets, and clean energy from various sources.

**CURRENT DATE: {current_date}**

When interpreting dates in the provided sources, always reference them relative to the current date above.{date_context}

Use the provided context to answer questions accurately and comprehensively. 
When answering analytical questions (e.g., "what technologies are mentioned most", "trending topics", "most discussed"):
1. COUNT and ANALYSE the information from the provided sources
2. Look for PATTERNS and FREQUENCIES in the data
3. Provide SPECIFIC NUMBERS and STATISTICS when possible
4. List technologies/topics in order of frequency or importance
5. Always verify dates from the source metadata to ensure accuracy

If the context doesn't contain relevant information, say so clearly.

CRITICAL CITATION RULES:
You MUST accurately cite all your outputs with your sources using markdown hyperlink format when referencing information.

IMPORTANT: 
1. The user message contains a "CITATION GUIDE" section with exact URLs you MUST use
2. ONLY use URLs that are explicitly provided in that Citation Guide
3. NEVER create, modify, shorten, or guess URLs - use them EXACTLY as provided
4. Format: [Source Title](EXACT_URL_FROM_CITATION_GUIDE)
5. If a source has no valid URL in the Citation Guide, reference it by title only without a hyperlink

Example of correct citation: 'According to [TechCrunch Article](https://techcrunch.com/2024/11/15/full-article-url), the technology...'
The URL must be copied EXACTLY from the Citation Guide - do not modify it in any way.

The knowledge base includes articles from sources like TechCrunch, Carbon Herald, Hydrogen Central, etc., with metadata about:
- Technology areas (AI, renewable energy, carbon tech, etc.)
- Technology Readiness Levels (TRL 1-9)
- Dimensions (energy, environment, technology, climate, etc.)
- Publication dates and sources"""
                    
                    # Create context from search results
                    context = "\n\n---\n\n".join(context_texts) if context_texts else "No relevant information found in the knowledge base."
                    
                    # Create source information for citations
                    source_info = []
                    metadata_summary = []
                    valid_sources = []  # Track sources with valid URLs
                    
                    for i, src in enumerate(sources, 1):
                        title = src.get('title', f'Source {i}')
                        url = src.get('url', '')
                        pub_date = src.get('publication_date', 'Unknown date')
                        tech = src.get('tech', 'N/A')
                        dimension = src.get('dimension', 'N/A')
                        
                        # Validate URL - must start with http:// or https://
                        is_valid_url = url and (url.startswith('http://') or url.startswith('https://'))
                        
                        # Format publication date
                        if pub_date and pub_date != 'Unknown date':
                            try:
                                if 'T' in pub_date:
                                    date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                else:
                                    date_obj = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                                pub_date = date_obj.strftime('%B %d, %Y')
                            except:
                                pass
                        
                        if is_valid_url:
                            # Create a clearer format for the model
                            source_info.append(f"**Source {i}**\n  - Title: {title}\n  - URL: {url}\n  - Published: {pub_date}")
                            metadata_summary.append(f"Source {i}: Tech: {tech}, Dimension: {dimension}")
                            valid_sources.append((i, title, url))
                        else:
                            # Skip sources without valid URLs
                            source_info.append(f"**Source {i}**\n  - Title: {title}\n  - URL: Not available\n  - Published: {pub_date}")
                            metadata_summary.append(f"Source {i}: Tech: {tech}, Dimension: {dimension}")
                    
                    sources_text = "\n\n".join(source_info)
                    metadata_text = "\n".join(metadata_summary)
                    
                    # Create a reference list of valid citations for the model
                    citation_guide = "\n".join([f"Source {i}: Use the format [{title}]({url})" for i, title, url in valid_sources]) if valid_sources else "No sources with valid URLs available."
                    
                    user_prompt = f"""Context information is below.
---------------------
{context}
---------------------

AVAILABLE SOURCES WITH URLS:
{sources_text}

CITATION GUIDE - Use EXACTLY these URLs when citing sources:
{citation_guide}

Source Metadata (for analysis):
{metadata_text}

CRITICAL CITATION INSTRUCTIONS:
- You MUST cite information using the EXACT URLs provided above
- Use the format: [Source Title](URL) where URL is taken EXACTLY from the "Citation Guide" above
- NEVER modify, shorten, or create your own URLs
- ONLY use the URLs explicitly listed in the Citation Guide
- If a source doesn't have a valid URL in the guide, reference it by title only without a link

For analytical questions about trends, frequencies, or patterns:
- Analyse ALL provided sources
- Count mentions across different articles
- Note publication dates to verify they match the requested timeframe
- Provide specific examples with citations using the EXACT URLs above

Question: {prompt}
Answer: """
                    
                    # Stream response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    stream = openai_client.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Sources (Database View)"):
                            # Validate URLs and show warning if needed
                            invalid_urls = []
                            for i, source in enumerate(sources, 1):
                                url = source.get('url', '')
                                if not url or not (url.startswith('http://') or url.startswith('https://')):
                                    invalid_urls.append(f"Source {i}: {url or 'No URL'}")
                            
                            if invalid_urls:
                                st.warning(f"⚠️ **URL Validation Warning:** {len(invalid_urls)} source(s) have invalid or missing URLs. These sources cannot be cited with hyperlinks.\n\n" + "\n".join(invalid_urls))
                            
                            for i, source in enumerate(sources, 1):
                                # Use URL as title if title is None or empty
                                title = source.get('title', 'Untitled')
                                if not title or title.lower() in ['none', 'untitled', 'n/a']:
                                    title = source.get('url', 'Untitled')
                                
                                st.markdown(f"**{i}. [{title}]({source['url']})**")
                                
                                # Display all fields in a structured database-like format
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.markdown("**📊 Database Fields:**")
                                    st.markdown(f"**Publication Date:** {source.get('publication_date', 'N/A')}")
                                    st.markdown(f"**Categories:** {source.get('categories', 'N/A')}")
                                    st.markdown(f"**Dimension:** {source.get('dimension', 'N/A')}")
                                    st.markdown(f"**Technology:** {source.get('tech', 'N/A')}")
                                    st.markdown(f"**TRL Level:** {source.get('trl', 'N/A')}")
                                    st.markdown(f"**Start-up:** {source.get('startup', 'N/A')}")
                                    st.markdown(f"**Source:** {source.get('source', 'N/A')}")
                                
                                with col2:
                                    st.markdown("**📝 Content Summary (Indicator):**")
                                    indicator_text = source.get('indicator', 'N/A')
                                    # Display full indicator text without truncation
                                    st.markdown(indicator_text)
                                
                                st.divider()
                    
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.stop()



elif page == "About":
    st.header("About Technology Intelligence Tool")

    st.markdown("""
    ### Overview

    The Technology Intelligence (TI) Tool is a comprehensive AI-powered platform for automated research,
    content extraction, analysis, and knowledge management. It streamlines the entire pipeline from discovery to insights 
    with cloud-native AWS S3 integration.

    ### Core Components

    **1. Web Search**
    - **Clarification** - AI asks targeted questions to refine research scope
    - **SERP Generation** - Creates optimized search queries from clarified intent
    - **Web Search** - Executes searches via SearxNG search engine
    - **Learning Extraction** - AI-powered analysis extracting key learnings, entities, and metrics
    - **Auto-save** - Results automatically saved locally and uploaded to S3

    **2. Database**
    - **Consolidated View** - All summarised content from multiple sources
    - **Advanced Filtering** - By category, source, date range, and keywords
    - **Full-text Search** - Search across summaries, titles, and content
    - **Multiple Views** - Cards, Table, and Detailed view modes
    - **Bulk Export** - Export filtered or complete database
    - **Text Selection** - Copy text from any cell in the table

    **3. Chatbot**
           
    **4. LinkedIn Home Feed Monitor**
    - **Automated Scraping** - Collects posts from LinkedIn home feed
    - **Smart Filtering** - Excludes promoted/suggested/reposted content
    - **Content Processing** - Extracts author, date, content, and URLs
    - **Auto-translation** - Non-English posts translated to English
    - **Deduplication** - Removes duplicate posts based on content
    - **URL Filtering** - Removes profile/company/hashtag links, keeps articles
    - **Date Limiting** - Configurable days back (1-90 days)
    - **S3 Management** - Upload, download, delete files via UI
    - **Targeted Network** - Monitors VCs and Companies (SOSV, Seed Capital, ADB Ventures)

    ### Technology Stack

    - **Frontend:** Streamlit with AgGrid
    - **AI Models:** OpenAI (gpt-4o, gpt-4o-mini, text-embedding-3-large)
    - **Alternative LLMs:** OpenAI, LM Studio (local)
    - **Search Engine:** SearxNG
    - **Vector Store:** Simple in-memory keyword search
    - **Cloud Storage:** AWS S3 (boto3)
    - **Validation:** Pydantic
    - **Agent Framework:** Pydantic AI
    - **Data Processing:** Pandas, NumPy

    ### AWS S3 Integration

    All pipeline outputs are automatically backed up to AWS S3:

    - **research_results/** - Web search outputs (JSON, Markdown)
    - **crawled_data/** - Web crawler outputs (CSV, JSON)
    - **processed_data/** - LLM extraction outputs (CSV, JSON)
    - **summarised_content/** - Summarisation outputs (CSV, JSON)
    - **linkedin_data/** - LinkedIn posts (CSV, JSON)

    **Features:**
    - ✅ Automatic upload after processing
    - ✅ Download and delete via UI (LinkedIn)
    - ✅ Graceful fallback if S3 unavailable
    ```

    ### Support & Troubleshooting

    - **Logs:** Check `research.log` for detailed operation logs
    - **S3 Status:** Run `python3 check_s3_status.py` to verify bucket contents
    - **Upload Missing Files:** Use `python3 upload_linkedin_to_s3.py` for backfill
    - **LinkedIn:** Posts are collected locally and uploaded to S3's `linkedin_data/` folder for display in the UI

    ---
    
    **Developed by Sharifah, with some guidance from AISG** | Technology Intelligence Tool © 2025
    """)

elif page == "LinkedIn Home Feed Monitor":
    st.header("📲 LinkedIn Home Feed Monitor")
    
    # Import AgGrid
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
    
    st.markdown("""
    ### Overview
    
    View and manage LinkedIn posts collected from your home feed. Data is loaded from AWS S3 storage.
    
    **Features:**
    - View all collected LinkedIn posts in an interactive table
    - Full-text search across all fields
    - Date filtering (last 7, 30, 90 days, or all time)
    - Download filtered results as CSV or Excel
    - S3 storage management (download/delete files)
    
    **Note:** LinkedIn posts must be collected locally using Selenium and uploaded to S3. 
    This interface displays the collected data from the `linkedin_data/` folder in S3.
    """)
    
    st.divider()
    
    # S3 Management Section
    with st.expander("☁️ S3 Storage Management", expanded=False):
        st.markdown("**Download or delete LinkedIn data from S3**")
        
        try:
            from aws_storage import get_storage
            s3_storage = get_storage()
            
            # List all LinkedIn files in S3
            linkedin_files = s3_storage.list_files(prefix="linkedin_data/", suffix=".csv")
            linkedin_files.extend(s3_storage.list_files(prefix="linkedin_data/", suffix=".json"))
            
            if linkedin_files:
                st.success(f"Found {len(linkedin_files)} files in S3")
                
                # Display files in a table
                file_data = []
                for file_key in linkedin_files:
                    file_name = file_key.split('/')[-1]
                    file_type = file_name.split('.')[-1].upper()
                    file_data.append({
                        "File Name": file_name,
                        "Type": file_type,
                        "S3 Key": file_key
                    })
                
                files_df = pd.DataFrame(file_data)
                st.dataframe(files_df, use_container_width=True)
                
                # Download and Delete options
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_file = st.selectbox(
                        "Select file to download",
                        options=[f["File Name"] for f in file_data]
                    )
                    
                    if st.button("⬇️ Download from S3", use_container_width=True):
                        # Find the S3 key
                        s3_key = next((f["S3 Key"] for f in file_data if f["File Name"] == selected_file), None)
                        if s3_key:
                            temp_path = Path(f"/tmp/{selected_file}")
                            if s3_storage.download_file(s3_key, str(temp_path)):
                                with open(temp_path, 'rb') as f:
                                    st.download_button(
                                        f"📥 Save {selected_file}",
                                        data=f.read(),
                                        file_name=selected_file,
                                        mime="text/csv" if selected_file.endswith('.csv') else "application/json"
                                    )
                                temp_path.unlink()
                                st.success(f"✅ Downloaded {selected_file}")
                            else:
                                st.error("Failed to download file")
                
                with col2:
                    selected_delete = st.selectbox(
                        "Select file to delete",
                        options=[f["File Name"] for f in file_data],
                        key="delete_select"
                    )
                    
                    if st.button("🗑️ Delete from S3", use_container_width=True, type="secondary"):
                        s3_key = next((f["S3 Key"] for f in file_data if f["File Name"] == selected_delete), None)
                        if s3_key:
                            if s3_storage.delete_file(s3_key):
                                st.success(f"✅ Deleted {selected_delete} from S3")
                                st.rerun()
                            else:
                                st.error("Failed to delete file")
            else:
                st.info("No LinkedIn files found in S3 yet. Scrape some posts to upload!")
                
        except Exception as e:
            st.warning(f"S3 not configured: {str(e)}")
    
    st.divider()
    
    # Load and display LinkedIn data from S3
    st.subheader("📊 LinkedIn Posts Database")
    
    try:
        from aws_storage import get_storage
        s3_storage = get_storage()
        
        # List all CSV files in linkedin_data folder
        csv_files = s3_storage.list_files(prefix="linkedin_data/", suffix=".csv")
        
        if not csv_files:
            st.info("📭 No LinkedIn posts found in S3. Upload some CSV files to the `linkedin_data/` folder.")
            st.stop()
        
        # Load all CSV files into a combined DataFrame
        all_data = []
        loaded_files = []
        
        with st.spinner(f"Loading {len(csv_files)} file(s) from S3..."):
            for csv_file in csv_files:
                try:
                    # Download to temporary location
                    temp_path = Path(f"/tmp/{csv_file.split('/')[-1]}")
                    if s3_storage.download_file(csv_file, str(temp_path)):
                        df = pd.read_csv(temp_path)
                        
                        # Add source file metadata
                        df['source_file'] = csv_file.split('/')[-1]
                        
                        all_data.append(df)
                        loaded_files.append(csv_file.split('/')[-1])
                        temp_path.unlink()
                except Exception as e:
                    st.warning(f"Failed to load {csv_file}: {e}")
        
        if not all_data:
            st.error("Failed to load any LinkedIn data from S3")
            st.stop()
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        original_count = len(combined_df)
        
        # Deduplicate based on "Content of post" column
        if 'Content of post' in combined_df.columns:
            # Remove duplicates based on content, keeping the first occurrence
            combined_df = combined_df.drop_duplicates(subset=['Content of post'], keep='first')
            duplicates_removed = original_count - len(combined_df)
            
            if duplicates_removed > 0:
                st.success(f"✅ Loaded {original_count} posts from {len(loaded_files)} file(s) | Removed {duplicates_removed} duplicates | {len(combined_df)} unique posts remaining")
            else:
                st.success(f"✅ Loaded {len(combined_df)} posts from {len(loaded_files)} file(s) | No duplicates found")
        else:
            st.success(f"✅ Loaded {len(combined_df)} posts from {len(loaded_files)} file(s)")
        
        # Keep original column names as specified
        # Ensure columns exist with exact names
        expected_columns = ['Person/Company name', 'Date of post', 'Content of post', 'URLs']
        
        # Convert date column to datetime
        if 'Date of post' in combined_df.columns:
            combined_df['Date of post'] = pd.to_datetime(combined_df['Date of post'], errors='coerce')
        
        # Filters Section
        st.markdown("### 🔍 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Start date filter
            if 'Date of post' in combined_df.columns:
                min_date = combined_df['Date of post'].min()
                start_date = st.date_input(
                    "📅 Start Date",
                    value=min_date if pd.notna(min_date) else datetime.now() - timedelta(days=90),
                    help="Select the start date for filtering posts"
                )
        
        with col2:
            # End date filter
            if 'Date of post' in combined_df.columns:
                max_date = combined_df['Date of post'].max()
                end_date = st.date_input(
                    "📅 End Date",
                    value=max_date if pd.notna(max_date) else datetime.now(),
                    help="Select the end date for filtering posts"
                )
        
        with col3:
            # Search box
            search_query = st.text_input(
                "🔎 Search (comma-separated for OR logic)",
                placeholder="e.g., AI, climate, startup",
                help="Search across all fields. Use commas to search multiple keywords (OR logic)."
            )
        
        # Apply date filter
        filtered_df = combined_df.copy()
        
        if 'Date of post' in filtered_df.columns:
            # Convert date inputs to datetime for comparison
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include end of day
            
            filtered_df = filtered_df[
                (filtered_df['Date of post'] >= start_datetime) & 
                (filtered_df['Date of post'] <= end_datetime)
            ]
        
        # Apply search filter
        if search_query:
            keywords = [k.strip() for k in search_query.split(',') if k.strip()]
            
            if not keywords:
                keywords = [search_query.strip()]
            
            # Apply OR logic across all columns
            final_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            
            for keyword in keywords:
                keyword_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
                
                # Search in all columns
                for col in filtered_df.columns:
                    if col != 'source_file':  # Skip source_file in search
                        keyword_mask |= filtered_df[col].astype(str).str.contains(keyword, case=False, na=False, regex=False)
                
                final_mask |= keyword_mask
            
            filtered_df = filtered_df[final_mask]
            
            if len(keywords) > 1:
                st.caption(f"🔍 Searching for any of: {', '.join([f'**{k}**' for k in keywords])} (any match)")
            else:
                st.caption(f"🔍 Searching for: **{keywords[0]}**")
        
        st.info(f"Showing {len(filtered_df)} of {len(combined_df)} posts")
        
        st.divider()
        
        # Display results with AgGrid
        st.subheader("📋 Results")
        
        if len(filtered_df) == 0:
            st.warning("No posts match your filters.")
        else:
            # Prepare dataframe for display with original column names
            display_columns = ['Person/Company name', 'Date of post', 'Content of post', 'URLs', 'source_file']
            display_df = filtered_df[[col for col in display_columns if col in filtered_df.columns]].copy()
            
            # Format date column
            if 'Date of post' in display_df.columns:
                display_df['Date of post'] = display_df['Date of post'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Fill NaN values with empty strings BEFORE any other processing
            display_df = display_df.fillna('')
            
            # Convert all object-type columns to string for consistent display
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    display_df[col] = display_df[col].astype(str)
            
            # Replace various representations of missing values with empty strings
            display_df = display_df.replace(['nan', 'None', 'NaN', 'NA', 'N/A', '<NA>'], '')
            
            # Reset index to show row numbers starting from 1
            display_df = display_df.reset_index(drop=True)
            
            st.info(f"📊 Showing {len(display_df)} posts | Use search and filters in the table below")
            
            # Configure AgGrid options
            gb = GridOptionsBuilder.from_dataframe(display_df)
            
            # Enable features
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
            gb.configure_side_bar(filters_panel=True, columns_panel=True)
            gb.configure_default_column(
                filterable=True,
                sortable=True,
                resizable=True,
                wrapText=True,
                autoHeight=True,
                enableCellTextSelection=True,
                editable=False
            )
            
            # Configure specific columns
            if 'Person/Company name' in display_df.columns:
                gb.configure_column(
                    'Person/Company name',
                    headerName='Person/Company name',
                    width=200,
                    wrapText=True,
                    enableCellTextSelection=True,
                    editable=False
                )
            
            if 'Date of post' in display_df.columns:
                gb.configure_column(
                    'Date of post',
                    headerName='Date of post',
                    width=150,
                    enableCellTextSelection=True,
                    editable=False
                )
            
            if 'Content of post' in display_df.columns:
                gb.configure_column(
                    'Content of post',
                    headerName='Content of post',
                    width=500,
                    wrapText=True,
                    autoHeight=True,
                    enableCellTextSelection=True,
                    editable=False
                )
            
            if 'URLs' in display_df.columns:
                gb.configure_column(
                    'URLs',
                    headerName='URLs',
                    width=300,
                    wrapText=True,
                    autoHeight=True,
                    cellStyle={'word-break': 'break-all', 'white-space': 'normal'},
                    enableCellTextSelection=True,
                    editable=False
                )
            
            if 'source_file' in display_df.columns:
                gb.configure_column(
                    'source_file',
                    headerName='Source File',
                    width=250,
                    enableCellTextSelection=True,
                    editable=False
                )
            
            # Enable text selection (read-only mode)
            gb.configure_grid_options(
                enableCellTextSelection=True,
                ensureDomOrder=True,
                suppressRowClickSelection=True
            )
            
            gridOptions = gb.build()
            
            # Configure default column options
            gridOptions['defaultColDef']['wrapText'] = True
            gridOptions['defaultColDef']['autoHeight'] = True
            gridOptions['defaultColDef']['enableCellTextSelection'] = True
            gridOptions['defaultColDef']['editable'] = False
            
            # Display AgGrid
            grid_response = AgGrid(
                display_df,
                gridOptions=gridOptions,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.NO_UPDATE,
                fit_columns_on_grid_load=False,
                theme='streamlit',
                height=600,
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False,
                reload_data=False
            )
            
            st.divider()
            
            # Export options
            st.subheader("📥 Export Data")
            
            # Prepare export dataframe without source_file column
            export_df = filtered_df[['Person/Company name', 'Date of post', 'Content of post', 'URLs']].copy()
            
            # Format date for export
            if 'Date of post' in export_df.columns:
                export_df['Date of post'] = pd.to_datetime(export_df['Date of post']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Export filtered data as CSV
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 CSV (Filtered)",
                    data=csv_data,
                    file_name=f"linkedin_posts_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export filtered data as JSON
                json_data = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📊 JSON (Filtered)",
                    data=json_data,
                    file_name=f"linkedin_posts_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Export complete data as CSV
                complete_export_df = combined_df[['Person/Company name', 'Date of post', 'Content of post', 'URLs']].copy()
                if 'Date of post' in complete_export_df.columns:
                    complete_export_df['Date of post'] = pd.to_datetime(complete_export_df['Date of post']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                csv_complete_data = complete_export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 CSV (Complete)",
                    data=csv_complete_data,
                    file_name=f"linkedin_posts_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col4:
                # Export complete data as JSON
                json_complete_data = complete_export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📊 JSON (Complete)",
                    data=json_complete_data,
                    file_name=f"linkedin_posts_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Error loading LinkedIn data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Footer
st.sidebar.divider()
    
st.sidebar.markdown("### ⚙️ Settings")

# LLM Provider Configuration - Display OpenAI info only
st.sidebar.markdown("#### 🤖 LLM Provider")

# Display OpenAI as the provider
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
st.sidebar.info(f"**Provider:** OpenAI")
st.sidebar.success(f"**Model:** `{model_name}`")

st.sidebar.divider()
st.sidebar.info(f"Session ID: {id(st.session_state)}")

if st.sidebar.button("Clear Session"):
    reset_session_state()
    st.rerun()
