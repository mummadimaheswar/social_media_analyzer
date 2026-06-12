import streamlit as st
import json
import os
from dotenv import load_dotenv
from graph import run_analysis

load_dotenv()

st.set_page_config(
    page_title="Advanced Social Media Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced UI CSS
custom_css = """
<style>
    :root {
        --primary-color: #ffffff;
        --background-color: #000000;
        --secondary-background-color: #111111;
        --text-color: #ffffff;
    }
    
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ensure regular Streamlit text is white */
    .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6, label {
        color: #ffffff !important;
    }
    
    /* Input fields (Sidebar and text areas) need visibility */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
    }
    
    /* Glassmorphism effects */
    .metric-card {
        background: rgba(17, 17, 17, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.5);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        border: none;
        border-radius: 0.5rem;
        color: white;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
        box-shadow: 0 6px 20px rgba(75, 108, 183, 0.4);
        transform: translateY(-2px);
    }

    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #aaaaaa;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #aaaaaa;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        color: #ffffff;
        font-weight: 700;
    }
    .trace-link {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #ffffff;
        color: #000000 !important;
        text-decoration: none;
        border-radius: 0.5rem;
        font-weight: 600;
        margin-top: 1rem;
        transition: background-color 0.2s;
    }
    .trace-link:hover {
        background-color: #cccccc;
    }
    
    /* Ensure sidebar matches */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #ffffff;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Social Media Post Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered insights for your social media content</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Input Post")
    user_input = st.text_area(
        "Paste your social media content here:", 
        placeholder="e.g. We are thrilled to announce our new product feature! Our team has worked so hard on this. Check out the link below! 👇🔥 #launch #startup", 
        height=200
    )
    
    analyze_btn = st.button("✨ Analyze Post", use_container_width=True, type="primary")

if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a post to analyze.")
    elif not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        st.error("❌ Please ensure GOOGLE_API_KEY is set in your .env file.")
    else:
        with st.spinner("🤖 Analyzing with Gemini & LangGraph..."):
            try:
                analysis_result = run_analysis(user_input)
                
                with col2:
                    st.subheader("📊 Analysis Results")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Tone</div><div class="metric-value">{analysis_result.get("tone", "N/A")}</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Intent</div><div class="metric-value">{analysis_result.get("intent", "N/A")}</div></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Style</div><div class="metric-value">{analysis_result.get("communication_style", "N/A")}</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    with st.expander("📝 Detailed Summary", expanded=True):
                        st.write(analysis_result.get("summary", "No summary provided."))
                        
                    trace_url = analysis_result.get("trace_url")
                    if trace_url:
                        st.markdown(f'<a href="{trace_url}" target="_blank" class="trace-link">🔍 View LLM Trace in Langfuse</a>', unsafe_allow_html=True)
                    elif os.environ.get("LANGFUSE_PUBLIC_KEY"):
                        st.info("ℹ️ Trace URL will be available shortly inside Langfuse dashboard.")
                        
                    # Also show full json for the curious
                    with st.expander("🛠 Raw JSON Output"):
                        st.json(analysis_result)
                        
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
