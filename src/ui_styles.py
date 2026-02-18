
# Enhanced UI Styles for Enterprise Look
HEADER_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #1a202c, #0e1117);
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* Typography */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #63b3ed, #4299e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.25rem;
        font-weight: 300;
        margin-bottom: 2.5rem;
    }
    
    /* Metrics Customization */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #63b3ed;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #cbd5e0;
        font-weight: 500;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s;
        box-shadow: 0 4px 6px rgba(49, 130, 206, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(49, 130, 206, 0.3);
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    /* JSON Viewer */
    .stJson {
        background: rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Fira Code', monospace;
    }
    
    /* Status Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #63b3ed;
        border-bottom: 2px solid #63b3ed;
    }
</style>
"""

FOOTER_STYLE = """
<div style="text-align: center; margin-top: 5rem; padding: 2rem; color: #718096; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.05);">
    <p>🛡️ <strong>Clarogent AI</strong> • Enterprise Document Forensics • Powered by Groq & Llama Vision</p>
</div>
"""
