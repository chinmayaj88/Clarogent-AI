import streamlit as st
import os
import time
import logging
import tempfile
import pandas as pd
from typing import Optional, List
from dotenv import load_dotenv
from src.engine import SolarVisionEngine, BatchProcessor
from src.ui_styles import HEADER_STYLE, FOOTER_STYLE

# --- Configuration & Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SolarAI.App")

st.set_page_config(
    page_title="Clarogent AI | Batch Document Processor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_MODELS = [
    "llama-3.2-90b-vision-preview",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-11b-vision-preview"
]

def load_css():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(HEADER_STYLE, unsafe_allow_html=True)

def render_sidebar() -> tuple[Optional[str], str]:
    """Renders the sidebar configuration and returns API Key and Model ID."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "Groq API Key", 
            type="password", 
            value=os.getenv("GROQ_API_KEY", ""), 
            help="Get yours securely at console.groq.com"
        )
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🧠 Logic Engine")
        env_models = os.getenv("GROQ_MODELS")
        model_options = env_models.split(",") if env_models else DEFAULT_MODELS
        selected_model = st.selectbox("Vision Model", model_options, index=0)
        
        st.info(f"**Active Model:** `{selected_model}`\n\n**Latency Target:** < 1.5s")
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("●  **Batch Engine Online**")
        st.caption("v2.6.0-Enterprise • Llama 4 Ready")
        
        return api_key, selected_model

def process_batch(uploaded_file, api_key: str, model_id: str, col_map: dict):
    """
    Handles the end-to-end batch processing logic.
    - Saves temp file
    - Initializes Engine
    - Runs Processing Loop
    - Handles Cleanup
    """
    # Create valid temp files using tempfile module for robustness
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_input:
        tmp_input.write(uploaded_file.getbuffer())
        temp_input_path = tmp_input.name

    output_filename = f"processed_{uploaded_file.name}"
    # We will save output to a path, but ultimately return bytes for download
    
    st.toast("Starting Batch Job...", icon="🚀")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    logs_container = st.expander("📜 Real-time Execution Logs", expanded=True)
    
    try:
        # Initialize Processor
        processor = BatchProcessor(temp_input_path, api_key, model_id=model_id)
        
        # Dev Limiter
        try:
            limit_rows = int(os.getenv("MAX_ROWS_LIMIT", 0))
            if limit_rows > 0:
                processor.df = processor.df.head(limit_rows)
                st.warning(f"⚠️ **DEV MODE:** Processing limited to first {limit_rows} rows.")
        except ValueError:
            pass
            
        total_rows = len(processor.df)
        
        start_time = time.time()
        
        # Define status callback for UI updates
        def ui_callback(msg):
            with logs_container:
                st.code(msg, language="bash")

        # Execution Loop
        for index, row in processor.df.iterrows():
            status_text.markdown(f"**Processing Row {index + 1}/{total_rows}**")
            
            if col_map.get('type') == 'solar_audit':
                 processor.process_solar_audit_row(
                     index,
                     row,
                     status_callback=ui_callback
                 )
            else:
                # Standard Mode
                processor.process_row(
                    index, 
                    row, 
                    col_map['img'], 
                    col_map['human'], 
                    col_map['remarks'], 
                    col_map['data'],
                    status_callback=ui_callback
                )
            
            # Update Progress
            progress = (index + 1) / total_rows
            progress_bar.progress(progress)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save results to a bytes buffer for download without temp file issues
        # Or save to a temp output file if pandas requires path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_output:
            # Save using the processor's save method to preserve styles via OpenPyXL
            tmp_output.close() # Close file handle so openpyxl can write to it
            processor.save(tmp_output.name)
            tmp_output_path = tmp_output.name

            
        st.success(f"✅ **Processing Complete!** {total_rows} rows processed in {duration:.2f}s.")
        st.balloons()
        
        return tmp_output_path

    except Exception as e:
        logger.error(f"Batch Processing Error: {e}", exc_info=True)
        st.error(f"❌ **Critical Batch Error:** {str(e)}")
        return None
        
    finally:
        # Cleanup Input File
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp input: {e}")

def main():
    load_css()
    
    # 1. Sidebar & Config
    api_key, selected_model = render_sidebar()

    if not api_key:
        st.warning("⚠️ **System Offline:** Please provide a valid Groq API Key to initialize the forensics engine.")
        st.stop()

    # 2. Hero Section
    st.markdown('<h1 class="hero-title">Clarogent AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Enterprise Batch Document Intelligence & Field Auditing.</p>', unsafe_allow_html=True)

    # 3. Main Interface
    col_upload, col_mapping = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 **Data Ingestion**")
        uploaded_file = st.file_uploader(
            "Upload Excel Sheet (.xlsx) with Google Drive Links", 
            type=["xlsx"],
            help="Ensure your sheet has a column with 'Anyone with the link' public Google Drive URLs."
        )

    
    if uploaded_file:
        try:
            df_preview = pd.read_excel(uploaded_file)
            columns = list(df_preview.columns)
            
            with col_mapping:
                # --- INTELLIGENT AUTO-CONFIGURATION ---
                st.markdown("### 🤖 **Auto-Analysis Config**")
                
                # 1. Detect Mode Based on Headers
                has_solar_headers = any("Module Serial Number Photo" in c for c in columns)
                
                if has_solar_headers:
                    mode = "solar_audit"
                    st.success("✅ **Detected: Solar Serial Number Audit**")
                    
                    # Scan for counts
                    module_count = sum(1 for c in columns if "Module Serial Number Photo" in c)
                    inverter_count = sum(1 for c in columns if "Inverter Serial Number Photo" in c or "Inverter Serial Number Image" in c)
                    
                    st.info(f"• Found {module_count} Module Photo Series\n• Found {inverter_count} Inverter Photo Series")
                    
                    col_map = {'type': 'solar_audit'}
                    
                else:
                    mode = "standard"
                    st.info("ℹ️ **Detected: Standard Batch Analysis**")
                    
                    # 2. HEURISTIC IMAGE COLUMN DETECTION
                    detected_img_col = None
                    sample = df_preview.head(5)
                    
                    for col in columns:
                        try:
                            sample_vals = sample[col].astype(str).tolist()
                            for val in sample_vals:
                                if "drive.google.com" in val or "http" in val:
                                    detected_img_col = col
                                    break
                            if detected_img_col: break
                        except:
                            continue
                    
                    if not detected_img_col:
                        keywords = ["image", "link", "url", "drive", "photo", "picture"]
                        for col in columns:
                            if any(k in col.lower() for k in keywords):
                                detected_img_col = col
                                break
                    
                    if not detected_img_col:
                        detected_img_col = columns[0]
                        st.warning(f"⚠️ Could not auto-detect image column. Defaulting to '{detected_img_col}'. Ensure links are present.")
                    else:
                        st.success(f"🔗 **Targeting Image Column:** `{detected_img_col}`")

                    col_map = {
                        'type': 'standard',
                        'img': detected_img_col,
                        'human': 'Human Detected',
                        'remarks': 'AI Remarks',
                        'data': 'Extracted Data'
                    }

            st.markdown("---")
            
            # Preview Data
            with st.expander("👀 Data Preview (First 5 Rows)", expanded=False):
                st.dataframe(df_preview.head())

            # Action Button
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                 output_path = process_batch(uploaded_file, api_key, selected_model, col_map)
                 
                 if output_path:
                    with open(output_path, "rb") as f:
                        file_data = f.read()
                        
                    st.download_button(
                        label="⬇️ Download Processed Report",
                        data=file_data,
                        file_name=f"processed_{uploaded_file.name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # Cleanup Output
                    try:
                        os.remove(output_path)
                    except:
                        pass
                    
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

    # Footer
    st.markdown(FOOTER_STYLE, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
