import streamlit as st
import os
import time
import logging
import tempfile
import pandas as pd
import threading
import queue
import concurrent.futures
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

# --- Session State Management ---
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = []

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
            help="Get yours securely at console.groq.com",
            disabled=st.session_state.is_processing
        )
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("### 🧠 Logic Engine")
        env_models = os.getenv("GROQ_MODELS")
        model_options = env_models.split(",") if env_models else DEFAULT_MODELS
        selected_model = st.selectbox(
            "Vision Model", 
            model_options, 
            index=0,
            disabled=st.session_state.is_processing
        )
        
        st.info(f"**Active Model:** `{selected_model}`\n\n**Latency Target:** < 1.5s")
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("●  **Batch Engine Online**")
        st.caption("v2.7.0-Enterprise • Llama 4 Ready")
        
        return api_key, selected_model

def start_batch_processing():
    """Callback to trigger processing state."""
    st.session_state.is_processing = True
    st.session_state.processed_file = None
    st.session_state.log_buffer = []

def process_batch_logic(uploaded_file, api_key: str, model_id: str, col_map: dict):
    """
    Handles the end-to-end batch processing logic with Queue-based UI updates.
    """
    # Create valid temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_input:
        tmp_input.write(uploaded_file.getbuffer())
        temp_input_path = tmp_input.name

    try:
        # UI Elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        logs_placeholder = st.empty()
        
        # Log Queue for Thread Safety
        log_queue = queue.Queue()
        
        # Initialize Processor
        processor = BatchProcessor(temp_input_path, api_key, model_id=model_id)
        
        # Dev Limiter
        env_limit = os.getenv("MAX_ROWS_LIMIT")
        if env_limit and env_limit.strip():
            try:
                limit_rows = int(env_limit)
                if limit_rows > 0:
                    processor.df = processor.df.head(limit_rows)
                    st.warning(f"⚠️ **DEV MODE:** Processing limited to first {limit_rows} rows. Remove MAX_ROWS_LIMIT from .env to process all.")
            except ValueError:
                pass # Invalid number in env var, defaulting to all rows
        
        total_rows = len(processor.df)
        start_time = time.time()
        
        # Define worker function
        def process_wrapper(idx, r):
            try:
                # Custom callback that puts messages into our thread-safe queue
                def thread_safe_callback(msg):
                    log_queue.put(msg)
                
                if col_map.get('type') == 'solar_audit':
                    processor.process_solar_audit_row(idx, r, status_callback=thread_safe_callback)
                else:
                    processor.process_row(
                        idx, r, 
                        col_map['img'], 
                        col_map['human'], 
                        col_map['remarks'], 
                        col_map['data'],
                        status_callback=thread_safe_callback
                    )
                return True
            except Exception as e:
                log_queue.put(f"❌ Row {idx+1} Failed: {str(e)}")
                return False

        # --- PARALLEL EXECUTION LOOP ---
        # Increased to 20 workers for maximum I/O throughput
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_wrapper, idx, row) for idx, row in processor.df.iterrows()]
            
            completed = 0
            while completed < total_rows:
                # 1. Update Logs from Queue
                while not log_queue.empty():
                    msg = log_queue.get()
                    st.session_state.log_buffer.append(msg)
                    # Keep buffer reasonable size to prevent UI lag
                    if len(st.session_state.log_buffer) > 10:
                        st.session_state.log_buffer.pop(0)
                
                # Render Logs (In-Place Update)
                logs_placeholder.code("\n".join(st.session_state.log_buffer), language="bash")

                # 2. Check Futures
                # We count how many satisfy "done()" without blocking
                new_completed = sum(1 for f in futures if f.done())
                if new_completed > completed:
                    completed = new_completed
                    progress = completed / total_rows
                    progress_bar.progress(progress)
                    status_text.markdown(f"**⚡ Processing... {completed}/{total_rows} Rows Completed**")
                
                time.sleep(0.1) # Prevent tight loop

            # Wait for any lingering threads
            concurrent.futures.wait(futures)

        end_time = time.time()
        duration = end_time - start_time
        
        # Save results
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_output:
            tmp_output.close()
            processor.save(tmp_output.name)
            return tmp_output.name

    except Exception as e:
        logger.error(f"Batch Processing Error: {e}", exc_info=True)
        st.error(f"❌ **Critical Batch Error:** {str(e)}")
        return None
        
    finally:
        # Cleanup Input File
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except:
                pass

def main():
    load_css()
    
    # 1. Sidebar & Config
    api_key, selected_model = render_sidebar()

    if not api_key:
        st.warning("⚠️ **System Offline:** Please provide a valid Groq API Key.")
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
            help="Ensure your sheet has a column with 'Anyone with the link' public Google Drive URLs.",
            disabled=st.session_state.is_processing
        )

    if uploaded_file:
        try:
            if not st.session_state.processed_file:
                # Only show preview if not currently processing or finished
                df_preview = pd.read_excel(uploaded_file)
                columns = list(df_preview.columns)
                
                with col_mapping:
                    # --- AUTO-CONFIG ---
                    st.markdown("### 🤖 **Auto-Analysis Config**")
                    has_solar_headers = any("Module Serial Number Photo" in c for c in columns)
                    
                    if has_solar_headers:
                        mode = "solar_audit"
                        st.success("✅ **Detected: Solar Serial Number Audit**")
                        col_map = {'type': 'solar_audit'}
                    else:
                        mode = "standard"
                        st.info("ℹ️ **Detected: Standard Batch Analysis**")
                        # Heuristic detection logic (simplified for brevity)
                        detected_img_col = next((c for c in columns if any(k in c.lower() for k in ["image", "link", "url", "drive"])), columns[0])
                        detected_remarks_col = next((c for c in columns if "ai remarks" in c.lower()), "AI Remarks")
                        
                        col_map = {
                            'type': 'standard',
                            'img': detected_img_col,
                            'human': 'Human Detected',
                            'remarks': detected_remarks_col,
                            'data': 'Extracted Data'
                        }

                st.markdown("---")
                with st.expander("👀 Data Preview (First 5 Rows)", expanded=False):
                    st.dataframe(df_preview.head())

                # START BUTTON
                # We use a button to trigger session state change
                if st.button("🚀 Start Processing", type="primary", use_container_width=True, disabled=st.session_state.is_processing, on_click=start_batch_processing):
                    # Force a rerun to apply disabled state immediately
                    pass

        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

    # --- PROCESSING BLOCK ---
    if st.session_state.is_processing and uploaded_file:
        with st.status("🚀 **AI Batch Processing Active**", expanded=True) as status:
            st.write("Initializing Worker Swarm...")
            
            # Re-infer col_map since it's local scope above (or persist it in session_state, but re-computing is fast)
            # For simplicity, repeating detection or assuming it's consistent
            df_temp = pd.read_excel(uploaded_file)
            cols = list(df_temp.columns)
            if any("Module Serial Number Photo" in c for c in cols):
                 col_map = {'type': 'solar_audit'}
            else:
                 col_map = {'type': 'standard', 'img': cols[0], 'human': 'Human Detected', 'remarks': 'AI Remarks', 'data': 'Data'} 

            output_path = process_batch_logic(uploaded_file, api_key, selected_model, col_map)
            
            if output_path:
                st.session_state.processed_file = output_path
                status.update(label="✅ Processing Complete!", state="complete", expanded=False)
            
            st.session_state.is_processing = False
            st.rerun()

    # --- DOWNLOAD BLOCK ---
    if st.session_state.processed_file:
        st.success("🎉 **Analysis Complete!** Your report is ready.")
        
        with open(st.session_state.processed_file, "rb") as f:
            file_data = f.read()
            
        col_dl_1, col_dl_2 = st.columns([1, 1])
        with col_dl_1:
             st.download_button(
                label="⬇️ Download Document",
                data=file_data,
                file_name=f"processed_report_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with col_dl_2:
            if st.button("🔄 Process New File", use_container_width=True):
                st.session_state.processed_file = None
                st.session_state.log_buffer = []
                st.rerun()

    # Footer
    st.markdown(FOOTER_STYLE, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
