
import logging
import json
import base64
import os
import requests
import pandas as pd
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional
from groq import Groq, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io

# Configure Enterprise Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SolarAI.Engine")

class SolarVisionEngine:
    """
    Enterprise-Grade Computer Vision Engine utilizing Groq's LPUs.
    Features:
    - Dynamic Document Parsing (Aadhaar, PAN, Solar Labels)
    - Rate Limiting with Exponential Backoff
    - Robust JSON Structure Enforcement
    """
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "llama-3.2-90b-vision-preview"):
        """Initialize the Groq client securely from Env or Argument."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Please provide it or set the environment variable.")
            
        self.client = Groq(api_key=self.api_key)
        # Using variable model ID for flexibility, default fallback from env if provided there
        self.model_id = os.getenv("GROQ_MODEL_ID", model_id)

    def _encode_image(self, image_path: str) -> str:
        """Optimized Base64 encoding for image payloads."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _preprocess_forensic(self, image_path: str) -> str:
        """
        Forensic Image Enhancement for Serial Number Extraction.
        Applies: Grayscale -> High Contrast -> Sharpening
        """
        try:
            with Image.open(image_path) as img:
                # 0. Handle EXIF Rotation (Crucial for mobile uploads)
                img = ImageOps.exif_transpose(img)
                
                # 1. Convert to Grayscale to remove color noise
                img = img.convert('L')
                
                # 2. Enhance Contrast (Make text blacker against background)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)  # Double the contrast
                
                # 3. Enhance Sharpness (Define edges of digits)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(2.0)  # Significant sharpening
                
                # 4. Resize if too small (width < 1000px) for better OCR
                if img.width < 1000:
                    ratio = 1000 / img.width
                    new_size = (1000, int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=95)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}. Falling back to raw image.")
            return self._encode_image(image_path)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def analyze_installation(self, image_path: str) -> Dict[str, Any]:
        """
        Executes a high-performance vision analysis with automatic rate-limit handling.
        Dynamically extracts key-value pairs based on document type.
        """
        logger.info(f"Initiating Vision Analysis on: {image_path}")
        try:
            base64_image = self._encode_image(image_path)
            
            # Universal Document Discovery Prompt (No Hardcoding)
            # CRITICAL: Includes Human Detection, Quality Check, and Domain Validation
            system_prompt = (
                "ACT AS A UNIVERSAL DOCUMENT/IMAGE PARSER. Your task is to digitize this image into a structured JSON format.\n"
                "RULES:\n"
                "1. NO PRE-DEFINED SCHEMA: Extract whatever fields are naturally present in the image.\n"
                "2. SMART KEYS: Convert visible labels to snake_case keys (e.g., 'Date of Birth' -> 'date_of_birth').\n"
                "3. HIERARCHY: If data is grouped (e.g., inside a box or under a header), nest it in the JSON.\n"
                "4. FULL COMPLETENESS: Capture every legible piece of text, including small print, headers, and footer codes.\n"
                "5. VALUES: Preserve exact text as seen on the document.\n"
                "6. VISION ENHANCEMENT: If the image is blurry, low-light, or contains noise (glare, shadows, scratches), use computer vision inference to reconstruct text. Filter out visual artifacts and focus on the data. Treat this as a forensic analysis task to recover data from compromised visibility.\n\n"
                "7. DOMAIN VALIDATION: The system is restricted to: Solar Energy sites/equipment, Field Audits, Official Documents (IDs, Invoices, Datasheets), Vehicles, and Industrial settings. IF THE IMAGE IS IRRELEVANT (e.g., food, scenic landscapes, random selfies, pets), set 'domain_relevant' to false.\n\n"
                "REQUIRED METADATA:\n"
                "- 'document_type': Infer the specific type (e.g., 'Vehicle Registration', 'Invoice', 'Solar Label').\n"
                "- 'domain_relevant': boolean (true if matches domain).\n"
                "- 'rejection_reason': string (null if relevant, else brief reason).\n"
                "- 'human_detected': boolean (true ONLY if a full human person, face, or distinct body is visible. DO NOT set to true for hands, fingers, or nails holding a document).\n"
                "- 'human_count': integer (count only full people/faces, exclude hands holding items).\n\n"
                "Output STRICTLY valid JSON. No markdown."
            )

            # Enterprise Implementation: Using Streaming for Real-time Latency Handling
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,           # Low temperature for deterministic output
                max_tokens=1024,
                top_p=1,
                stream=True,               # Enabling Streaming
                stop=None
            )

            # Stream aggregation mechanism
            full_response = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                
            logger.debug(f"Raw Model Output: {full_response[:100]}...")

            # Robust JSON Parsing with Fallback Logic
            try:
                # Sanitization: Locate the first '{' and last '}' to handle potential preamble leaks
                json_start = full_response.find('{')
                json_end = full_response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    clean_json = full_response[json_start:json_end]
                    return json.loads(clean_json)
                else:
                    raise ValueError("No JSON structure found in response")
            except json.JSONDecodeError:
                logger.error("JSON Parsing Failed. Attempting simplified extraction.")
                return {"error": "Invalid Metadata Format", "raw": full_response}

        except Exception as e:
            logger.error(f"Critical Vision Engine Failure: {str(e)}", exc_info=True)
            return {"error": str(e)}

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def extract_serial_only(self, image_path: str, asset_type: str = "Solar Module") -> str:
        """
        Specialized extraction for Serial Numbers with STRICT Validation.
        Uses a 2-Pass Logic to prevent digit omission.
        """
        def call_model(img_b64, rigorous=False):
            intensity = "EXTREME" if rigorous else "HIGH"
            system_prompt = (
                f"Your Task: Extract the SERIAL NUMBER from this {asset_type} photo.\n"
                f"MODE: {intensity} PRECISION FORENSICS.\n\n"
                "Target: A unique alphanumeric string identifying this unit.\n"
                "CRITICAL VALIDATION RULES:\n"
                "1. LENGTH CHECK: Solar serials are almost always 14-20 characters.\n"
                "   - If you see a short string (e.g., '6723'), it is WRONG. Look for the longer string.\n"
                "   - Example Format: '1646M6625L817662' (16 chars).\n"
                "2. NO OMISSION: It is better to include a questionable character than to skip it. Do not drop leading/trailing zeros.\n"
                "3. ORIENTATION: Text may be VERTICAL (rotated). Read carefully.\n"
                "4. CONTEXT: Ignore 'Model No', 'Date', 'Rating'. Look for the UNLABELED barcode string.\n\n"
                "OUTPUT FORMAT: Return ONLY the raw serial number. No JSON. No markdown."
            )
            
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=60,
                stream=False
            )
            return completion.choices[0].message.content.strip()

        try:
            # Pass 1: Standard Forensic Enhancement
            base64_v1 = self._preprocess_forensic(image_path)
            result_v1 = call_model(base64_v1, rigorous=False)
            
            clean_v1 = ''.join(filter(str.isalnum, result_v1))
            
            # Validation: If it looks good (14+ chars), accept it.
            if len(clean_v1) >= 14:
                return clean_v1
                
            # Pass 2: If result is suspicious (<14 chars), try 'Thresholding' to force B/W
            # This helps if the image was too gray/washed out.
            logger.info(f"Pass 1 result '{clean_v1}' suspicious. Retrying Pass 2...")
            
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img).convert('L')
                # Aggressive Thresholding (Binarization)
                # Any pixel < 128 becomes 0 (black), else 255 (white)
                img = img.point( lambda p: 255 if p > 100 else 0 )
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                base64_v2 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
            result_v2 = call_model(base64_v2, rigorous=True)
            clean_v2 = ''.join(filter(str.isalnum, result_v2))
            
            # Decision: Return the longest valid-looking string
            if len(clean_v2) > len(clean_v1):
                return clean_v2
            return clean_v1 if clean_v1 else "N/A"

        except Exception as e:
            logger.error(f"Serial Extraction Failed: {str(e)}")
            return "Error"



class BatchProcessor:
    """
    Handles bulk processing of Excel sheets containing Google Drive links.
    Integrates with SolarVisionEngine for row-by-row analysis.
    """
    def __init__(self, excel_path: str, api_key: Optional[str] = None, model_id: str = "llama-3.2-90b-vision-preview"):
        self.excel_path = excel_path
        # Initialize Engine (it handles env variable fallback)
        self.engine = SolarVisionEngine(api_key=api_key, model_id=model_id)
        
        # Load Excel with Pandas for processing logic
        try:
            self.df = pd.read_excel(excel_path)
            
            # Load Excel with OpenPyXL for style preservation
            self.wb = load_workbook(excel_path)
            self.ws = self.wb.active
            
            # Map column names to 1-based indices for OpenPyXL
            # Assumption: Headers are in the first row
            self.col_map_xl = {}
            for idx, col_cell in enumerate(self.ws[1], start=1):
                if col_cell.value:
                    self.col_map_xl[col_cell.value] = idx
                    
            logger.info(f"Loaded Excel file: {excel_path} with {len(self.df)} rows.")
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise

    def save(self, output_path: str):
        """Saves the processed workbook, preserving styles but auto-fitting columns."""
        try:
            # Auto-resize columns
            for i, col in enumerate(self.ws.columns, start=1):
                col_letter = get_column_letter(i)
                max_length = 0
                
                for cell in col:
                    try:
                        if cell.value:
                            val_len = len(str(cell.value))
                            if val_len > max_length:
                                max_length = val_len
                    except:
                        pass
                
                # Add padding and cap width
                adjusted_width = (max_length + 2)
                # Cap at 60 chars for readability
                if adjusted_width > 60: 
                    adjusted_width = 60
                
                self.ws.column_dimensions[col_letter].width = adjusted_width

            self.wb.save(output_path)
            logger.info(f"Saved processed file to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output file: {e}")
            raise

    def _update_xl(self, row_idx, col_name, value):
        """Updates the OpenPyXL worksheet cell, preserving style."""
        try:
            # Pandas index is 0-based. Excel rows are 1-based.
            # Usually row 1 is header. So data starts at row 2.
            # Row in Excel = row_idx + 2
            xl_row = row_idx + 2
            
            if col_name in self.col_map_xl:
                xl_col = self.col_map_xl[col_name]
                self.ws.cell(row=xl_row, column=xl_col).value = value
        except Exception as e:
            logger.warning(f"Failed to update Excel cell: {e}")

    def _get_drive_id(self, url: str) -> Optional[str]:
        """Extracts Google Drive file ID from URL."""
        if not isinstance(url, str):
            return None
            
        parsed = urlparse(url)
        if 'drive.google.com' in parsed.netloc:
            # Handle /file/d/ID/view format
            path_parts = parsed.path.split('/')
            if 'd' in path_parts:
                try:
                    return path_parts[path_parts.index('d') + 1]
                except IndexError:
                    pass
            
            # Handle ?id=ID format
            query = parse_qs(parsed.query)
            if 'id' in query:
                return query['id'][0]
                
        return None

    def _download_image(self, drive_id: str, save_path: str) -> bool:
        """Downloads image from Google Drive using ID."""
        url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return True
            else:
                logger.warning(f"Failed to download image: {drive_id} (Status: {response.status_code})")
                return False
        except Exception as e:
            logger.error(f"Error downloading image {drive_id}: {e}")
            return False


    def process_solar_audit_row(self, index, row, status_callback=None):
        """
        Specialized processor for Solar Audit Sheets.
        Automatically detects 'Module Serial Number Photo X' and fills 'AI Module Serial Number X'.
        """
        # Define the logic for pairs
        # We look for "Module Serial Number Photo X" and writes to "AI Module Serial Number X"
        # We look for "Inverter Serial Number Photo" (or similar) and writes to "AI Inverter Serial Number"
        
        cols = row.index.tolist()
        processed_any = False
        
        # 1. Helper to process a single pair
        def process_pair(photo_col, target_col, asset_type):
            img_url = row.get(photo_col)
            if not img_url or pd.isna(img_url) or not isinstance(img_url, str):
                return False
                
            drive_id = self._get_drive_id(img_url)
            if not drive_id:
                val = "Invalid Link"
                self.df.at[index, target_col] = val
                self._update_xl(index, target_col, val)
                return False
                
            temp_path = f"temp_{asset_type}_{index}.jpg"
            if self._download_image(drive_id, temp_path):
                serial = self.engine.extract_serial_only(temp_path, asset_type)
                self.df.at[index, target_col] = serial
                self._update_xl(index, target_col, serial)
                if os.path.exists(temp_path): os.remove(temp_path)
                return True
            else:
                 val = "Download Failed"
                 self.df.at[index, target_col] = val
                 self._update_xl(index, target_col, val)
                 return False

        # 2. Iterate through columns to find Photo columns
        count = 0
        for col in cols:
            # Check for Module Photos
            if "Module Serial Number Photo" in col:
                # Deduce target: Replace "Photo" with nothing, prefix with "AI" 
                # e.g., "Module Serial Number Photo 1" -> "AI Module Serial Number 1"
                # But user said "AI Module Serial Number 1" is the target.
                
                # Logic: Target = "AI " + col.replace(" Photo", "") 
                # Let's try to find the matching AI column dynamically
                suffix = col.replace("Module Serial Number Photo", "").strip()
                target_candidates = [c for c in cols if f"AI Module Serial Number {suffix}" in c]
                
                if target_candidates:
                    target_col = target_candidates[0]
                    if process_pair(col, target_col, "Solar Module"):
                        count += 1
                        
            # Check for Inverter Photos (DISABLED FOR NOW)
            # elif "Inverter" in col and ("Photo" in col or "Image" in col):
            #      # Find target "AI Inverter..."
            #      target_candidates = [c for c in cols if "AI Inverter Serial Number" in c]
            #      if target_candidates:
            #          target_col = target_candidates[0]
            #          if process_pair(col, target_col, "Inverter"):
            #              count += 1
            pass

        if status_callback:
            if count > 0:
                status_callback(f"✅ Row {index+1}: Extracted {count} Serial Numbers")
            else:
                status_callback(f"⚠️ Row {index+1}: No valid photos found")

    def process_row(self, index, row, img_col, human_col, ai_remarks_col, data_col, status_callback=None):
        """Processes a single row."""
        # ... (Old Logic - Unchanged) ...
        image_url = row.get(img_col)
        # Re-insert the start of the original method to ensure it's still available for generic mode
        if not image_url or pd.isna(image_url):
            logger.info(f"Row {index}: No image URL found. Skipping.")
            return

        drive_id = self._get_drive_id(image_url)
        if not drive_id:
            logger.warning(f"Row {index}: Could not extract Drive ID from URL: {image_url}")
            return
            
        temp_img_path = f"temp_batch_{index}.jpg"
        
        if self._download_image(drive_id, temp_img_path):
            try:
                # Analyze image
                result = self.engine.analyze_installation(temp_img_path)
                
                # Check for errors
                if "error" in result:
                    val = f"Error: {result['error']}"
                    self.df.at[index, ai_remarks_col] = val
                    self._update_xl(index, ai_remarks_col, val)
                    
                    if status_callback: status_callback(f"⚠️ Row {index+1}: Vision Error")
                else:
                    # Fill columns
                    human_detected_val = "YES" if result.get('human_detected', False) else "NO"
                    self.df.at[index, human_col] = human_detected_val
                    self._update_xl(index, human_col, human_detected_val)
                    
                    # AI Remarks Logic
                    remarks = []
                    if not result.get('domain_relevant', True):
                        remarks.append(f"Irrelevant: {result.get('rejection_reason', 'Unknown')}")
                    
                    remarks.append(f"Type: {result.get('document_type', 'Unknown')}")
                    remarks.append(f"Confidence: {result.get('confidence', 'N/A')}")
                    
                    # Human Count
                    h_count = result.get('human_count', 0)
                    if h_count > 0:
                         remarks.append(f"Humans: {h_count}")

                    remarks_val = " | ".join(remarks)
                    self.df.at[index, ai_remarks_col] = remarks_val
                    self._update_xl(index, ai_remarks_col, remarks_val)
                    
                    # Extracted Data (JSON Dump or Summary)
                    # Filter out metadata keys for cleaner data
                    excluded_keys = ['document_type', 'domain_relevant', 'rejection_reason', 'human_detected', 'human_count', 'confidence', 'error']
                    data_only = {k: v for k, v in result.items() if k not in excluded_keys}
                    data_val = str(data_only)
                    self.df.at[index, data_col] = data_val
                    self._update_xl(index, data_col, data_val) 
                    
                    if status_callback: status_callback(f"✅ Row {index+1}: Processed")
                
                logger.info(f"Row {index}: Processed successfully.")
                
            except Exception as e:
                logger.error(f"Row {index}: Processing error: {e}")
                val = f"Processing Error: {str(e)}"
                self.df.at[index, ai_remarks_col] = val
                self._update_xl(index, ai_remarks_col, val)
                if status_callback: status_callback(f"❌ Row {index+1}: Exception")
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
        else:
            val = "Download Failed"
            self.df.at[index, ai_remarks_col] = val
            self._update_xl(index, ai_remarks_col, val)
            if status_callback: status_callback(f"⚠️ Row {index+1}: Download Failed (Check Permissions)")
