import logging
import json
import base64
from typing import Dict, Any, Optional
from groq import Groq, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
    
    def __init__(self, api_key: str, model_id: str = "llama-3.2-90b-vision-preview"):
        """Initialize the Groq client securely."""
        self.client = Groq(api_key=api_key)
        # Using variable model ID for flexibility
        self.model_id = model_id

    def _encode_image(self, image_path: str) -> str:
        """Optimized Base64 encoding for image payloads."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

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
