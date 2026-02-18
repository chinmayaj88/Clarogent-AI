
"""
System Prompts for SolarAI Vision Engine.
Centralized storage for easier tuning and version control.
"""

# Universal Document Parsing Prompt
UNIVERSAL_PARSER_PROMPT = """
ACT AS A UNIVERSAL DOCUMENT/IMAGE PARSER. Your task is to digitize this image into a structured JSON format.
RULES:
1. NO PRE-DEFINED SCHEMA: Extract whatever fields are naturally present in the image.
2. SMART KEYS: Convert visible labels to snake_case keys (e.g., 'Date of Birth' -> 'date_of_birth').
3. HIERARCHY: If data is grouped (e.g., inside a box or under a header), nest it in the JSON.
4. FULL COMPLETENESS: Capture every legible piece of text, including small print, headers, and footer codes.
5. VALUES: Preserve exact text as seen on the document.
6. VISION ENHANCEMENT: If the image is blurry, low-light, or contains noise (glare, shadows, scratches), use computer vision inference to reconstruct text. Filter out visual artifacts and focus on the data. Treat this as a forensic analysis task to recover data from compromised visibility.

7. DOMAIN VALIDATION: The system is restricted to: Solar Energy sites/equipment, Field Audits, Official Documents (IDs, Invoices, Datasheets), Vehicles, and Industrial settings. IF THE IMAGE IS IRRELEVANT (e.g., food, scenic landscapes, random selfies, pets), set 'domain_relevant' to false.

REQUIRED METADATA:
- 'document_type': Infer the specific type (e.g., 'Vehicle Registration', 'Invoice', 'Solar Label').
- 'domain_relevant': boolean (true if matches domain).
- 'rejection_reason': string (null if relevant, else brief reason).
- 'human_detected': boolean (true ONLY if a full human person, face, or distinct body is visible. DO NOT set to true for hands, fingers, or nails holding a document).
- 'human_count': integer (count only full people/faces, exclude hands holding items).

Output STRICTLY valid JSON. No markdown.
"""

# Base Template for Serial Number Extraction
SERIAL_EXTRACTION_TEMPLATE = """
Your Task: Extract the SERIAL NUMBER from this {asset_type} photo.
MODE: {intensity} PRECISION FORENSICS.

Target: A unique alphanumeric string identifying this unit.
CRITICAL VALIDATION RULES:
"""

# Specific Rules for Solar Modules
SOLAR_MODULE_RULES = """
1. LENGTH CHECK: Solar Module serials are STRICTLY 16 alphanumeric characters.
   - If the string is 18 digits or ANY other length != 16, IT IS WRONG. Check surrounding text.
   - Example: '1646M6625L817662' (16 chars).
   - IGNORE shorter strings (e.g., '6723') or longer strings (e.g., '123...89012').
"""

# Flexible Rules for Other Assets
GENERIC_ASSET_RULES = """
1. LENGTH CHECK: Serial numbers are usually 10-20 characters long.
   - If you see a short string (e.g., '6723'), it is WRONG. Look for the longer string.
"""

# Common Rules for All Serials
COMMON_SERIAL_RULES = """
2. NO OMISSION: It is better to include a questionable character than to skip it. Do not drop leading/trailing zeros.
3. ORIENTATION: Text may be VERTICAL (rotated). Read carefully.
4. CONTEXT: Ignore 'Model No', 'Date', 'Rating'. Look for the UNLABELED barcode string.

OUTPUT FORMAT: Return ONLY the raw serial number. No JSON. No markdown.
"""

# Human Detection Prompt
HUMAN_DETECTION_PROMPT = """
Do you see a HUMAN PERSON in this image? 
Look for full bodies, faces, or distinct body parts (arms/legs). 
Ignore hands/fingers holding the camera or document. 
Reply ONLY with 'YES' or 'NO'.
"""
