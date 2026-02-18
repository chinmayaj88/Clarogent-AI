# 🦅 Clarogent: AI-Powered Batch Document Intelligence

**Clarogent** is an enterprise-grade AI forensics platform designed for high-volume document processing and field audit intelligence.

Unlike traditional OCR tools that require templates, Clarogent uses **Llama 3.2 90B Vision** on **Groq Cloud** to dynamically understand and extract structured data from thousands of images instantly. It is specifically optimized for processing Google Drive links in bulk from Excel sheets.

![Batch Processor](https://img.shields.io/badge/Clarogent-Batch_AI-blue?style=for-the-badge&logo=python)

---

## 🚀 Key Features

### 1. ⚡ High-Volume Batch Processing

- **Excel-Driven Workflow**: Upload an `.xlsx` file containing Google Drive image links.
- **Smart Column Mapping**: Automatically detects and maps your Excel columns to the AI engine.
- **Real-Time Progress**: Watch as the AI processes each row, updating status and error logs instantly.

### 2. 🧠 Advanced Vision capabilities

- **Universal Extraction**: Extracts key-value pairs from any document type (Invoices, IDs, Solar Labels) without training.
- **Human Detection**: Automatically flags images containing people for compliance or privacy audits.
- **Quality Analysis**: Identifies blurry, low-light, or irrelevant images and provides actionable remarks.

### 3. 🛡️ Enterprise-Ready

- **Secure Handling**: Processes images in-memory without storing sensitive data permanently.
- **Rate Limiting**: Built-in exponential backoff handles API limits gracefully.
- **Detailed Reporting**: Generates a comprehensive Excel report with extracted JSON data and AI insights.

---

## 🛠️ Usage Guide

### prerequisites

1. **Python 3.10+**
2. **Groq API Key**: Get one for free at [Groq Console](https://console.groq.com/keys).
3. **Google Drive Images**: Ensure your image links are set to **"Anyone with the link"** (Public) so the AI can access them.

### Quick Start

1. **Clone & Install**

   ```bash
   git clone https://github.com/yourusername/clarogent.git
   cd clarogent
   .\setup.bat
   ```

2. **Prepare Your Data**
   - Create a Google Sheet with your image links.
   - **Download it as an Excel file (.xlsx)**.
   - Example Columns: `Image Link`, `Human Detected`, `AI Remarks`, `Extracted Data`.

3. **Run the Application**

   ```powershell
   streamlit run app.py
   ```

4. **Process & Download**
   - Upload your Excel file.
   - Map the columns in the sidebar.
   - Click **Start Batch Processing**.
   - Download the final report when done!

---

## 🏗️ Technical Architecture

| Component     | Technology      | Description                                                     |
| :------------ | :-------------- | :-------------------------------------------------------------- |
| **Frontend**  | Streamlit       | Reactive web interface for file handling and real-time updates. |
| **Engine**    | Python + Pandas | Core logic for batch processing, data mapping, and export.      |
| **AI Model**  | Llama 3.2 90B   | State-of-the-art vision model for deep document understanding.  |
| **Inference** | Groq LPU        | Ultra-low latency inference for rapid batch processing.         |

---

_Built for the Future of AI Automation._
