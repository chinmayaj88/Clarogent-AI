# 🦅 Clarogent: Universal AI Document & Audit Intelligence

**Clarogent** is an AI forensics platform built to automate field audits, document verification, and asset intelligence.

Architected to meet the high-precision standards of the **Solar Industry**, Clarogent goes beyond generic OCR to handle the specific constraints of field operations. Whether capturing technical specifications from solar panel labels under harsh glare or verifying installer safety compliance, the system is tuned for industrial-grade accuracy while maintaining universal applicability for field documents.

Powered by **Groq Cloud** and **Llama 4 Maverick Vision**, it delivers millisecond-latency OCR and scene understanding, instantly digitizing unique assets alongside standard government IDs (Aadhaar, PAN, Vehicle RC).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clarogent-ai.streamlit.app/)

---

## 🚀 Key Features

### 1. ⚡ Ultra-Fast Vision Engine

- Uses **meta-llama/llama-4-maverick-17b-128e-instruct** on Groq's LPU Inference Engine.
- Parses complex documents in <1.5 seconds.

### 2. 📄 Universal Document Parsing

- **No Templates Required**: The AI dynamically discovers fields.
- **Smart Classification**: Auto-detects if an image is a Solar Panel, Vehicle RC, Aadhaar, or Invoice.
- **Deep Extraction**: Captures nested details like `technical_specs` (Pmax, Voc, Isc for solar modules) and `vehicle_details`.

### 3. 👥 Field Intelligence

- **Human Detection**: Instantly counts personnel in a scene for compliance (e.g., "2 Installers Detected").
- **Safety Compliance**: Real-time verification of installer PPE (hard hats, vests) for on-site safety audits.

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10+
- A free API Key from [Groq Console](https://console.groq.com/keys)

### Quick Start

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/clarogent.git
   cd clarogent
   ```

2. **Run Setup Script (Windows)**
   This script automatically:
   - Creates a virtual environment (`venv`) using your system Python.
   - Installs all dependencies.
   - Launches the dashboard.

   ```powershell
   .\setup.bat
   ```

3. **Configure API Key**
   - Create a `.env` file in the root directory:
     ```env
     GROQ_API_KEY=gsk_your_key_here...
     ```
   - _Alternatively, enter the key in the sidebar when the app runs._

---

## 🏗️ Technical Architecture

| Component      | Technology   | Description                                          |
| -------------- | ------------ | ---------------------------------------------------- |
| **Frontend**   | Streamlit    | Responsive, real-time reactive dashboard.            |
| **AI Brain**   | Llama Vision | 17B parameter model for high-fidelity OCR.           |
| **Inference**  | Groq Cloud   | LPU-based acceleration for instant token generation. |
| **Resilience** | Tenacity     | Exponential backoff for robust API handling.         |

---

_Built with ❤️ for the Future of AI Auditing._
