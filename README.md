# 🦙 Llama Training

A project to train a Llama model for extracting specific fields from invoices.

## 📘 Getting Started

Follow the steps below to clone the repository, install dependencies, set up the Hugging Face token, and run the application.

---

### ✅ Prerequisites

- **Python 3.8 or higher** 
- **Git** 
- **Hugging Face Account** (to obtain an API token)

---

### 📂 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/damlois/Llama-Training
   cd Llama-Training
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate with Hugging Face**:
   - Get your Hugging Face API token from [Hugging Face](https://huggingface.co/settings/tokens).
   - Run the following command to log in to Hugging Face:
     ```bash
     huggingface-cli login
     ```
   - When prompted, paste your API token.

---

### 🚀 **Run the application**

Once you have set up the Hugging Face token, you can run the application:

```bash
python index.py
```