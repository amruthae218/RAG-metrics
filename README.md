# How to Run Your Streamlit RAG QA App in Google Colab


### 1. Install Required Packages

Open a Google Colab notebook and run:

```python
!pip install streamlit langchain langchain-community langchain-groq langchain-huggingface sentence-transformers faiss-cpu pypdf bert-score scikit-learn python-dotenv plotly pyngrok
```

### 2. Save Your Streamlit App Script

Save your complete Streamlit Python script (`app.py`) from your local machine to Colab. You can do this by copying the code and writing it to a file:

```python
app_code = '''
# Paste your full app.py code here
'''
with open("app.py", "w") as f:
    f.write(app_code)
```

### 3. Set Up Groq API Key and Any Other Environment Variables

You should set environment variables in the Colab cell before running the Streamlit app:

```python
import os
os.environ["GROQ_KEY"] = "your_groq_api_key"  # Replace with your Groq API key
```

### 4. Start Streamlit with a Tunnel

Run the Streamlit app in the background and use a tunnel to expose the web port:

```python
from pyngrok import ngrok
import threading

def run():
    os.system("streamlit run app.py --server.port 8501 --server.enableCORS false")

threading.Thread(target=run).start()

# Open the tunnel and print the public URL
public_url = ngrok.connect(8501)
print(f"Visit this Streamlit app URL:\n{public_url}")
```
- Wait for the URL (it will look like `https://xxxx-8501.ngrok-free.app`).
- Open this link in your browser to use your full Streamlit UI from Colab[1][2].

### 5. Upload Files and Use the App

- Use the upload widgets in your Streamlit sidebar to upload the PDF and QA CSV.
- Run batch evaluations, select Groq models, and use the visualization features exactly as you would on your local machine.

