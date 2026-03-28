# DukeGPT

AIPI 590 -- Large Language Models

Developer: Keese Phillips

---

## About

DukeGPT is a retrieval-augmented chatbot built for current and prospective Duke University students. It answers commonly asked questions about student life, admissions, academics, and upcoming campus events by grounding every response in Duke's own source material rather than relying on the model's pretrained knowledge.

The system uses a LangChain agent backed by DeepSeek-R1-Distill-Llama-8B (4-bit quantized via BitsAndBytes) to route user queries to the appropriate knowledge source. Three FAISS vector stores serve as the retrieval layer, each built from web-scraped content covering Duke University general information, the Pratt School of Engineering, and the AIPI master's program. A fourth tool queries the Duke Events Calendar API to surface upcoming lectures, ceremonies, games, and other campus events within the next 30 days. The frontend is a Streamlit application styled with Duke's brand guidelines.

## Architecture

The query pipeline works as follows. A user submits a question through the Streamlit interface. The LangChain zero-shot ReAct agent analyzes the question and selects the most relevant tool: `duke_knowledge_search` for general university questions, `pratt_knowledge_search` for engineering school questions, `aipi_knowledge_search` for AIPI program questions, or `events_search` for upcoming events. The selected tool either performs a FAISS similarity search against the appropriate vector store or queries the Duke Events Calendar JSON API. The agent then summarizes the retrieved content into a natural language response and returns it to the user.

Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`. The knowledge base documents are stored as Parquet files and indexed into FAISS at first run, with the resulting indices cached to disk for subsequent startups.

## Project Structure

```
.
|-- app.py                          Streamlit frontend application
|-- model.py                        LangChain agent, tools, and FAISS vector stores
|-- assets/                         Duke branding images used in the UI
|-- data/                           Parquet files for the knowledge base
|   |-- duke.parquet                General Duke University content
|   |-- pratt.parquet               Pratt School of Engineering content
|   |-- aipi.parquet                AIPI master's program content
|-- notebooks/
|   |-- webscrapping.ipynb          Recursive web scraper for building the knowledge base
|-- .streamlit/
|   |-- config.toml                 Streamlit theme configuration (Duke Navy palette)
|-- streamlit.service               systemd unit file for production deployment
|-- .env                            HuggingFace API key (not committed)
|-- .gitignore                      Ignores __pycache__ and .env
|-- .gitattributes                  Git line-ending normalization
|-- README.md                       This file
```

## Requirements

The project requires Python 3.12+ and a CUDA-capable GPU for model inference. The core dependencies are as follows.

**Model and Agent:**
DeepSeek-R1-Distill-Llama-8B served through HuggingFace Transformers with 4-bit NF4 quantization via BitsAndBytes, orchestrated by a LangChain zero-shot ReAct agent with conversation buffer memory.

**Retrieval:**
FAISS for vector similarity search, sentence-transformers/all-MiniLM-L6-v2 for embedding generation, and Pandas for Parquet data loading.

**Frontend:**
Streamlit with custom Duke University theming (EB Garamond headings, Open Sans body text, Duke Navy primary color).

**Data Collection:**
BeautifulSoup4 and Requests for recursive web scraping of Duke domains.

Install all dependencies:

```bash
pip install torch transformers bitsandbytes accelerate
pip install langchain langchain-core langchain-community
pip install faiss-gpu sentence-transformers
pip install streamlit pandas pyarrow
pip install python-dotenv huggingface_hub
pip install beautifulsoup4 requests
```

## Setup

Clone the repository and create a `.env` file containing your HuggingFace API token. This token is required to download the DeepSeek model weights on first run.

```bash
git clone https://github.com/<your-username>/DukeGPT.git
cd DukeGPT
echo "API_KEY=hf_your_token_here" > .env
```

The knowledge base Parquet files should already be present in the `data/` directory. If you need to rebuild them from scratch, run the `notebooks/webscrapping.ipynb` notebook, which recursively scrapes Duke web pages up to a configurable depth and exports the chunked content to Parquet format.

## Usage

Launch the Streamlit application locally:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`. On first startup, the model weights are downloaded from HuggingFace Hub and the FAISS indices are built from the Parquet files. Subsequent startups load the cached indices from disk and skip the indexing step.

For production deployment, the included `streamlit.service` file provides a systemd unit configuration that runs the application as a background service with automatic restart on failure.

```bash
sudo cp streamlit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
```

## Limitations

DukeGPT is scoped to Duke University content and cannot answer questions outside the Duke ecosystem. The knowledge base covers general Duke University information, the Pratt School of Engineering, and the AIPI master's program. Event queries are limited to events occurring within the next 30 days as returned by the Duke Events Calendar API. Responses are grounded entirely in retrieved content -- the agent is instructed not to supplement answers with its pretrained knowledge, which means questions on topics not covered by the knowledge base will receive an acknowledgment rather than a speculative answer.
