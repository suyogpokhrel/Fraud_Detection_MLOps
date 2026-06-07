# Running FastAPI + Streamlit locally

1. Install dependencies (inside `fraud_mlops` environment):

```bash
conda activate fraud_mlops
pip install -r requirements.txt
```

2. Start FastAPI server:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

3. In another terminal, start Streamlit UI:

```bash
streamlit run src/ui/streamlit_app.py
```

4. Quick test (script expects FastAPI running):

```bash
conda run -n fraud_mlops python src/api/test_predict.py
```

Notes:
- Streamlit posts engineered features to the FastAPI `/predict` endpoint. For production use, secure and containerize both services.
