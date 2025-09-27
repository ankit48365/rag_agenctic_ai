
### test uv directory
`uv run python -c "import sys; print(sys.executable)"`

### check package in uv running
`uv run python -c "from sentence_transformers import SentenceTransformer; print('sentence_transformers imported successfully!')"`


### One time 
`uv add jupyter ipykernel`

### One time  // try skipping
`uv run python -m ipykernel install --user --name="rag_agenctic_ai" --display-name="RAG Agentic AI (uv)"`

### check kernels list
`uv run jupyter kernelspec list`

### start notebook
`uv run jupyter lab --port=8888`
`uv run jupyter lab --no-browser --port=8888`




