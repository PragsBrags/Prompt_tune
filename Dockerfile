FROM unsloth/unsloth:latest

WORKDIR /app

COPY requirements.txt /rq/requirements.txt

RUN pip install --no-cache-dir \
    -r /rq/requirements.txt

COPY . /app

ENV PYTHONUNBUFFEREF=1 \
    HF_HOME=/cache/huggingface \
    NLTK_DATA=/cache/nltk_data \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app/src
CMD ["python", "cli.py"]