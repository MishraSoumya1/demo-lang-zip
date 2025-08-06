from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sentence-transformers/paraphrase-albert-small-v2",
    local_dir="./app/local_models/paraphrase-albert-small-v2",
    allow_patterns=[
        "config.json",
        "pytorch_model.bin",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "modules.json",
        "0_Pooling/config.json"  # 👈 THIS is required to avoid your current error
    ]
)
