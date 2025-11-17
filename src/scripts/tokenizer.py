from transformers import AutoTokenizer

# Using a pretrained tokenizer for this project (This particular tokenizer is heavily trained on coding corpus)
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-15b", trust_remote_code=True)
