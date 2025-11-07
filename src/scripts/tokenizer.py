from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-15b", trust_remote_code=True)
