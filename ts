from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_text = "Hello, how are you?"
tokens = tokenizer.encode(input_text, add_special_tokens=True)
import torch

input_ids = torch.tensor([tokens])
outputs = model(input_ids)
embeddings = outputs.last_hidden_state
