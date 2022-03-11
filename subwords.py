# hide_output
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "Tokenizing text is a core task of NLP."

encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))

print(tokenizer.vocab_size)
print(tokenizer.model_max_length)


print(tokenizer.model_input_names)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

from datasets import load_dataset
emotions = load_dataset("emotion")

print(tokenize(emotions["train"][:2]))

#hide_input
import pandas as pd
tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
data = sorted(tokens2ids, key=lambda x : x[-1])
df = pd.DataFrame(data, columns=["Special Token", "Special Token ID"])
print(df.T)

# hide_output
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

print(emotions_encoded["train"].column_names)

# hide_output
from transformers import AutoModel, TFAutoModel
import torch

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_model = AutoModel.from_pretrained(model_ckpt).to(device)
tf_model = TFAutoModel.from_pretrained(model_ckpt)

tf_xlmr = TFAutoModel.from_pretrained("xlm-roberta-base", from_pt=True)
