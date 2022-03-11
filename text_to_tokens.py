import pandas as pd

text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]})
print(categorical_df)

print(pd.get_dummies(categorical_df["Name"]))

import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
print("Input Ids: {}".format(input_ids.shape))
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print("One Hot encodings: {}".format(one_hot_encodings.shape))

tokenized_text = text.split()
print(tokenized_text)