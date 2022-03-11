from datasets import load_dataset, list_datasets
import pandas as pd
import matplotlib.pyplot as plt

# all_datasets = list_datasets()
# print(len(all_datasets))
# print(all_datasets[:10])

emotions = load_dataset("emotion")
# print(emotions)
#
# print(emotions["train"][0])

emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())

def label_int2sr(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2sr)
print(df.head())

# df["label_name"].value_counts(ascending=True).plot.barh()
# plt.title("Frequency of classes")
# plt.show()

df["Words per tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words per tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

