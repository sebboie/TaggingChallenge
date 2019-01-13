"""Generate labels and images for train/test."""
import pandas as pd
import numpy as np
from global_vars import *


photo_data = pd.read_json("metadata/yelp_academic_dataset_photo.json",
                          lines=True)
photo_data = photo_data[photo_data["label"] == "food"]

business_data = pd.read_json("metadata/yelp_academic_dataset_business.json",
                             lines=True)
df = pd.merge(photo_data, business_data, how="inner", on="business_id")

df.drop(df[df["categories"].isna()].index, inplace=True, axis=0)

for i in range(len(FOOD_CATEGORIES)):
    df[FOOD_CATEGORIES[i]] = 0
    df.loc[df["categories"].str.contains(FOOD_CATEGORIES[i]),
           FOOD_CATEGORIES[i]] = 1

# The number of samples per category are quite unbalanced. To remedy this, and
# ease the computational burden we take the category with the least amount of
# samples and pick the same amount from all categories.
samples_per_cat = df.loc[:, FOOD_CATEGORIES].sum(axis=0)
samples_per_cat = np.min(samples_per_cat)

columns = ["photo_id"]
columns.extend(FOOD_CATEGORIES)
train_data = pd.DataFrame([], columns=columns)
test_data = pd.DataFrame([], columns=columns)
for cat in FOOD_CATEGORIES:
    # Choose some images that have the corresponding tag
    selected_data = df[df[cat] == 1].sample(
        n=samples_per_cat, random_state=RANDOM_SEED).loc[:, columns]

    train, test = np.split(selected_data, [int(0.8*len(selected_data))])
    train_data = train_data.append(train, ignore_index=True)
    test_data = test_data.append(test, ignore_index=True)

    # Choose some images that DO NOT have the corresponding tag
    selected_data = df[df[cat] == 0].sample(
        n=samples_per_cat, random_state=RANDOM_SEED).loc[:, columns]

    train, test = np.split(selected_data, [int(0.8*len(selected_data))])
    train_data = train_data.append(train, ignore_index=True)
    test_data = test_data.append(test, ignore_index=True)

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)
