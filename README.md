import pandas as pd

import numpy as np

train_df = pd.read_csv("titanic.csv")

test_df = pd.read_csv("test.csv")

train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

train_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

test_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

bins = [0, 12, 20, 40, 60, 100]

labels = ['Child', 'Teen', 'Adult', 'Senior', 'Elder']

train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=bins, labels=labels)

test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=bins, labels=labels)

combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

correlation = combined_df.corr(numeric_only=True)

print("Correlation Matrix:")

print(correlation)

train_df.to_csv("cleaned_train.csv", index=False)

test_df.to_csv("cleaned_test.csv", index=False)

