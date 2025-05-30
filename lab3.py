import pandas as pd
import math
from collections import Counter

df = pd.read_csv("3.csv")
print("Dataset:\n", df)

def entropy(col):
    total = len(col)
    return sum([- (count / total) * math.log2(count / total) 
                for count in Counter(col).values()])

def info_gain(df, attr, target):
    total = len(df)
    grouped = df.groupby(attr)
    weighted = sum((len(g)/total) * entropy(g[target]) for _, g in grouped)
    return entropy(df[target]) - weighted

def id3(df, target, attrs, depth=0):
    if len(set(df[target])) == 1:
        return df[target].iloc[0]
    if not attrs:
        return Counter(df[target]).most_common(1)[0][0]
    
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    gain = info_gain(df, best, target)
    print(f"{'|  '*depth}Best Attribute: {best}, Gain: {gain:.10f}")
    
    tree = {best: {}}
    for val, group in df.groupby(best):
        sub_attrs = [a for a in attrs if a != best]
        tree[best][val] = id3(group.drop(columns=[best]), target, sub_attrs, depth+1)
    return tree

def classify(tree, sample):
    while isinstance(tree, dict):
        attr = next(iter(tree))
        tree = tree[attr].get(sample.get(attr), "Unknown")
    return tree

target = "Play Tennis"
features = [col for col in df.columns if col != target]
tree = id3(df, target, features)

print("\nDecision Tree:\n", tree)

sample = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
print("\nSample:", sample, "=>", classify(tree, sample))