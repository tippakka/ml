import csv
import math
import random
import statistics

def calc_prob(x, mean, stdev):
    if stdev == 0: return 1 if x == mean else 0
    exp = math.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exp


dataset = []
with open('5.csv', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append([float(x) for x in row])

print("Size of database =", len(dataset))

random.shuffle(dataset)
split = int(0.7 * len(dataset))
train_set = dataset[:split]
test_set = dataset[split:]

print("Training size of database =", len(train_set))

classes = {}
for row in train_set:
    label = row[-1]
    classes.setdefault(label, []).append(row)


model = {}
for label, rows in classes.items():
    stats = [(statistics.mean(col), statistics.stdev(col)) for col in zip(*rows)]
    model[label] = stats[:-1]  


def predict(row):
    probs = {}
    for label, stats in model.items():
        prob = 1
        for i in range(len(stats)):
            mean, stdev = stats[i]
            prob *= calc_prob(row[i], mean, stdev)
        probs[label] = prob
    return max(probs, key=probs.get), probs

x_test = test_set[0][:-1]
print("x_test:", x_test)
y_pred, probs = predict(x_test)
print("Predicted y:", y_pred)
print("Probabilities:", list(probs.values()))

correct = sum(1 for row in test_set if predict(row[:-1])[0] == row[-1])
accuracy = (correct / len(test_set)) * 100
print("Accuracy: {:.2f}%".format(accuracy))