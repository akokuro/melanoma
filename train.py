import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("results.csv", index_col=None).to_numpy()

neg = data[data[:, -1] == 0]
pos = data[data[:, -1] == 1]

max_len = min(neg.shape[0], pos.shape[0])

data = np.vstack([neg[:max_len, :], pos[:max_len, :]])

print(np.min(data, axis=0))
print(np.max(data, axis=0))

print(data.shape)

x = data[:, 1:-1]
y = data[:, -1]
print(x.shape, y.shape)

reg = LinearRegression().fit(x, y)
print(reg.score(x, y))

print(reg.coef_)

res = reg.predict(x)

res[res > 0.5] = 1
res[res <= 0.5] = 0

total_neg = 0
true_neg = 0
false_pos = 0


for pred, truth in zip(res, y):
    if truth == 0:
        total_neg += 1
    if pred == 0 and truth == 0:
        true_neg += 1
    if pred == 1 and truth == 0:
        false_pos += 1

print(f"меланомы нет {total_neg} всего {true_neg}({true_neg / total_neg:.2%}) распознано {false_pos}({false_pos / total_neg:.2%}) ошибочно")

total_pos = 0
true_pos = 0
false_neg = 0


for pred, truth in zip(res, y):
    if truth == 1:
        total_pos += 1
    if pred == 1 and truth == 1:
        true_pos += 1
    if pred == 0 and truth == 1:
        false_neg += 1

print(f"меланомa     {total_pos} всего {true_pos}({true_pos / total_pos:.2%}) распознано {false_neg}({false_neg / total_pos:.2%}) ошибочно")
