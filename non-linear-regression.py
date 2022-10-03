import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/china_gdp.csv")

x_data, y_data = (df["Year"].values, df["Value"].values)
x_data = (x_data - min(x_data)) / (max(x_data) - min(x_data))
y_data = (y_data - min(y_data)) / (max(y_data) - min(y_data))

def non_linear(x, b1, b2):
  y = 1 / (1 + np.exp(-b1*(x-b2)))
  return y

def loss(f, b1, b2, x_data, y_data):
  sum = 0
  for i in range(len(x_data)):
    sum += (f(x_data[i], b1, b2) - y_data[i])**2
  return sum

def optimize():
  learning_rate = 0.01
  b1 = 1
  b2 = 0
  f = non_linear
  l = loss(f, b1, b2, x_data, y_data)

  for _ in range(10000):
    lb1 = (loss(f, b1+0.01, b2, x_data, y_data) - l)  / 0.01
    lb2 = (loss(f, b1, b2+0.01, x_data, y_data) - l)  / 0.01
    b1 -= learning_rate * lb1
    b2 -= learning_rate * lb2
    nl = loss(f, b1, b2, x_data, y_data)
    print(nl)
    l = nl

  return (b1, b2)

(b1, b2) = optimize()

Y_pred = non_linear(x_data, b1, b2)

plt.figure(figsize=(8, 5))
plt.ylabel('GDP')
plt.xlabel('Year')
plt.plot(x_data, Y_pred)
plt.plot(x_data, y_data, 'ro')
plt.show()