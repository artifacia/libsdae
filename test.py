# some quick testing; comment if not needed
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from deepautoencoder import StackedAutoEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

# shuffle
idx = np.arange(iris_x.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
iris_x = iris_x[idx]
iris_y = iris_y[idx]
print(iris_x.shape, iris_y.shape)

# train = np.genfromtxt('data/train.csv', delimiter=',')[:, :-1]
# train = StandardScaler().fit_transform(train)
# print('loaded and scaled data')

split = 80
lr = LogisticRegression(fit_intercept=True, n_jobs=-1)
lr.fit(iris_x[:split], iris_y[:split])
p = lr.predict(iris_x[split:])
print(accuracy_score(iris_y[split:], p))

m = StackedAutoEncoder(dims=[5, 5], activations=['relu', 'relu'], epoch=[10000, 10000],
                       loss='rmse', lr=0.005, batch_size=50, print_step=2000)

iris_x = m.fit_transform(iris_x, iris_y)

lr = LogisticRegression(fit_intercept=True, n_jobs=-1)
lr.fit(iris_x[:split], iris_y[:split])
p = lr.predict(iris_x[split:])
print(accuracy_score(iris_y[split:], p))

# print(iris.data[0])
