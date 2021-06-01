import numpy as np
import torch
from keel import keel_data
from sklearn.model_selection import train_test_split

from fuzzynet.anfis import SkAnfis, MyBaggingClassifier

datasets = ["movement_libras", "wine"]

init_method = ['fcm clustering', 'essc clustering', 'no clustering']


def test_BDANFIS(dataset_name, prob):
    print(dataset_name, prob)

    x, y, classes, labels = keel_data(dataset_name)
    x, xt, y, yt = train_test_split(x, y, test_size=0.3, shuffle=True)
    x_train_tensor, y_train_tensor = torch.tensor(x, dtype=torch.float32), torch.tensor(y,
                                                                                        dtype=torch.float32)
    xt, yt = torch.tensor(xt, dtype=torch.float32), torch.tensor(yt, dtype=torch.float32)

    max_samples = x.shape[0]
    y_train_tensor = y_train_tensor.squeeze().long()
    # Dropout-ANFIS
    D_ANFIS = SkAnfis(num_mfs=2, num_out=classes, mf_type='gaussmf', dropout=True, proba=prob,
                      labels=labels, init_method=init_method[2])
    # BD-ANFIS
    model = MyBaggingClassifier(base_estimator=D_ANFIS, n_estimators=10, max_samples=max_samples,
                                max_features=5, n_jobs=5)

    model.fit(x_train_tensor, y_train_tensor)

    percentiles = np.array([50, 70, 80, 90, 95, 97.5])

    model.drop_rule(xt, 0)

    return np.array(model.evaluate(xt, yt))


if __name__ == '__main__':
    test_BDANFIS('wine', 0.5)
