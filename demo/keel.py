'''
    DropoutANFIS in torch
    @author: Fei Guo
'''

import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, Normalizer


def keel_data(dataset_name):
    print("Loading data file...")

    # with open(f'./data/{dataset_name}.dat') as f_in:
    with open(f'../data/{dataset_name}.dat') as f_in:
        # 处理头
        # 是否以 @ 开头
        # 返回属性名以及属性范围
        reg_range = '\[.*\]'
        reg_label = '\{(.*)\}'
        ranges = []
        print("Reading data header...")
        while True:
            line = f_in.readline()
            line = line.strip('\n')
            if line.startswith('@attribute'):
                if line.find('Class') == -1:
                    ranges.append(eval(re.findall(reg_range, line)[0]))
                else:
                    labels = re.findall(reg_label, line)[0].split(',')
            elif line.startswith('@inputs'):
                cols = line.split(" ")[1:]
                cols = [col.strip(',') for col in cols]
            elif line.startswith('@data'):
                break
        # 处理数据
        print("Reading data content...")
        x = []
        y = []
        for line in f_in.readlines():
            # 替换标签为 0，1,...n 值
            values = line.strip('\n').split(',')
            values = [v.strip() for v in values]
            x.append([float(v) for v in values[0:-1]])
            y.append(values[-1])
    labels = [label.strip() for label in labels]
    ranges = np.array(ranges).T
    x_encoder = MinMaxScaler().fit(x)
    # x_encoder = Normalizer().fit(x)
    # x_encoder = MinMaxScaler().fit(ranges)
    y_encoder = LabelEncoder().fit(labels)
    x = np.array(x)
    y = np.array(y)
    x = x_encoder.transform(x)
    y = y_encoder.transform(y)
    return x, y, len(labels), labels


if __name__ == '__main__':
    keel_data('wine')
