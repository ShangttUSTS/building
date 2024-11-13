import  torch
import numpy as np
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to script.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
def preprocess_data(df):
    std = df['meter_reading'].std()
    mean = df['meter_reading'].mean()
    left = mean - 3 * std
    right = mean + 3 * std
    df['meter_reading'].loc[(df['meter_reading'] < left) | (df['meter_reading'] > right)] = None
    df['meter_reading'] = df['meter_reading'].interpolate()
    return df

# 数据集构建
def dataset_func(data_pro, sequence_length):
    data = []
    for i in range(len(data_pro) - sequence_length + 1):
        data.append(data_pro[i: i + sequence_length])
    reshaped_data = np.array(data)
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    return x, y
def dataProcess(sub_data):
    train_data = sub_data[:6000]
    test_data = sub_data[6000:]

    X_train, y_train = dataset_func(train_data['meter_reading'].values, 21)
    X_test, y_test = dataset_func(test_data['meter_reading'].values, 21)
    missing_values = np.isnan(X_train).sum()
    print(f'Missing values in X_train: {missing_values}')

    missing_values_y = np.isnan(y_train).sum()
    print(f'Missing values in y_train: {missing_values_y}')
    # 用均值填充缺失值
    X_train = np.where(np.isnan(X_train), np.nanmean(X_train, axis=0), X_train)
    y_train = np.where(np.isnan(y_train), np.nanmean(y_train), y_train)
    inf_values = np.isinf(X_train).sum()
    print(f'Infinite values in X_train: {inf_values}')

    inf_values_y = np.isinf(y_train).sum()
    print(f'Infinite values in y_train: {inf_values_y}')
    X_train[np.isinf(X_train)] = np.nanmean(X_train)  # 或替换为其他值
    y_train[np.isinf(y_train)] = np.nanmean(y_train)
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # 添加通道维度
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                # Choose appropriate reduced int type for integer columns
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Choose appropriate reduced float type for floating point columns
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df