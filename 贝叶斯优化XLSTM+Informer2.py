import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
import random
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

plt.rc('font',family='Arial')
plt.style.use("ggplot")
from models import Informer
from utils.timefeatures import time_features
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def tslib_data_loader(window, length_size, batch_size, data, data_mark, shuffle=True):
    """
    数据加载器函数，用于加载和预处理时间序列数据，以用于训练模型。
    """
    seq_len = window
    sequence_length = seq_len + length_size
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]
    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark

def _make_dec_inp(labels, pred_len):
    """构造 decoder 输入：已知部分保留，预测区间置 0。"""
    label_len = labels.shape[1] - pred_len
    # keep known part, zero-pad future part
    return torch.cat([labels[:, :label_len, :], torch.zeros_like(labels[:, -pred_len:, :])], dim=1)

def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device,
                print_train=False, target_idx=-1):
    """训练（无验证）。"""
    train_loss = []
    print_frequency = max(1, int(num_epochs / 20))

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        net.train()

        for datapoints, labels, datapoints_mark, labels_mark in train_loader:
            datapoints = datapoints.to(device)
            labels = labels.to(device)
            datapoints_mark = datapoints_mark.to(device)
            labels_mark = labels_mark.to(device)

            optimizer.zero_grad()

            dec_inp = _make_dec_inp(labels, length_size)
            preds = net(datapoints, datapoints_mark, dec_inp, labels_mark, None)

            # preds: [B, pred_len, 1] -> [B, pred_len]
            if preds.dim() == 3 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)

            y_true = labels[:, -length_size:, target_idx]
            if y_true.dim() == 3 and y_true.size(-1) == 1:
                y_true = y_true.squeeze(-1)

            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_loss.append(avg_train_loss)

        if print_train and ((epoch + 1) % print_frequency == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}")

    return net, train_loss, epoch + 1

def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler,
                    num_epochs, device, early_patience=0.15, print_train=False, target_idx=-1):
    """训练（含验证集与早停）。"""
    train_loss, val_loss = [], []
    print_frequency = max(1, int(num_epochs / 20))

    early_patience_epochs = max(1, int(early_patience * num_epochs))
    best_val = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # ---- train ----
        net.train()
        total_train_loss = 0.0
        for datapoints, labels, datapoints_mark, labels_mark in train_loader:
            datapoints = datapoints.to(device)
            labels = labels.to(device)
            datapoints_mark = datapoints_mark.to(device)
            labels_mark = labels_mark.to(device)

            optimizer.zero_grad()

            dec_inp = _make_dec_inp(labels, length_size)
            preds = net(datapoints, datapoints_mark, dec_inp, labels_mark, None)

            if preds.dim() == 3 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)

            y_true = labels[:, -length_size:, target_idx]
            if y_true.dim() == 3 and y_true.size(-1) == 1:
                y_true = y_true.squeeze(-1)

            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())

        avg_train = total_train_loss / max(1, len(train_loader))
        train_loss.append(avg_train)

        # ---- val ----
        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                val_x_mark = val_x_mark.to(device)
                val_y_mark = val_y_mark.to(device)

                dec_inp_val = _make_dec_inp(val_y, length_size)
                pred_val = net(val_x, val_x_mark, dec_inp_val, val_y_mark, None)

                if pred_val.dim() == 3 and pred_val.size(-1) == 1:
                    pred_val = pred_val.squeeze(-1)

                y_true_val = val_y[:, -length_size:, target_idx]
                if y_true_val.dim() == 3 and y_true_val.size(-1) == 1:
                    y_true_val = y_true_val.squeeze(-1)

                loss_val = criterion(pred_val, y_true_val)
                total_val_loss += float(loss_val.item())

        avg_val = total_val_loss / max(1, len(val_loader))
        val_loss.append(avg_val)

        if scheduler is not None:
            scheduler.step(avg_val)

        if print_train and ((epoch + 1) % print_frequency == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(f"Epoch: {epoch + 1}, Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}")

        # early stopping
        if avg_val < best_val - 1e-12:
            best_val = avg_val
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                if print_train:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {best_val:.6f}")
                break

    net.train()
    return net, train_loss, val_loss, epoch + 1

def cal_eval(y_real, y_pred):
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100

    df_eval = pd.DataFrame({'R2': r2,
                           'MSE': mse, 'RMSE': rmse,
                           'MAE': mae, 'MAPE': mape},
                          index=['Eval'])
    return df_eval
start_time = time.time()

# 加载数据
df = pd.read_csv('D:/water predict/新xlstm+informer/XLSTM+informer/inf/data/1TNS8.csv')

DATE_COL = 'date'
TARGET_COL = 'Target'   # 改成你的目标列名，例如 'DO'

# 特征列（默认：除 DATE_COL 外的全部列；包含 TARGET_COL 本身）
feature_cols = [c for c in df.columns if c != DATE_COL]
data = df[feature_cols]
data_target = df[TARGET_COL]

data_dim = data.shape[1]
target_idx = list(data.columns).index(TARGET_COL)

# 处理时间戳
df_stamp = df[[DATE_COL]]
df_stamp = df_stamp.copy()
df_stamp[DATE_COL] = pd.to_datetime(df_stamp[DATE_COL])
df_stamp_for_tf = df_stamp.copy()
if DATE_COL != 'date':
    df_stamp_for_tf = df_stamp_for_tf.rename(columns={DATE_COL: 'date'})
data_stamp = time_features(df_stamp_for_tf, timeenc=1, freq='h')

# 数据划分比例
dev_ratio = 0.8                 # development = 80%
val_ratio_in_dev = 0.125          # validation is 12.5% of development
data_length = len(data)

dev_end = int(dev_ratio * data_length)                 # 80%
train_end = int((1 - val_ratio_in_dev) * dev_end)

# 先划分再归一化（scaler 只在 train 上拟合）
scaler_x = MinMaxScaler()
data_raw = np.array(data)

scaler_x.fit(data_raw[:train_end, :])
data_scaled = scaler_x.transform(data_raw)

data_train = data_scaled[:train_end, :]
data_val   = data_scaled[train_end:dev_end, :]
data_test  = data_scaled[dev_end:, :]

data_train_mark = data_stamp[:train_end, :]
data_val_mark   = data_stamp[train_end:dev_end, :]
data_test_mark  = data_stamp[dev_end:, :]

# 目标变量的 scaler（用于反归一化），同样只在 train 上拟合
y_scaler = MinMaxScaler()
y_scaler.fit(np.array(data_target[:train_end]).reshape(-1, 1))

# =======================
# 序列/训练基础参数
# seq_len=window, pred_len=length_size, label_len=window/2
window = 10        # 输入历史长度（seq_len）
length_size = 12   # 预测步长（pred_len）
batch_size = 64
# =======================

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(
    window, length_size, batch_size, data_train, data_train_mark, shuffle=True
)
val_loader, x_val, y_val, x_val_mark, y_val_mark = tslib_data_loader(
    window, length_size, batch_size, data_val, data_val_mark, shuffle=False
)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(
    window, length_size, batch_size, data_test, data_test_mark, shuffle=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 50
learning_rate = 0.0010974
scheduler_patience = int(0.25 * num_epochs)
early_patience = 0.2

# 定义贝叶斯优化的搜索空间
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'early_patience': hp.uniform('early_patience', 0.1, 0.3),
    'd_model': hp.choice('d_model', [64, 128, 256, 512]),
    'n_heads': hp.choice('n_heads', [4, 8, 16]),
    'd_ff': hp.choice('d_ff', [64, 128, 256, 512]),
    'dropout': hp.uniform('dropout', 0.0, 0.3),
    'e_layers': hp.choice('e_layers', [1, 2, 3]),
    'd_layers': hp.choice('d_layers', [1, 2, 3])
}

# 定义目标函数
def objective(params):
    class Config:
        def __init__(self, params):
            self.seq_len = window
            self.label_len = int(window / 2)
            self.pred_len = length_size
            self.freq = 'h'
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.learning_rate = params['learning_rate']
            self.stop_ratio = params['early_patience']
            self.dec_in = data_dim
            self.enc_in = data_dim
            self.c_out = 1
            self.d_model = params['d_model']
            self.n_heads = params['n_heads']
            self.dropout = params['dropout']
            self.e_layers = params['e_layers']
            self.d_layers = params['d_layers']
            self.d_ff = params['d_ff']
            self.factor = 5
            self.activation = 'gelu'
            self.channel_independence = 0
            self.top_k = 5
            self.num_kernels = 6
            self.distil = 1
            self.embed = 'timeF'
            self.output_attention = 0
            self.task_name = 'short_term_forecast'
    
    config = Config(params)
    net = Informer.Model(config).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(0.25 * num_epochs))
    
    _, train_loss, val_loss, final_epoch = model_train_val(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,  
        length_size=length_size,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        early_patience=config.stop_ratio,
        print_train=False,
        target_idx=target_idx
    )
    
    best_val_loss = min(val_loss)
    return {
        'loss': best_val_loss,
        'status': STATUS_OK,
        'params': params,
        'epochs': final_epoch
    }

# 运行贝叶斯优化
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=1,
    trials=trials,
)

print("Best parameters found: ", best)

# 使用最佳参数重新训练模型
best_params = {
    'learning_rate': best['learning_rate'],
    'early_patience': best['early_patience'],
    'd_model': [64, 128, 256, 512][best['d_model']],  # 转换为实际值
    'n_heads': [4, 8, 16][best['n_heads']],
    'd_ff': [64, 128, 256, 512][best['d_ff']],
    'dropout': best['dropout'],
    'e_layers': [1, 2, 3][best['e_layers']],
    'd_layers': [1, 2, 3][best['d_layers']]
}

class BestConfig:
    def __init__(self, best_params):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = length_size
        self.freq = 'h'
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = best_params['learning_rate']
        self.stop_ratio = best_params['early_patience']
        self.dec_in = data_dim
        self.enc_in = data_dim
        self.c_out = 1
        self.d_model = best_params['d_model']
        self.n_heads = best_params['n_heads']
        self.dropout = best_params['dropout']
        self.e_layers = best_params['e_layers']
        self.d_layers = best_params['d_layers']
        self.d_ff = best_params['d_ff']
        self.factor = 5
        self.activation = 'gelu'
        self.channel_independence = 0
        self.top_k = 5
        self.num_kernels = 6
        self.distil = 1
        self.embed = 'timeF'
        self.output_attention = 0
        self.task_name = 'short_term_forecast'

best_config = BestConfig(best_params)
best_net = Informer.Model(best_config).to(device)

# 训练最终模型（scheduler 必须绑定同一个 optimizer）
final_optimizer = optim.Adam(best_net.parameters(), lr=best_config.learning_rate)
final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=0.1, patience=int(0.25 * num_epochs))

trained_model, train_loss, val_loss, final_epoch = model_train_val(
    net=best_net,
    train_loader=train_loader,
    val_loader=val_loader,
    length_size=length_size,
    optimizer=final_optimizer,
    criterion=nn.MSELoss().to(device),
    scheduler=final_scheduler,
    num_epochs=num_epochs,
    device=device,
    early_patience=best_config.stop_ratio,
    print_train=True,
    target_idx=target_idx
)

# 在测试集上评估
trained_model.eval()
with torch.no_grad():
    dec_inp_test = _make_dec_inp(y_test.to(device), length_size)

    pred = trained_model(
        x_test.to(device),
        x_test_mark.to(device),
        dec_inp_test,
        y_test_mark.to(device),
        None
    )

# pred: [B, pred_len, 1] -> [B, pred_len]
if pred.dim() == 3 and pred.size(-1) == 1:
    pred = pred.squeeze(-1)

# true: [B, pred_len] (ground truth for forecast horizon)
true = y_test[:, -length_size:, target_idx]

# 1) 统一成 [B, pred_len]
if pred.dim() == 3:
    pred = pred[:, :, 0]
if true.dim() == 3:
    true = true[:, :, 0]

# 2) 只取最后一步 -> [B]
pred_last = pred[:, -1]
true_last = true[:, -1]

# 3) 反归一化（目标变量） -> [B,1]
true_np = true_last.detach().cpu().numpy().reshape(-1, 1)
pred_np = pred_last.detach().cpu().numpy().reshape(-1, 1)

true_uninverse = y_scaler.inverse_transform(true_np)
pred_uninverse = y_scaler.inverse_transform(pred_np)

# 可视化结果
df_pred_true = pd.DataFrame({'Predict': pred_uninverse.flatten(), 'Real': true_uninverse.flatten()})
df_pred_true.plot(figsize=(12, 4))
plt.title('Result')
plt.show()

# 保存
result_df = pd.DataFrame({"真实值": true_uninverse.ravel(), "预测值": pred_uninverse.ravel()})
result_df.to_csv("results_last_step.csv", index=False, encoding="utf-8")

# 评估
df_eval = cal_eval(true_uninverse, pred_uninverse)
print(df_eval)

# 计算并打印总运行时间
end_time = time.time()
total_time = end_time - start_time
print(f"\n代码总运行时间:{total_time:.2f}秒")
