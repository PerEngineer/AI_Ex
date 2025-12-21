"""
实验五：循环神经网络LSTM的实现及应用
- 使用LSTM和GRU实现天气温度预测
- 数据集：Jena Climate Dataset
- 对比分析LSTM和GRU模型性能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import urllib.request
import zipfile
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ======================= 数据下载与预处理 =======================
def download_jena_climate_data(data_dir='./data'):
    """下载Jena Climate数据集"""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
    
    if not os.path.exists(csv_path):
        print("正在下载Jena Climate数据集...")
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        zip_path = os.path.join(data_dir, 'jena_climate.zip')
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)
            print("数据下载完成！")
        except Exception as e:
            print(f"下载失败: {e}")
            print("正在生成模拟天气数据...")
            return generate_synthetic_weather_data(data_dir)
    
    return csv_path


def generate_synthetic_weather_data(data_dir='./data'):
    """生成模拟天气数据（备用方案）"""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'weather_data.csv')
    
    # 生成2年的模拟数据（每10分钟一条记录）
    n_samples = 365 * 24 * 6 * 2  # 2年
    
    # 时间序列
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='10T')
    
    # 模拟温度：基础温度 + 年周期 + 日周期 + 噪声
    t = np.arange(n_samples)
    yearly_cycle = 10 * np.sin(2 * np.pi * t / (365 * 24 * 6))  # 年周期
    daily_cycle = 5 * np.sin(2 * np.pi * t / (24 * 6))  # 日周期
    noise = np.random.normal(0, 2, n_samples)  # 随机噪声
    temperature = 15 + yearly_cycle + daily_cycle + noise
    
    # 模拟其他气象特征
    humidity = 60 + 20 * np.sin(2 * np.pi * t / (24 * 6) + np.pi) + np.random.normal(0, 5, n_samples)
    pressure = 1013 + 10 * np.sin(2 * np.pi * t / (7 * 24 * 6)) + np.random.normal(0, 3, n_samples)
    wind_speed = np.abs(5 + 3 * np.sin(2 * np.pi * t / (24 * 6)) + np.random.normal(0, 2, n_samples))
    
    df = pd.DataFrame({
        'Date Time': dates,
        'T (degC)': temperature,
        'rh (%)': np.clip(humidity, 0, 100),
        'p (mbar)': pressure,
        'wv (m/s)': wind_speed
    })
    
    df.to_csv(csv_path, index=False)
    print(f"模拟数据已生成: {csv_path}")
    return csv_path


def load_and_preprocess_data(csv_path, target_col='T (degC)', sample_rate=6):
    """加载并预处理数据"""
    df = pd.read_csv(csv_path)
    
    print(f"\n数据集形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    print(f"\n数据预览:")
    print(df.head())
    
    # 每小时采样一次（原始数据每10分钟一条）
    df = df[::sample_rate].reset_index(drop=True)
    print(f"\n采样后数据形状: {df.shape}")
    
    # 选择特征列（数值列）
    feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    
    # 确保目标列在特征中
    if target_col not in feature_cols:
        # 尝试查找温度列
        temp_cols = [col for col in feature_cols if 'T' in col or 'temp' in col.lower()]
        if temp_cols:
            target_col = temp_cols[0]
        else:
            target_col = feature_cols[0]
    
    print(f"\n使用特征: {feature_cols}")
    print(f"预测目标: {target_col}")
    
    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)
    
    return data, target_idx, feature_cols


# ======================= 数据集类 =======================
class WeatherDataset(Dataset):
    """天气时间序列数据集"""
    def __init__(self, data, seq_length, pred_length, target_idx):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.target_idx = target_idx
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, self.target_idx]
        return x, y


# ======================= 模型定义 =======================
class LSTMModel(nn.Module):
    """LSTM天气预测模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 使用最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out


class GRUModel(nn.Module):
    """GRU天气预测模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # GRU层
        gru_out, h_n = self.gru(x)
        # 使用最后一个时间步的输出
        out = self.fc(gru_out[:, -1, :])
        return out


# ======================= 训练与评估 =======================
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, model_name, save_dir='./models'):
    """训练模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name}")
    print(f"{'='*60}")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pth'))
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.6f} | '
                  f'Val Loss: {val_loss:.6f}')
    
    print(f"\n{model_name} 训练完成！最佳验证Loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, scaler, target_idx, model_name):
    """评估模型"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # 反归一化
    # 创建临时数组用于反归一化
    pred_full = np.zeros((predictions.shape[0], scaler.n_features_in_))
    actual_full = np.zeros((actuals.shape[0], scaler.n_features_in_))
    
    pred_full[:, target_idx] = predictions[:, 0] if predictions.ndim > 1 else predictions
    actual_full[:, target_idx] = actuals[:, 0] if actuals.ndim > 1 else actuals
    
    pred_inv = scaler.inverse_transform(pred_full)[:, target_idx]
    actual_inv = scaler.inverse_transform(actual_full)[:, target_idx]
    
    # 计算评估指标
    mse = mean_squared_error(actual_inv, pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_inv, pred_inv)
    r2 = r2_score(actual_inv, pred_inv)
    
    print(f"\n{model_name} 测试集评估结果:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return pred_inv, actual_inv, metrics


# ======================= 可视化 =======================
def plot_training_curves(lstm_losses, gru_losses, save_path='./results'):
    """绘制训练曲线"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LSTM训练曲线
    axes[0].plot(lstm_losses['train'], label='Train Loss', color='blue')
    axes[0].plot(lstm_losses['val'], label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('LSTM Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # GRU训练曲线
    axes[1].plot(gru_losses['train'], label='Train Loss', color='blue')
    axes[1].plot(gru_losses['val'], label='Val Loss', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('GRU Training Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存到 {save_path}/training_curves.png")


def plot_predictions(lstm_results, gru_results, save_path='./results'):
    """绘制预测结果对比"""
    os.makedirs(save_path, exist_ok=True)
    
    lstm_pred, lstm_actual = lstm_results
    gru_pred, gru_actual = gru_results
    
    # 选择展示的样本数量
    n_show = min(500, len(lstm_pred))
    x = np.arange(n_show)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # LSTM预测结果
    axes[0].plot(x, lstm_actual[:n_show], label='Actual', color='blue', alpha=0.7)
    axes[0].plot(x, lstm_pred[:n_show], label='LSTM Prediction', color='red', alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('LSTM Weather Temperature Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # GRU预测结果
    axes[1].plot(x, gru_actual[:n_show], label='Actual', color='blue', alpha=0.7)
    axes[1].plot(x, gru_pred[:n_show], label='GRU Prediction', color='green', alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('GRU Weather Temperature Prediction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"预测结果已保存到 {save_path}/predictions.png")


def plot_comparison(lstm_metrics, gru_metrics, lstm_params, gru_params, save_path='./results'):
    """绘制模型对比图"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 性能指标对比
    metrics_names = ['RMSE', 'MAE']
    lstm_values = [lstm_metrics['RMSE'], lstm_metrics['MAE']]
    gru_values = [gru_metrics['RMSE'], gru_metrics['MAE']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, lstm_values, width, label='LSTM', color='steelblue')
    bars2 = axes[0].bar(x + width/2, gru_values, width, label='GRU', color='coral')
    
    axes[0].set_ylabel('Value')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值
    for bar, val in zip(bars1, lstm_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, gru_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 参数量对比
    params = ['LSTM', 'GRU']
    param_values = [lstm_params, gru_params]
    colors = ['steelblue', 'coral']
    
    bars = axes[1].bar(params, param_values, color=colors)
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Model Parameters Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, param_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"对比结果已保存到 {save_path}/comparison.png")


def plot_scatter(lstm_results, gru_results, save_path='./results'):
    """绘制预测值与真实值散点图"""
    os.makedirs(save_path, exist_ok=True)
    
    lstm_pred, lstm_actual = lstm_results
    gru_pred, gru_actual = gru_results
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # LSTM散点图
    axes[0].scatter(lstm_actual, lstm_pred, alpha=0.3, s=10, color='steelblue')
    min_val = min(lstm_actual.min(), lstm_pred.min())
    max_val = max(lstm_actual.max(), lstm_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('Actual Temperature (°C)')
    axes[0].set_ylabel('Predicted Temperature (°C)')
    axes[0].set_title('LSTM: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # GRU散点图
    axes[1].scatter(gru_actual, gru_pred, alpha=0.3, s=10, color='coral')
    min_val = min(gru_actual.min(), gru_pred.min())
    max_val = max(gru_actual.max(), gru_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('Actual Temperature (°C)')
    axes[1].set_ylabel('Predicted Temperature (°C)')
    axes[1].set_title('GRU: Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'scatter_plots.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"散点图已保存到 {save_path}/scatter_plots.png")


# ======================= 主程序 =======================
def main():
    # ============ 超参数配置 ============
    print("="*60)
    print("实验五：循环神经网络LSTM的实现及应用 - 天气预测")
    print("="*60)
    
    # 模型超参数
    HIDDEN_SIZE = 64        # 隐藏层大小
    NUM_LAYERS = 2          # RNN层数
    DROPOUT = 0.2           # Dropout比率
    
    # 训练超参数
    SEQ_LENGTH = 24         # 输入序列长度（24小时）
    PRED_LENGTH = 1         # 预测长度（1小时后的温度）
    BATCH_SIZE = 64         # 批次大小
    LEARNING_RATE = 0.001   # 学习率
    NUM_EPOCHS = 50         # 训练轮数
    
    # 数据划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    print("\n超参数配置:")
    print(f"  隐藏层大小: {HIDDEN_SIZE}")
    print(f"  RNN层数: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  输入序列长度: {SEQ_LENGTH} (小时)")
    print(f"  预测长度: {PRED_LENGTH} (小时)")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    
    # ============ 数据加载与预处理 ============
    print("\n" + "="*60)
    print("数据加载与预处理")
    print("="*60)
    
    # 下载/加载数据
    csv_path = download_jena_climate_data()
    data, target_idx, feature_cols = load_and_preprocess_data(csv_path)
    
    # 数据归一化
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 数据划分
    n = len(data_scaled)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_data = data_scaled[:train_end]
    val_data = data_scaled[train_end:val_end]
    test_data = data_scaled[val_end:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")
    
    # 创建数据集
    train_dataset = WeatherDataset(train_data, SEQ_LENGTH, PRED_LENGTH, target_idx)
    val_dataset = WeatherDataset(val_data, SEQ_LENGTH, PRED_LENGTH, target_idx)
    test_dataset = WeatherDataset(test_data, SEQ_LENGTH, PRED_LENGTH, target_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    INPUT_SIZE = data.shape[1]  # 特征数量
    OUTPUT_SIZE = PRED_LENGTH   # 输出大小
    
    # ============ 模型创建 ============
    print("\n" + "="*60)
    print("模型创建")
    print("="*60)
    
    # LSTM模型
    lstm_model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT
    ).to(device)
    
    # GRU模型
    gru_model = GRUModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT
    ).to(device)
    
    # 统计参数量
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    gru_params = sum(p.numel() for p in gru_model.parameters())
    
    print(f"\nLSTM模型结构:")
    print(lstm_model)
    print(f"\nGRU模型结构:")
    print(gru_model)
    
    # ============ 模型训练 ============
    print("\n" + "="*60)
    print("模型训练")
    print("="*60)
    
    criterion = nn.MSELoss()
    
    # 训练LSTM
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, train_loader, val_loader, criterion, lstm_optimizer,
        NUM_EPOCHS, 'LSTM'
    )
    
    # 训练GRU
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)
    gru_train_losses, gru_val_losses = train_model(
        gru_model, train_loader, val_loader, criterion, gru_optimizer,
        NUM_EPOCHS, 'GRU'
    )
    
    # ============ 模型评估 ============
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    # 加载最佳模型
    lstm_model.load_state_dict(torch.load('./models/LSTM_best.pth'))
    gru_model.load_state_dict(torch.load('./models/GRU_best.pth'))
    
    # 评估模型
    lstm_pred, lstm_actual, lstm_metrics = evaluate_model(
        lstm_model, test_loader, scaler, target_idx, 'LSTM'
    )
    gru_pred, gru_actual, gru_metrics = evaluate_model(
        gru_model, test_loader, scaler, target_idx, 'GRU'
    )
    
    # ============ 结果可视化 ============
    print("\n" + "="*60)
    print("结果可视化")
    print("="*60)
    
    # 训练曲线
    lstm_losses = {'train': lstm_train_losses, 'val': lstm_val_losses}
    gru_losses = {'train': gru_train_losses, 'val': gru_val_losses}
    plot_training_curves(lstm_losses, gru_losses)
    
    # 预测结果
    plot_predictions((lstm_pred, lstm_actual), (gru_pred, gru_actual))
    
    # 模型对比
    plot_comparison(lstm_metrics, gru_metrics, lstm_params, gru_params)
    
    # 散点图
    plot_scatter((lstm_pred, lstm_actual), (gru_pred, gru_actual))
    
    # ============ 实验总结 ============
    print("\n" + "="*60)
    print("实验总结与分析")
    print("="*60)
    
    print("\n1. 模型参数对比:")
    print(f"   LSTM参数量: {lstm_params:,}")
    print(f"   GRU参数量:  {gru_params:,}")
    print(f"   参数减少比例: {(1 - gru_params/lstm_params)*100:.2f}%")
    
    print("\n2. 性能指标对比:")
    print(f"   {'指标':<10} {'LSTM':<15} {'GRU':<15} {'差异':<15}")
    print(f"   {'-'*50}")
    for metric in ['RMSE', 'MAE', 'R2']:
        lstm_val = lstm_metrics[metric]
        gru_val = gru_metrics[metric]
        diff = gru_val - lstm_val
        better = "GRU更优" if (diff < 0 and metric != 'R2') or (diff > 0 and metric == 'R2') else "LSTM更优"
        print(f"   {metric:<10} {lstm_val:<15.4f} {gru_val:<15.4f} {better}")
    
    print("\n3. 实验结论:")
    print("   - LSTM和GRU都能有效捕捉天气温度的时序模式")
    print(f"   - GRU相比LSTM减少了约{(1 - gru_params/lstm_params)*100:.1f}%的参数")
    if gru_metrics['RMSE'] <= lstm_metrics['RMSE']:
        print("   - GRU在本实验中性能与LSTM相当或更优，验证了其高效性")
    else:
        print("   - LSTM在本实验中性能略优，但GRU训练更快，参数更少")
    print("   - 两种模型都适合用于天气温度的短期预测任务")
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()
