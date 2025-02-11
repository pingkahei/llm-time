import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator, DeepSpeedPlugin
from torch.utils.data import DataLoader
import os
# 其他必要的导入

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
import time
import random


# 命令行参数解析器
parser = argparse.ArgumentParser(description='Time-LLM')
parser.add_argument('--task_name', type=str, required=True, help='Task name')
parser.add_argument('--is_training', type=str, required=True, help='Whether it is training phase')
parser.add_argument('--model_id', type=str, required=True, help='Model ID')
parser.add_argument('--model', type=str, required=True, help='Model type')
parser.add_argument('--data', type=str, required=True, help='Data type')
parser.add_argument('--root_path', type=str, required=True, help='Root path of the dataset')
parser.add_argument('--data_path', type=str, required=True, help='File name of the dataset')
parser.add_argument('--embed', type=str, required=True, help='Embedding type')
parser.add_argument('--freq', type=str, required=True, help='Frequency for time encoding')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size')

# 解析参数
args = parser.parse_args()

# 禁用 DeepSpeed 的分布式环境检测
os.environ["DEEPSPEED_DISABLE_DIST_INIT"] = "1"

deepspeed_plugin = None  # 禁用 DeepSpeed


# 配置类
class Configs:
    def __init__(self):
        self.task_name = "long_term_forecast"
        self.seq_len = 60
        self.pred_len = 10
        self.moving_avg = 25
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.learning_rate = 0.001
        self.train_epochs = 10
        self.batch_size = 32
        self.eval_batch_size = 32
        self.patience = 3
        self.num_workers = 0
        self.data_path = "financial_data.csv"
        self.root_path = "C:\\Users\\XIA PING\\Desktop\\Time-LLM\\Time-LLM\\dataset"

# 数据集类
class FinancialDataset(Dataset):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self.data = pd.read_csv(file_path)[['Open', 'High', 'Close', 'Adj Close']].dropna()
        if self.data.empty:
            raise ValueError("The dataset is empty after applying dropna.")

        self.features = self.data[['Open', 'High', 'Close']].astype('float32').to_numpy()
        self.target = self.data['Adj Close'].astype('float32').to_numpy()

        # 验证数据是否有效
        if np.any(np.isnan(self.features)) or np.any(np.isnan(self.target)):
            raise ValueError("Dataset contains NaN or invalid values.")

        print(f"Dataset initialized with {len(self.features)} samples.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        print(f"Index: {idx}, x: {x}, y: {y}")
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 模型类
class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 训练函数
def train_model(configs, model, train_loader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

    for epoch in range(configs.train_epochs):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{configs.train_epochs}"):
            optimizer.zero_grad()  # 清零梯度
            output = model(x)  # 前向传播
            loss = criterion(output.squeeze(), y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

        # 保存模型
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch + 1}.pt")

def evaluate_model(model, loader, device):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x).squeeze()
            predictions.extend(output.cpu().numpy())  # 模型预测值
            true_values.extend(y.cpu().numpy())  # 真实值
    return predictions, true_values

# 主函数
def main():
    # # 初始化配置
    # configs = Configs()
    #
    # # 加载数据集
    # dataset = FinancialDataset(configs.data_path)
    # print(f"Dataset initialized with {len(dataset)} samples.")  # 检查数据集样本数
    #
    # # 定义 DataLoader 并添加自定义 collate_fn
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=min(len(dataset), configs.batch_size),
    #     num_workers=0,
    #     shuffle=True,
    #     collate_fn=custom_collate_fn  # 使用自定义 collate_fn
    # )
    #
    # # 检查 DataLoader 数据
    # print("Checking train_loader batches...")
    # for i, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
    #     # Training logic here
    #     pass
    #     print(f"Batch {i}: Features shape {batch_x.shape}, Target shape {batch_y.shape}")
    #     break  # 只检查一个 batch
    #
    # # 初始化模型
    # model = SimpleModel(input_dim=3, output_dim=1)
    #
    # # 训练模型
    # train_model(configs, model, train_loader)
    # 实例化配置

    parser = argparse.ArgumentParser(description='Time-LLM')

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=str, required=True, default='1',
                        help='status (True/False)')
    parser.add_argument('--model_id', type=str, required=True, default='test',
                        help='model id')
    parser.add_argument('--model_comment', type=str, required=False, default='none',
                        help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')  # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='4096',
                        help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)

    args = parser.parse_args()
    configs = Configs()

    # 从命令行参数更新配置
    configs.data_path = args.data_path if args.data_path else configs.data_path
    configs.root_path = args.root_path if args.root_path else configs.root_path

    # 数据集路径
    dataset_path = os.path.join(configs.root_path, configs.data_path)
    print(f"Dataset path: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # 加载数据集
    dataset = FinancialDataset(dataset_path)
    print(f"Dataset loaded successfully with {len(dataset)} samples.")
    train_loader = DataLoader(dataset, batch_size=min(len(dataset), configs.batch_size), num_workers=0, shuffle=True)

    # 数据集拆分：训练集、验证集、测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), configs.batch_size), num_workers=0,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(len(val_dataset), configs.batch_size), num_workers=0,
                            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=min(len(test_dataset), configs.batch_size), num_workers=0,
                             shuffle=False)

    # 调试打印每个批次的数据形状
    print("Checking train_loader batches...")
    for i, (batch_x, batch_y) in enumerate(train_loader):
        print(f"Batch {i + 1}: x shape {batch_x.shape}, y shape {batch_y.shape}")
        if i >= 2:  # 只检查前两个批次，避免输出过多内容
            break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(input_dim=3, output_dim=1).to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

    # 初始化损失函数
    loss_fn = torch.nn.MSELoss()

    # 训练循环
    for epoch in range(configs.train_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in tqdm(train_loader, total=len(train_loader),
                                     desc=f"Epoch {epoch + 1}/{configs.train_epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            output = model(batch_x)

            # 计算损失
            loss = loss_fn(output.squeeze(), batch_y)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累计损失
            epoch_loss += loss.item()

        # 打印当前 epoch 的损失
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        # 验证模型
        predictions, true_values = evaluate_model(model, val_loader, device)
        val_mse = mean_squared_error(true_values, predictions)
        print(f"Epoch {epoch + 1}, Validation MSE: {val_mse:.4f}")

    # 保存验证结果
    np.save("predictions.npy", predictions)
    np.save("true_values.npy", true_values)
    print("Predictions and true values saved for visualization.")

    # 训练完成后，检查并删除 checkpoints
    path = './checkpoints'
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)  # 删除目录
        print(f"Directory {path} deleted successfully.")
    else:
        print(f"Directory {path} does not exist. No deletion necessary.")

    # 添加验证和可视化
    # 使用 test_loader 验证模型性能
    predictions, true_values = evaluate_model(model, test_loader, device)

    # 可视化
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 绘制预测值 vs 真实值
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(true_values)), y=true_values, label='True Values', color='blue')
    sns.lineplot(x=range(len(predictions)), y=predictions, label='Predictions', color='orange')
    plt.title('Predictions vs True Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 测试模型性能
    test_predictions, test_true_values = evaluate_model(model, test_loader, device)
    test_mse = mean_squared_error(test_true_values, test_predictions)
    print(f"Test MSE: {test_mse:.4f}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # 保留内存分配优化

    from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

    # Convert is_training to boolean with error handling
    if args.is_training.lower() not in ['true', '1', 'yes', 'y', 'false', '0', 'no', 'n']:
        raise ValueError(
            "Invalid value for is_training. Expected one of ['true', '1', 'yes', 'y', 'false', '0', 'no', 'n'].")
    args.is_training = args.is_training.lower() in ['true', '1', 'yes', 'y']

    # Initialize Accelerator (if necessary)
    accelerator = Accelerator()  # Simplified, without DeepSpeed or distributed-related plugins

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

        # 数据加载器
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        # 模型初始化
        model_dict = {
            'Autoformer': Autoformer.Model,
            'DLinear': DLinear.Model,
            'TimeLLM': TimeLLM.Model,
        }
        if args.model in model_dict:
            model = model_dict[args.model](args).float()
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        # 检查点路径设置
        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
        criterion = nn.MSELoss()

        # 学习率调度器
        train_steps = len(train_loader)
        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                    accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = './checkpoints'  # unique checkpoint saving path
        del_files(path)  # delete checkpoint files
        accelerator.print('success delete checkpoints')


if __name__ == "__main__":
    main()
