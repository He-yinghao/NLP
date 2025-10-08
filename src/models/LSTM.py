import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)

        self.fc = nn.Linear(hidden_size, output_size, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_0=None, c_0=None, mask=None):
        batch_size, seq_len, _ = x.size()
        if h_0 is None:
            h_0 = Variable(torch.zeros(batch_size, self.hidden_size)).to(device)
        if c_0 is None:
            c_0 = Variable(torch.zeros(batch_size, self.hidden_size)).to(device)
        h_t = h_0  # [batch_size,hidden_size]
        c_t = c_0  # [batch_size,hidden_size]
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size,input_size]
            combined = torch.cat((h_t, x_t), dim=1)  # [batch_size,input_size + hidden_size]
            f_t = self.sigmoid(self.W_f(combined))  # [batch_size,hidden_size]
            i_t = self.sigmoid(self.W_i(combined))  # [batch_size,hidden_size]
            c_hat_t = self.tanh(self.W_c(combined))  # [batch_size,hidden_size]
            c_t = i_t * c_hat_t + f_t * c_t  # [batch_size,hidden_size]
            o_t = self.sigmoid(self.W_o(combined))  # [batch_size,hidden_size]
            h_t = o_t * self.tanh(c_t)  # [batch_size,hidden_size]
            outputs.append(h_t)
        outputs = torch.stack(outputs, dim=1)  # [batch_size,seq_len,hidden_size]
        if mask is not None:
            outputs = outputs * mask.unsqueeze(-1)  # 使用掩码
        output = self.fc(outputs.contiguous().view(-1, self.hidden_size))  # [batch_size * seq_len,output_size]
        return output.view(batch_size, seq_len, -1)  # [batch_size,seq_len,output_size]


# 设置随机种子以确保结果可复现
torch.manual_seed(1234)
np.random.seed(1234)

# 定义一些超参数
BATCH_SIZE = 64
SEQ_LEN = 10  # 序列长度
INPUT_SIZE = 100  # 输入特征维度
HIDDEN_SIZE = 256
OUTPUT_SIZE = 1  # 二分类任务
N_EPOCHS = 5000


# 生成随机训练数据
# 生成随机训练数据
def generate_random_data(num_samples, max_seq_len, input_size):
    seq_lens = np.random.randint(1, max_seq_len + 1, size=num_samples)  # 随机序列长度
    X = np.zeros((num_samples, max_seq_len, input_size), dtype=np.float32)  # 填充后输入数据
    for i in range(num_samples):
        X[i, :seq_lens[i], :] = np.random.rand(seq_lens[i], input_size)  # 填充前的随机输入数据
    y = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)  # 随机标签（0或1）
    return torch.from_numpy(X), torch.from_numpy(y), seq_lens


# 生成训练和验证数据
train_X, train_y, train_seq_lens = generate_random_data(1000, SEQ_LEN, INPUT_SIZE)  # 1000个训练样本
valid_X, valid_y, valid_seq_lens = generate_random_data(200, SEQ_LEN, INPUT_SIZE)  # 200个验证样本

# 将数据移至GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X, train_y = train_X.to(device), train_y.to(device)
valid_X, valid_y = valid_X.to(device), valid_y.to(device)


# 创建掩码
def create_mask(seq_lens, max_len):
    mask = torch.zeros((len(seq_lens), max_len), dtype=torch.float32)
    for i, l in enumerate(seq_lens):
        mask[i, :l] = 1  # 非填充部分设为1
    return mask.to(device)


train_mask = create_mask(train_seq_lens, SEQ_LEN)
valid_mask = create_mask(valid_seq_lens, SEQ_LEN)

# 创建模型实例
model = MyLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model = model.to(device)

# 定义损失函数和优化器
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()


# 计算二分类准确率的辅助函数
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# 训练函数
def train(model, X, y, mask, optimizer, criterion):
    model.train()

    # 进行一次前向传播
    predictions = model(X, mask=mask)  # [batch_size,seq_len,output_size]
    predictions = predictions[:, -1, :]  # [batch_size,output_size]
    loss = criterion(predictions, y)  # y:[batch_size,output_size]
    acc = binary_accuracy(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


# 验证函数
def evaluate(model, X, y, mask, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X, mask=mask)[:, -1, :]
        loss = criterion(predictions, y)
        acc = binary_accuracy(predictions, y)

    return loss.item(), acc.item()


# 训练模型
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_X, train_y, train_mask, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_X, valid_y, valid_mask, criterion)

    print(f'Epoch: {epoch + 1}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
