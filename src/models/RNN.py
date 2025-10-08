import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义权重矩阵和偏置
        self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))  # [input_size, hidden_size]
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # [hidden_size,hidden_size]
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))  # [hidden_size,]

        self.W_hy = nn.Parameter(torch.Tensor(hidden_size, output_size))  # [hidden_size, output_size]
        self.b_y = nn.Parameter(torch.Tensor(output_size))  # [output_size,]

        # 参数初始化
        self.reset_parameters()
        # 定义激活函数
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    # 没有考虑长度不相等（序列填充）的情况
    def forward(self, x, h_0=None):
        # x:[batch_size,seq_len,input_size]
        if h_0 is None:
            h_0 = torch.zeros(x.size(0), self.hidden_size).to(device)
        h_t = h_0  # [batch_size,hidden_size]
        outputs = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]  # [batch_size,input_size]
            # x_t @ self.W_xh:[batch_size,input_size] @ [input_size,hidden_size] ==> [batch_size,hidden_size]
            # h_t @ self.W_hh:[batch_size,hidden_size] @ [hidden_size,hidden_size] ==> [batch_size,hidden_size]
            h_t = self.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)  # [batch_size,hidden_size]
            # h_t @ self.W_hy:[batch_size,hidden_size] @ [hidden_size, output_size] ==> [batch_size, output_size]
            y_t = self.softmax(h_t @ self.W_hy + self.b_y)  # [batch_size, output_size]
            outputs.append(y_t)
        outputs = torch.stack(outputs, dim=1)  # [batch_size,seq_len,output_size]
        return outputs, h_t

    def reset_parameters(self):
        # 使用Kaiming初始化方法初始化权重
        nn.init.kaiming_uniform_(self.W_xh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_hh, a=math.sqrt(5))
        nn.init.zeros_(self.b_h)

        nn.init.kaiming_uniform_(self.W_hy, a=math.sqrt(5))
        nn.init.zeros_(self.b_y)


# 参数设置
input_size = 5
hidden_size = 10
output_size = 3
batch_size = 2
seq_len = 4
num_epochs = 3000
learning_rate = 0.01

rnn = MyRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

# 准备数据
input_data = Variable(torch.randn(batch_size, seq_len, input_size)).to(device)
# target_data = Variable(torch.LongTensor(batch_size, seq_len).random_(0, output_size)).to(device)
target_data = Variable(torch.LongTensor(batch_size).random_(0, output_size)).to(device)

# 训练循环
for epoch in range(num_epochs):
    # 清除之前的梯度
    optimizer.zero_grad()

    # 前向传播
    outputs, _ = rnn(input_data)

    # 计算损失
    # 需要将输出从 [batch_size, seq_len, output_size] 转换为 [batch_size * seq_len, output_size]
    # 目标也需要从 [batch_size, seq_len] 转换为 [batch_size * seq_len]
    output_last = outputs[:, -1, :]
    loss = criterion(output_last, target_data)
    # loss = criterion(outputs.view(-1, output_size), target_data.view(-1))

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed.")
