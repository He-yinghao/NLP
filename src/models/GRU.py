import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()
        self.hidden_size = hidden_size

        # 重置门投影
        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        # 更新门投影
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        # 候选隐藏状态投影
        self.h = nn.Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_t_minus_1):
        # x: [batch_size, input_size]
        # h_t_minus_1: [batch_size, hidden_size]

        combined_data = torch.cat(
            (x, h_t_minus_1), dim=1
        )  # [batch_size, input_size + hidden_size]

        r_t = self.sigmoid(self.r(combined_data))  # 重置门：[batch_size, hidden_size]
        z_t = self.sigmoid(self.z(combined_data))  # 更新门：[batch_size, hidden_size]
        h_hat_t = self.tanh(
            self.h(torch.cat((r_t * h_t_minus_1, x), dim=1))
        )  # 候选隐藏状态：[batch_size, hidden_size]

        h_t = (
            z_t * h_t_minus_1 + (1 - z_t) * h_hat_t
        )  # 最终隐藏状态：[batch_size, hidden_size]

        return h_t


class MultiLayerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MultiLayerGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 创建每一层的GRU单元
        self.gru_cells = nn.ModuleList(
            [
                MyGRUCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态
        if h_0 is None:
            h_0 = [
                torch.zeros(batch_size, self.hidden_size).to(x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_0 = list(h_0)

        # 存储每一步的输出
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]

            # 通过每一层GRU
            for layer_idx in range(self.num_layers):
                h_t = self.gru_cells[layer_idx](x_t, h_0[layer_idx])
                h_0[layer_idx] = h_t  # 更新隐藏状态
                x_t = h_t  # 下一层的输入是当前层的输出

            y_t = self.fc(h_t)  # [batch_size, output_size]
            outputs.append(y_t)

        # 将输出堆叠成一个张量
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, output_size]

        # 返回最终的输出和最后一层的隐藏状态
        return outputs, h_0[-1]


if __name__ == "__main__":
    # 参数设置
    input_size = 5
    hidden_size = 10
    output_size = 3
    batch_size = 2
    seq_len = 4
    num_epochs = 3000
    learning_rate = 0.01
    num_layers = 5

    multilayer_gru = MultiLayerGRU(input_size, hidden_size, output_size, num_layers).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(multilayer_gru.parameters(), lr=learning_rate)

    # 准备数据
    input_data = Variable(torch.randn(batch_size, seq_len, input_size)).to(device)
    # target_data = Variable(torch.LongTensor(batch_size, seq_len).random_(0, output_size)).to(device)
    target_data = Variable(torch.LongTensor(batch_size).random_(0, output_size)).to(
        device
    )

    # 训练循环
    for epoch in range(num_epochs):
        # 清除之前的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs, _ = multilayer_gru(input_data)

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

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training completed.")
