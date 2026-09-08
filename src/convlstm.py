import torch.nn as nn
import torch

# 核心思想：在每个时间步，使用卷积操作代替全连接层来处理输入和隐藏状态，从而保留空间结构。
# 通过堆叠多个 ConvLSTM 层，可以提取更高层次的时空特征。

# 单个时间步处理单元，包含输入门、遗忘门、输出门和候选记忆，更新隐藏状态h和记忆细胞c
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            输入张量通道数，例如视频帧通道数（rgb为3）
        hidden_dim: int
            隐藏状态通道数，即 ConvLSTM 单元输出的特征图通道数
        kernel_size: (int, int)
            卷积核大小，二元组（height，width）
        bias: bool
            是否在卷积中使用偏置项
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        #定义一个 2D卷积层，它的输入通道数是 input_dim + hidden_dim（因为要把当前输入和上一时刻的隐藏状态拼接在一起）
        #输出通道数是 4 * hidden_dim。为什么是 4 倍？因为 ConvLSTM 内部有四个门（输入门 i、遗忘门 f、输出门 o、候选记忆 g），每个门都需要一个 hidden_dim 通道的输出。卷积后会将四个门的信息分别取出。
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    # input_tensor:当前时间步输入,(batch,input_dim,height,width)
    # cur_state:当前时间步状态，元祖(h_cur,c_cur)构成，代表隐藏状态和记忆细胞
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 在通道维度（dim=1）上拼接输入和隐藏状态，得到形状(batch, input_dim+hidden_dim, height, width) 的张量
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # 将拼接后的张量送入卷积层，输出形状为(batch, 4*hidden_dim, height, width)
        combined_conv = self.conv(combined)
        # 将卷积输出沿通道维度拆分成四个等大小的块，每个块大小为hidden_dim
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # Sigmoid：输出 0~1，控制信息通过的比例
        i = torch.sigmoid(cc_i) # 输入门
        f = torch.sigmoid(cc_f) # 遗忘门
        o = torch.sigmoid(cc_o) # 输出门
        # Tanh：输出 -1~1，生成候选记忆值，保持数据在合理范围
        g = torch.tanh(cc_g) # 候选记忆

        # 更新记忆细胞：遗忘门控制保留多少旧记忆，输入门控制加入多少新信息。两者相加得到新的记忆细胞
        c_next = f * c_cur + i * g
        # 更新隐藏状态：输出门控制从记忆细胞中输出多少信息。先对新的记忆细胞应用 tanh 压缩到 -1~1，再与输出门逐元素相乘，得到新的隐藏状态
        h_next = o * torch.tanh(c_next)

        return h_next, c_next   # 供下一个时间步使用

    # 用于初始化隐藏状态h和记忆细胞c，初始化为全零张量
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# 将多个 ConvLSTMCell 堆叠起来，形成多层 ConvLSTM，并处理整个时间序列
class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: 输入通道数
        hidden_dim: 隐藏状态通道数
        kernel_size: 卷积核大小
        num_layers: 堆叠层数
        batch_first: true的话batch在seq_len前面
        bias: 是否使用偏置
        return_all_layers: false只返回最后一层的输出和最后状态
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # 调用静态方法检查 kernel_size 的一致性，必须是元祖或元祖列表
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # 将可能为单值的参数扩展成与层数相同的列表：简化用户输入，同时支持逐层定制
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        # 确保扩展后的列表长度等于层数，否则报错
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # 创建每一层的 ConvLSTMCell，并放入 nn.ModuleList 中
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # 适配输入维度：(t, b, c, h, w) -> (b, t, c, h, w)  (批次大小，时间步帧数，通道数，高，宽)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()  # 获取批量大小和空间尺寸（高度、宽度）

        # Implement stateful ConvLSTM
        # 提供隐藏状态，抛出异常，功能尚未实现
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            # 如果没有提供隐藏状态，则调用init_hidden初始化所有层的隐藏状态（全零）
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []  # 存储每层输出
        last_state_list = []    # 存储最后状态

        seq_len = input_tensor.size(1)  # 时间步数
        cur_layer_input = input_tensor  # 原始输入

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx] # 取出初始隐藏状态
            output_inner = []
            for t in range(seq_len): # 遍历时间步
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            # 将这一层所有时间步的输出堆叠起来，得到形状 (batch, seq_len, hidden_dim[layer_idx], h, w) 的张量
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # 如果不需要返回所有层的输出和状态，则只保留最后一层的
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    # 调用每个 cell 的 init_hidden 方法，为每一层生成初始隐藏状态列表
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    # 检查 kernel_size 是否为元组或全为元组的列表，否则抛出异常
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    # 如果 param 不是列表，则将其复制为长度为 num_layers 的列表
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
