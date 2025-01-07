import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPatternAttention(nn.Module):
    """
    时序模式注意力机制，用于捕捉时间序列中的关键模式。
    类似于 TPA-LSTM 中的注意力机制。
    """
    def __init__(self, hidden_size, pattern_size):
        super(TemporalPatternAttention, self).__init__()
        self.hidden_size = hidden_size
        self.pattern_size = pattern_size
        self.pattern = nn.Parameter(torch.randn(pattern_size, hidden_size))  # 学习的模式参数
        self.linear = nn.Linear(pattern_size, 62)

    def forward(self, lstm_outputs):
        """
        lstm_outputs: (batch, seq_len, hidden_size)
        """
        # 计算模式与LSTM输出的相似度
        similarity = torch.matmul(lstm_outputs, self.pattern.t())  # (batch, seq_len, pattern_size)
        attention_weights = F.softmax(similarity, dim=1)  # 在时间维度上归一化

        # 加权求和得到上下文向量
        context = torch.matmul(attention_weights.transpose(1, 2), lstm_outputs)  # (batch, pattern_size, hidden_size)
        # context = context.view(-1, self.pattern_size * self.hidden_size)  # 展平
        #
        # # 通过线性层进行变换
        context = self.linear(context.transpose(1, 2))  # (batch, hidden_size)
        return context.transpose(1, 2), attention_weights

class CrossAttention(nn.Module):
    """
    简单的跨模态注意力实现：
    Q: 来自一个模态的查询 (batch, seq_len, d_model)
    K, V: 来自另一个模态的键和值 (batch, seq_len, d_model)
    """
    def __init__(self, d_model, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # 定义Q, K, V的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换
        Q = self.W_q(Q)  # (batch, seq_len_q, d_model)
        K = self.W_k(K)  # (batch, seq_len_k, d_model)
        V = self.W_v(V)  # (batch, seq_len_k, d_model)

        # 分头
        d_k = self.d_model // self.num_heads
        Q = Q.view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)  # (batch, num_heads, seq_len_q, d_k)
        K = K.view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)  # (batch, num_heads, seq_len_k, d_k)
        V = V.view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)  # (batch, num_heads, seq_len_k, d_k)

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (batch, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len_q, seq_len_k)

        # 加权求和
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len_q, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch, seq_len_q, d_model)

        # 输出层
        out = self.out(context)  # (batch, seq_len_q, d_model)
        return out, attn_weights

class TPA_LSTM_CrossAttention(nn.Module):
    """
    结合 TPA-LSTM 和跨模态注意力的多模态模型示例。
    """
    def __init__(self, audio_input_size, visual_input_size, hidden_size, pattern_size, num_layers=2, num_heads=4):
        super(TPA_LSTM_CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.pattern_size = pattern_size

        # 音频编码器
        self.audio_lstm = nn.LSTM(input_size=audio_input_size, hidden_size=hidden_size, num_layers=num_layers,  bidirectional=True)
        self.audio_tpa = TemporalPatternAttention(hidden_size*2, pattern_size)

        # 视觉编码器
        self.visual_lstm = nn.LSTM(input_size=visual_input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.visual_tpa = TemporalPatternAttention(hidden_size*2, 2)

        # 跨模态注意力
        self.cross_attention_audio_to_visual = CrossAttention(hidden_size*2, num_heads)
        self.cross_attention_visual_to_audio = CrossAttention(hidden_size*2, num_heads)

        # 输出层
        self.fc = nn.Linear(hidden_size * 4, hidden_size*2)  # 结合音频和视觉信息

    def forward(self, audio_feat, visual_feat):
        """
        audio_feat: (batch, seq_len_audio, audio_input_size)
        visual_feat: (batch, seq_len_visual, visual_input_size)
        """
        # 音频编码
        audio_output, _ = self.audio_lstm(audio_feat)  # (batch, seq_len_audio, hidden_size)
        audio_context, audio_attn = self.audio_tpa(audio_output)  #  (batch, seq_len_audio, pattern_size), (batch, seq_len_audio, pattern_size)
        audio_context = audio_context + audio_output
        # 视觉编码
        visual_output, _ = self.visual_lstm(visual_feat)  # (batch, seq_len_visual, hidden_size)
        visual_context, visual_attn = self.visual_tpa(visual_output)  # (batch, hidden_size), (batch, seq_len_visual, pattern_size)
        visual_context = visual_context + visual_output
        # 跨模态注意力
        # 使用音频上下文作为查询，视觉上下文作为键和值
        fused_audio, attn_audio_to_visual = self.cross_attention_audio_to_visual(audio_context, visual_context, visual_context)

        # 使用视觉上下文作为查询，音频上下文作为键和值
        # fused_visual, attn_visual_to_audio = self.cross_attention_visual_to_audio(visual_context, audio_output, audio_output)
        #
        # # 结合音频和视觉信息
        # combined = torch.cat([fused_audio.squeeze(1), fused_visual.squeeze(1)], dim=-1)  # (batch, hidden_size * 2)
        # output = self.fc(combined)  # (batch, hidden_size)

        return fused_audio, {
            'audio_attn': audio_attn,
            'visual_attn': visual_attn,
            'attn_audio_to_visual': attn_audio_to_visual,

        }

# ===== 测试示例 =====
if __name__ == "__main__":
    batch_size = 8
    seq_len_audio = 100
    seq_len_visual = 50
    audio_input_size = 128
    visual_input_size = 256
    hidden_size = 256
    pattern_size = 10
    num_layers = 2
    num_heads = 4

    # 随机生成音频和视觉特征
    audio_feat = torch.randn(batch_size, seq_len_audio, audio_input_size)
    visual_feat = torch.randn(batch_size, seq_len_visual, visual_input_size)

    # 初始化模型
    model = TPA_LSTM_CrossAttention(
        audio_input_size=audio_input_size,
        visual_input_size=visual_input_size,
        hidden_size=hidden_size,
        pattern_size=pattern_size,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # 前向传播
    output, attentions = model(audio_feat, visual_feat)

    print("输出特征形状:", output.shape)  # (batch, hidden_size)
    print("注意力权重:", {k: v.shape for k, v in attentions.items()})
    # 输出示例:
    # 输出特征形状: torch.Size([8, 256])
    # 注意力权重: {
    #     'audio_attn': torch.Size([8, 100, 10]),
    #     'visual_attn': torch.Size([8, 50, 10]),
    #     'attn_audio_to_visual': torch.Size([8, 4, 1, 100]),
    #     'attn_visual_to_audio': torch.Size([8, 4, 1, 50])
    # }