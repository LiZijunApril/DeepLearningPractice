import torch

# %% 预测 在循环遍历prefix中的初始字符时，我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。这被称为预热，
# 在此期间模型会进行更新，但不会进行预测。预热期结束后，隐状态的值通常比初始值更是预测
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]: # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds): # Predict 'num_preds' steps
        y, state = net(get_input(), state)
        # outputs.append(int(y.argmax(dim=1).reshape(1)))
        _, y = y.max(dim=1)
        outputs.append(int(y.reshape(1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])