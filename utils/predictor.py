import torch
import dataset
import math
import collections

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

# %% BLEU
def bleu(pred_seq: str, label_seq: str, k: str):
    """Compute the BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k+1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1          
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# %% 预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab: dataset.Vocab, num_steps, device, save_attention_weights=False):
    """Predict for sequence to sequence"""
    #* Set 'net' to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = dataset.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    #* Add the batch size
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    #* Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用预测可能性最大的词元，作为解码器在下一个时间步的输入
        # dec_X = Y.argmax(dim=2)
        dec_X = Y.argmax(dim=2)

        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦<eos>被预测出，输出序列的预测就结束了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq