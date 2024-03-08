# %%
import collections
from utils import dataset
import re

# %% 读取数据集《时光机器》
dataset.DATA_HUB['time_machine'] = ('http://d2l-data.s3-accelerate.amazonaws.com/' + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(dataset.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数：{len(lines)}')
print(lines[0])
print(lines[10])

# %% 词元化
# 词元是文本的基本单位
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
# for i in range(11):
#     print(tokens[i])

# %% 词表
"""
    词元的类型是字符串，而模型要的输入是数字,构建一个字典，通常也叫词表（vocabulary）,用来将字符串类型的词元映射到从0开始的数字索引中。
    先将训练集中的所有文档合并在一起，对他们的唯一词元进行统计，得到的统计结果称为语料库（corpus）；然后根据每个唯一词元出现的频率，
    为其分配一个数字索引。
    语料库中不存在或已移除的任何词元都将映射到一个特定的未知词元'<unk>'。我们可以选择增加一个列表，用于保存那些被保留的词元，例如
    填充词元('<pad>')，序列开始词元('<bos>')，序列结束词元('<eos>')。
"""

class Vocab:
    """文本词表"""
    # def __init__(self, tokens=None, min_freq=0, reserved_tokens=None) -> None:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按照出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知元素索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self): # 未知词元的索引为0
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
    
def count_corpus(tokens):
    """统计词元出现的频率"""
    #  这里的tokens是一维列表或二维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# %%
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print('文本：', tokens[i])
    print('索引：', vocab[tokens[i]])

# %%
from utils import dataset
corpus, vocab = dataset.load_corpus_time_machine()
len(corpus), len(vocab)
# %%