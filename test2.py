# %%
from utils import dataset

raw_text = dataset.read_data_nmt()
print(raw_text[:75])

text = dataset.preprocess_nmt(raw_text)

source, target = dataset.tokenize_nmt(text)
source[:6], target[:6]

# %%
from utils import plot

plot.show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target),
# %%

src_vocab = dataset.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
dataset.truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
# %%
