# %%
import torch
from torch import nn
from utils import plot

attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
plot.show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
# %%
n_train = 50 # No. of traning example
x_train, _ = torch.sort(torch.rand(n_train) * 5)
def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,)) # Training outputs
x_test = torch.arange(0, 5, 0.1) # Testing example
y_truth = f(x_test) # 测试样本的真实输出
n_test = len(x_test) # No. of testing example
n_test
# %%
from utils import d2l
def plot_kernel_reg(y_hat):
    plot.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    plot.plt.plot(x_train, y_train, 'o', alpha=0.5);
# %%
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
y_hat, y_train, y_train.mean(), n_test
plot_kernel_reg(y_hat)
# %%
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
print('Attention_weights: \n', attention_weights)
print('attention_weights: ', attention_weights.shape)

y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
X_repeat, x_train, y_hat
X_repeat.shape, x_train.shape, y_hat.shape
# %%
plot.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                   xlabel='Sorted training inputs',
                   ylabel='Sorted testing inputs')
# %%
def sort_texts(texts, verbs):
    # 使用列表推导式和排序函数对文本列表进行排序
    sorted_texts = sorted(texts, key=lambda text: [verbs.index(verb) if verb in text else len(verbs) for verb in verbs])
    return sorted_texts

# 示例文本列表和动词列表
texts = ["他强调了这个观点", "表示同意这个决定", "她说了一些话", "他强调了重要性", "她说了一句话"]
verbs = ['强调', '表示', '说']

# 调用函数进行排序
sorted_texts = sort_texts(texts, verbs)

# 打印排序后的结果
for text in sorted_texts:
    print(text)
# %%

