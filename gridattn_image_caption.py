#!/usr/bin/env python
# coding: utf-8

# # 实战案例：基于注意力的图像描述模型（ARCTIC）
# 
# ## 模型训练流程
# 
# ![模型训练的一般流程](img/cr-traning_process.png)
# 

# 在现代的深度学习框架基础下，模型训练的一般流程包括读取数据、前馈计算、计算损失、更新参数、选择模型五个步骤。每个步骤需要实现相应的模块。
# 
# - 在读取数据阶段，我们首先需要下载数据集，然后对整理数据集的格式，以方便接下来构造数据集类，最后在数据集类的基础上构建能够按批次产生训练、验证和测试数据的对象。
# 
# - 在前馈计算阶段，我们需要实现具体的模型，使得模型能够根据输入产生相应的输出。
# 
# - 在计算损失阶段，我们需要将模型输出和预期输出进行对比，实现损失函数。
# 
# - 在更新参数阶段，我们需要给出具体的参数更新方法，即优化方法；由于现代深度学习框架能够自动计算参数梯度，并实现了绝大多数优化方法，我们通常只需要从中进行选择即可。
# 
# - 在选择模型阶段，我们需要实现具体的评估指标，选出在验证集上表现最优的模型参数。
# 
# 下面，我们将按照这个次序介绍一个图像编码器为CNN网格表示提取器、文本解码器为RNN+注意力的图像描述方法的具体实现。我们的实现大体上是在复现ARCTIC模型，但是在细节上有一些改变，下面的实现过程会对这些改变做具体说明。此外，[链接](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)给出了一个更接近原始ARCTIC模型的代码库，非常推荐大家阅读。本节的部分代码也是受到该代码库的启发。

# ## 读取数据
# 
# ### 下载数据集
# 
# - 我们使用的数据集为flickr8k([下载地址](https://www.kaggle.com/adityajn105/flickr8k))。
# - 下载解压后，将其图片放在指定目录(本代码中将该目录设置为../data/flickr8k)下的images文件夹里。
# - 数据集包括8000张图片，每张图片对应5个句子描述。
# - 数据集划分采用Karpathy提供的方法([下载地址](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip))，下载解压后，将其中的dataset_flickr8k.json文件拷贝到指定目录下。该划分方法将数据集分成3个子集：6,000张图片和其对应句子描述组成训练集，1,000张图片和描述为验证集，剩余的1,000张图片和描述为测试集。
# 
# 

# ### 整理数据集
# 
# 数据集下载完成后，我们需要对其进行处理，以适合之后构造的PyTorch数据集类读取。
# - 对于文本描述，我们首先构建词典，然后根据词典将文本描述转化为向量。
# - 对于图像，我们这里仅记录文件路径。
# - - 如果机器的内存和硬盘空间就比较大，这里也可以将图片读取并处理成三维数组，这样在模型训练和测试的阶段，就不需要再直接读取图片。




# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import json
import random 
from collections import defaultdict, Counter
from PIL import Image
from matplotlib import pyplot as plt

from argparse import Namespace 
import numpy as np
import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import nltk
from torch.multiprocessing import freeze_support




train_dataset_path = "./data/flickr8k/train_captions.json"
test_dataset_path = "./data/flickr8k/test_captions.json"
vocab_path = "./data/flickr8k/vocab.json"
image_path = "./data/flickr8k/images/"
output_path = "./data/flickr8k/"





def create_dataset(train_json_file = "./data/flickr8k/train_captions.json", test_json_file = "./data/flickr8k/test_captions.json" , vocab_file = "./data/flickr8k/vocab.json", output_folder = "./data/flickr8k/"):
    """
    从 json 文件创建词典

    Args:
        train_json_file: 输入 json 文件
        vocab_file: 输出词典文件

    Returns:
        无
    """

    with open(train_json_file, 'r') as f:
        data = json.load(f)

    image_paths = defaultdict(list)
    image_captions = defaultdict(list)
    vocab = Counter()
    for image_name, description in data.items():
        for word in nltk.word_tokenize(description):
            vocab.update([word])

    # 添加 <pad>、<unk>、<start>、<end> 这四个 token

    vocab.update(['<pad>', '<unk>', '<start>', '<end>'])

    # 将词典转换为字典格式

    vocab = {word: i for i, word in enumerate(vocab.keys())}

    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
  

    # 整理数据集
    with open(train_json_file, 'r') as f:
        data = json.load(f)
        for image_name, description in data.items():
            image_name = os.path.join(image_path, image_name)
            image_paths['train'].append(image_name)
            image_captions['train'].append(nltk.word_tokenize(description))

    with open(test_json_file, 'r') as f:
        data = json.load(f)
        for image_name, description in data.items():
            image_name = os.path.join(image_path, image_name)
            image_paths['test'].append(image_name)
            image_captions['test'].append(nltk.word_tokenize(description))

    for split in image_paths:
        imgpaths = image_paths[split]
        imcaps = image_captions[split]

        enc_captions:list[list] = []

        for i, path in enumerate(imgpaths):
            captions = imcaps[i]
            # for j, c in enumerate(captions):
                # 对文本描述进行编码
            enc_c: list[int] = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in captions] + [vocab['<end>']] 
            enc_captions.append(enc_c)
        # 合法性检查
        # assert len(imgpaths) == len(enc_captions)
      
        # 存储数据
        data = {'IMAGES': imgpaths, 
                'CAPTIONS': enc_captions}
        print("数据集大小：", len(imgpaths))
        print("数据集：", len(enc_captions))
        with open(os.path.join(output_folder, split + '_data.json'), 'w') as fw:
            json.dump(data, fw)




# create_dataset()


# 在调用该函数生成需要的格式的数据集文件之后，我们可以展示其中一条数据，简单验证下数据的格式是否和我们预想的一致。




# 读取词典和验证集
with open('./data/flickr8k/vocab.json', 'r') as f:
    vocab = json.load(f)
vocab_idx2word = {idx:word for word,idx in vocab.items()}

with open('./data/flickr8k/train_captions.json', 'r') as f:
    train_data = json.load(f)

with open('./data/flickr8k/test_captions.json', 'r') as f:
    test_data = json.load(f)

print("词典长度：", len(vocab))
print("训练集json：", len(train_data))
print("测试集json：", len(test_data))
print("训练集json+测试集json：", len(train_data)+len(test_data))
print("images数量：", len(os.listdir('./data/flickr8k/images')))

# # 展示第12张图片，其对应的文本描述序号是60到64
# content_img = Image.open(data['IMAGES'][12])
# plt.imshow(content_img)
# for i in range(5):
#     print(' '.join([vocab_idx2word[word_idx] for word_idx in data['CAPTIONS'][12*5+i]]))


# ### 定义数据集类
# 
# 在准备好的数据集的基础上，我们需要进一步定义PyTorch Dataset类，以使用PyTorch DataLoader类按批次产生数据。PyTorch中仅预先定义了图像、文本和语音的单模态任务中常见的数据集类。因此，我们需要定义自己的数据集类。
# 
# 在PyTorch中定义数据集类非常简单，仅需要继承torch.utils.data.Dataset类，并实现\_\_getitem\_\_和\_\_len\_\_两个函数即可。




class ImageTextDataset(Dataset):
    """
    PyTorch数据类，用于PyTorch DataLoader来按批次产生数据
    """

    def __init__(self, dataset_path, vocab_path, split, captions_per_image=1, max_len=120, transform=None):
        """
        参数：
            dataset_path：json格式数据文件路径
            vocab_path：json格式词典文件路径
            split：train、val、test
            captions_per_image：每张图片对应的文本描述数
            max_len：文本描述包含的最大单词数
            transform: 图像预处理方法
        """
        self.split = split
        assert self.split in {'train', 'test'}
        self.cpi = captions_per_image
        self.max_len = max_len

        # 载入数据集
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        # 载入词典
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        # PyTorch图像预处理流程
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.data['IMAGES'])

    def __getitem__(self, i):
        # 第i个文本描述对应第(i // captions_per_image)张图片
        img = Image.open(self.data['IMAGES'][i // self.cpi]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        caplen = len(self.data['CAPTIONS'][i])
        caption = torch.LongTensor(self.data['CAPTIONS'][i]+ [self.vocab['<pad>']] * (self.max_len + 2 - caplen))
        
        return img, caption, caplen
        

    def __len__(self):
        return self.dataset_size



# ### 批量读取数据
# 
# 利用刚才构造的数据集类，借助DataLoader类构建能够按批次产生训练、验证和测试数据的对象。




def mktrainval(data_dir, vocab_path, batch_size, workers=0):
    train_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_set = ImageTextDataset(os.path.join(data_dir, 'train_data.json'), 
                                 vocab_path, 'train',  transform=train_tx)
    test_set = ImageTextDataset(os.path.join(data_dir, 'test_data.json'), 
                                 vocab_path, 'test', transform=test_tx)
    print("训练集大小：", len(train_set))
    print("测试集大小：", len(test_set))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)

    return train_loader, test_loader    


# ## 定义模型
# 
# ARCTIC模型是一个典型的基于注意力的编解码模型，其编码器为图像网格表示提取器，解码器为循环神经网络。解码器在每生成一个词时，都利用注意力机制考虑当前生成的词和图像中的哪些网格更相关。
# 
# ![ARCTIC的模型结构示意图](img/mt-cnn-attn.png)
# 

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

# ### 图像编码器
# 
# ARCTIC原始模型使用在ImageNet数据集上预训练过的分类模型VGG19作为图像编码器，VGG19最后一个卷积层作为网格表示提取层。而我们这里使用ResNet-101作为图像编码器，并将其最后一个非全连接层作为网格表示提取层。



from torchvision.models import vit_b_16
class ImageEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        self.grid_rep_extractor = model
        self.fc = nn.Linear(1000, 512)
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned
        
    def forward(self, images):
        out = self.grid_rep_extractor(images)
        out = self.fc(out)
        out = out.reshape(out.shape[0], -1, out.shape[1])
        return out
        # -> (batch_size, 1, 512)


# ### 文本解码器
# 
# ARCTIC原始模型使用结合注意力的LSTM作为文本解码器，我们这里使用结合注意力的GRU作为文本解码器，注意力评分函数采用的是加性注意力。下面给出加性注意力和解码器的具体实现。
# 

# 加性注意力评分函数的具体形式为 $W_2^T{\rm tanh}(W_1 [\mathbf{q}_i; \mathbf{k}_j])$ 。
# 
# - 首先将权重 $W_1$ 依照查询q和键k的维度，相应地拆成两组权重，分别将单个查询和一组键映射到到注意力函数隐藏层表示空间；
# - 然后将二者相加得到一组维度为attn_dim的表示，并在经过非线性变换后，使用形状为(attn_dim, 1) 的权重 $W_2$ 将其映射为一组数值；
# - 再通过softmax函数获取单个查询和所有键的关联程度，即归一化的相关性分数；
# - 最后以相关性得分为权重，对值进行加权求和，计算输出特征。这里的值和键是同一组向量表示。




import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=122):
        """
        位置编码器类的初始化函数
        
        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings
        pe = torch.zeros(max_len, d_model) # 构建位置编码的张量
        position = torch.arange(0, max_len).unsqueeze(1) # 将每个token的位置（即从0到max_len-1）变为列向量
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) # 计算
        pe[:, 0::2] = torch.sin(position * div_term) # 使用了广播机制分配pe的不同列。这代表着从第0行开始每隔两列取一列。返回所有偶数位置的编码。
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 最后增加了一个维度，也就是batch的维度。
        self.register_buffer('pe', pe) #将位置编码张量存储在模型中。这是因为位置编码不需要进行训练，因此可以将其视为模型的一个固定部分。
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach() # 从位置编码张量中取出对应长度的位置编码，并与输入的词嵌入相加。
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_size, num_layers, dropout=0.2) -> None:
        super(TransformerDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.PE = PositionalEncoding(word_dim, dropout)

        decoder_layers = nn.TransformerDecoderLayer(
            d_model=word_dim, # 输入维度
            nhead=8,  # Number of heads in the multiheadattention models
            # dim_feedforward=768,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_code, captions, cap_lens = None):
        embed_captions = self.embed(captions)
        positional_encoding = self.PE(embed_captions)

        self_attn_mask = get_attn_pad_mask(captions, captions)
        dec_mask = get_attn_subsequent_mask(captions).to(image_code.device)

        tgt_mask = torch.gt((self_attn_mask + dec_mask), 0)

        tgt_mask = tgt_mask.unsqueeze(1).expand(-1, 8, -1, -1).reshape(-1, captions.size(1), captions.size(1))
        
        out = self.transformer_decoder(positional_encoding, image_code, tgt_mask=tgt_mask)
        out = self.fc2(out)
        # -> (batchsize, seq_len, vocab_size)
        return out, captions
# ### ARCTIC模型
# 
# 在定义编码器和解码器完成之后，我们就很容易构建图像描述模型ARCTIC了。仅需要在初始化函数时声明编码器和解码器，然后在前馈函数实现里，将编码器的输出和文本描述作为解码器的输入即可。
# 
# 这里我们额外定义了束搜索采样函数，用于生成句子，以计算BLEU值。下面的代码详细标注了其具体实现。

from torch.nn import functional as F
# torch.set_printoptions(threshold=torch.inf)
class ARCTIC(nn.Module):
    def __init__(self, image_code_dim, vocab, word_dim, attention_dim, hidden_size, num_layers):
        super(ARCTIC, self).__init__()
        self.vocab = vocab
        self.encoder = ImageEncoder()
        self.decoder = TransformerDecoder(len(vocab), word_dim, hidden_size, num_layers)

    def forward(self, images, captions, cap_lens):
        image_code = self.encoder(images)
        return self.decoder(image_code, captions, cap_lens)
    
    def generate(self, image_code, captions, max_len=120):
        """
        generate next word
        """
        preds, _ = self.decoder(image_code, captions)
        log_soft = nn.LogSoftmax(dim=2)
        # -> (k, len=1, vocab_size)
        preds_log = log_soft(preds)
        max_values, max_indices = torch.max(preds_log[:,-1,:], dim=-1)
        return max_values.view(-1)
        

    def sample(model, images, beam_k=1, max_len=120, temperature=1.0):
        vocab_size = len(model.vocab)
        # image -> (batchsize, 3, 224, 224)
        image_codes = model.encoder(images)
        # -> (batchsize, 1, 512)
        texts = []
        sample_logprobs = []  # 用于存储对数概率
        device = images.device
        # 对batchsize中每个图像样本执行生成
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k, 1, 1)
            # -> (beam_k, 1, 512)
            # 生成一个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), model.vocab['<start>'], dtype=torch.long, device=device)
            # -> (beam_k, 1)
            cur_sent_embed = model.decoder.embed(cur_sents)[:, 0, :]
            # -> (beam_k, 1, 512)
            sent_lens = torch.LongTensor([1] * beam_k).to(device)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1, device=device)
            # 存储每个词生成的概率
            word_probs = []

            k = beam_k
            while True:
                preds, _ = model.decoder(image_code[:k], cur_sents)
                log_soft = F.log_softmax(preds, dim=2)
                # -> (k, len=1, vocab_size)
                probs = torch.exp(log_soft[:, -1, :])
                
                # 对每个候选句子进行采样
                if temperature != 1.0:
                    probs = probs / temperature

                it = torch.multinomial(probs, 1)
                sampleLogprobs = log_soft.gather(2, it.unsqueeze(2))
                it = it.view(-1).long()
                # 更新候选句子
                cur_sents = torch.cat([cur_sents, it.unsqueeze(1)], dim=1)
                # 记录每个词生成的概率
                word_probs.append(probs.gather(1, it.unsqueeze(1)))

                # 查找包含结束符<end>的句子
                end_indices = it == model.vocab['<end>']
                if end_indices.any():
                    end_probs.extend(sampleLogprobs[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    # 如果所有的句子都包含结束符，则停止生成
                    k -= end_indices.sum().item()
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = ~end_indices
                if cur_indices.any():
                    cur_sents = cur_sents[cur_indices]
                    probs = sampleLogprobs[cur_indices].view(-1, 1)

                # 句子太长，停止生成
                if cur_sents.size(1) >= max_len:
                    break

            if len(end_sents) == 0:
                # 如果没有包含结束符的句子，则选取第一个句子作为生成句子
                gen_result = cur_sents[0, 1:]  # 去掉第一个元素
            else:
                # 否则选取包含结束符的句子中概率最大的句子
                gen_result = end_sents[end_probs.index(max(end_probs))][1:]

            texts.append(gen_result)
            # 将 word_probs 转换为形状与生成句子一致的张量
            word_probs_tensor = torch.cat(word_probs, dim=1)

            sample_logprobs.append(word_probs_tensor)
        
        # 将结果转换为张量
        texts_tensor = torch.stack(texts).detach().clone()
        return texts_tensor, sample_logprobs




    def generate_by_beamsearch(self, images, beam_k=5, max_len=120):
        vocab_size = len(self.vocab)
        # image -> (batchsize, 3, 224, 224)
        image_codes = self.encoder(images)
        # -> (batchsize, 1, 512)
        texts = []
        device = images.device
        # 对batchsize中每个图像样本执行束搜索
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k,1,1)
            # -> (beam_k, 1, 512)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            # -> (beam_k, 1)
            cur_sent_embed = self.decoder.embed(cur_sents)[:,0,:]
            # -> (beam_k, 1, 512)
            sent_lens = torch.LongTensor([1]*beam_k).to(device)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            # -> (beam_k, 1)
            k = beam_k
            while True:
                preds, _ = self.decoder(image_code[:k], cur_sents)
                # print(preds)
                log_soft = nn.LogSoftmax(dim=2)
                # -> (k, len=1, vocab_size)
                preds = log_soft(preds)
                # print(preds)
                # 对每个候选句子采样概率值最大的前k个单词生成k个新的候选句子，并计算概率
                # -> (k, len=1, vocab_size)
                probs = probs.repeat(1,preds.size(-1)) + preds[:, -1, :].view(k, -1)
                # print(preds[:, -1, :].view(k, -1))
                # -> (k, k, vocab_size)
                if cur_sents.size(1) == 1:
                    # 第一步时，所有句子都只包含开始标识符，因此，仅利用其中一个句子计算topk
                    values, indices = probs[0].topk(k, 0, True, True)
                    for index ,(v, i) in enumerate(zip(values, indices)):
                        if i == 113:
                            values_temp, indices_temp = probs[0].topk(k + 1, 0, True, True)
                            values[index] = values_temp[-1]
                            indices[index] = indices_temp[-1]
                else:
                    # probs: (k, vocab_size) 是二维张量
                    # topk函数直接应用于二维张量会按照指定维度取最大值，这里需要在全局取最大值
                    # 因此，将probs转换为一维张量，再使用topk函数获取最大的k个值
                    values, indices = probs.view(-1).topk(k, 0, True, True)
                # 计算最大的k个值对应的句子索引和词索引
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc') 
                word_indices = indices % vocab_size 
                # 将词拼接在前一轮的句子后，获得此轮的句子
                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)
                # 查找此轮生成句子结束符<end>的句子
                end_indices = [idx for idx, word in enumerate(word_indices) if word == self.vocab['<end>']]
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    # 如果所有的句子都包含结束符，则停止生成
                    k -= len(end_indices)
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = [idx for idx, word in enumerate(word_indices) 
                            if word != self.vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    # 仅保留还需要继续生成的句子、句子概率、隐状态、词嵌入
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1,1)
                    # hidden_state = hidden_state[:,cur_sent_indices,:]
                    cur_sent_embed = self.decoder.embed(
                        cur_word_indices.view(-1,1))[:,0,:]
                # 句子太长，停止生成
                if cur_sents.size(1) >= max_len:
                    break
            
            # end_sents = [sublist for sublist in end_sents if sublist != [112, 113]]

            if len(end_sents) == 0:
                # 如果没有包含结束符的句子，则选取第一个句子作为生成句子
                gen_sent = cur_sents[0].tolist()
            else: 
                # 否则选取包含结束符的句子中概率最大的句子
                gen_sent = end_sents[end_probs.index(max(end_probs))]
            texts.append(gen_sent)
        return texts

# ## 定义损失函数
# 
# 这里采用了最常用的交叉熵损失作为损失函数。由于同一个训练批次里的文本描述的长度不一致，因此，有大量的不需要计算损失的<pad>目标。为了避免计算资源的浪费，这里先将数据按照文本长度排序，再利用pack_padded_sequence函数将预测目标为\<pad\>的数据去除，最后再利用交叉熵损失计算实际的损失。




class PackedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PackedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, lengths):
        """
        参数：
            predictions：按文本长度排序过的预测结果
            targets：按文本长度排序过的文本描述
            lengths：文本长度
        """
        lengths = lengths.to(torch.device('cpu'))
        predictions = pack_padded_sequence(predictions, lengths, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)[0]
        return self.loss_fn(predictions, targets)
        


# ## 选择优化方法
# 
# 这里选用Adam优化算法来更新模型参数，由于数据集较小，训练轮次少，因此，学习速率在训练过程中并不调整。但是对编码器和解码器采用了不同的学习速率。具体来说，预训练的图像编码器的学习速率小于需要从头开始训练的文本解码器的学习速率。




def get_optimizer(model, config):
    return torch.optim.Adam([{"params": filter(lambda p: p.requires_grad, model.encoder.parameters()), 
                              "lr": config.encoder_learning_rate},
                             {"params": filter(lambda p: p.requires_grad, model.decoder.parameters()), 
                              "lr": config.decoder_learning_rate}])

def adjust_learning_rate(optimizer, epoch, config):
    """
        每隔lr_update个轮次，学习速率减小至当前十分之一，
        实际上，我们并未使用该函数，这里是为了展示在训练过程中调整学习速率的方法。
    """
    optimizer.param_groups[0]['lr'] = config.encoder_learning_rate * (0.1 ** (epoch // config.lr_update))
    optimizer.param_groups[1]['lr'] = config.decoder_learning_rate * (0.1 ** (epoch // config.lr_update))


# ## 评估指标
# 
# 这里借助nltk库实现了图像描述中最常用的评估指标BLEU值，需要注意的是，再调用计算BLEU值之前，要先将文本中人工添加的文本开始符、结束符和占位符去掉。
# 




from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
rouge = Rouge()

def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]

@torch.no_grad()
def evaluate(data_loader, model, config):

    new_vocab = {v : k for k, v in model.vocab.items()}

    # model.eval()
    # 存储候选文本
    cands = []
    # 存储参考文本
    refs = []
    # 需要过滤的词
    filterd_words = set({model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']})
    cpi = config.captions_per_image
    device = next(model.parameters()).device
    for i, (imgs, caps, caplens) in enumerate(data_loader):
        with torch.no_grad():
            # 通过束搜索，生成候选文本
            texts = model.generate_by_beamsearch(imgs.to(device), config.beam_k, config.max_len+2)
            sentence_list = [new_vocab[i] for i in texts[0]]
            sentence = ' '.join(sentence_list)
            print(sentence)
            # 候选文本
            cands.extend([filter_useless_words(text, filterd_words) for text in texts])
            # 参考文本
            refs.extend([filter_useless_words(cap, filterd_words) for cap in caps.tolist()])
    # 实际上，每个候选文本对应cpi条参考文本
    multiple_refs = []
    for idx in range(len(refs)):
        multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])
    # 计算BLEU-4值，corpus_bleu函数默认weights权重为(0.25,0.25,0.25,0.25)
    # 即计算1-gram到4-gram的BLEU几何平均值
    bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))
    # bleu4 = meteor_score(multiple_refs, cands)
    # bleu4 = rouge.get_scores(multiple_refs, cands)
    model.train()
    return bleu4


# ## 训练模型
# 
# 训练模型过程仍然是分为读取数据、前馈计算、计算损失、更新参数、选择模型五个步骤。
# 
# 模型训练的具体方案为一共训练30轮，编码器和解码器的学习速率分别为0.0001和0.0005。
# 
# <!-- 模型训练的具体方案为一共训练30轮，初始编码器和解码器的学习速率分别为0.0001和0.0005，每10轮将学习速率变为原数值的1/10。 -->


def main():
    # 设置模型超参数和辅助变量
    # freeze_support()
    config = Namespace(
        max_len = 120,
        captions_per_image = 1,
        batch_size = 1,
        image_code_dim = 2048,
        word_dim = 512,
        hidden_size = 512,
        attention_dim = 512,
        num_layers = 3,
        encoder_learning_rate = 0.00001,
        decoder_learning_rate = 0.00005,
        num_epochs = 30,
        grad_clip = 5.0,
        alpha_weight = 1.0,
        evaluate_step = 2000, # 900, # 每隔多少步在验证集上测试一次
        checkpoint = "./model_1/best_flickr8k.ckpt", # 如果不为None，则利用该变量路径的模型继续训练
        best_checkpoint = './model_1/best_flickr8k.ckpt', # 验证集上表现最优的模型的路径
        last_checkpoint = './model_1/last_flickr8k.ckpt', # 训练完成时的模型的路径
        beam_k = 5
    )



    # 设置GPU信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # print("使用设备：", device)
    # device = torch.device("cpu")

    # 数据
    data_dir = './data/flickr8k/'
    vocab_path = './data/flickr8k/vocab.json'
    train_loader, test_loader = mktrainval(data_dir, vocab_path, config.batch_size)

    # 模型
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # 随机初始化 或 载入已训练的模型
    start_epoch = 0
    checkpoint = config.checkpoint
    if checkpoint is None:
        model = ARCTIC(config.image_code_dim, vocab, config.word_dim, config.attention_dim, config.hidden_size, config.num_layers)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
    
    last_epoch = start_epoch
    # 优化器
    optimizer = get_optimizer(model, config)

    # 将模型拷贝至GPU，并开启训练模式
    model.to(device)
    model.train()

    # 损失函数
    loss_fn = PackedCrossEntropyLoss().to(device)

    best_res = 0
    print("开始训练")
    bleu_score = evaluate(test_loader, model, config)
    print('Validation@BLEU-4=%.2f' % 
    (bleu_score))
    fw = open('log.txt', 'w')
    for epoch in range(start_epoch, config.num_epochs):
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            optimizer.zero_grad()
            # 1. 读取数据至GPU
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # 2. 前馈计算
            predictions, captions = model(imgs, caps, caplens)
            
            # 3. 计算损失
            # captions从第2个词开始为targets
            loss = loss_fn(predictions, captions[:, 1:], caplens)
            # 重随机注意力正则项，使得模型尽可能全面的利用到每个网格
            # 要求所有时刻在同一个网格上的注意力分数的平方和接近1
            # loss += config.alpha_weight * ((1. - alphas.sum(axis=1)) ** 2).mean()

            loss.backward()
            # 梯度截断
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # 4. 更新参数
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('epoch %d, step %d: loss=%.2f' % (epoch, i+1, loss.cpu()))
                fw.write('epoch %d, step %d: loss=%.2f \n' % (epoch, i+1, loss.cpu()))
                fw.flush()

            state = {
                    'epoch': epoch,
                    'step': i,
                    'model': model,
                    'optimizer': optimizer
                    }
            if epoch - last_epoch == 3: #(i+1) % config.evaluate_step == 0:
                last_epoch = epoch
                bleu_score = evaluate(test_loader, model, config)
                # 5. 选择模型
                if best_res < bleu_score:
                    best_res = bleu_score
                    torch.save(state, config.best_checkpoint)
                torch.save(state, config.last_checkpoint)
                fw.write('Validation@epoch, %d, step, %d, BLEU-4=%.2f\n' % 
                    (epoch, i+1, bleu_score))
                fw.flush()
                print('Validation@epoch, %d, step, %d, BLEU-4=%.2f' % 
                    (epoch, i+1, bleu_score))
    checkpoint = torch.load(config.best_checkpoint)
    model = checkpoint['model']
    bleu_score = evaluate(test_loader, model, config)
    print("Evaluate on the test set with the model that has the best performance on the validation set")
    print('Epoch: %d, BLEU-4=%.2f' % 
        (checkpoint['epoch'], bleu_score))
    fw.write('Epoch: %d, BLEU-4=%.2f' % 
        (checkpoint['epoch'], bleu_score))
    fw.close()

if __name__ == '__main__':
    main()

# 这段代码完成训练，最后一行会输出在验证集上表现最好的模型在测试集上的结果，具体如下：
# 
# ```
# Epoch: 4, BLEU-4=0.23
# ```
# 
