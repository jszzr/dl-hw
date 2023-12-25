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
      
        enc_captions = []

        for i, path in enumerate(imgpaths):
            captions = imcaps[i]
          
            for j, c in enumerate(captions):
                # 对文本描述进行编码
                enc_c = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in c] + [vocab['<end>']] 
                enc_captions.append(enc_c)
        # 合法性检查
        # assert len(imgpaths) == len(enc_captions)
      
        # 存储数据
        data = {'IMAGES': imgpaths, 
                'CAPTIONS': enc_captions}
        with open(os.path.join(output_folder, split + '_data.json'), 'w') as fw:
            json.dump(data, fw)
  



# create_dataset()

# 读取词典和验证集
with open('./data/flickr8k/vocab.json', 'r') as f:
    vocab = json.load(f)
vocab_idx2word = {idx:word for word,idx in vocab.items()}

with open('./data/flickr8k/train_captions.json', 'r') as f:
    train_data = json.load(f)

with open('./data/flickr8k/test_captions.json', 'r') as f:
    test_data = json.load(f)

print("词典长度：", len(vocab))
print("词典：", vocab)
print("训练集json：", len(train_data))
print("测试集json：", len(test_data))
print("训练集json+测试集json：", len(train_data)+len(test_data))
print("images数量：", len(os.listdir('./data/flickr8k/images')))




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

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader    


from torchvision.models import vit_b_16
class ImageEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
        self.conv_layer = nn.Conv2d(in_channels=768, out_channels=2048, kernel_size=2, stride=2)
        # self.grid_rep_extractor = model
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned
        
    def forward(self, images):
        self.grid_rep_extractor.to(images.device)
        self.conv_layer.to(images.device)
        out = self.grid_rep_extractor(images)
        out = self.conv_layer(out)
        # print(out.shape)
        return out




class AdditiveAttention(nn.Module):
    def  __init__(self, query_dim, key_dim, attn_dim):
        """
        参数：
            query_dim: 查询Q的维度
            key_dim: 键K的维度
            attn_dim: 注意力函数隐藏层表示的维度
        """
        super(AdditiveAttention, self).__init__()
        self.attn_w_1_q = nn.Linear(query_dim, attn_dim)
        self.attn_w_1_k = nn.Linear(key_dim, attn_dim)
        self.attn_w_2 = nn.Linear(attn_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, key_value):
        """
        Q K V：Q和K算出相关性得分，作为V的权重，K=V
        参数：
            query: 查询 (batch_size, q_dim)
            key_value: 键和值，(batch_size, n_kv, kv_dim)
        """
        # （2）计算query和key的相关性，实现注意力评分函数
        # -> (batch_size, 1, attn_dim)
        queries = self.attn_w_1_q(query).unsqueeze(1)
        # -> (batch_size, n_kv, attn_dim)
        keys = self.attn_w_1_k(key_value)
        # -> (batch_size, n_kv)
        attn = self.attn_w_2(self.tanh(queries+keys)).squeeze(2) 
        # （3）归一化相关性分数
        # -> (batch_size, n_kv)
        attn = self.softmax(attn) 
        # （4）计算输出
        # (batch_size x 1 x n_kv)(batch_size x n_kv x kv_dim)
        # -> (batch_size, 1, kv_dim)
        output = torch.bmm(attn.unsqueeze(1), key_value).squeeze(1)
        return output, attn




class AttentionDecoder(nn.Module):
    def __init__(self, image_code_dim, vocab_size, word_dim, attention_dim, hidden_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.attention = AdditiveAttention(hidden_size, image_code_dim, attention_dim)
        self.init_state = nn.Linear(image_code_dim, num_layers*hidden_size)
        self.rnn = nn.GRU(word_dim + image_code_dim, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        # RNN默认已初始化
        self.init_weights()
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, image_code, captions, cap_lens):
        """
        参数：
            image_code：图像编码器输出的图像表示 
                        (batch_size, image_code_dim, grid_height, grid_width)
        """
        # 将图像网格表示转换为序列表示形式 
        batch_size, image_code_dim = image_code.size(0), image_code.size(1)
        # -> (batch_size, grid_height, grid_width, image_code_dim) 
        image_code = image_code.permute(0, 2, 3, 1)  
        # -> (batch_size, grid_height * grid_width, image_code_dim)
        image_code = image_code.view(batch_size, -1, image_code_dim)
        # （1）按照caption的长短排序
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        captions = captions[sorted_cap_indices]
        image_code = image_code[sorted_cap_indices]
         #（2）初始化隐状态
        hidden_state = self.init_state(image_code.mean(axis=1))
        hidden_state = hidden_state.view(
                            batch_size, 
                            self.rnn.num_layers, 
                            self.rnn.hidden_size).permute(1, 0, 2)
        return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state

    def forward_step(self, image_code, curr_cap_embed, hidden_state):
        #（3.2）利用注意力机制获得上下文向量
        # query：hidden_state[-1]，即最后一个隐藏层输出 (batch_size, hidden_size)
        # context: (batch_size, hidden_size)
        context, alpha = self.attention(hidden_state[-1], image_code)
        #（3.3）以上下文向量和当前时刻词表示为输入，获得GRU输出
        x = torch.cat((context, curr_cap_embed), dim=-1).unsqueeze(0)
        # x: (1, real_batch_size, hidden_size+word_dim)
        # out: (1, real_batch_size, hidden_size)
        out, hidden_state = self.rnn(x, hidden_state)
        #（3.4）获取该时刻的预测结果
        # (real_batch_size, vocab_size)
        preds = self.fc(self.dropout(out.squeeze(0)))
        return preds, alpha, hidden_state
        
    def forward(self, image_code, captions, cap_lens):
        """
        参数：
            hidden_state: (num_layers, batch_size, hidden_size)
            image_code:  (batch_size, feature_channel, feature_size)
            captions: (batch_size, )
        """
        # （1）将图文数据按照文本的实际长度从长到短排序
        # （2）获得GRU的初始隐状态
        image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state \
            = self.init_hidden_state(image_code, captions, cap_lens)
        batch_size = image_code.size(0)
        # 输入序列长度减1，因为最后一个时刻不需要预测下一个词
        lengths = sorted_cap_lens.cpu().numpy() - 1
        # 初始化变量：模型的预测结果和注意力分数
        predictions = torch.zeros(batch_size, lengths[0], self.fc.out_features).to(captions.device)
        alphas = torch.zeros(batch_size, lengths[0], image_code.shape[1]).to(captions.device)
        # 获取文本嵌入表示 cap_embeds: (batch_size, num_steps, word_dim)
        cap_embeds = self.embed(captions)
        # Teacher-Forcing模式
        for step in range(lengths[0]):
            #（3）解码
            #（3.1）模拟pack_padded_sequence函数的原理，获取该时刻的非<pad>输入
            real_batch_size = np.where(lengths>step)[0].shape[0]
            preds, alpha, hidden_state = self.forward_step(
                            image_code[:real_batch_size], 
                            cap_embeds[:real_batch_size, step, :],
                            hidden_state[:, :real_batch_size, :].contiguous())            
            # 记录结果
            predictions[:real_batch_size, step, :] = preds
            alphas[:real_batch_size, step, :] = alpha
        return predictions, alphas, captions, lengths, sorted_cap_indices


class ARCTIC(nn.Module):
    def __init__(self, image_code_dim, vocab, word_dim, attention_dim, hidden_size, num_layers):
        super(ARCTIC, self).__init__()
        self.vocab = vocab
        self.encoder = ImageEncoder()
        self.decoder = AttentionDecoder(image_code_dim, len(vocab), word_dim, attention_dim, hidden_size, num_layers)

    def forward(self, images, captions, cap_lens):
        image_code = self.encoder(images)
        return self.decoder(image_code, captions, cap_lens)
    
    def generate_by_beamsearch(self, images, beam_k, max_len):
        vocab_size = len(self.vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        # 对每个图像样本执行束搜索
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k,1,1,1)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            cur_sent_embed = self.decoder.embed(cur_sents)[:,0,:]
            sent_lens = torch.LongTensor([1]*beam_k).to(device)
            # 获得GRU的初始隐状态
            image_code, cur_sent_embed, _, _, hidden_state = \
                self.decoder.init_hidden_state(image_code, cur_sent_embed, sent_lens)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k
            while True:
                preds, _, hidden_state = self.decoder.forward_step(image_code[:k], cur_sent_embed, hidden_state.contiguous())
                # -> (k, vocab_size)
                preds = nn.functional.log_softmax(preds, dim=1)
                # 对每个候选句子采样概率值最大的前k个单词生成k个新的候选句子，并计算概率
                # -> (k, vocab_size)
                probs = probs.repeat(1,preds.size(1)) + preds
                if cur_sents.size(1) == 1:
                    # 第一步时，所有句子都只包含开始标识符，因此，仅利用其中一个句子计算topk
                    values, indices = probs[0].topk(k, 0, True, True)
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
                    hidden_state = hidden_state[:,cur_sent_indices,:]
                    cur_sent_embed = self.decoder.embed(
                        cur_word_indices.view(-1,1))[:,0,:]
                # 句子太长，停止生成
                if cur_sents.size(1) >= max_len:
                    break
            if len(end_sents) == 0:
                # 如果没有包含结束符的句子，则选取第一个句子作为生成句子
                gen_sent = cur_sents[0].tolist()
            else: 
                # 否则选取包含结束符的句子中概率最大的句子
                gen_sent = end_sents[end_probs.index(max(end_probs))]
            texts.append(gen_sent)
        return texts


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
        predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
        return self.loss_fn(predictions, targets)
        


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




from nltk.translate.bleu_score import corpus_bleu

def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]

@torch.no_grad()
def evaluate(data_loader, model, config):
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
    model.train()
    return bleu4




# 设置模型超参数和辅助变量
if __name__ == '__main__':
    config = Namespace(
        max_len = 60,
        captions_per_image = 1,
        batch_size = 8,
        image_code_dim = 2048,
        word_dim = 512,
        hidden_size = 512,
        attention_dim = 512,
        num_layers = 1,
        encoder_learning_rate = 0.00001,
        decoder_learning_rate = 0.00005,
        num_epochs = 10,
        grad_clip = 5.0,
        alpha_weight = 1.0,
        evaluate_step = 900, # 每隔多少步在验证集上测试一次
        checkpoint = None, # 如果不为None，则利用该变量路径的模型继续训练
        best_checkpoint = './model/ARCTIC/best_flickr8k.ckpt', # 验证集上表现最优的模型的路径
        last_checkpoint = './model/ARCTIC/last_flickr8k.ckpt', # 训练完成时的模型的路径
        beam_k = 5
    )

    # 设置GPU信息
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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

    # 优化器
    optimizer = get_optimizer(model, config)

    # 将模型拷贝至GPU，并开启训练模式
    model.to(device)
    model.train()

    # 损失函数
    loss_fn = PackedCrossEntropyLoss().to(device)

    best_res = 0
    print("开始训练")
    fw = open('log.txt', 'w')

    for epoch in range(start_epoch, config.num_epochs):
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            optimizer.zero_grad()
            # 1. 读取数据至GPU
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # 2. 前馈计算
            predictions, alphas, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)
            # 3. 计算损失
            # captions从第2个词开始为targets
            loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
            # 重随机注意力正则项，使得模型尽可能全面的利用到每个网格
            # 要求所有时刻在同一个网格上的注意力分数的平方和接近1
            loss += config.alpha_weight * ((1. - alphas.sum(axis=1)) ** 2).mean()

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
            if (i+1) % config.evaluate_step == 0:
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


# 这段代码完成训练，最后一行会输出在验证集上表现最好的模型在测试集上的结果，具体如下：
# 
# ```
# Epoch: 4, BLEU-4=0.23
# ```
# 
