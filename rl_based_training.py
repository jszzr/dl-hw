import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from gridattn_image_caption import *
from Vit_GRU import *
from misc import *


rl_crit = RewardCriterion()


def main():
    # 设置模型超参数和辅助变量
    # freeze_support()
    config = Namespace(
        max_len = 120,
        captions_per_image = 1,
        batch_size = 16,
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    optimizer = get_optimizer(model, config)
    model.to(device)

    # 环境


    for epoch in range(start_epoch, config.num_epochs):
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            
            gen_result, sample_logprobs = model.sample(imgs)
            # gen_result = [sublist[1:] for sublist in gen_result]
            # gen_result = nn.utils.rnn.pad_sequence([torch.tensor(sublist) for sublist in gen_result], batch_first=True, padding_value=-1).to(device)
            # sample_logprobs = nn.utils.rnn.pad_sequence([torch.tensor(sublist) for sublist in sample_logprobs], batch_first=True, padding_value=0).to(device)

            reward = get_self_critical_reward(model, imgs, gen_result, caps, config.batch_size)
            loss = rl_crit(sample_logprobs, gen_result, torch.from_numpy(reward).float().cuda())
            
            loss.backward()

            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()




if __name__ == '__main__':
    main()
