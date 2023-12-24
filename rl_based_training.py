import torch
import torch.optim as optim
from torch.distributions import Categorical
from nltk.translate.bleu_score import sentence_bleu
from gridattn_image_caption import *

class RLBasedTrainer:
    def __init__(self, model, optimizer, bleu_score_fn):
        self.model = model
        self.optimizer = optimizer
        self.bleu_score_fn = bleu_score_fn

    def train(self, images, captions):
        self.model.train()
        self.optimizer.zero_grad()

        # Generate sentences using the current model parameters
        sampled_captions, log_probs = self.model.sample(images)
        greedy_captions = self.model.greedy_search(images)

        # Compute rewards
        sampled_rewards = self.compute_rewards(sampled_captions, captions)
        greedy_rewards = self.compute_rewards(greedy_captions, captions)

        # Compute advantages
        advantages = sampled_rewards - greedy_rewards

        # Compute loss
        loss = -log_probs * advantages
        loss = loss.sum()

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_rewards(self, generated_captions, target_captions):
        rewards = []
        for gen_cap, tar_cap in zip(generated_captions, target_captions):
            reward = self.bleu_score_fn(gen_cap, tar_cap)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float)

# Initialize the model and the optimizer
model = ARCTIC()
optimizer = optim.Adam(model.parameters(), lr=0.001)
data_dir = './data/flickr8k/'
vocab_path = './data/flickr8k/vocab.json'
train_loader, test_loader = mktrainval(data_dir, vocab_path, 8)

# Initialize the trainer
trainer = RLBasedTrainer(model, optimizer, sentence_bleu)
# Train the model
for images, captions in train_loader:
    loss = trainer.train(images, captions)
    print('Loss:', loss)