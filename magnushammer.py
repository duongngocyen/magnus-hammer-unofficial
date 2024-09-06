from transformers import GPT2Model, GPT2Tokenizer, AdamW
import torch
import torch.nn as nn
from datasets import load_dataset
import random

# Load dataset
dataset = load_dataset("Simontwice/premise_selection_in_isabelle")

class MagnusHammer(nn.Module):
    def __init__(self, model_name='gpt2', hidden_dim=512):
        super(MagnusHammer, self).__init__()
        self.transformer = GPT2Model.from_pretrained(model_name)
        self.projection_q = nn.Linear(hidden_dim, hidden_dim)
        self.projection_k = nn.Linear(hidden_dim, hidden_dim)
        self.relevance_projection = nn.Linear(hidden_dim, 1)  # for RERANK stage
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        proof_embedding = self.projection_q(outputs[:, 0, :])  # Use the first token
        premise_embedding = self.projection_k(outputs[:, -1, :])  # Use the last token
        relevance_score = torch.sigmoid(self.relevance_projection(outputs[:, -1, :]))
        return proof_embedding, premise_embedding, relevance_score

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)

def info_nce_loss(proof_state, positive_premise, negative_premises, temperature=0.07):
    positive_similarity = cosine_similarity(proof_state, positive_premise)
    negative_similarities = torch.stack([cosine_similarity(proof_state, neg) for neg in negative_premises])
    numerator = torch.exp(positive_similarity / temperature)
    denominator = numerator + torch.sum(torch.exp(negative_similarities / temperature), dim=0)
    return -torch.log(numerator / denominator).mean()

def rerank_loss(model, positive_pairs, negative_pairs):
    loss_fn = nn.BCELoss()
    positive_scores = torch.cat([model(p)[2] for p in positive_pairs])
    negative_scores = torch.cat([model(n)[2] for n in negative_pairs])
    labels = torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)])
    all_scores = torch.cat([positive_scores, negative_scores])
    return loss_fn(all_scores, labels)

def train_magnushammer(model, select_data, rerank_data, num_steps=10000, T=1000):
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    for step in range(num_steps):
        proof_state, positive_premise, negative_premises = select_data.sample()
        proof_embedding, premise_embedding, _ = model(proof_state)
        select_loss = info_nce_loss(proof_embedding, premise_embedding, negative_premises)
        optimizer.zero_grad()
        select_loss.backward()
        optimizer.step()
        if step % T == 0:
            positive_pairs, negative_pairs = rerank_data.sample()
            rerank_loss_val = rerank_loss(model, positive_pairs, negative_pairs)
            optimizer.zero_grad()
            rerank_loss_val.backward()
            optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, SELECT Loss: {select_loss.item()}, RERANK Loss: {rerank_loss_val.item()}")

class SelectDataset:
    def __init__(self, dataset, num_negatives=3):
        self.dataset = dataset['train'] 
        self.num_negatives = num_negatives  # M negative premises
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    def sample(self):
        proof_state_data = random.sample(self.dataset, 1)
        
        proof_state = proof_state_data[0]['statement']
        positive_premise = proof_state_data[0]['premise_statement']
        
        # Negative premises: sample M premises that are not ground truth for the proof state
        negative_premises = [
            d['premise_statement'] for d in self.dataset 
            if d['premise_name'] != proof_state_data[0]['premise_name']
        ]
        negative_premises = random.sample(negative_premises, self.num_negatives)

        proof_state_tensor = self.tokenizer(proof_state, return_tensors='pt')['input_ids']
        positive_premise_tensor = self.tokenizer(positive_premise, return_tensors='pt')['input_ids']
        negative_premise_tensors = [self.tokenizer(neg, return_tensors='pt')['input_ids'] for neg in negative_premises]

        return proof_state_tensor, positive_premise_tensor, negative_premise_tensors

class RerankDataset:
    def __init__(self, dataset):
        self.dataset = dataset['train']
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def sample(self):
        data_point = random.sample(self.dataset, 1)[0]
        positive_pairs = [self.tokenizer(p, return_tensors='pt')['input_ids'] for p in data_point['positive_pairs']]
        negative_pairs = [self.tokenizer(n, return_tensors='pt')['input_ids'] for n in data_point['negative_pairs']]
        return positive_pairs, negative_pairs

# Initialize datasets
select_data = SelectDataset(dataset)
rerank_data = RerankDataset(dataset)

# Initialize the model
model = MagnusHammer(hidden_dim=512)

# Train the model
train_magnushammer(model, select_data, rerank_data, num_steps=10000, T=1000)