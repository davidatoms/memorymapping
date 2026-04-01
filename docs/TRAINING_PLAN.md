# Training & Testing Plan: Octopus Brain GPT-2

## Phase 1: Verification (15-30 minutes)

### Step 1.1: Test the Demo
```bash
cd /home/david/Desktop/MemoryMapping
source /home/david/Research/ArtificialIntelligence/modelsForUse/venvgpt2/bin/activate

# Run the built-in demo
python octopus_gpt2_trainer.py
```

**Expected output**:
- ✅ Model loads successfully
- ✅ Arms probe attention
- ✅ Main brain makes decisions
- ✅ Some batches skipped (~30-50%)
- ✅ Memory shells accumulate
- ✅ Energy savings reported

**Red flags**:
- ❌ CUDA out of memory → Add `device='cpu'` in code
- ❌ No batches skipped → Lower similarity thresholds
- ❌ All batches skipped → Raise similarity thresholds

### Step 1.2: Verify Memory Shell Behavior
```bash
# Create a test script
cat > test_memory_shells.py << 'EOF'
from octopus_gpt2_trainer import OctopusGPT2Trainer
import torch

trainer = OctopusGPT2Trainer(model_name='gpt2', device='cpu')

# Process same batch twice
batch = {
    'input_ids': torch.randint(0, 1000, (2, 32)),
    'attention_mask': torch.ones(2, 32)
}

optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=5e-5)

# First pass - should PROCEED
result1 = trainer.train_step(batch, optimizer)
print(f"First pass: {result1['decision']['decision']}")

# Second pass - should SKIP (redundant)
result2 = trainer.train_step(batch, optimizer)
print(f"Second pass: {result2['decision']['decision']}")

# Verify it learned
assert result1['decision']['decision'] in ['PROCEED', 'PARTIAL'], "First pass should process"
print("✅ Memory shells working!")
EOF

python test_memory_shells.py
```

---

## Phase 2: Baseline Comparison (1-2 hours)

### Step 2.1: Standard GPT-2 Training Baseline
```bash
cat > baseline_trainer.py << 'EOF'
#!/usr/bin/env python3
"""Standard GPT-2 training for comparison"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import time

# Load model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Create simple dataset
texts = [
    "The octopus is a fascinating creature.",
    "Machine learning transforms AI.",
    "Neural networks require data.",
] * 10

dataset = tokenizer(texts, truncation=True, padding='max_length', 
                   max_length=64, return_tensors='pt')

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }

dataloader = DataLoader(SimpleDataset(dataset), batch_size=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
start_time = time.time()
total_loss = 0
batches = 0

for epoch in range(2):
    for batch in dataloader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1

elapsed = time.time() - start_time

print(f"\n{'='*60}")
print("BASELINE TRAINING RESULTS")
print(f"{'='*60}")
print(f"Total batches: {batches}")
print(f"Batches processed: {batches} (100%)")
print(f"Average loss: {total_loss/batches:.4f}")
print(f"Training time: {elapsed:.2f}s")
print(f"Batches/sec: {batches/elapsed:.2f}")
print(f"{'='*60}\n")
EOF

python baseline_trainer.py > baseline_results.txt
```

### Step 2.2: Octopus Brain Training
```bash
cat > octopus_comparison.py << 'EOF'
#!/usr/bin/env python3
"""Octopus brain training for comparison"""
from octopus_gpt2_trainer import OctopusGPT2Trainer
import torch
from torch.utils.data import DataLoader
import time

# Same setup as baseline
trainer = OctopusGPT2Trainer(model_name='gpt2', num_arms=4)

texts = [
    "The octopus is a fascinating creature.",
    "Machine learning transforms AI.",
    "Neural networks require data.",
] * 10

dataset = trainer.tokenizer(texts, truncation=True, padding='max_length',
                            max_length=64, return_tensors='pt')

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }

dataloader = DataLoader(SimpleDataset(dataset), batch_size=2)
optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=5e-5)

# Training loop
start_time = time.time()

for epoch in range(2):
    for batch in dataloader:
        trainer.train_step(batch, optimizer)

elapsed = time.time() - start_time

print(f"\n{'='*60}")
print("OCTOPUS BRAIN TRAINING RESULTS")
print(f"{'='*60}")
trainer.print_statistics()
print(f"Training time: {elapsed:.2f}s")
print(f"Speedup: {1.0:.2f}x (baseline) vs current")
print(f"{'='*60}\n")
EOF

python octopus_comparison.py > octopus_results.txt
```

### Step 2.3: Compare Results
```bash
echo "COMPARISON REPORT" > comparison.txt
echo "==================" >> comparison.txt
echo "" >> comparison.txt
echo "BASELINE:" >> comparison.txt
cat baseline_results.txt >> comparison.txt
echo "" >> comparison.txt
echo "OCTOPUS BRAIN:" >> comparison.txt
cat octopus_results.txt >> comparison.txt

cat comparison.txt
```

**Metrics to verify**:
- [ ] Octopus is faster (time)
- [ ] Similar or better final loss
- [ ] Computational savings reported
- [ ] Memory shells accumulated

---

## Phase 3: Real Dataset Testing (2-4 hours)

### Step 3.1: Download Small Dataset
```bash
cat > download_wikitext.py << 'EOF'
from datasets import load_dataset
import os

print("Downloading WikiText-2 dataset...")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1000]')

# Save to file
with open('wikitext_sample.txt', 'w') as f:
    for example in dataset:
        if example['text'].strip():
            f.write(example['text'] + '\n')

print(f"✅ Saved {len(dataset)} examples to wikitext_sample.txt")
EOF

python download_wikitext.py
```

### Step 3.2: Train on Real Data
```bash
cat > train_wikitext.py << 'EOF'
#!/usr/bin/env python3
from octopus_gpt2_trainer import OctopusGPT2Trainer
import torch
from torch.utils.data import DataLoader, Dataset

class WikiTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        with open(file_path, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }

# Initialize trainer
trainer = OctopusGPT2Trainer(model_name='gpt2', num_arms=4)

# Load dataset
dataset = WikiTextDataset('wikitext_sample.txt', trainer.tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Setup optimizer
optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=5e-5)

# Train for 3 epochs
print("Starting training on WikiText-2...")
for epoch in range(3):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch + 1}/3")
    print(f"{'='*70}")
    
    for batch_idx, batch in enumerate(dataloader):
        result = trainer.train_step(batch, optimizer)
        
        if batch_idx % 10 == 0:
            status_emoji = {
                'SKIP': '🔋',
                'PARTIAL': '🔄', 
                'PROCEED': '⚡'
            }[result['decision']['decision']]
            
            print(f"Batch {batch_idx:4d} | {status_emoji} {result['decision']['decision']:7s} | "
                  f"Loss: {result['loss']:.4f}")
    
    trainer.print_statistics()

# Save model
trainer.model.save_pretrained('octopus_gpt2_trained')
trainer.tokenizer.save_pretrained('octopus_gpt2_trained')
print("\n✅ Model saved to octopus_gpt2_trained/")
EOF

python train_wikitext.py
```

---

## Phase 4: Validation & Quality Testing (1 hour)

### Step 4.1: Test Generation Quality
```bash
cat > test_generation.py << 'EOF'
#!/usr/bin/env python3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load trained model
model = GPT2LMHeadModel.from_pretrained('octopus_gpt2_trained')
tokenizer = GPT2Tokenizer.from_pretrained('octopus_gpt2_trained')

# Test prompts
prompts = [
    "The octopus",
    "Machine learning",
    "In the future",
    "Scientists discovered"
]

print("GENERATION QUALITY TEST")
print("="*70)

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 70)
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=3,
            temperature=0.8,
            do_sample=True,
            top_p=0.9
        )
    
    for i, seq in enumerate(output):
        text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"  {i+1}. {text}")

print("\n" + "="*70)
EOF

python test_generation.py
```

### Step 4.2: Perplexity Evaluation
```bash
cat > eval_perplexity.py << 'EOF'
#!/usr/bin/env python3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math

model = GPT2LMHeadModel.from_pretrained('octopus_gpt2_trained')
tokenizer = GPT2Tokenizer.from_pretrained('octopus_gpt2_trained')

# Test sentences
test_sentences = [
    "The cat sat on the mat.",
    "Scientists discovered a new species.",
    "Technology is advancing rapidly.",
]

print("PERPLEXITY EVALUATION")
print("="*70)

total_loss = 0
total_tokens = 0

for sentence in test_sentences:
    encodings = tokenizer(sentence, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss
    
    ppl = math.exp(loss.item())
    total_loss += loss.item()
    total_tokens += encodings['input_ids'].size(1)
    
    print(f"Sentence: {sentence}")
    print(f"Perplexity: {ppl:.2f}")
    print()

avg_ppl = math.exp(total_loss / len(test_sentences))
print(f"Average Perplexity: {avg_ppl:.2f}")
print("="*70)
EOF

python eval_perplexity.py
```

---

## Phase 5: Analysis & Optimization (2 hours)

### Step 5.1: Analyze Memory Shells
```bash
cat > analyze_shells.py << 'EOF'
#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from octopus_gpt2_trainer import OctopusGPT2Trainer

# Load trained model
trainer = OctopusGPT2Trainer(model_name='gpt2')

# Assuming you have training data
# ... run training first to populate memory shells ...

# Analyze shells
shells = trainer.main_brain.memory_shells.shells
print(f"Total memory shells: {len(shells)}")

if shells:
    # Analyze loss distribution
    losses = [s['loss'] for s in shells if s['loss'] is not None]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(losses, bins=30, color='steelblue', edgecolor='black')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Distribution of Losses in Memory Shells')
    plt.axvline(np.mean(losses), color='red', linestyle='--', 
                label=f'Mean: {np.mean(losses):.3f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.7)
    plt.xlabel('Shell Index (Time)')
    plt.ylabel('Loss')
    plt.title('Loss Over Time in Memory Shells')
    plt.axhline(np.mean(losses), color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('memory_shells_analysis.png', dpi=300)
    print("✅ Analysis saved to memory_shells_analysis.png")
EOF

python analyze_shells.py
```

### Step 5.2: Tune Thresholds
```bash
cat > tune_thresholds.py << 'EOF'
#!/usr/bin/env python3
"""Test different threshold configurations"""
from octopus_gpt2_trainer import OctopusGPT2Trainer
import torch

configurations = [
    {'reject': 0.85, 'partial': 0.5, 'name': 'Aggressive skipping'},
    {'reject': 0.92, 'partial': 0.6, 'name': 'Balanced (default)'},
    {'reject': 0.98, 'partial': 0.7, 'name': 'Conservative skipping'},
]

for config in configurations:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")
    
    trainer = OctopusGPT2Trainer(model_name='gpt2')
    trainer.main_brain.similarity_reject_threshold = config['reject']
    trainer.main_brain.similarity_partial_threshold = config['partial']
    
    # Run mini training
    # ... add training code here ...
    
    trainer.print_statistics()
EOF

python tune_thresholds.py
```

---

## Phase 6: Production Preparation (Optional)

### Step 6.1: Create Training Script
```bash
cat > train_production.py << 'EOF'
#!/usr/bin/env python3
"""Production-ready training script with checkpointing"""
import argparse
from octopus_gpt2_trainer import OctopusGPT2Trainer
import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

def main(args):
    # Initialize
    trainer = OctopusGPT2Trainer(
        model_name=args.model_name,
        device=args.device,
        num_arms=args.num_arms
    )
    
    # Load dataset
    # ... implement dataset loading ...
    
    # Training loop with checkpointing
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            result = trainer.train_step(batch, optimizer)
            
            # Save checkpoint every N batches
            if batch_idx % args.checkpoint_every == 0:
                checkpoint_path = Path(args.output_dir) / f'checkpoint_epoch{epoch}_batch{batch_idx}'
                trainer.model.save_pretrained(checkpoint_path)
                
                # Save memory shells
                shells_data = {
                    'shells': [
                        {
                            'embedding': s['embedding'].tolist(),
                            'attention_signature': s['attention_signature'],
                            'loss': s['loss']
                        }
                        for s in trainer.main_brain.memory_shells.shells
                    ],
                    'stats': trainer.stats
                }
                
                with open(checkpoint_path / 'memory_shells.json', 'w') as f:
                    json.dump(shells_data, f)
        
        trainer.print_statistics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_arms', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--output_dir', default='./checkpoints')
    args = parser.parse_args()
    
    main(args)
EOF

chmod +x train_production.py
```

---

## Success Criteria Checklist

### Functionality:
- [ ] Demo runs without errors
- [ ] Memory shells accumulate over time
- [ ] Batches are skipped when redundant
- [ ] Energy savings are positive
- [ ] Model generates coherent text

### Performance:
- [ ] Training time < baseline (or similar with better quality)
- [ ] Computational savings: 30-60%
- [ ] Final loss comparable to baseline
- [ ] Perplexity reasonable (<100 for small dataset)

### Efficiency:
- [ ] Skip rate increases from epoch 1 → 2 → 3
- [ ] Efficiency score > 50%
- [ ] Memory shell size reasonable (<100MB for 1000 samples)

### Quality:
- [ ] Generated text is coherent
- [ ] Model hasn't catastrophically forgotten
- [ ] Attention patterns make sense
- [ ] Memory shells capture meaningful patterns

---

## Quick Start Command Sequence

```bash
# 1. Setup
cd /home/david/Desktop/MemoryMapping
source /home/david/Research/ArtificialIntelligence/modelsForUse/venvgpt2/bin/activate

# 2. Verify installation
python -c "import torch; from transformers import GPT2LMHeadModel; print('✅ Ready')"

# 3. Run demo
python octopus_gpt2_trainer.py

# 4. Run comparison
python baseline_trainer.py
python octopus_comparison.py
cat comparison.txt

# 5. Train on real data
python download_wikitext.py
python train_wikitext.py

# 6. Evaluate
python test_generation.py
python eval_perplexity.py

# 7. Analyze
python analyze_shells.py
```

---

## Expected Timeline

| Phase | Task | Time | Output |
|-------|------|------|--------|
| 1 | Verification | 15-30 min | Demo works ✅ |
| 2 | Baseline comparison | 1-2 hours | Performance metrics |
| 3 | Real data training | 2-4 hours | Trained model |
| 4 | Quality testing | 1 hour | Generation samples |
| 5 | Analysis | 2 hours | Visualizations |
| **Total** | | **6-9 hours** | Production-ready system |

---

## Troubleshooting Guide

### Issue: "No batches skipped"
**Solution**: Lower thresholds
```python
trainer.main_brain.similarity_reject_threshold = 0.85
trainer.main_brain.similarity_partial_threshold = 0.5
```

### Issue: "All batches skipped"
**Solution**: Raise thresholds or check if shells are too generic
```python
trainer.main_brain.similarity_reject_threshold = 0.95
```

### Issue: "CUDA out of memory"
**Solution**: Use CPU or reduce batch size
```python
trainer = OctopusGPT2Trainer(device='cpu')
# or
dataloader = DataLoader(dataset, batch_size=1)
```

### Issue: "Quality degraded"
**Solution**: Process more batches fully
```python
trainer.main_brain.similarity_reject_threshold = 0.98  # Very strict
```

---

Ready to start? Begin with:
```bash
python octopus_gpt2_trainer.py
```

Then follow the phases in order! 🐙🚀
