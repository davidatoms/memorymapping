# LLM Training Choreography Design 🧠🐙

## Goal

Connect the Python GPT-2 trainer (`octopus_gpt2_enhanced.py`) with the Elixir Chorex choreography for **formally verified, distributed LLM training coordination**.

---

## Architecture Options

### Option 1: Python ↔ Elixir Bridge (Recommended)

```
Python GPT-2 Trainer
       ↓
   [Port/NIFs]
       ↓
Elixir Chorex Coordinator
       ↓
Choreography: MainBrain + 8 Arms
```

**Benefits**:
- ✅ Keep existing Python trainer code
- ✅ Add formal coordination via Chorex
- ✅ Best of both worlds (PyTorch + Elixir distribution)

**Implementation**:
- Use **Erlang Ports** for Python ↔ Elixir communication
- Python sends batches via JSON
- Elixir choreography decides: skip/partial/proceed
- Python receives decision and processes accordingly

### Option 2: Pure Elixir with Nx/Axon

```
Pure Elixir GPT-2 Trainer
       ↓
Chorex Choreography
       ↓
Distributed across nodes
```

**Benefits**:
- ✅ Full choreographic coordination
- ✅ Native Elixir distribution
- ✅ No bridge complexity

**Challenges**:
- ❌ Need to rewrite trainer in Nx/Axon
- ❌ Less mature ML ecosystem than PyTorch

---

## Option 1 Design (Recommended)

### Python Side

```python
# octopus_gpt2_enhanced.py (modified)

import json
import subprocess

class ElixirCoordinator:
    def __init__(self):
        # Start Elixir choreography as subprocess
        self.elixir_process = subprocess.Popen(
            ["elixir", "chorex_coordinator.exs"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
    
    def get_decision(self, batch_data):
        """Send batch to Elixir, receive coordination decision"""
        # Format batch as JSON
        payload = {
            "id": batch_data["id"],
            "embedding": batch_data["embedding"].tolist(),
            "attention_pattern": batch_data.get("attention", [])
        }
        
        # Send to Elixir
        self.elixir_process.stdin.write(json.dumps(payload) + "\n")
        self.elixir_process.stdin.flush()
        
        # Receive decision
        response = self.elixir_process.stdout.readline()
        decision = json.loads(response)
        
        return decision  # e.g., {"action": "instant_skip", "energy": 0.1}
    
    def train_with_chorex(self):
        for batch in dataloader:
            # Get coordination decision from Elixir
            decision = self.get_decision(batch)
            
            if decision["action"] == "instant_skip":
                print(f"⚡ Instant skip! (saved {decision['energy']} energy)")
                continue
            elif decision["action"] == "skip":
                print(f"🔋 Skip (saved {decision['energy']} energy)")
                continue
            elif decision["action"] == "partial":
                # Partial forward pass
                outputs = model(**batch, output_attentions=True)
                loss = outputs.loss * decision["scale"]
            elif decision["action"] == "puzzle_mode":
                print(f"🧩 Puzzle mode! (deep exploration)")
                outputs = model(**batch, output_attentions=True)
                loss = outputs.loss
                # ... extra analysis ...
            else:  # proceed
                outputs = model(**batch)
                loss = outputs.loss
            
            # Standard training
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### Elixir Side

```elixir
# chorex_coordinator.exs

defmodule LLMTrainingCoordinator do
  def start do
    # Read from stdin (Python sends JSON)
    # Dispatch to choreography
    # Write decision to stdout (JSON)
    
    loop()
  end
  
  defp loop do
    case IO.gets("") do
      :eof -> :ok
      line ->
        # Parse JSON batch
        batch = Jason.decode!(line)
        
        # Run choreography
        actors = %{
          MainBrain => OctopusBrain.ChorexActors.MainBrainActor,
          Arm1 => OctopusBrain.ChorexActors.Arm1Actor,
          # ... all 8 arms
        }
        
        Chorex.start(OctopusBrain.ChorexCoordination.Chorex, actors, [batch])
        
        # Receive result
        decision = receive do
          {:chorex_return, MainBrain, result} -> result
        after
          5000 -> {:error, :timeout}
        end
        
        # Send decision back to Python
        IO.puts(Jason.encode!(decision))
        
        loop()
    end
  end
end

LLMTrainingCoordinator.start()
```

---

## Communication Protocol

### Python → Elixir (Batch Request)

```json
{
  "id": 1234,
  "embedding": [0.1, 0.2, ..., 0.768],
  "attention_pattern": [0.5, 0.8, ..., 0.3],
  "tokens": ["the", "quick", "brown", "fox"]
}
```

### Elixir → Python (Decision Response)

```json
{
  "action": "instant_skip" | "skip" | "partial" | "proceed" | "curious_exploration" | "puzzle_mode",
  "energy_cost": 0.1,
  "confidence": 0.95,
  "reason": "shell_match_found",
  "scale": 0.5  // for partial mode
}
```

---

## Choreography for LLM Training

```elixir
defchor [Coordinator, Worker1, Worker2, Worker3, Worker4] do
  def llm_training_step(Coordinator.(batch)) do
    # Broadcast batch to workers
    Coordinator.(batch) ~> Worker1.(data1)
    Coordinator.(batch) ~> Worker2.(data2)
    Coordinator.(batch) ~> Worker3.(data3)
    Coordinator.(batch) ~> Worker4.(data4)
    
    # Workers check shells instantly
    Worker1.check_shell(data1) ~> Coordinator.(match1)
    Worker2.check_shell(data2) ~> Coordinator.(match2)
    Worker3.check_shell(data3) ~> Coordinator.(match3)
    Worker4.check_shell(data4) ~> Coordinator.(match4)
    
    if Coordinator.any_match([match1, match2, match3, match4]) do
      instant_skip_decision(Coordinator.(batch))
    else
      # Compute gradients in parallel
      Worker1.forward_pass(data1) ~> Coordinator.(grad1)
      Worker2.forward_pass(data2) ~> Coordinator.(grad2)
      Worker3.forward_pass(data3) ~> Coordinator.(grad3)
      Worker4.forward_pass(data4) ~> Coordinator.(grad4)
      
      # Aggregate gradients
      aggregated = Coordinator.aggregate_gradients([grad1, grad2, grad3, grad4])
      
      # Broadcast updated weights
      Coordinator.(aggregated) ~> Worker1.(weights)
      Coordinator.(aggregated) ~> Worker2.(weights)
      Coordinator.(aggregated) ~> Worker3.(weights)
      Coordinator.(aggregated) ~> Worker4.(weights)
      
      # Workers update
      Worker1.update_weights(weights)
      Worker2.update_weights(weights)
      Worker3.update_weights(weights)
      Worker4.update_weights(weights)
      
      Coordinator.({:training_complete, aggregated})
    end
  end
end
```

---

## Implementation Steps

### Phase 1: Basic Bridge
1. Create `chorex_coordinator.exs` (stdin/stdout JSON handler)
2. Modify `octopus_gpt2_enhanced.py` to spawn Elixir subprocess
3. Test simple batch → decision flow

### Phase 2: Choreography Integration
1. Adapt existing choreography for LLM batches
2. Implement shell checking with embeddings
3. Test all decision modes (skip/partial/proceed/puzzle)

### Phase 3: Performance Optimization
1. Use **Erlang Ports** instead of subprocess (faster)
2. Batch multiple decisions together
3. Add async coordination (don't block Python)

### Phase 4: Distributed Workers
1. Extend choreography to multiple Python workers
2. Each worker has own Elixir coordinator
3. Chorex handles distributed gradient aggregation

---

## Example: Distributed Training

```elixir
defchor [MainCoordinator, Node1, Node2, Node3, Node4] do
  def distributed_training(MainCoordinator.(global_batch)) do
    # Split batch across nodes
    batch1 = MainCoordinator.split_batch(global_batch, 1)
    batch2 = MainCoordinator.split_batch(global_batch, 2)
    batch3 = MainCoordinator.split_batch(global_batch, 3)
    batch4 = MainCoordinator.split_batch(global_batch, 4)
    
    # Send to nodes
    MainCoordinator.(batch1) ~> Node1.(data)
    MainCoordinator.(batch2) ~> Node2.(data)
    MainCoordinator.(batch3) ~> Node3.(data)
    MainCoordinator.(batch4) ~> Node4.(data)
    
    # Each node runs local octopus choreography
    Node1.run_octopus_choreo(data) ~> MainCoordinator.(result1)
    Node2.run_octopus_choreo(data) ~> MainCoordinator.(result2)
    Node3.run_octopus_choreo(data) ~> MainCoordinator.(result3)
    Node4.run_octopus_choreo(data) ~> MainCoordinator.(result4)
    
    # Aggregate results
    final = MainCoordinator.aggregate([result1, result2, result3, result4])
    
    MainCoordinator.(final)
  end
end
```

This gives you:
- ✅ **4 machines**, each with its own **octopus brain** (1 main + 8 arms = 9 actors)
- ✅ **Main coordinator** orchestrates across machines
- ✅ **Formal verification** of entire distributed protocol
- ✅ **Shell recognition** happens locally on each machine (fast!)

---

## Performance Expectations

### Without Chorex (Traditional)
```
Batch 1: Process (100 energy)
Batch 2: Process (100 energy)  [similar to Batch 1, but no memory!]
Batch 3: Process (100 energy)
Batch 4: Process (100 energy)
---
Total: 400 energy
```

### With Chorex Octopus Brain
```
Batch 1: Process (100 energy) → Store shell
Batch 2: Instant skip (0.1 energy)  ⚡ Arms see shell!
Batch 3: Partial (25 energy)       🔋 Low attention
Batch 4: Instant skip (0.1 energy) ⚡ Shell match!
---
Total: 125.2 energy (68.7% savings!)
```

---

## Next Steps

1. **Create bridge prototype** (`chorex_coordinator.exs`)
2. **Test with sample batches** (JSON communication)
3. **Integrate with Python trainer**
4. **Benchmark energy savings**
5. **Scale to distributed workers**

---

## Files to Create

1. **`chorex_coordinator.exs`** - Elixir coordinator (stdin/stdout bridge)
2. **`octopus_gpt2_chorex.py`** - Python trainer with Elixir coordination
3. **`llm_training_choreo.ex`** - Choreography for LLM training
4. **`chorex_bridge_test.exs`** - Integration tests

---

## Alternative: Pure Elixir (Future Work)

If you want pure Elixir (no Python):

```elixir
# Pure Elixir GPT-2 trainer with Nx/Axon

defmodule OctopusGPT2 do
  use Axon
  
  def model do
    Axon.input("input_ids")
    |> Axon.embedding(50257, 768)
    |> transformer_blocks(12)
    |> Axon.dense(50257)
  end
  
  def train_with_choreo(model, data) do
    Enum.reduce(data, model, fn batch, model ->
      # Run choreography for coordination
      decision = run_chorex_choreo(batch)
      
      case decision.action do
        :instant_skip -> model  # No update!
        :skip -> model
        :partial -> partial_update(model, batch, decision.scale)
        _ -> full_update(model, batch)
      end
    end)
  end
end
```

---

## Summary

**Recommended path**: Python-Elixir bridge (Option 1)
- Keep PyTorch ecosystem
- Add Chorex formal coordination
- Incremental adoption

**Choreography benefits for LLM training**:
- ⚡ **Instant shell recognition** (huge savings!)
- 🧠 **Intelligent skipping** (avoid redundant computation)
- 🐙 **Distributed coordination** (scale to multiple nodes)
- ✅ **Formal verification** (no race conditions!)

**The octopus learns efficiently! 🐙📚**
