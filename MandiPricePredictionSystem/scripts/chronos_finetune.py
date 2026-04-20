import pandas as pd
import torch
import torch.optim as optim
from chronos import ChronosPipeline
from tqdm import tqdm
import math

def finetune_chronos():
    print("=== Amazon Chronos Fine-Tuning Optimisation ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Pre-trained Chronos into memory
    print("Loading T5-Mini Base Architecture...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-mini",
        device_map=device,
        torch_dtype=torch.float32, # Safe for gradients
    )
    
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    model.train()
    
    # 2. Extract our Agmarknet trajectories
    print("Isolating spatial trajectories...")
    df = pd.read_csv("dl_30_features_data.csv")
    df = df.sort_values(["Mandi", "Commodity", "date"])
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 3
    context_length = 60
    prediction_length = 14
    
    print(f"Executing Looped Fine-Tuning across dataset isolated lengths (Epochs: {epochs})...")
    
    # We will extract highest volume top 100 crops so it finishes fast but learns the unique variance
    top_groups = df.groupby(["Mandi", "Commodity"]).size().nlargest(100).index
    
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        
        for name in top_groups:
            crop_df = df[(df["Mandi"] == name[0]) & (df["Commodity"] == name[1])]
            prices = torch.tensor(crop_df["ModalPrice"].values, dtype=torch.float32)
            
            if len(prices) < context_length + prediction_length:
                continue
                
            # Slide over trajectory
            for i in range(0, len(prices) - context_length - prediction_length, 30):
                context = prices[i : i+context_length].unsqueeze(0)
                target = prices[i+context_length : i+context_length+prediction_length].unsqueeze(0)
                
                # Tokenize on CPU
                input_ids, attention_mask, scale = tokenizer.context_input_transform(context)
                target_ids, target_mask, _ = tokenizer.context_input_transform(target)
                
                # Ensure device match
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                target_ids = target_ids.to(device)
                
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_ids
                )
                
                loss = outputs.loss
                if not math.isnan(loss.item()):
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1
                    
        print(f"Epoch {epoch+1}/{epochs} | Avg Volatility Loss: {(epoch_loss/max(1, batches)):.4f}")
        
    print("Fine-Tuned Chronos model optimized successfully!")

if __name__ == "__main__":
    finetune_chronos()
