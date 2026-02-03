from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from policy import PolicyNet

class ReplayDataset(Dataset):
    def __init__(self, replays_dir: str):
        self.samples = []
        path = Path(replays_dir)

        # Load and flatten immediately
        for replay_file in sorted(path.glob("*.replay")):
            try:
                episode_data = torch.load(replay_file, weights_only=False)
                # episode_data is a list of dicts (check terminal_player)
                self.samples.extend(episode_data)
            except Exception as e:
                print(f"Could not load file {replay_file}: {e}")

        print(f"Loaded {len(self.samples)} samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            "obs": sample["obs"],
            "mask": sample["mask"].to(torch.bool),
            # converting numpy array action to a tensor
            "action": torch.tensor(sample["action"], dtype=torch.long),
        }

def train_behavior_cloning(replays_path):
    # Configs
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-4

    # Load dataset
    dataset = ReplayDataset(replays_path)

    if len(dataset) == 0:
        print("No data available for training.")
        return
    
    '''
        Create train/val split.
        Episode data is consisting of ~700 random files for 100 episodes.
        Probably in cronological order with each file representing a turn.
        So we can just do a simple split as random split would cause data leakage to occur.
    '''
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(
        dataset, list(range(train_size))
    )
    val_dataset = torch.utils.data.Subset(
        dataset, list(range(train_size, len(dataset)))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Get dimensions
    sample = dataset[0]

    # Initialize model, loss, optimizer
    policy = PolicyNet()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, eps=1e-5)  

    # Run training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            # Get batch data
            obs = batch["obs"]
            mask = batch["mask"]
            target = batch["action"]
            
            optimizer.zero_grad()
            
            log_prob, _, _ = policy.evaluate_actions(obs, target, mask)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                logits = policy.get_policy_masked_logits(obs, target, mask)
                _, preds = torch.max(logits, dim=2)
                train_total += target.shape[0] * 2 # two pokemon actions
                train_correct += (preds == target).sum().item()


        print(f"Training accuracy: {train_correct / train_total}")

        policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"]
                mask = batch["mask"]
                target = batch["action"]
                
                optimizer.zero_grad()
                
                log_prob, _, _ = policy.evaluate_actions(obs, target, mask)
                loss = -log_prob.mean()
                val_loss += loss.item()

                logits = policy.get_policy_masked_logits(obs, target, mask)
                _, preds = torch.max(logits, dim=2)
                val_total += target.shape[0] * 2 # two pokemon actions
                val_correct += (preds == target).sum().item()
    
        print(f"Validation accuracy: {val_correct / val_total}")
        print("")

    return policy

if __name__ == "__main__":
    PATH_VAR = "C:/Users/oprea/Projects/vgc-rl/replays" # replace with actual path
    dataset = ReplayDataset(PATH_VAR)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    '''
    for batch in loader:
        obs = batch["obs"]  # (B, *OBS_DIM)
        mask = batch["mask"]  # (B, 2, ACT_SZ)
        target = batch["action"]  # (B, 2)

        # print("obs", obs.shape, obs.dtype)
        # print("mask", mask.shape, mask.dtype)
        # print("target", target.shape, target.dtype)

        # TODO: implement a behaviour cloning loop here
        pass
    '''
    if len(dataset) > 0:
        # Implementation of the training loop here
        model = train_behavior_cloning(PATH_VAR)
    else:
        print("No data available for training.")