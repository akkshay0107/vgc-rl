from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

        # Convert to tensors here and Flatten all tensors to 1D
        return {
            "obs": sample["obs"].to(torch.float32).reshape(-1),
            "mask": sample["mask"].to(torch.bool).reshape(-1),
            # converting numpy array action to a tensor
            "action": torch.tensor(sample["action"], dtype=torch.long).reshape(-1),
        }

class VGCBehaviorCloningModel(nn.Module):
    def __init__(self, obs_dim, action_size):
        super().__init__()
        action_size = action_size // 2  # Assuming action_size includes both pokemon
        
        # Simple feedforward network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Output heads for action logits
        self.p1_head = nn.Linear(128, action_size)
        self.p2_head = nn.Linear(128, action_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, obs, mask):
        x = self.net(obs)
        m1, m2 = mask[:, :47], mask[:, 47:]
        
        p1_logits = self.p1_head(x)
        p2_logits = self.p2_head(x)

        p1_masked_logits = p1_logits.masked_fill(~m1, float("-inf"))
        p2_masked_logits = p2_logits.masked_fill(~m2, float("-inf"))

        p1_log_probs = self.log_softmax(p1_masked_logits)
        p2_log_probs = self.log_softmax(p2_masked_logits)

        return p1_log_probs, p2_log_probs

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
    obs_dim = sample["obs"].shape[0]
    num_actions = sample["mask"].shape[-1]

    # Initialize model, loss, optimizer
    model = VGCBehaviorCloningModel(obs_dim, num_actions)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

    # Run training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            # Get batch data
            obs = batch["obs"]
            mask = batch["mask"]
            target = batch["action"]
            
            optimizer.zero_grad()
            output = model(obs, mask)  # Assuming single action for simplicity

            # Unpack outputs
            p1_preds = output[0]
            p2_preds = output[1]

            p1_correct_moves = target[:, 0]
            p2_correct_moves = target[:, 1]

            p1_loss = criterion(p1_preds, p1_correct_moves)
            p2_loss = criterion(p2_preds, p2_correct_moves)
            loss = p1_loss + p2_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, p1_preds = torch.max(p1_preds, 1)
            _, p2_preds = torch.max(p2_preds, 1)

            train_total += target.shape[0] * 2 # two pokemon actions
            train_correct += (p1_preds == p1_correct_moves).sum().item()
            train_correct += (p2_preds == p2_correct_moves).sum().item()

        print(f"Training accuracy: {train_correct / train_total}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                obs = batch["obs"]
                mask = batch["mask"]
                target = batch["action"]

                output = model(obs, mask) # Assuming single action for simplicity

                # Unpack outputs
                p1_preds = output[0]
                p2_preds = output[1]

                p1_correct_moves = target[:, 0]
                p2_correct_moves = target[:, 1]

                p1_loss = criterion(p1_preds, p1_correct_moves)
                p2_loss = criterion(p2_preds, p2_correct_moves)
                loss = p1_loss + p2_loss

                val_loss += loss.item()

                _, p1_preds = torch.max(p1_preds, 1)
                _, p2_preds = torch.max(p2_preds, 1)

                val_total += target.shape[0] * 2 # two pokemon actions
                val_correct += (p1_preds == p1_correct_moves).sum().item()
                val_correct += (p2_preds == p2_correct_moves).sum().item()
        
        print(f"Validation accuracy: {val_correct / val_total}")
        print("")

    return model

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