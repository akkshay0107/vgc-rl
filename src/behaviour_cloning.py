from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ReplayDataset(Dataset):
    def __init__(self, replays_dir: str):
        self.samples = []
        path = Path(replays_dir)

        # Load and flatten immediately
        for replay_file in sorted(path.glob("*.replay")):
            try:
                episode_data = torch.load(replay_file)
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
            "mask": sample["mask"],
            # converting numpy array action to a tensor
            "action": torch.tensor(sample["action"], dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = ReplayDataset("./replays")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    for batch in loader:
        obs = batch["obs"]  # (B, *OBS_DIM)
        mask = batch["mask"]  # (B, 2, ACT_SZ)
        target = batch["action"]  # (B, 2)

        # TODO: implement a behaviour cloning loop here
        pass
