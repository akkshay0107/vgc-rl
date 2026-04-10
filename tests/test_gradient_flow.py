import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "src"))

from policy import PolicyNet
from lookups import OBS_DIM, ACT_SIZE


def test_full_gradient_flow():
    print("Testing full gradient flow through Transformer to GRU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Increase nlayer and d_model to ensure some attention
    policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE, d_model=256, nhead=8, nlayer=4).to(
        device
    )

    obs = torch.randn(2, *OBS_DIM).to(device)

    # Forward pass
    policy.zero_grad()
    _, _, _, values, _ = policy(obs)
    values.sum().backward()

    # Check if HG type embedding has gradient
    # HG is type 13 in CLSReducer
    hg_type_grad = policy.reducer.type_emb.weight.grad[13]
    print(f"HG (Type 13) Type Embedding Grad Sum: {hg_type_grad.abs().sum().item():.6e}")

    # Check H1 type embedding (Type 10)
    h1_type_grad = policy.reducer.type_emb.weight.grad[10]
    print(f"H1 (Type 10) Type Embedding Grad Sum: {h1_type_grad.abs().sum().item():.6e}")

    # Check if any other type embedding has gradient
    cls_type_grad = policy.reducer.type_emb.weight.grad[0]
    print(f"CLS (Type 0) Type Embedding Grad Sum: {cls_type_grad.abs().sum().item():.6e}")

    # Check in_proj grad
    in_proj_grad = policy.reducer.in_proj.weight.grad
    print(f"In-Proj Grad Sum: {in_proj_grad.abs().sum().item():.6e}")

    # Check history_gru grad
    gru_grad = policy.reducer.history_gru.weight_ih.grad
    if gru_grad is not None and gru_grad.abs().sum() > 0:
        print(f"SUCCESS: Gradient reached GRU. Sum: {gru_grad.abs().sum().item():.6e}")
    else:
        print("FAILURE: No gradient reached GRU.")
        # Check if next_state[0] (which is hg_curr) has grad if we backward from it
        policy.zero_grad()
        _, _, _, _, next_state = policy(obs)
        next_state[0].sum().backward()
        gru_grad_direct = policy.reducer.history_gru.weight_ih.grad
        if gru_grad_direct is not None and gru_grad_direct.abs().sum() > 0:
            print(f"INFO: GRU parameters DO get gradients when backwarding from next_state[0]. Sum: {gru_grad_direct.abs().sum().item():.6e}")
        else:
            print("ERROR: GRU parameters STILL have no gradients when backwarding from next_state[0] directly!")


if __name__ == "__main__":
    test_full_gradient_flow()
