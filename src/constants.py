# Shared constants

# Observation dimensions: (sequence_length, feature_dim)
# 1 field row + 1 info row + 3 tokens * 12 pokemon = 38
TINYBERT_SZ = 624
EXTRA_SZ = 28
OBS_DIM = (38, TINYBERT_SZ)

# Action space parameters
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 1
ACT_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * (NUM_GIMMICKS + 1)
