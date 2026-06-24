import torch
import torch.nn.functional as F

from harmonic_transformer import HarmonicTransformer, generate_synthetic_data

VOCAB_SIZE = 50
EMBED_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 2
FF_DIM = 64
MAX_SEQ_LEN = 24


def test_transformer_forward_and_backward():
    model = HarmonicTransformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_SEQ_LEN)
    data, targets = generate_synthetic_data(2, MAX_SEQ_LEN, VOCAB_SIZE)

    logits = model(data)
    assert logits.shape == (2, MAX_SEQ_LEN, VOCAB_SIZE)

    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
    loss.backward()
    # If we reach here without exception, backward succeeded


def test_rpe_device_transfer():
    model = HarmonicTransformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, MAX_SEQ_LEN)
    data, _ = generate_synthetic_data(1, MAX_SEQ_LEN, VOCAB_SIZE)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        data = data.to(device)
        logits = model(data)
        assert logits.device.type == 'cuda'
    else:
        logits = model(data)
        assert logits.device.type == 'cpu'
