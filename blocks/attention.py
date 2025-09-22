from deepen.backend import active_backend as bx
from deepen.core.tensor import Tensor
from deepen.layers import Layer, Linear, Dropout

_bx = bx() # backend singleton

class MultiHeadAttention(Layer):
    """
    Minimal scaffold for Multi-Head Attention.

    Implementations to fill in:
    - Set head_dim = d_model // num_heads
    - Initialize Linear layers: w_q, w_k, w_v, w_o
    - Optional: attn_drop when dropout > 0
    - Forward pass: project to Q/K/V, split heads, compute scaled dot-product attention,
      apply masks, softmax, dropout, combine heads, output projection.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = None  # set to d_model // num_heads
        self.causal = causal
        self.dropout_p = dropout

        # Projections (set to Linear(d_model, d_model))
        self.w_q: Linear | None = None
        self.w_k: Linear | None = None
        self.w_v: Linear | None = None
        self.w_o: Linear | None = None

        # Dropout on attention weights
        self.attn_drop: Dropout | None = None

    def _shape_qkv(self, Q: Tensor, K: Tensor, V: Tensor):
        """
        Reshape and transpose to split heads.
        Expected steps:
        - reshape Q/K/V from [B, T, D] -> [B, T, H, head_dim]
        - transpose to [B, H, T, head_dim]
        Return: (Qh, Kh, Vh)
        """
        Qh = None
        Kh = None
        Vh = None
        return Qh, Kh, Vh

    def _apply_masks(self, scores: Tensor, key_padding_mask: Tensor | None):
        """
        Apply padding/causal masks to attention scores.
        Expected steps:
        - If key_padding_mask [B, Tk] provided, expand to [B, 1, 1, Tk] and add large negative where masked.
        - If self.causal, apply an upper-triangular mask over [Tq, Tk].
        Return masked scores.
        """
        masked_scores = None
        return masked_scores

    def forward(self, X: Tensor, context: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Forward scaffold:
        - src = context if provided else X
        - Q = w_q(X); K = w_k(src); V = w_v(src)
        - Qh, Kh, Vh = _shape_qkv(Q, K, V)
        - scores = Qh.matmul(Kh.transpose(axes=(0, 1, 3, 2))) ; scale by 1/sqrt(head_dim)
        - scores = _apply_masks(scores, key_padding_mask)
        - attn = scores.softmax(axes=-1)
        - if attn_drop: attn = attn_drop(attn)
        - ctx = attn.matmul(Vh)
        - merged = transpose/reshape back to [B, T, D]
        - out = w_o(merged)
        - return out
        """
        src = None
        Q = None
        K = None
        V = None

        Qh, Kh, Vh = None, None, None

        scores = None
        attn = None
        ctx = None
        merged = None
        out = None
        return out


