import torch
import torch.nn.functional as F
import torch.nn as nn

class EnhancerAttention(nn.Module):
    def __init__(self, embed_size):
        super(EnhancerAttention, self).__init__()
        self.embed_size = embed_size
        # Define linear transformations for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.classifier_head = nn.Linear(embed_size, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute the dot products between Q and K, then scale by the square root of the key dimension
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # Softmax to normalize scores, producing attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute the final output as weighted values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x, mask=None):
        
        # add the [cls] token to the end of the input sequence
        cls = self.cls_token.expand(x.size(0), 1, -1)
        x = torch.cat([x, cls], dim=1)

        # need to extend mask by 1
        if mask is not None:
            cls_mask = torch.ones(x.size(0), 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, cls_mask], dim=1)  # (B, L+1)
            

        # Generate Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention using our scaled dot-product function
        out, _ = self.scaled_dot_product_attention(Q, K, V, mask) # (batch, N, embed_dim)
        
        cls_out = out[:, -1, :].squeeze(1) # (batch, embed_dim)

        # Enhancer probability for each input sequence
        logits = self.classifier_head(cls_out).squeeze(-1) # (batch) 

        return logits