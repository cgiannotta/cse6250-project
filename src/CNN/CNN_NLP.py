import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_2 = nn.Conv1d(in_channels = self.embed_dim, out_channels = num_filters[0], kernel_size = filter_sizes[0])
        self.conv1d_3 = nn.Conv1d(in_channels = self.embed_dim, out_channels = num_filters[1], kernel_size = filter_sizes[1])
        self.conv1d_4 = nn.Conv1d(in_channels = self.embed_dim, out_channels = num_filters[2], kernel_size = filter_sizes[2])
        self.conv1d_5 = nn.Conv1d(in_channels = self.embed_dim, out_channels = num_filters[3], kernel_size = filter_sizes[3])



        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """
        print('input_ids shape (batch, max_length): ', input_ids.shape)
        print('meow')

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()
        print('x_embed shape (batch, max_length, embed_dim): ', x_embed.shape)

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        print('x_reshaped shape (batch, embed_dim, max_len): ', x_reshaped.shape)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv1d_2 = F.relu(self.conv1d_2(x_reshaped))
        print('x_conv1d_2 shape (b, num_filters[i], L_out): ', x_conv1d_2.shape)
        x_conv1d_3 = F.relu(self.conv1d_3(x_reshaped))
        print('x_conv1d_3 shape (b, num_filters[i], L_out): ', x_conv1d_3.shape)
        x_conv1d_4 = F.relu(self.conv1d_4(x_reshaped))
        print('x_conv1d_4 shape (b, num_filters[i], L_out): ', x_conv1d_4.shape)
        x_conv1d_5 = F.relu(self.conv1d_5(x_reshaped))
        print('x_conv1d_5 shape (b, num_filters[i], L_out): ', x_conv1d_5.shape)

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_2 = F.max_pool1d(x_conv1d_2, kernel_size=x_conv1d_2.shape[2])
        print('x_pool_2 shape (b, num_filters[i], 1): ', x_pool_2.shape)
        x_pool_3 = F.max_pool1d(x_conv1d_3, kernel_size=x_conv1d_3.shape[2])
        print('x_pool_3 shape (b, num_filters[i], 1): ', x_pool_3.shape)
        x_pool_4 = F.max_pool1d(x_conv1d_4, kernel_size=x_conv1d_4.shape[2])
        print('x_pool_4 shape (b, num_filters[i], 1): ', x_pool_4.shape)
        x_pool_5 = F.max_pool1d(x_conv1d_5, kernel_size=x_conv1d_5.shape[2])
        print('x_pool_5 shape (b, num_filters[i], 1): ', x_pool_5.shape)
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc2 = x_pool_2.squeeze(dim=2)
        x_fc3 = x_pool_3.squeeze(dim=2)
        x_fc4 = x_pool_4.squeeze(dim=2)
        x_fc5 = x_pool_5.squeeze(dim=2)

        x_fc = torch.cat([x_fc2, x_fc3, x_fc4, x_fc5], dim=1)
        print('x_fc shape (b, sum(num_filters)): ', x_fc.shape)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))
        print('logits shape (batch, n_classes): ', logits.shape)

        return logits