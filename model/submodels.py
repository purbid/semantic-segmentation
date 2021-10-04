import torch
import random
from torch import nn
import torch.nn.functional as F
from torchtext.legacy.data import Field, BucketIterator
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

'''
    Sentences are represented as a sequence of word tokens. These sequences are converted into fixed-length sentence embeddings by passing them through a Bi-LSTM. The hidden embedding at the last word is considered to represent the entire sentence.   
'''
class LSTM_Sentence_Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, drop = 0.5, device = 'cpu'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden = None
        self.device = device        
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
    
    def forward(self, sentences, sent_lengths):
        ## sentences: tensor[batch_size, max_sent_len]
        ## sent_lengths: list[batch_size]        
        
        # initialize hidden state
        batch_size = sentences.shape[0]
        self.hidden = self.init_hidden(batch_size)
        
        # convert word tokens to word embeddings
        ## tensor[batch_size, max_sent_len] --> tensor[batch_size, max_sent_len, emb_dim] 
        x = self.emb(sentences)
        x = self.dropout(x)
        
        # generate sentence embedding from sequence of word embeddings
        ## tensor[batch_size, max_sent_len, emb_dim] --> tensor[2, batch_size, hidden_dim // 2]
        x = nn.utils.rnn.pack_padded_sequence(x, list(sent_lengths), batch_first = True, enforce_sorted = False)
        _, (x, _) = self.lstm(x, self.hidden)

        ## tensor[2, batch_size, hidden_dim // 2] --> tensor[batch_size, hidden_dim]        
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return x
    

'''
    A Bi-LSTM is used to generate feature vectors for each sentence from the sentence embeddings. The feature vectors are actually context-aware sentence embeddings. These are then fed to a feed-forward network to obtain emission scores for each class at each sentence.
'''
class LSTM_Emitter(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop = 0.5, device = 'cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        # print(n_tags)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.hidden = None
        self.device = device
        
    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device), torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))
    
    def forward(self, sequences):
        ## sequences: tensor[batch_size, max_seq_len, emb_dim]
        
        # initialize hidden state
        self.hidden = self.init_hidden(sequences.shape[0])
        
        # generate context-aware sentence embeddings (feature vectors)
        ## tensor[batch_size, max_seq_len, emb_dim] --> tensor[batch_size, max_seq_len, hidden_dim]
        x, self.hidden = self.lstm(sequences, self.hidden)
        x = self.dropout(x)
        # print(x.shape)

        # generate emission scores for each class at each sentence
        # tensor[batch_size, max_seq_len, hidden_dim] --> tensor[batch_size, max_seq_len, n_tags]
        x = self.hidden2tag(x)
        return x
    
'''
    A linear-chain CRF is fed with the emission scores at each sentence, and it finds out the optimal sequence of tags by learning the transition scores.
'''
class CRF(nn.Module):    
    def __init__(self, n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx = None):
        super().__init__()
        
        self.n_tags = n_tags
        self.SOS_TAG_IDX = sos_tag_idx
        self.EOS_TAG_IDX = eos_tag_idx
        self.PAD_TAG_IDX = pad_tag_idx
        
        self.transitions = nn.Parameter(torch.empty(self.n_tags, self.n_tags))
        self.init_weights()
        
    def init_weights(self):
        # initialize transitions from random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        
        # enforce constraints (rows = from, cols = to) with a big negative number.
        # exp(-1000000) ~ 0
        
        # no transitions to SOS
        self.transitions.data[:, self.SOS_TAG_IDX] = -1000000.0
        # no transition from EOS
        self.transitions.data[self.EOS_TAG_IDX, :] = -1000000.0
        
        if self.PAD_TAG_IDX is not None:
            # no transitions from pad except to pad
            self.transitions.data[self.PAD_TAG_IDX, :] = -1000000.0
            self.transitions.data[:, self.PAD_TAG_IDX] = -1000000.0
            # transitions allowed from end and pad to pad
            self.transitions.data[self.PAD_TAG_IDX, self.EOS_TAG_IDX] = 0.0
            self.transitions.data[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0
            
    def forward(self, emissions, tags, mask = None):
        ## emissions: tensor[batch_size, seq_len, n_tags]
        ## tags: tensor[batch_size, seq_len]
        ## mask: tensor[batch_size, seq_len], indicates valid positions (0 for pad)
        return -self.log_likelihood(emissions, tags, mask = mask)
    
    def log_likelihood(self, emissions, tags, mask = None):                   
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
            
        scores = self._compute_scores(emissions, tags, mask = mask)
        partition = self._compute_log_partition(emissions, mask = mask)
        return torch.sum(scores - partition)
    
    # find out the optimal tag sequence using Viterbi Decoding Algorithm
    def decode(self, emissions, mask = None):      
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
        '''
        emissions => document_size*numberofsentences*tags
        
        '''
        scores, sequences = self._viterbi_decode(emissions, mask)
        print(len(sequences), len(scores), scores)
        exit()
        return scores, sequences
    
    def _compute_scores(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        # scores = torch.zeros(batch_size).cuda()
        scores = torch.zeros(batch_size).to(device)
        
        # save first and last tags for later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()
        
        # add transition from SOS to first tags for each sample in batch
        t_scores = self.transitions[self.SOS_TAG_IDX, first_tags]
        
        # add emission scores of the first tag for each sample in batch
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores
        
        # repeat for every remaining word
        for i in range(1, seq_len):
            
            is_valid = mask[:, i]
            prev_tags = tags[:, i - 1]
            curr_tags = tags[:, i]
            
            e_scores = emissions[:, i].gather(1, curr_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[prev_tags, curr_tags]
                        
            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            
            scores += e_scores + t_scores
            
        # add transition from last tag to EOS for each sample in batch
        scores += self.transitions[last_tags, self.EOS_TAG_IDX]
        return scores
    
    # compute the partition function in log-space using forward algorithm
    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        
        # in the first step, SOS has all the scores
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        
        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1) 
            
            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            
            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            
            scores = e_scores + t_scores + a_scores
            new_alphas = torch.logsumexp(scores, dim = 1)
            
            # set alphas if the mask is valid, else keep current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas
            
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)
        
        # return log_sum_exp
        return torch.logsumexp(end_scores, dim = 1)
    
    # return a list of optimal tag sequence for each example in the batch
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape

        ##### 32, 658, 10
        #### 1, 8, 10


        # in the first iteration, SOS will have all the scores and then, the max
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        
        backpointers = []
        
        for i in range(1, seq_len):
            ## tensor[batch_size, n_tags] -> tensor[batch_size, 1, n_tags]
            e_scores = emissions[:, i].unsqueeze(1) 
            
            ## tensor[n_tags, n_tags] -> tensor[batch_size, n_tags, n_tags]
            t_scores = self.transitions.unsqueeze(0)
            
            ## tensor[batch_size, n_tags] -> tensor[batch_size, n_tags, 1]
            a_scores = alphas.unsqueeze(2)
            
            scores = e_scores + t_scores + a_scores
            
            # find the highest score and tag, instead of log_sum_exp
            max_scores, max_score_tags = torch.max(scores, dim = 1)
            
            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            
            backpointers.append(max_score_tags.t())
            
        # add scores for final transition
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences
    
    # auxiliary function to find the best path sequence for a specific example
    def _find_best_path(self, sample_id, best_tag, backpointers):
        ## backpointers: list[tensor[seq_len_i - 1, n_tags, batch_size]], seq_len_i is the length of the i-th sample of the batch
        
        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path



class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size


        #### embedding layer is not required since we use pretrained embeddings

        # embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        #### input from lstm-crf is Batch_size * Seq_len * embedding
        ### but the model expects seq_len * batch_size *embedding, hence permute

        x = x.permute(1,0,2)

        encoder_states, (hidden, cell) = self.rnn(x)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor, teacher_force_ratio=0.5):
        batch_size = input_tensor.shape[0]
        target_len = input_tensor.shape[1]
        target_vocab_size = 10
        print("forward of seq2seq")
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(input_tensor)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 32

encoder_embedding_size = 200
decoder_embedding_size = 200
hidden_size = 1024
input_size_encoder = 100
input_size_decoder = 100
output_size = 10
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0
# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
