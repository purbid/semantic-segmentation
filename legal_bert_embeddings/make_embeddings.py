import os
import torch

from transformers import AutoTokenizer, AutoModel

bert_model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased', output_hidden_states = True)

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
texts=['Purbid really enjoyed this movie a lot.']


def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT

    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.

    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids

    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids


    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model

    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids

    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token

    """

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        return outputs[1].squeeze().tolist()

        # Removing the first hidden state
    #     # The first state is the input state
    #     hidden_states = outputs[2][1:]
    #
    # # Getting embeddings from the final BERT layer
    # token_embeddings = hidden_states[-1]
    # print(token_embeddings.shape)


    #
    # # Collapsing the tensor into 1-dimension
    # token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # # Converting torchtensors to lists
    # list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]
    #
    # return list_token_embeddings



    print((input_embeddings.shape))





dir_path ="C:/Users/User/PycharmProjects/law_ai/semantic-segmentation/data/text/"
new_embeddings = "C:/Users/User/PycharmProjects/law_ai/semantic-segmentation/legal_bert_embeddings/pretrained_embeddings/"

for file_num, filename in enumerate(os.listdir(dir_path)):
    ###### iterate over each document
    if file_num%10==0:
        print("done for "+str(file_num))
    f = os.path.join(dir_path, filename)
    if os.path.isfile(f):
        sentence_embeddings = []
        with open(f) as file:
            lines = file.readlines()
            #######each sentence in single document
            for sent in lines:
                input = sent.split("\t")[0]
                label = sent.split("\t")[-1]
                tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(input, tokenizer)
                input_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, bert_model)
                input_embeddings.append(label)
                input_embeddings = [str(vec) for vec in input_embeddings]
                sentence_embeddings.append(" ".join(input_embeddings))
            file.close()


        f = open(new_embeddings+filename+'.txt', 'w')
        for embedding in sentence_embeddings:
            f.write(embedding)

        f.close()


