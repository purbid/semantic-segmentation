import torch
import time
import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class LegalBertDataSet(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)




'''
    Randomly shuffle the data and divide into batches
'''
def batchify(x, y, batch_size):
    idx = list(range(len(x)))
    random.shuffle(idx)
    
    # convert to numpy array for ease of indexing
    x = np.array(x)[idx]
    y = np.array(y)[idx]

    # print(len(x[0]))

    
    i = 0
    while i < len(x):
        # print("making_batch: "+str(i)+str(len(x)))
        j = min(i + batch_size, len(x))
        
        batch_idx = idx[i : j]
        batch_x = x[i : j]
        batch_y = y[i : j]
        
        yield batch_idx, batch_x, batch_y
        
        i = j


'''
    Perform a single training step by iterating over the entire training data once. Data is divided into batches.
'''
def train_step(model, opt, x, y, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example] (number of documents, label of each sentence inside that document)

    model.train()
    
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    idx = [] # example index

    
    for i, (batch_idx, batch_x, batch_y) in enumerate(batchify(x, y, batch_size)):

        ##batch_x : [batch_size, sentence_len, embeddingd_size] (number of documents, each sentence, sentence embedding)
        ##batch_y : [bach_size, sentence_len] (number of documents, label for each sentence)
        #### passing labels for teacher forcing


        pred = model(batch_x, batch_y)
        loss = model._loss(batch_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
     
        y_pred.extend(pred)
        y_gold.extend(batch_y)
        idx.extend(batch_idx)

    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    return total_loss / (i + 1), idx, y_gold, y_pred

'''
    Perform a single evaluation step by iterating over the entire training data once. Data is divided into batches.
'''
def val_step(model, x, y, batch_size):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    ## y: list[num_examples, sents_per_example]
    
    model.train()
    
    total_loss = 0
    y_pred = [] # predictions
    y_gold = [] # gold standard
    idx = [] # example index
    
    for i, (batch_idx, batch_x, batch_y) in enumerate(batchify(x, y, batch_size)):
        pred = model(batch_x, batch_y)
        loss = model._loss(batch_y)
               
        total_loss += loss.item()
     
        y_pred.extend(pred)
        y_gold.extend(batch_y)
        idx.extend(batch_idx)
        
    assert len(sum(y, [])) == len(sum(y_pred, [])), "Mismatch in predicted"
    return total_loss / (i + 1), idx, y_gold, y_pred

'''
    Infer predictions for un-annotated data
'''
def infer_step(model, x):
    ## x: list[num_examples, sents_per_example, features_per_sentence]
    
    model.eval()
    y_pred = model(x) # predictions
    
    return y_pred    


'''
    Report all metrics in format using sklearn.metrics.classification_report
'''
def statistics(data_state, tag2idx):
    idx, gold, pred = data_state['idx'], data_state['gold'], data_state['pred']
    
    rev_tag2idx = {v: k for k, v in tag2idx.items()}
    tags = [rev_tag2idx[i] for i in range(len(tag2idx)) if rev_tag2idx[i] not in ['<start>', '<end>', '<pad>']]
    
    # flatten out
    gold = sum(gold, [])
    pred = sum(pred, [])
    
    print(classification_report(gold, pred, target_names = tags, digits = 3))

'''
    Train the model on entire dataset and report loss and macro-F1 after each epoch.
'''
def learn(model, x, y, tag2idx, val_fold, args):
    samples_per_fold = args.dataset_size // args.num_folds

    val_idx = list(range(val_fold * samples_per_fold, val_fold * samples_per_fold + samples_per_fold))
    train_idx = list(range(val_fold * samples_per_fold)) + list(range(val_fold * samples_per_fold + samples_per_fold, args.dataset_size))
    
    train_x = [x[i] for i in train_idx]

    train_y = [y[i] for i in train_idx]
    
    val_x = [x[i] for i in val_idx]
    val_y = [y[i] for i in val_idx]

    
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.reg)
    
    print("{0:>7}  {1:>10}  {2:>6}  {3:>10}  {4:>6}".format('EPOCH', 'Tr_LOSS', 'Tr_F1', 'Val_LOSS', 'Val_F1'))
    print("-----------------------------------------------------------")
    
    best_val_f1 = 0.0
    
    model_state = {}
    data_state = {}
    
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        train_loss, train_idx, train_gold, train_pred = train_step(model, opt, train_x, train_y, args.batch_size)
        val_loss, val_idx, val_gold, val_pred = val_step(model, val_x, val_y, args.batch_size)
        # print(train_idx, train_gold, train_pred)
        # print(val_idx, val_gold, val_pred)
        # exit()
        train_f1 = f1_score(sum(train_gold, []), sum(train_pred, []), average = 'macro')
        val_f1 = f1_score(sum(val_gold, []), sum(val_pred, []), average = 'macro')

        if train_f1>0.996 and epoch > 110:
            break

        if epoch % args.print_every == 0:
            print("{0:7d}  {1:10.3f}  {2:6.3f}  {3:10.3f}  {4:6.3f}".format(epoch, train_loss, train_f1, val_loss, val_f1))


        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_state = {'epoch': epoch, 'arch': model, 'name': model.__class__.__name__, 'state_dict': model.state_dict(), 'best_f1': val_f1, 'optimizer' : opt.state_dict()}
            data_state = {'idx': val_idx, 'loss': val_loss, 'gold': val_gold, 'pred': val_pred}
            
    end_time = time.time()
    
    print("Dumping model and data ...", end = ' ')
    
    torch.save(model_state, args.save_path + 'model_state' + str(val_fold) + '.tar')
    
    with open(args.save_path + 'data_state' + str(val_fold) + '.json', 'w') as fp:
        json.dump(data_state, fp)
    
    print("Done")    

    print('Time taken:', int(end_time - start_time), 'secs')

    statistics(data_state, tag2idx)


'''
    Train the bert based model on entire dataset
'''

def learn_bert(model, x, y, tag2idx, args):


    #### add all data to one list, and labels to another
    data_list = []
    label_list = []

    labels_to_id_mapping = {
        'Facts': 0,
        'Argument': 1,
        'Ratio of the decision': 2,
        'Statute': 3,
        'Ruling by Present Court': 4,
        'Ruling by Lower Court': 5,
        'Precedent': 6
    }


    for doc in x:
        data_list.extend(doc)

    for doc in y:
        label_list.extend([labels_to_id_mapping[retoric] for retoric in doc])

    opt = torch.optim.Adam(model.parameters(), lr=args.bert_lr, weight_decay=args.reg)


    print("{0:>7}  {1:>10}  {2:>6}  {3:>10}  {4:>6}".format('EPOCH', 'Tr_LOSS', 'Tr_F1', 'Val_LOSS', 'Val_F1'))
    print("-----------------------------------------------------------")

    best_val_f1 = 0.0
    model_state = {}
    data_state = {}
    start_time = time.time()

    '''
    split datasets into training and testing.
    
    '''
    train_texts, val_texts, train_labels, val_labels = train_test_split(data_list, label_list , test_size=.25,
                                                                        random_state=2018)


    train_encodings = model.tokenizer(train_texts, truncation=True, max_length = 250,
    pad_to_max_length=True)
    val_encodings = model.tokenizer(val_texts, truncation=True, max_length = 250,
    pad_to_max_length=True)

    train_dataset = LegalBertDataSet(train_encodings, train_labels)
    val_dataset = LegalBertDataSet(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    model.to(device)
    model.train()


    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            opt.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    #     train_loss, train_idx, train_gold, train_pred = train_step(model, opt, train_x, train_y, args.batch_size)
    #     val_loss, val_idx, val_gold, val_pred = val_step(model, val_x, val_y, args.batch_size)
    #
    #     train_f1 = f1_score(sum(train_gold, []), sum(train_pred, []), average='macro')
    #     val_f1 = f1_score(sum(val_gold, []), sum(val_pred, []), average='macro')
    #
    #     if epoch % args.print_every == 0:
    #         print("{0:7d}  {1:10.3f}  {2:6.3f}  {3:10.3f}  {4:6.3f}".format(epoch, train_loss, train_f1, val_loss,
    #                                                                         val_f1))
    #
    #     if val_f1 > best_val_f1:
    #         best_val_f1 = val_f1
    #         model_state = {'epoch': epoch, 'arch': model, 'name': model.__class__.__name__,
    #                        'state_dict': model.state_dict(), 'best_f1': val_f1, 'optimizer': opt.state_dict()}
    #         data_state = {'idx': val_idx, 'loss': val_loss, 'gold': val_gold, 'pred': val_pred}
    #
    # end_time = time.time()
    #
    # print("Dumping model and data ...", end=' ')
    #
    # torch.save(model_state, args.save_path + 'model_state' + str(val_fold) + '.tar')
    #
    # with open(args.save_path + 'data_state' + str(val_fold) + '.json', 'w') as fp:
    #     json.dump(data_state, fp)
    #
    # print("Done")
    #
    # print('Time taken:', int(end_time - start_time), 'secs')
    #
    # statistics(data_state, tag2idx)

