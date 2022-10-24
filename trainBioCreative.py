
import torch.nn as nn
from transformers import BertModel
import os
import numpy as np
import torch
import random
from ast import iter_fields
import torch
from torch import nn, optim
from tqdm import tqdm
import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
from random import choice
from transformers import  AutoTokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

BERT_MAX_LEN = 512


torch.cuda.set_device(0)

class setuper(nn.Module):
    def __init__(self):

            super().__init__()
            name ="bert-base-cased"
            self.bert = BertModel.from_pretrained(name)


    def forward(self, token, att_mask):
            x = self.bert(token, attention_mask=att_mask)
            return x[0]

class Main_NN(nn.Module):

    def __init__(self):
        super(Main_NN, self).__init__()
        
        self.sentence_encoder = setuper()
        self.gat = GNN()


    def forward(self, token, mask, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails):

        # token
        hidden = self.sentence_encoder(token, mask)

        sub_heads_logits, sub_tails_logits, obj_heads_logits, obj_tails_logits = self.gat(hidden, sub_head, sub_tail, mask)


        #loss
        sub_head_loss = F.binary_cross_entropy(sub_heads_logits, sub_heads.float(), reduction='none')
        sub_head_loss = (sub_head_loss * mask.float()).sum() / mask.float().sum()
        sub_tail_loss = F.binary_cross_entropy(sub_tails_logits, sub_tails.float(), reduction='none')
        sub_tail_loss = (sub_tail_loss * mask.float()).sum() / mask.float().sum()
        obj_head_loss = F.binary_cross_entropy(obj_heads_logits, obj_heads.float(), reduction='none').sum(2)
        obj_head_loss = (obj_head_loss * mask.float()).sum() / mask.float().sum()
        obj_tail_loss = F.binary_cross_entropy(obj_tails_logits, obj_tails.float(), reduction='none').sum(2)
        obj_tail_loss = (obj_tail_loss * mask.float()).sum() / mask.float().sum()
        loss = sub_head_loss + sub_tail_loss + obj_head_loss + obj_tail_loss
        return loss

    def Subject_output(self, token, mask=None):
        
        t = self.sentence_encoder(token, mask)
        sub_heads_logits, sub_tails_logits = self.gat(t, mask=mask)
        return sub_heads_logits, sub_tails_logits

    def Object_output(self, token, sub_head, sub_tail, mask=None):
        hidden = self.sentence_encoder(token, mask)
        _, _, obj_heads_logits, obj_tails_logits = self.gat(hidden, sub_head, sub_tail, mask)


        return obj_heads_logits, obj_tails_logits


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        number_of_classes = 171
        hidden_size = 768
        self.embeding = nn.Embedding(number_of_classes, hidden_size)
        self.relation = nn.Linear(hidden_size, hidden_size)
        self.down = nn.Linear(3 * hidden_size, hidden_size)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)
        self.start_tail = nn.Linear(hidden_size, 1)
        self.end_tail = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList([LLayer(hidden_size) for _ in range(2)])

    def forward(self, x, sub_head =None, sub_tail =None, mask=None):
        # relation
        
        p = torch.arange(171).long()
        if torch.cuda.is_available():
            p = p.cuda()
        p = self.relation(self.embeding(p))
        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))  # bcd
        x, p = self.gat_layer(x, p, mask)  # x bcd
        tail_start, tail_end = self.head_tail_finder(x)
        if sub_head is not None and sub_tail is not None:
            e1 = self.subject_candidates(x, sub_head, sub_tail)
            head_start, head_end = self.pre_tail(x, e1, p)
            return tail_start, tail_end, head_start, head_end
        return tail_start, tail_end

        
    def subject_candidates(self, x, sub_head, sub_tail):
        batch = x.size(0)
        avai_len = x.size(1)
        sub_head, sub_tail = sub_head.view(-1, 1), sub_tail.view(-1, 1)
        mask = []
        for i in range(batch):
            pos = torch.zeros(avai_len, device=x.device).float()
            h, t = sub_head[i].item(), sub_tail[i].item()
            pos[h:t+1] = 1.0
            mask.append(pos)
        mask = torch.stack(mask, 0)
        e1 = x * mask.unsqueeze(2).expand(-1, -1, x.size(2))
        divied = torch.sum(mask, 1)
        e1 = torch.sum(e1, 1) / divied.unsqueeze(1)

        return e1

    def head_tail_finder(self, x):
        x = torch.tanh(x)
        ts = self.start_head(x).squeeze(2)
        ts = ts.sigmoid()
        te = self.end_head(x).squeeze(2)
        te = te.sigmoid()
        return ts, te

    def pre_tail(self, x, e1, p):
        e1 = e1.unsqueeze(1).expand_as(x)
        e1 = e1.unsqueeze(2).expand(-1, -1, p.size(1), -1)
        x = x.unsqueeze(2).expand(-1,-1,p.size(1),-1)
        p = p.unsqueeze(1).expand(-1,x.size(1),-1,-1)
        t = self.down(torch.cat([x, p, e1], 3))
        t = torch.tanh(t)
        ts = self.start_tail(t).squeeze(3)
        ts = ts.sigmoid()
        te = self.end_tail(t).squeeze(3)
        te = te.sigmoid()
        return ts, te

    def gat_layer(self, x, p, mask=None):

        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p


class LLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = GraphAttention(hidden_size)
        self.ra2 = GraphAttention(hidden_size)

    def forward(self, x, p, mask=None):
        x_ = self.ra1(x, p)
        x = x_+ x
        p_ = self.ra2(p, x, mask)
        p =  p_ + p
        return x, p

class GraphAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GraphAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.cancatenator(q, k)
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(score, 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        g = self.gate(torch.cat([out, p], 2)).sigmoid()
        out = g * out + (1 - g) * p
        return out

    def cancatenator(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        return self.score(temp).squeeze(3)


def h_finder(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def pad(batch, padding=0):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])

class DatasetOpener(data.Dataset):
    def __init__(self, path, rel_dict_path):
        super().__init__()
        self.path = path
        self.data = json.load(open(path, encoding='utf-8'))
        id2rel, rel2id = json.load(open(rel_dict_path, encoding='utf-8'))
        id2rel = {int(i): j for i, j in id2rel.items()}
        self.num_rels = len(id2rel)
        self.id2rel = id2rel
        self.rel2id = rel2id
        self.maxlen = 512
        self.berttokenizer = AutoTokenizer.from_pretrained('bert-base-cased',do_lower_case=False, do_basic_tokenize = False)#do_lower_case=False
        # self.berttokenizer.do_basic_tokenize = False
        for sent in self.data:
            ## to tuple
            
            triple_list = []
            for triple in sent['triple_list']:
                # print(triple)
                triple_list.append(tuple(triple))
            sent['triple_list'] = triple_list
        self.data_length = len(self.data)
        print("new loder")



    def _tokenizer(self, line):
        
        # print("line['text'] is: ", line['text'])
        text = ' '.join(line['text'].split()[:self.maxlen])
        tokens = self._tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[:BERT_MAX_LEN]
        text_len = len(tokens)
        tagger = {}
        for triple in line['triple_list']:
            # print(triple)
            triple = (self._tokenize(triple[0])[1:-1], triple[1], self._tokenize(triple[2])[1:-1])
            
            sub_head_idx = h_finder(tokens, triple[0])
            obj_head_idx = h_finder(tokens, triple[2])
            # print(triple[0])
            # print(triple[1])
            # print(triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1:
                sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                if sub not in tagger:
                    tagger[sub] = []
                tagger[sub].append((obj_head_idx,
                                      obj_head_idx + len(triple[2]) - 1,
                                      self.rel2id[triple[1]]))
            
        if tagger:
            
            token_ids = self.berttokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            for s in tagger:
                sub_heads[s[0]] = 1
                sub_tails[s[1]] = 1
            sub_head, sub_tail = choice(list(tagger.keys()))
            #Creating one-hot vector for relations
            obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
            for ro in tagger.get((sub_head, sub_tail), []):
                obj_heads[ro[0]][ro[2]] = 1
                obj_tails[ro[1]][ro[2]] = 1
            att_mask = torch.ones(len(token_ids)).long()
           
          
            # set_trace()
            return [token_ids, att_mask, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails]
        else:
            return []

    def _tokenize(self, tokens):
        re_tokens = ['[CLS]']
        for token in tokens.strip().split():
            re_tokens += self.berttokenizer.tokenize(token)
        re_tokens.append('[SEP]')
        return re_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        ret = self._tokenizer(item)
        return ret
        
    def calculator(self, model, h_bar = 0.5, t_bar=0.5, exact_match=False, output_path=None):
        save_data = []
        orders = ['subject', 'relation', 'object']
        correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
        for line in tqdm(self.data):
            Pred_triples = set(self.extract_items(model, line['text'], h_bar=h_bar, t_bar=t_bar))
            Gold_triples = set(line['triple_list'])

            Pred_triples_eval, Gold_triples_eval = self.partial_match(Pred_triples, Gold_triples) if not exact_match else (
            Pred_triples, Gold_triples)

            correct_num += len(Pred_triples_eval & Gold_triples_eval)
            predict_num += len(Pred_triples_eval)
            gold_num += len(Gold_triples_eval)

            if output_path:
                temp = {
                    'text': line['text'],
                    'triple_list_gold': [
                        dict(zip(orders, triple)) for triple in Gold_triples
                    ],
                    'triple_list_pred': [
                        dict(zip(orders, triple)) for triple in Pred_triples
                    ],
                    'new': [
                        dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                    ],
                    'lack': [
                        dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                    ]
                }
                save_data.append(temp)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f)

        precision = correct_num / predict_num
        recall = correct_num / gold_num
        f1_score = 2 * precision * recall / (precision + recall)

        print(f'correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}')
#         print('f1-score:{}'.format(f1_score))
        print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
        return precision, recall, f1_score

    def partial_match(self, pred_set, gold_set):
        pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
                 i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
        gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1],
                 i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
        return pred, gold

    def extract_items(self, model, text_in, h_bar=0.5, t_bar=0.5):
        tokens = self._tokenize(text_in)
        token_ids = self.berttokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > BERT_MAX_LEN:
            token_ids = token_ids[:, :BERT_MAX_LEN]
        token_ids_np = np.array([token_ids])
        token_ids = torch.tensor(token_ids).unsqueeze(0).long().cuda()
        sub_heads_logits, sub_tails_logits = model.Subject_output(token_ids)
        sub_heads, sub_tails = np.where(sub_heads_logits[0].cpu() > h_bar)[0], np.where(sub_tails_logits[0].cpu() > h_bar)[0]
        # print(sub_heads_logits[0])
        subjects = []
        for sub_head in sub_heads:
            sub_tail = sub_tails[sub_tails >= sub_head]
            if len(sub_tail) > 0:
                sub_tail = sub_tail[0]
                subject = tokens[sub_head: sub_tail+1]
                subjects.append((subject, sub_head, sub_tail))
        if subjects:
            triple_list = []
            token_ids = torch.from_numpy(np.repeat(token_ids_np, len(subjects), 0)).long().cuda()
            sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
            sub_heads, sub_tails = torch.from_numpy(sub_heads).cuda(), torch.from_numpy(sub_tails).cuda()
            obj_heads_logits, obj_tails_logits = model.Object_output(token_ids, sub_heads, sub_tails)

            for i, subject in enumerate(subjects):
                sub = subject[0]
                sub = self.detokeenizer(sub)
                obj_heads, obj_tails = np.where(obj_heads_logits[i].cpu() > t_bar), np.where(obj_tails_logits[i].cpu() > t_bar)
                for obj_head, rel_head in zip(*obj_heads):
                    for obj_tail, rel_tail in zip(*obj_tails):
                        if obj_head <= obj_tail and rel_head == rel_tail:
                            rel = self.id2rel[rel_head]
                            obj = tokens[obj_head: obj_tail+1]
                            obj = self.detokeenizer(obj)
                            triple_list.append((sub, rel, obj))
                            break
            triple_set = set()
            for s, r, o in triple_list:
                triple_set.add((s, r, o))
            return list(triple_set)
        else:
            return []
    def detokeenizer(self, x):
        new_x = []
        for i in range(len(x)-1):
            sub_x = x[i]
            rear = x[i+1]
            new_x.append(sub_x)
            if "##" not in rear:
                new_x.append("[blank]")
        if len(x)>0:
            new_x.append(x[-1])
        new_x = ''.join([i.lstrip("##") for i in new_x])
        new_x = ' '.join(new_x.split('[blank]'))
        return new_x

    @staticmethod
    def collate_fn(data):
        
        data = list(zip(*data))
        token_ids,att_mask, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails = data
        num_rels = obj_heads[0].shape[1]
        tokens_batch = torch.from_numpy(pad(token_ids)).long()
        att_mask_batch = pad_sequence(att_mask,batch_first=True,padding_value=0)
        sub_heads_batch = torch.from_numpy(pad(sub_heads))
        sub_tails_batch = torch.from_numpy(pad(sub_tails))
        obj_heads_batch = torch.from_numpy(pad(obj_heads, np.zeros(num_rels)))
        obj_tails_batch = torch.from_numpy(pad(obj_tails, np.zeros(num_rels)))
        sub_head_batch, sub_tail_batch = torch.from_numpy(np.array(sub_head)).long(), torch.from_numpy(np.array(sub_tail)).long()
        
        return tokens_batch, att_mask_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch




def TrainLoader(path, rel2id, batch_size,
                     shuffle, num_workers=8, collate_fn=DatasetOpener.collate_fn):

    dataset = DatasetOpener(path=path, rel_dict_path=rel2id)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
 
    return data_loader


class First_RE(nn.Module):

    def __init__(self,model,train = os.path.join('data', 'WebNLG/train_triples.json'),val=os.path.join('data', 'WebNLG/dev_triples.json'),test=os.path.join('data', 'WebNLG/test_triples.json'),
                 rel2id=os.path.join('data', 'WebNLG/rel2id.json'),ckpt='checkpoint/WebNLG.pth.tar',batch_size=6,max_epoch=100,lr=1e-1,num_workers= 4,weight_decay=1e-5):

        super().__init__()
        self.max_epoch = max_epoch
    
        self.train_loder = TrainLoader(path=train,rel2id=rel2id,batch_size=batch_size,shuffle=True,num_workers=num_workers)

        self.val_set = DatasetOpener(path=val, rel_dict_path=rel2id)
        self.test_set = DatasetOpener(path=test, rel_dict_path=rel2id)
        
        # Model
        # self.model = nn.DataParallel(model)
        self.model = model
        # Criterion
        self.loss_func = nn.BCELoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, params), lr, weight_decay=weight_decay)
        # Cuda
        
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, warmup=True):
        best_f1 = 0
        global_step = 0
        wait =0
        for epoch in range(self.max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_sent_loss = Meter()
            
            t = tqdm(self.train_loder)
         
            
            for _, data in enumerate(t):

             
                if torch.cuda.is_available():

                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
       
                loss = self.model(*data)
                # Log
                avg_sent_loss.update(loss.item(), 1)
                t.set_postfix(sent_loss=avg_sent_loss.avg)

                # Optimize
                if warmup == True:
                    warmup_step = 300
                    if global_step < warmup_step:
                        warmup_rate = float(global_step) / warmup_step
                    else:
                        warmup_rate = 1.0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * warmup_rate

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                ###
            # Val
            print("=== Epoch %d val ===" % epoch)
            self.eval()
            precision, recall, f1 = self.val_set.calculator(self.model)
            if f1 > best_f1 or f1 < 1e-4:
                if f1 > 1e-4:
                    best_f1 = f1
                    print("Best ckpt and saved.")
                    torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
            else:
                wait +=1
                if wait>20:
                    print('Epoch %05d: early stopping' % (epoch + 1))
                    break
            print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, best_f1))
        #torch.save({'state_dict': self.model.state_dict()}, 'checkpoint/final.pth.tar')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, best_f1))

    def load_state_dict(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['state_dict'])



class Meter(object):


    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):

        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)





 
 


if __name__ == '__main__':
    root_path = 'data'
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Main_NN()  
    GNNModel = First_RE(model)
    output_path = 'save_result/WebNLG_result.json'
    GNNModel.train_model()
    GNNModel.load_state_dict('checkpoint/WebNLG.pth.tar')
    GNNModel.test_set.calculator(GNNModel.model, output_path=output_path)
    