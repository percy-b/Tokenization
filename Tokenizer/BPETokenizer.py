#Class to handle special tokens
class Handle_special():
    def __init__(self, allowed_special, ids, vocab_length):
        self.allowed_special = allowed_special
        self.ids = ids
        self.vocab_length = vocab_length
        self.special_dict = {}
        
    #replace sequence
    def replace_sequence(self, id_lst, target, replacement):
        n = len(target)
        i=0
        if isinstance(replacement, list):
            m = len(replacement)
        else:
            m=1
            replacement = [replacement]
        while i <= len(id_lst)-n+1:
            if id_lst[i:i+n]==target:
                id_lst[i:i+n]=replacement
                i+=m
                #print("replcaing")
            else:
                i+=1

        return id_lst
    #small vocab for special tokens
    def special_tokens(self):
        special_ids = [tuple(map(int, i.encode('utf-8'))) for i in self.allowed_special]
        j=0
        for id_ in special_ids:
            self.special_dict[id_] = self.vocab_length+j
            j+=1
        #return special_dict

    #small
    def add_tokens(self):
        newer = self.ids.copy()
        #print(newer)
        #print(len(newer))
        #print("-"*50)
        self.special_tokens() #populate special_dict
        for k in self.special_dict.keys():
            newer = self.replace_sequence(newer, list(k), self.special_dict[k])
        return newer

#usage
#encoder = MyTokenizer(corpus, vocab_length)
#encoder.train()
#then u can use it to enccode and decode 

class MyTokenizer():
    def __init__(self, corpus, vocab_size=1000, allowed_special=None, verbose=False):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.merges = {} # (int, int) -> int
        self._vocab = {idx:bytes([idx]) for idx in range(256)}
        self.allowed_special = allowed_special #list of special tokens
        self.h=None
        self.verbose = verbose
        
    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]): #pythonic way to iterate over 
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        newids = []
        i=0
        while i<len(ids):
            #if not at last position and pair matches, replace it
            if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1] ==pair[1]:
                newids.append(idx)
                i+=2
            else:
                newids.append(ids[i])
                i+=1
        return newids
    
    def train(self):
        tokens = self.corpus.encode("utf-8") #raw bytes
        tokens = list(map(int, tokens))
        
        ids = list(tokens) # copy not to destroy orig list
        self.h = Handle_special(self.allowed_special, ids,self.vocab_size)
        ids = self.h.add_tokens() # merge special tokens and return new tokens
        #
        # add new tokens to end of vocab
        spec_dict =  self.h.special_dict
        for i in list(spec_dict.keys()):
            v = b''
            for j in range(len(i)):
                v+=self._vocab[i[j]]
            self._vocab[spec_dict[i]]=v

        
        num_merges = self.vocab_size - 256


        #for each run, get top pair and replace it
        for i in range(num_merges):
            stats = self._get_stats(ids)
            # Check if stats is empty
            if not stats:
                print(f"No more pairs to merge at iteration {i}.")
                print(f"You are overfitting, reduce the vocabulary size to an appropriate one")
                break
            pair = max(stats, key = stats.get)
            idx = 256+i
            #print(f"merging {pair} into new token {idx}")
            ids = self._merge(ids, pair, idx) #replcae occuances
            self.merges[pair] = idx
            if i%100==0 and self.verbose:
                print(f"Merging the {i}th pair")
            
        
        for (p0, p1), idx in self.merges.items():
            self._vocab[idx] = self._vocab[p0]+self._vocab[p1]

        
            
        print("tokens length: ", len(tokens))
        print("ids length: ", len(ids))
        print(f"compression ratio: {len(tokens)/len(ids):.2f}X")

        
    def decode(self,ids): #enter list of tokens i.e [23,44,55]
        tokens = b"".join(self._vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors = "replace")
        return text
    
    def encode(self, text): #enter text
        tokens = list(text.encode('utf-8'))
        #first replace the special tokens if present
        new_tokens = list(tokens)
        for i in range(len(self.allowed_special)):
            target = tuple(map(int,self.allowed_special[i].encode('utf-8')))
            new_tokens = self.h.replace_sequence(new_tokens, list(target), self.h.special_dict[target])
        
        print("Length of original tokens: ",len(tokens))
        print("Length of tokens after handling special words: ", len(new_tokens))
        
        tokens = list(new_tokens)
        while len(tokens)>=2:
            stats = self._get_stats(tokens)

            #we gonna check pair with minimum value in vocab, if it exist the first one actually
            #until all are done
            pair = min(stats, key=lambda p:self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break #nothing else to merge, break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        print("Length of tokens after merging: ",len(tokens))
        return tokens
    
    def get_vocabulary(self):
        return dict(sorted(self._vocab.items()))
    
    

"""
# usage
1. prepare corpus/text data with relevant special tokens if need be
2. call the tokenizer class with input parameters, text, vocab_size and allowed_special characters 
    e.g `tokenizer = MyTokenizer(text, vocab_size=500, allowed_special=["|<endoftext>|", "|<eos>|"])`
3. train the tokenizer
    e.g `tokenizer.train()`
4. then use it to encode any new text to get tokens
    e.g `ids = tokenizer.encode("Hello world")`
5. you can finally do any decoding
    e.g `tokenizer.decode(ids)`
6. to get vocabulary, u can use the get_vocabulary function
   e.g `tokenizer.get_vocabulary()`
"""
        
        