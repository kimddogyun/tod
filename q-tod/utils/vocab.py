class Vocab(object):
    def __init__(self, model, tokenizer):
        self.SPECIAL_TOKENS = ["<eos_u>", "<eos_r>", "<eos_q>", "<eos_k>", "<pad>", "[NOTHING]"] #eos_q -> query, #eos_k -> knowledge retrieval [NOTHING] -> NO QUERY
        self.tokenizer = tokenizer
        self.vocab_size = self.add_special_tokens_(model, tokenizer)

    def add_special_tokens(self, model, tokenizer):
        orig_num_tokens = len(tokenizer)
        num_added_tokens = tokenizer.add_special_tokens(self.SPECIAL_TOKENS)

        if num_added_tokens > 0:
            model.resize_token_embeddings(new_num_tokens=orig_num_tokens+num_added_tokens)
            model.tie_decoder()

        return orig_num_tokens + num_added_tokens

    def encode(self, word):
        """ customize for damd script """
        return self.tokenizer.encode(word)[0]

    def sentence_encode(self, word_list):
        """ customize for damd script """
        return self.tokenizer.encode(" ".join(word_list))

    def decode(self, idx):
        """ customize for damd script """
        return self.tokenizer.decode(idx)

    def sentence_decode(self, index_list, eos=None):
        """ customize for damd script """
        l = self.tokenizer.decode(index_list)
        l = l.split()
        if not eos or eos not in l:
            text = ' '.join(l)
        else:
            idx = l.index(eos)
            text = ' '.join(l[:idx])
        return puntuation_handler(text)

# T5 cannot seperate the puntuation for some reason
def puntuation_handler(text):
    text = text.replace("'s", " 's")
    text = text.replace(".", " .")
    text = text.replace("!", " !")
    text = text.replace(",", " ,")
    text = text.replace("?", " ?")
    text = text.replace(":", " :")
    return text