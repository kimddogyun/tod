import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import time
from . import Vocab
from . import NoamLR
from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration, AdamW, WEIGHTS_NAME, CONFIG_NAME, get_linear_schedule_with_warmup)


class Model(object):
    def __init__(self, args, test=False):
        if args.backbone =='t5-large':
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        elif 't5-3b' in args.model:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        vocab = Vocab(self.model, self.tokenizer)
        self.optim = AdamW(self.model.parameters(), lr=args.lr, correct_bias=True) #Q-TOD USE ADAMW
        self.args = args
        self.model.to(args.device)

    def load_model(self):
        self.model = type(self.model).from_pretrained(self.args.model_path)
        self.model.to(self.args.device)

    def train(self):
        torch.save(self.args, self.args.model_path + '/model_training_args.bin')
        start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.tokenizer.save_pretrained(self.args.model_path)
        self.model.config.to_json_file(os.path.join(self.args.model_path, CONFIG_NAME))
        self.model.train()
        scheduler = NoamLR(self.optim, warmup_steps=self.args.warmup_steps)

        for epoch in range(self.args.epoch_nums): #epoch = 50
            step = 0
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                prev_query = {'query_annotation':None}
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch,prev_query, first_turn=first_turn)
                
                batch_size = inputs["input_ids"].shape[0]
                input_seq_len = inputs["input_ids"].shape[1]
                query_seq_len = inputs["state_input"].shape[1]
                resp_seq_len = inputs["response"].shape[1]
                print(f"batch_size:{batch_size},seq_len:{input_seq_len}, query:{query_seq_len}, resp:{resp_seq_len}")

                outputs = self.model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["masks"],
                    decoder_input_ids=inputs["state_input"],
                    lm_labels=inputs["state_update"]
                    )

                query_loss = outputs[0]

                # outputs = self.model(encoder_outputs=outputs[-1:], #skip loss and logits
                #                     attention_mask=inputs["masks"],
                #                     decoder_input_ids=inputs["response_input"],
                #                     lm_labels=inputs["response"]
                #                     )
                # resp_loss = outputs[0]

                prev_query['query_annotation'] = turn_batch['query_annotation']

                total_loss = (query_loss) / self.args.gradient_accumulation_steps #+ resp_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                if step % self.args.gradient_accumulation_steps  == 0:
                    self.optim.step()
                    self.otpim.zero_grad()
                step += 1


