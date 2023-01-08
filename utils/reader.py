import os
import logging
from utils.io_utils import load_json, load_pickle, save_pickle, get_or_create_logger
from transformers import T5Tokenizer
from utils import definition

logger = get_or_create_logger(__name__)

class BaseReader(object):
    def __init__(self,backbone):
        self.tokenizer = self.init_tokenizer(backbone)
        self.data_dir = self.get_data_dir()

        encoded_data_path = os.path.join(self.data_dir, "encoded_data.pkl")

        if os.path.exists(encoded_data_path):
            logger.info("Load encoded data from {}".format(encoded_data_path))

            self.data = load_pickle(encoded_data_path)

        else:
            logger.info("Encode data and save to {}".format(encoded_data_path))
            train = self.encode_data("train")
            dev = self.encoded_data("dev")
            test = self.encoded_data("test")

            self.data = {"train":train, "dev":dev, "test": test}

            save_pickle(self.data, encoded_data_path)


    def get_data_dir(self):
        raise NotImplementedError


    def encode_data(self, data_type):
        raise NotImplementedError



    def init_tokenizer(backbone):
        tokenizer = T5Tokenizer.from_pretained(backbone)
        special_tokens = []

        special_tokens.extend(definition.SPECIAL_TOKENS)

        #add Nothing
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

class  MultiWOZReader(BaseReader):
    def __init__(self, backbone):
        super(MultiWOZReader, self).__init__(backbone)


    def get_data_dir(self):
        return os.path.join(
            "data", "MultiWOZ_{}".format(self.version), "processed")