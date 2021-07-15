import logging
import os
import pickle
import json
import random

import torch
from torch.utils.data import Dataset
from transformers_custom import (
    PreTrainedTokenizer,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class JsonlCoconDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, cs_len, hs_len, tis_len, block_size=None, text_json_key="text", evaluate=False, prepended_text_to_remove=None):
        print(f"{file_path=}")
        assert os.path.isfile(file_path)

        self.cs_len = cs_len
        self.hs_len = hs_len
        self.tis_len = tis_len

        if block_size is None:
            block_size = hs_len + max(cs_len, tis_len)
        self.block_size = block_size

        directory, filename = os.path.split(file_path)
        if evaluate and text_json_key != 'text':
            cached_features_file = os.path.join(
                directory, args.model_type + "_cached_cocon_" + str(block_size) + text_json_key + "_" + filename
            )
        else:
            cached_features_file = os.path.join(
                directory, args.model_type + "_cached_cocon_" + str(block_size) + "_" + filename
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            if prepended_text_to_remove is not None:
                if ';' in prepended_text_to_remove:
                    prepended_texts = prepended_text_to_remove.split(';')
                    logger.info("prepended_texts: {}".format(prepended_texts))
                else:
                    prepended_texts = [prepended_text_to_remove]
            else:
                prepended_texts = None

            lines = []
            with open(file_path, encoding="utf-8") as f:
                for jsonl in tqdm(f):
                    json_dict = json.loads(jsonl)
                    if 'length' in json_dict.keys() and evaluate == False:
                        if json_dict['length'] >= block_size:
                            line = json_dict[text_json_key]
                            if prepended_text_to_remove is not None and len(prepended_texts) == 1 and prepended_text_to_remove in line:
                                line = line[line.index(prepended_text_to_remove)+len(prepended_text_to_remove):]
                            else:
                                if prepended_texts is not None:
                                    for prepended_text in prepended_texts:
                                        if prepended_text in line:
                                            line = line[line.index(prepended_text_to_remove)+len(prepended_text_to_remove):]
                                            break
                            lines.append(line)
                    else:
                        line = json_dict[text_json_key]
                        if prepended_text_to_remove is not None:
                            if len(prepended_texts) == 1 and prepended_text_to_remove in line:
                                line = line[line.index(prepended_text_to_remove)+len(prepended_text_to_remove):]
                            else:
                                for prepended_text in prepended_texts:
                                    if prepended_text in line:
                                        line = line[len(prepended_text):]
                                        break

                        lines.append(line)

            logger.info("Encoding with tokenizer")
            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=None)["input_ids"]

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        overflow_len = len(example) - self.block_size
        if overflow_len > 0:
            random_ind = random.randint(0, overflow_len) # random integer between 0 and overflow_len (both inclusive)
        else:
            random_ind = 0
        example_block = example[random_ind:random_ind+self.block_size]

        return torch.tensor(example_block, dtype=torch.long)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512, prepend_bos_token=False):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            if prepend_bos_token:
                lines = [tokenizer.bos_token + line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            else:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False, file_path=None, generate=False, line_by_line=False, prepend_bos_token=False, text_json_key="text", prepended_text_to_remove=None):
    if generate:
        cs_len = args.gen_cs_len
        hs_len = args.gen_hs_len
        tis_len = args.gen_tis_len
    else:
        cs_len = args.cs_len
        hs_len = args.hs_len
        tis_len = args.tis_len

    if file_path is None:
        file_path = args.eval_data_file if evaluate else args.train_data_file

    if line_by_line:
        logger.info("Creating LineByLineTextDataset")
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, prepend_bos_token=prepend_bos_token)
    else:
        if evaluate:
            logger.info("Creating JsonlCoconDataset for eval")
            return JsonlCoconDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, text_json_key=text_json_key, cs_len=cs_len, hs_len=hs_len, tis_len=tis_len, evaluate=True, prepended_text_to_remove=prepended_text_to_remove)
        else:
            return JsonlCoconDataset(tokenizer, args, file_path=file_path, cs_len=cs_len, hs_len=hs_len, tis_len=tis_len)
