#coding:utf-8
from transformers.trainer_pt_utils import LabelSmoother
from typing import List, Tuple, Dict, Any

class QwenDataProcess:
    def __init__(self, custom_args, tokenizer, is_train):
        self.max_length = custom_args.max_length
        self.max_source_length = custom_args.max_source_length
        self.max_target_length = custom_args.max_target_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.instruction_column = custom_args.instruction_column ##system
        self.input_column = custom_args.input_column ##user
        self.output_column = custom_args.output_column ##answer
        self.training_mode = custom_args.training_mode #training_mode

    def filter_fn(self, example: Dict[str, Any]) -> bool:
        """用于过滤无效样本：input_column 不为 None 且非空字符串"""
        input_val = example.get(self.input_column)
        answer_val = example.get(self.output_column)
        return isinstance(input_val, str) and len(input_val.strip()) > 0  \
            and isinstance(answer_val, str) and len(answer_val.strip()) > 0

    def __call__(self, example):
        input = example[self.input_column]
        if self.output_column in example:
            output = example[self.output_column]
        instruction = None
        if self.instruction_column in example:
            instruction = example[self.instruction_column]
        if self.training_mode == 'pretrain':
            data_list = self._encode_data_for_pretrain(input)
        else:
            data_list = self._encode_data(input, output if self.is_train else None, instruction)

        input_ids, labels = [], []
        for _, input_id, label in data_list:
            input_ids += input_id
            labels += label

        if self.max_length and self.max_length > 0 and self.max_length < len(input_ids):
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        if self.is_train:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def _tokenize_text(self, text: str, max_length: int = None) -> List[int]:
        if text is None or len(text) == 0:
            text = " "
        ids = self.tokenizer(text).input_ids
        if max_length > 0 and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    # from git: https://github.com/QwenLM/Qwen-7B/blob/main/finetune.py#L123
    def _encode_data(self, input: str, output: str,
                     instruction: str) -> List[Tuple[str, List[int], List[int]]]:
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START).input_ids
        im_end = self.tokenizer(IM_END).input_ids
        nl_tokens = self.tokenizer(NL).input_ids ## 换行符
        _system = self.tokenizer('system').input_ids + nl_tokens
        _user = self.tokenizer('user').input_ids + nl_tokens
        _assistant = self.tokenizer('assistant').input_ids + nl_tokens
        default_instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        data_list = []
        # instruct,
        if not instruction:
            instruction = default_instruction
        input_ids = im_start + _system + self.tokenizer(instruction).input_ids + im_end + nl_tokens
        labels = [IGNORE_TOKEN_ID] * len(im_start) + [IGNORE_TOKEN_ID] * (len(input_ids) - 3) + [IGNORE_TOKEN_ID] * len(im_end) + [IGNORE_TOKEN_ID] * len(nl_tokens)
        data_list.append(("instruct", input_ids, labels)) ## system

        assert len(input_ids) == len(labels)
        # conversations
        history = []
        history.append([input, output])
        for turn_prompt, turn_response in history:
            # prompt IGNORE_TOKEN_ID
            prompt_ids = self._tokenize_text(turn_prompt, max_length=self.max_source_length)
            input_ids = im_start + _user + prompt_ids + im_end + nl_tokens + im_start + _assistant
            labels = [IGNORE_TOKEN_ID]*len(im_start) + [IGNORE_TOKEN_ID] * len(_user + prompt_ids) + [IGNORE_TOKEN_ID]*(len(im_end) + len(nl_tokens) + len(im_start)) + [
                IGNORE_TOKEN_ID] * len(_assistant)
            assert len(input_ids) == len(labels)
            data_list.append(("prompt", input_ids, labels))

            # response LABEL
            if turn_response:
                response_ids = self._tokenize_text(turn_response, max_length=self.max_target_length)
                input_ids = response_ids + im_end + nl_tokens
                labels = response_ids + [IGNORE_TOKEN_ID] * (len(im_end) + len(nl_tokens))
                assert len(input_ids) == len(labels)
                data_list.append(("response", input_ids, labels))
        return data_list


    def _encode_data_for_pretrain(self, input: str):
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        data_list = []

        # response LABEL
        response_ids = self._tokenize_text(input, max_length=self.max_target_length)
        input_ids = response_ids
        labels = response_ids
        assert len(input_ids) == len(labels)
        data_list.append(("corpus", input_ids, labels))
        return data_list