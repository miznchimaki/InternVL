import re, os
import random
from typing import Union, List, Tuple, Dict
from pathlib import Path
import conversation as conversation_lib
from conversation import default_conversation
from constants import IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN
from constants import IGNORE_INDEX
import torch
import transformers


TASKS_TYPE = (
              "caption", "visual_grounding", "ocr", 
              "visual_reasoning", "vqa", "multi_image", "text"
             )


def taisu2_preprocess_internvl2_5(
                                  imgnames: Union[str | List[str] | Tuple[str]], 
                                  anns: str, 
                                  task_type: str, 
                                  context_token_per_img: int, 
                                  sub_img_num: Union[int | List[int] | Tuple[int]], 
                                  tokenizer: transformers.PreTrainedTokenizer, 
                                  padding: Union[str | bool] = "do_not_pad", 
                                  padding_side: Union[str | None] = "right", 
                                  return_tensors: Union[str | None] = "pt", 
                                  return_attention_mask: Union[bool | None] = True, 
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    if task_type.lower() not in TASKS_TYPE:
        raise ValueError(f"task for Taisu2 preprocessing function could only be `{TASKS_TYPE}`, but get {task_type}")
    if task_type.lower() == "caption":
        prompt_file = "taisu2_caption_prompt.txt"
    elif task_type.lower() == "visual_grounding":
        prompt_file = "taisu2_grounding_prompt.txt"
    elif task_type.lower() == "ocr":
        prompt_file = "taisu2_ocr_prompt.txt"
    elif task_type.lower() == "visual_reasoning":
        prompt_file = "taisu2_reasoning_prompt.txt"
    elif task_type.lower() == "multi_image":
        prompt_file = "taisu2_multi_image_prompt.txt"
    elif task_type.lower() == "text":
        prompt_file = None
    prompt_dir = Path(os.path.abspath(__file__)).parent / "taisu2_prompts"
    prompt_p = prompt_dir / prompt_file if prompt_file is not None else None
    if prompt_p is not None:
        if not prompt_p.exists():
            raise FileNotFoundError(f"prompt file for task {task_type.lower()} - {prompt_p}, doesn't exist!")
        all_prompts = []
        with open(prompt_p, mode="r", encoding="utf-8") as prompt_fp:
            for prompt in prompt_fp:
                all_prompts.append(prompt.strip())

    img_tokens = set((IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN))
    for token in tokenizer.added_tokens_decoder.values():
        if token.content in img_tokens:
            img_tokens.remove(token.content)
    if img_tokens:
        raise ValueError(f"InternVL2-5/InternVL3 tokenizer doesn't have special image tokens: {img_tokens}")
    conv = default_conversation.copy()
    conv.messages = []
    if conv.sep_style != conversation_lib.SeparatorStyle.MPT:
        raise ValueError(f"the separator style of InternVL2_5/InternVL3 must be SeparatorStyle.MPT")
    roles = conv.roles
    roles_map = {"user": roles[0], "gpt": roles[1]}

    if task_type.lower() == "caption":
        if all_prompts[0] != "native_prompts:":
            raise ValueError(f"first line of caption prompt file shoud be `native_prompts:`")
        if "recaption_prompts:" not in all_prompts:
            raise ValueError(f"`recaption_prompts:` should be in caption prompt file")
        if "native_and_recaption_prompts:" not in all_prompts:
            raise ValueError(f"`native_and_recaption_prompts:` should be in caption prompt file")
        native_caption_prompts = all_prompts[1: all_prompts.index("recaption_prompts:")]
        re_caption_prompts = all_prompts[all_prompts.index("recaption_prompts:") + 1: all_prompts.index("native_and_recaption_prompts:")]
        two_caption_prompts = all_prompts[all_prompts.index("native_and_recaption_prompts:") + 1: ]

        pat_strs = (
                    r"(\<\|native_caption_start\|>)([\s\S]*)(\<\|native_caption_end\|>)", 
                    r"(\<\|recaption_start\|>)([\s\S]*)(\<\|recaption_end\|>)", 
                    r"(\<\|native_caption_start\|>)([\s\S]*)(\<\|native_caption_end\|>)(\<\|recaption_start\|>)([\s\S]*)(\<\|recaption_end\|>)"
                   )
        native_caption = None
        re_caption = None
        match_res = re.fullmatch(pat_strs[2], anns)
        if match_res is not None:
            native_caption = match_res.group(2)
            re_caption = match_res.group(5)
        else:
            match_res = re.fullmatch(pat_strs[1], anns)
            if match_res is not None:
                re_caption = match_res.group(2)
            else:
                match_res = re.fullmatch(pat_strs[0], anns)
                if match_res is not None:
                    native_caption = match_res.group(2)
        if native_caption is None and re_caption is None:
            native_caption = anns
        if (not native_caption) and re_caption is None:
            raise ValueError(f"image-alttext pairs with name {imgnames} does not have an effective native caption and re-caption")
        if native_caption and re_caption:
            prompt_str = random.choice(two_caption_prompts)
            user_prompt = prompt_str.format(native_caption=native_caption)
            conv.append_message(roles[0], "<image>\n" + user_prompt)
            conv.append_message(roles[1], re_caption)
        if native_caption and (not re_caption):
            user_prompt = random.choice(native_caption_prompts)
            conv.append_message(roles[0], "<image>\n" + user_prompt)
            conv.append_message(roles[1], native_caption)
        if (not native_caption) and re_caption:
            user_prompt = random.choice(re_caption_prompts)
            conv.append_message(roles[0], "<image>\n" + user_prompt)
            conv.append_message(roles[1], re_caption)
        conv_prompt = conv.get_prompt()

    if isinstance(sub_img_num, int):
        conv_prompt = conv_prompt.replace("<image>", IMG_START_TOKEN + sub_img_num * context_token_per_img * IMG_CONTEXT_TOKEN + IMG_END_TOKEN, 1)
    elif isinstance(sub_img_num, (list, tuple)):
        for sub_img_num_per in sub_img_num:
            conv_prompt = conv_prompt.replace("<image>", IMG_START_TOKEN + sub_img_num_per * context_token_per_img * IMG_CONTEXT_TOKEN + IMG_END_TOKEN, 1)
    remained_img_token_num = conv_prompt.count("<image>")
    if remained_img_token_num:
        raise ValueError(f"after replacing all `<image>` into image context tokens, there're still `<image>` left: {conv_prompt}")
    tokenized_res = tokenizer(conv_prompt, padding=padding, padding_side=padding_side, 
                              return_tensors=return_tensors, 
                              return_attention_mask=return_attention_mask)
    if return_tensors == "pt":
        input_ids = tokenized_res.input_ids[0]
    elif return_tensors is None:
        input_ids = torch.tensor(tokenized_res.input_ids)
    else:
        raise TypeError(f"got an unspported `return_tensors` param: {return_tensors}")
    targets = input_ids.clone()

    # Mask targets
    sep = conv.sep + roles[1]
    total_len = targets.ne(tokenizer.pad_token_id).sum().item()
    rounds = conv_prompt.split(conv.sep)
    re_rounds = [conv.sep.join(rounds[: 3])]  # system + user + gpt
    for conv_idx in range(3, len(rounds), 2):
        re_rounds.append(conv.sep.join(rounds[conv_idx: conv_idx + 2]))
    cur_len = 0
    targets[: cur_len] = IGNORE_INDEX
    for i, rou in enumerate(re_rounds):
        if rou == "":
            break
        parts = rou.split(sep)
        if len(parts) != 2:
            break
        parts[0] += sep
        round_len = len(tokenizer(rou, padding=False, padding_side=None, return_tensors=None, return_attention_mask=False).input_ids)
        round_len += len(tokenizer(conv.sep, padding=False, padding_side=None, return_tensors=None, return_attention_mask=False).input_ids)
        instruction_len = len(tokenizer(parts[0], padding=False, padding_side=None, return_tensors=None, return_attention_mask=False).input_ids)
        targets[cur_len: cur_len + instruction_len] = IGNORE_INDEX

        cur_len += round_len
    targets[cur_len: ] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            targets[:] = IGNORE_INDEX
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                  f" (ignored)"
                 )

    return {"input_ids": input_ids, "labels": targets}


def taisu2_text_preprocess(
                           sources: str, 
                           imgnames: Union[str | List[str] | Tuple[int]], 
                           task_type: str, 
                           sub_img_num: Union[int | List[int] | Tuple[int]], 
                           tokenizer: transformers.PreTrainedTokenizer = None, 
                           data_args = None
                          ) -> Dict[str, torch.Tensor]:
    if "internvl2_5" in conversation_lib.default_conversation.name or "internvl3" in conversation_lib.default_conversation.name:
        return taisu2_preprocess_internvl2_5(
                                             imgnames=imgnames, 
                                             anns=sources, 
                                             task_type=task_type, 
                                             context_token_per_img=data_args.context_token_per_img, 
                                             sub_img_num=sub_img_num, 
                                             tokenizer=tokenizer, 
                                             padding=data_args.padding, 
                                             padding_side=data_args.padding_side, 
                                             return_tensors=data_args.return_tensors, 
                                             return_attention_mask=data_args.return_attention_mask
                                            )
