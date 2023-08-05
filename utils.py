import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

import openai
import tqdm
from openai import openai_object
import copy

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

import torch
import math
import random
from torch import nn
import torch.nn.functional as F
from scipy.stats import norm


def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param    reduced_rank: the final rank
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    print(f"W: ({H},{W}) | Rank: {rank} | U:{U.shape} | S:{S.shape} | Vh:{Vh.shape}")
    print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
    print(f"L: {L.shape} | R: {R.shape}")

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'reduced_rank': reduced_rank}


class NFQuantizer:
    def __init__(self, num_bits=2, device='cuda', differentiable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
        if differentiable:
            self.norm_lookup_table = nn.Parameter(self.norm_lookup_table)
        self.norm_lookup_table = self.norm_lookup_table.to(device)

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        variations = 2 ** num_bits

        if symmetric:
            print("symmetric nf4")
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            print("asymmetric nf4")
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            # print(torch.linspace(offset, 0.5, 9)[:-1])
            # print(v1)
            v2 = [0]
            # v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            # print(torch.linspace(offset, 0.5, 8)[:-1])
            # print(v3)
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        # print(values)
        return values
        # assert values.

    def quantize_tensor(self, weight):
        max_abs = torch.abs(weight).max()
        weight_normed = weight / max_abs

        weight_normed_expanded = weight_normed.unsqueeze(-1)

        # Reshape L to have the same number of dimensions as X_expanded
        L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)

        # Calculate the absolute difference between X_expanded and L_reshaped
        abs_diff = torch.abs(weight_normed_expanded - L_reshaped)

        # Find the index of the minimum absolute difference for each element
        qweight = torch.argmin(abs_diff, dim=-1)
        # print(min_index)
        return qweight, max_abs

    def dequantize_tensor(self, qweight, max_abs):
        qweight_flatten = qweight.flatten()

        weight_normed = self.norm_lookup_table[qweight_flatten]
        weight = weight_normed * max_abs

        weight = weight.reshape(qweight.shape)

        return weight

    def quantize_block(self, weight, block_size=64):
        assert len(weight.shape) == 2 and weight.shape[0] * weight.shape[1] % block_size == 0
        M, N = weight.shape
        device = weight.device

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, block_size)  # (L, B), L = M * N / B
        weight_max = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        weight_max = weight_max.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        # Pack multiple k-bit into uint8
        qweight = qweight.reshape(-1, 8 // self.num_bits)
        qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)

        # data format example:
        # [1, 0, 3, 2] or [01, 00, 11, 10]  -> [10110001], LIFO
        for i in range(8 // self.num_bits):
            qweight[:, i] = qweight[:, i] << i * self.num_bits
            qweight_pack[:, 0] |= qweight[:, i]

        return qweight_pack, weight_max, weight.shape

    def dequantize_block(self, qweight, weight_max, weight_shape, block_size=64):
        # unpack weight
        device = qweight.device
        weight = torch.zeros((qweight.shape[0], 8 // self.num_bits), dtype=torch.float32, device=device)
        for i in range(8 // self.num_bits):
            lookup_table_idx = qweight % 2 ** self.num_bits  # get the most right 2 bits
            lookup_table_idx = lookup_table_idx.to(torch.int)
            weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
            qweight = qweight >> self.num_bits  # right shift 2 bits of the original data

        weight_block = weight.reshape(-1, block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)

        return weight


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = input @ self.weight.T
        if self.bias is None:
            return ret
        else:
            return ret + self.bias


class QLinearLR(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 reduced_rank: int,
                 num_bits: int,
                 block_size=64,
                 enable_lora=True,
                 bias=False,
                 ):
        super().__init__()
        self.num_bits = num_bits
        self.enable_lora = enable_lora
        self.quantizer = NFQuantizer(num_bits=num_bits)

        self.register_buffer('qweight', torch.empty((in_features * out_features // 8 * num_bits, 1), dtype=torch.uint8))
        self.register_buffer('absmax', torch.empty((in_features * out_features // block_size, 1), dtype=torch.float32))
        self.lora_A = nn.Parameter(torch.empty((reduced_rank, in_features), dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty((out_features, reduced_rank), dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.bias = None

        self.weight_size = torch.Size([out_features, in_features])
        self.weight_type = torch.float32
        self.block_size = block_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.quantizer.dequantize_block(self.qweight, self.absmax, self.weight_size, self.block_size)
        ret = input @ weight.T
        lora = (input @ self.lora_A.T) @ self.lora_B.T if self.enable_lora else 0

        return ret + lora + self.bias if self.bias else ret + lora

    def initial_backbone(self, qweight, absmax, bias=None):
        self.qweight = qweight
        self.absmax = absmax
        if not self.bias and not bias:
            self.bias = bias
        elif not bias:
            print("Warning: No bias at initialization, but pass bias")

    def initial_lora(self, lora_A, lora_B):
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B


def substitute_layer_weights_iter_quant(module,
                                        allow_name=None,
                                        block_name=None,
                                        reduced_rank=32,
                                        num_bits=4,
                                        num_iter=5,
                                        load=False,
                                        enable_lora=True):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
    assert (num_bits == 8 or num_bits == 4 or num_bits == 2) and num_iter > 0

    allow_module = [nn.Linear, Linear]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)
            linear_loras = QLinearLR(target_attr.in_features, target_attr.out_features,
                                     reduced_rank,
                                     num_bits,
                                     block_size=64,
                                     enable_lora=enable_lora,
                                     bias=target_attr.bias)

            if not load:
                weight = target_attr.weight.data
                out_feature, in_feature = weight.size()
                device = weight.device
                calibration = False
                quantizer = NFQuantizer(num_bits=num_bits, device=device, differentiable=calibration)
                res = weight.clone()

                for i in range(num_iter):
                    # Quantization
                    quantized_weight, max_abs, shape = quantizer.quantize_block(res)
                    dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)
                    res = weight - dequantized_weight

                    # Decompose the residual by SVD
                    output = low_rank_decomposition(res, reduced_rank=reduced_rank)
                    L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
                    res = weight - torch.mm(L, R)

                if num_iter == 0:
                    quantized_weight, max_abs, shape = quantizer.quantize_block(res)
                    L = torch.randn((reduced_rank, in_feature), device=device)
                    R = torch.zeros((out_feature, reduced_rank), device=device)
                linear_loras.initial_backbone(quantized_weight, max_abs)
                linear_loras.initial_lora(R, L)

            delattr(module, attr_str)
            torch.cuda.empty_cache()
            setattr(module, attr_str, linear_loras)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            substitute_layer_weights_iter_quant(immediate_child_module,
                                                allow_name=allow_name,
                                                block_name=block_name,
                                                reduced_rank=reduced_rank,
                                                num_bits=num_bits,
                                                num_iter=num_iter,
                                                load=load,
                                                enable_lora=enable_lora)

