import random
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from flight_controller import episode
from jym import logits

BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_RANK = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FLY_BATCH = 32  # decisions per forward pass when flying


class Pilot:
    """Qwen3 0.6B with a LoRA adapter, which answers each question with an option letter"""

    def __init__(self, init: Path | None, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.cache: dict[str, list[int]] = {}
        # decisions with the same prompt start share its keys and values
        self.starts: list[list[int]] = []
        self.start_index: dict[tuple, int] = {}
        self.kv: dict[int, list] = {}  # keys and values per prompt start, until the weights change
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)
        if init:
            self.model = PeftModel.from_pretrained(model, str(init), is_trainable=True)
        else:
            config = LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.0, target_modules=LORA_TARGETS)
            self.model = get_peft_model(model, config)
        self.transformer = self.model.base_model.model.model  # Qwen3 with the LoRA layers
        self.letters = logits.letter_ids(self.tokenize)
        # only the option letters' rows of the output layer are needed
        weight = model.get_output_embeddings().weight
        self.letter_rows = weight[torch.tensor(self.letters, device=device)].detach().float()

    def tokenize(self, text: str) -> list[int]:
        if text not in self.cache:
            if len(self.cache) > 50_000:
                self.cache.clear()
            self.cache[text] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.cache[text]

    def prompt(self, state: dict, questions: dict) -> logits.Prompt:
        return logits.Prompt(self.tokenize, self.letters, state, questions)

    def start_id(self, prompt: logits.Prompt) -> int:
        key = tuple(prompt.start)
        if key not in self.start_index:
            self.start_index[key] = len(self.starts)
            self.starts.append(prompt.start)
        return self.start_index[key]

    def encode(self, prompt: logits.Prompt, chosen: list[int]) -> dict:
        """Encode a decision for scoring: its prompt start, the ids after it, and each answer letter's position"""
        suffix, positions = prompt.after_start(chosen)
        return {"start": self.start_id(prompt), "suffix": suffix, "positions": positions, "counts": counts(prompt)}

    def merge(self) -> None:
        """Merge the adapter into the weights for faster flying, never for training"""
        self.model = self.model.merge_and_unload()
        self.transformer = self.model.model
        self.kv.clear()

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def start_cache(self, start: int, batch: int) -> tuple[DynamicCache, int]:
        """Return a prompt start's keys and values repeated for a batch, and its length"""
        base = self.kv.get(start)
        if base is None:
            ids = torch.tensor([self.starts[start]], device=self.device)
            cache = DynamicCache()
            with torch.no_grad():
                self.transformer(input_ids=ids, past_key_values=cache, use_cache=True)
            base = [(layer.keys, layer.values) for layer in cache.layers]
            if not self.model.training:  # caching them during training would fragment GPU memory
                self.kv[start] = base
        cache = DynamicCache()
        for index, (keys, values) in enumerate(base):
            cache.update(keys.repeat_interleave(batch, 0), values.repeat_interleave(batch, 0), index)
        return cache, len(self.starts[start])

    def updated(self) -> None:
        """Drop the cached keys and values after the weights change"""
        self.kv.clear()

    def letter_logits(self, items: list[dict]) -> list[torch.Tensor]:
        """Return teacher-forced letter logits at each answer position, as a (questions x 8) tensor per decision"""
        out: list = [None] * len(items)
        groups: dict[int, list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault(item["start"], []).append(index)
        for start, members in groups.items():
            cache, length = self.start_cache(start=start, batch=len(members))
            width = max(len(items[i]["suffix"]) for i in members)
            ids = torch.zeros((len(members), width), dtype=torch.long)
            mask = torch.zeros((len(members), length + width), dtype=torch.long)
            mask[:, :length] = 1
            for row, i in enumerate(members):
                suffix = items[i]["suffix"]
                ids[row, : len(suffix)] = torch.tensor(suffix)
                mask[row, length : length + len(suffix)] = 1
            positions = (length + torch.arange(width))[None, :].expand(len(members), -1)
            hidden = self.transformer(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                position_ids=positions.to(self.device),
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
            rows = torch.tensor(
                [row for row, i in enumerate(members) for _ in items[i]["positions"]], device=self.device
            )
            cols = torch.tensor([p for i in members for p in items[i]["positions"]], device=self.device)
            sizes = [len(items[i]["positions"]) for i in members]
            scores = (hidden[rows, cols].float() @ self.letter_rows.T).split(sizes)
            for i, values in zip(members, scores, strict=True):
                out[i] = values
        return out

    def log_probabilities(self, items: list[dict]) -> list[torch.Tensor]:
        """Return the log-softmax over each question's offered letters, masking the rest"""
        out = []
        for values, item in zip(self.letter_logits(items), items, strict=True):
            limit = torch.tensor(item["counts"], device=values.device)[:, None]
            offered = torch.arange(values.shape[1], device=values.device)[None, :] < limit
            out.append(torch.log_softmax(values.masked_fill(~offered, -1e9), dim=-1))
        return out

    def act(self, prompts: list[logits.Prompt], greedy: bool) -> list[list[int]]:
        """Choose each question's option in turn, greedily or by sampling"""
        results: list = [None] * len(prompts)
        groups: dict[int, list[int]] = {}
        for index, prompt in enumerate(prompts):
            groups.setdefault(self.start_id(prompt), []).append(index)
        self.model.eval()
        with torch.no_grad():
            for start, members in groups.items():
                for at in range(0, len(members), FLY_BATCH):
                    chunk = members[at : at + FLY_BATCH]
                    picks = self.decode(start, [prompts[i] for i in chunk], greedy)
                    for index, chosen in zip(chunk, picks, strict=True):
                        results[index] = chosen
        return results

    def decode(self, start: int, prompts: list[logits.Prompt], greedy: bool) -> list[list[int]]:
        """Answer each prompt's questions in turn, all sharing one prompt start"""
        cache, length = self.start_cache(start=start, batch=len(prompts))
        first = [prompt.end + prompt.answer_prefixes[0] for prompt in prompts]
        width = max(map(len, first))
        ids = torch.zeros((len(first), width), dtype=torch.long)
        mask = torch.zeros((len(first), length + width), dtype=torch.long)
        positions = torch.full((len(first), width), length, dtype=torch.long)
        mask[:, :length] = 1
        for row, sequence in enumerate(first):  # left-pad so every row ends in the same column
            pad = width - len(sequence)
            ids[row, pad:] = torch.tensor(sequence)
            mask[row, length + pad :] = 1
            positions[row, pad:] = length + torch.arange(len(sequence))
        ids, mask, positions = ids.to(self.device), mask.to(self.device), positions.to(self.device)
        chosen: list[list[int]] = [[] for _ in prompts]
        for question in range(len(prompts[0].questions)):
            if question:
                step = [
                    [p.letters[question - 1][chosen[row][-1]], *p.answer_prefixes[question]]
                    for row, p in enumerate(prompts)
                ]
                ids = torch.tensor(step, device=self.device)
                positions = positions[:, -1:] + torch.arange(1, ids.shape[1] + 1, device=self.device)[None, :]
                mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            hidden = self.transformer(
                input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True
            ).last_hidden_state[:, -1]
            scores = hidden.float() @ self.letter_rows.T
            count = torch.tensor([len(p.options[question]) for p in prompts], device=self.device)[:, None]
            scores = scores.masked_fill(torch.arange(scores.shape[1], device=self.device)[None, :] >= count, -1e9)
            picks = scores.argmax(-1) if greedy else torch.multinomial(torch.softmax(scores, -1), 1)[:, 0]
            for row, pick in enumerate(picks.tolist()):
                chosen[row].append(pick)
        return chosen

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(folder))


def choice_indices(prompt: logits.Prompt, choices: dict[str, str]) -> list[int]:
    return [options.index(choices[qid]) for qid, options in zip(prompt.questions, prompt.options, strict=True)]


def choice_names(prompt: logits.Prompt, chosen: list[int]) -> dict[str, str]:
    return {qid: options[i] for qid, options, i in zip(prompt.questions, prompt.options, chosen, strict=True)}


def counts(prompt: logits.Prompt) -> list[int]:
    return [len(options) for options in prompt.options]


def by_start(items: list[dict], size: int, rng: random.Random) -> list[list[int]]:
    """Batch the indices of items that share a prompt start, in random order"""
    groups: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item["start"], []).append(index)
    batches = []
    for members in groups.values():
        rng.shuffle(members)
        batches += [members[at : at + size] for at in range(0, len(members), size)]
    rng.shuffle(batches)
    return batches


class Critic:
    """Estimate a state's value from Episode.features, which the policy never sees"""

    def __init__(self, device: str):
        self.device = device
        layers = [torch.nn.Linear(episode.FEATURES, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh()]
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(256, 1)).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def __call__(self, features: list[list[float]]) -> torch.Tensor:
        return self.net(torch.tensor(features, dtype=torch.float32, device=self.device))[:, 0]
