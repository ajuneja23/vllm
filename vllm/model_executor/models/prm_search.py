from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
import os


class BeamForCausalLM(nn.Module):
    def __init__(self, n=8, m=2, beam_expansions=5, temp=0.7):
        super(BeamForCausalLM, self).__init__()
        self.n, self.m, self.beam_expansions, self.temp = n, m, beam_expansions, temp

    def load_prm(self):
        model_path = "models/Qwen2.5-Math-PRM-7B"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-Math-PRM-7B",
                trust_remote_code=True,
                local_files_only=False,
            )
            tokenizer.save_pretrained(model_path)
            model = AutoModel.from_pretrained(
                "Qwen/Qwen2.5-Math-PRM-7B",
                trust_remote_code=True,
                local_files_only=False,
            )
            model.save_pretrained(model_path)
            return model, tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            return model, tokenizer

    def load_math_model(self):
        model_path = "models/Qwen2.5-Math-1.5B-Instruct"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Math-1.5B-Instruct"
            )
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            return model, tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            return model, tokenizer

    def beamSearch(self, query, n=8, m=2, beam_expansions=5, temp=0.7):
        device = "auto"
        prmModel, prmTokenizer = self.load_prm()
        model, tokenizer = self.load_math_model()
        model = model.to(device)
        tokenizer = tokenizer.to(device)
        paths = []
        for i in range(beam_expansions):
            if i == 0:
                for _ in range(n):
                    tokens = tokenizer(
                        query,
                        return_tensors="pt",
                        add_special_tokens=False,
                        max_length=100,
                        truncation=True,
                        padding="longest",
                    )
                    input_ids = tokens["input_ids"]
                    for step in range(50):
                        output = model(
                            input_ids=input_ids, output_hidden_states=True
                        ).logits
                        next_token_logits = output[:, -1, :]

                        next_token_logits = next_token_logits / temp

                        probabilities = F.softmax(next_token_logits, dim=-1)

                        next_token = torch.multinomial(probabilities, num_samples=1)

                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        eos_token_id = tokenizer.eos_token_id
                        next_token = next_token.squeeze(-1).item()
                        newline_token_id = tokenizer.encode(
                            "\n", add_special_tokens=False
                        )[0]
                        if (
                            next_token == newline_token_id
                            or next_token == eos_token_id
                            or step == 49
                        ):
                            decoded_output = tokenizer.decode(
                                input_ids[0], skip_special_tokens=True
                            )
                            paths.append(decoded_output)
                            break
                for j in range(len(paths)):  # score with PRM
                    scores = []
                    prmTokens = prmTokenizer.encode(paths[j], return_tensors="pt").to(
                        model.device
                    )
                    output = prmModel(input_ids=prmTokens)
                    scores.append(output[0])
                scores = [[scores[i], i] for i in range(len(scores))]
                scores.sort()
                cutoff = int(n / m)
                scores = scores[:cutoff]
                paths2 = []
                for [_, i] in scores:
                    paths2.append([paths[i]])
                paths = paths2
            else:
                paths2 = []
                for path in paths:
                    for _ in range(m):
                        baseContext = "".join(path)
                        tokens = tokenizer(
                            baseContext,
                            return_tensors="pt",
                            add_special_tokens=False,
                            max_length=200,
                            truncation=True,
                            padding="longest",
                        )
                        input_ids = tokens["input_ids"]
                        for step in range(50):
                            output = model(
                                input_ids=input_ids, output_hidden_states=True
                            ).logits
                            next_token_logits = output[:, -1, :]

                            next_token_logits = next_token_logits / temp

                            probabilities = F.softmax(next_token_logits, dim=-1)

                            next_token = torch.multinomial(probabilities, num_samples=1)

                            input_ids = torch.cat([input_ids, next_token], dim=-1)
                            eos_token_id = tokenizer.eos_token_id
                            next_token = next_token.squeeze(-1).item()
                            newline_token_id = tokenizer.encode(
                                "\n", add_special_tokens=False
                            )[0]
                            if (
                                next_token == newline_token_id
                                or next_token == eos_token_id
                                or step == 49
                            ):
                                decoded_output = tokenizer.decode(
                                    input_ids[0], skip_special_tokens=True
                                )
                                paths2.append(decoded_output)
                                break
                for j in range(len(paths2)):
                    scores = []
                    prmTokens = prmTokenizer.encode(paths2[j], return_tensors="pt").to(
                        model.device
                    )
                    output = prmModel(input_ids=prmTokens)
                    scores.append(output[0])
                scores = [[scores[i], i] for i in range(len(scores))]
                scores.sort()
                cutoff = int(n / m)
                scores = scores[:cutoff]
                paths2 = []
                for score, i in scores:
                    paths2.append([paths2[i]])
                paths = paths2
        return paths[0]

    def forward(self, prompt):
        return self.beamSearch(prompt, self.n, self.m, self.beam_expansions, self.temp)
