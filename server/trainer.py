import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

from tqdm.auto import tqdm
# import wandb

from utils import compute_f1
import time

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


def fp16_zo_eval(func):
    def wrapper(self, *args, **kwargs):
        zo_eval_dtype = torch.float16
        dtype = self.model.dtype
        if self.args.optimizer == "zo" and dtype != torch.float16:
            # self.model.to(zo_eval_dtype)
            # print("Switched to fp16 for evaluation")
            copy_learnable_params = dict()
            # print(f"Max memory 0: {torch.cuda.max_memory_allocated() / 1024**3}")
            for name, param in self.trainable_parameters.items():
                copy_learnable_params[name] = param.clone().detach()
            # print(f"Max memory 1: {torch.cuda.max_memory_allocated() / 1024**3}")
            self.model.to(zo_eval_dtype)
            # Make sure the model is in fp16
            for name, param in self.trainable_parameters.items():
                assert param.dtype == torch.float16, f"Param {name} is not in fp16"
            print("Switched to fp16 for evaluation")
        
        # print(f"Max memory 2: {torch.cuda.max_memory_allocated() / 1024**3}")
        # Clear memory
        torch.cuda.empty_cache()
        result = func(self, *args, **kwargs)
        # print(f"Max memory 3: {torch.cuda.max_memory_allocated() / 1024**3}")

        if self.args.optimizer == "zo" and dtype != torch.float16:
        #     self.model.to(dtype)
        #     print("Switched back to original dtype")
            for name, param in self.trainable_parameters.items():
                param.data = copy_learnable_params[name].data
            self.model.to(dtype)
            # Make sure the model is back to original dtype
            for name, param in self.trainable_parameters.items():
                assert param.dtype == dtype, f"Param {name} is not in original dtype"
            print("Switched back to original dtype")
        
        return result
    return wrapper



class Trainer:
    def __init__(self, args, model, tokenizer, train_dataloader, eval_dataloader, accelerator, cls_idx=None, optimizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        # self.embedding_keywords = ["embed_tokens", "lm_head", "wte", "wpe"]
        # for name, param in model.named_parameters():
        #     if any(k in name for k in self.embedding_keywords):
        #         param.requires_grad = False

        self.trainable_parameters = dict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.trainable_parameters[name] = param
        self.rand_time = 0.0
        self.quantized_parameter_names = set()
        self.zo_generator = torch.Generator(device=self.accelerator.device)
        self.zo_candidate_names = list(self.trainable_parameters.keys())
        run_name_file = getattr(self.args, "run_name_file", getattr(self.args, "run_name", "run"))
        os.makedirs("output", exist_ok=True)
        self.eval_log_path = os.path.join("output", f"{run_name_file}.jsonl")
        os.makedirs("loss", exist_ok=True)
        self.loss_log_path = os.path.join("loss", f"{run_name_file}.jsonl")
        with open(self.eval_log_path, "w", encoding="utf-8") as f:
            meta = {
                "run_name": getattr(self.args, "run_name", run_name_file),
                "format": "jsonl",
                "int4_ratio": f"{getattr(self.args, 'mixed_int4_ratio', 0.0):.2f}",
                "int8_ratio": f"{getattr(self.args, 'mixed_int8_ratio', 0.0):.2f}",
                "enable_early_exit": bool(getattr(self.args, "enable_early_exit", False)),
                "gamma": float(getattr(self.args, "gamma", 1.0)),
                "max_resample_k": int(getattr(self.args, "max_resample_k", 0)),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.eval_history = []
        self.total_step = 0
        self.loss_ema_mu = None
        self.loss_ema_sq = None
        self.loss_ema_beta = 0.95
        
        get_trainable_parameters(model)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criteria = nn.CrossEntropyLoss()

        if self.args.optimizer == "fo":
            assert args.n == 1, "Only n=1 is supported for first-order optimization"
            self.one_step = self.one_step_fo
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
            self.optimizer = optimizer
        else:
            if self.args.zo_mode == "single":
                self.one_step = self.one_step_zo_single
            else:
                self.one_step = self.one_step_zo_dual
        
        if self.args.task_name == "copa" or self.args.task_name == "winogrande":
            self.eval = self.eval_mch
        elif self.args.task_name == "squad" or self.args.task_name == "drop":
            self.eval = self.eval_qa
        else:
            self.eval = self.eval_cls
            self.cls_idx_list = list(cls_idx)

        if self.args.enable_mixed_layer_quantization:
            self.apply_layerwise_mixed_quantization()
            self.refresh_trainable_state()
        self.refresh_zo_candidates()

    def refresh_trainable_state(self):
        self.trainable_parameters = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.trainable_parameters[name] = param

    def refresh_zo_candidates(self):
        self.zo_candidate_names = [
            name for name in self.trainable_parameters.keys()
            if name not in self.quantized_parameter_names
        ]
        self.device_generators = {}
        for name in self.zo_candidate_names:
            device_key = str(self.trainable_parameters[name].device)
            if device_key not in self.device_generators:
                self.device_generators[device_key] = torch.Generator(device=self.trainable_parameters[name].device)

    def _record_train_loss(self, current_loss):
        record = {
            "step": int(self.global_step),
            "total_step": int(self.total_step),
            "loss": float(current_loss)
        }
        with open(self.loss_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _record_eval(self, accuracy=None, metric_name=None, metric_value=None):
        record = {
            "step": int(self.global_step),
            "total_step": int(self.total_step),
            "epoch": int(self.epoch),
        }
        if accuracy is not None:
            record["accuracy"] = float(accuracy)
        if metric_name is not None and metric_value is not None:
            record[metric_name] = float(metric_value)
        self.eval_history.append(record)
        with open(self.eval_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_loss_ema(self, loss_value):
        if self.loss_ema_mu is None:
            self.loss_ema_mu = float(loss_value)
            self.loss_ema_sq = float(loss_value) * float(loss_value)
            return
        beta = self.loss_ema_beta
        self.loss_ema_mu = beta * self.loss_ema_mu + (1.0 - beta) * float(loss_value)
        self.loss_ema_sq = beta * self.loss_ema_sq + (1.0 - beta) * float(loss_value) * float(loss_value)

    def _get_loss_stats(self):
        if self.loss_ema_mu is None:
            return None, None
        var = max(self.loss_ema_sq - self.loss_ema_mu * self.loss_ema_mu, 0.0)
        return self.loss_ema_mu, var ** 0.5

    @staticmethod
    def _pseudo_quantize_tensor(weight, bits):
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        max_val = weight.abs().max()
        if max_val == 0:
            return weight
        scale = max_val / qmax
        q = torch.clamp(torch.round(weight / scale), qmin, qmax)
        return q * scale

    def _get_calibration_batches(self, max_batches):
        batches = []
        for step, batch in enumerate(self.train_dataloader):
            batches.append(batch)
            if step + 1 >= max_batches:
                break
        return batches

    def _compute_avg_loss(self, batches):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in batches:
                outputs = self.model(**batch)
                losses.append(outputs.loss.detach())
        self.model.train()
        if not losses:
            return 0.0
        return torch.stack(losses).mean().item()

    def _estimate_layer_sensitivity(self, param_name, param, base_loss, calib_batches):
        original = param.data.clone()
        quantized = self._pseudo_quantize_tensor(original, bits=4)
        param.data.copy_(quantized)
        quantized_loss = self._compute_avg_loss(calib_batches)
        mse = torch.mean((quantized - original) ** 2).item()
        param.data.copy_(original)
        return (quantized_loss - base_loss) / (mse + 1e-3)

    def _compute_quant_gain(self, param_num, bits):
        delta_memory = param_num * (16 - bits)
        delta_compute = param_num / float(bits)
        return self.args.quant_alpha * delta_memory + self.args.quant_beta * delta_compute

    def _compute_layer_score(self, risk, param_num, bits):
        return self._compute_quant_gain(param_num, bits) - risk

    def apply_layerwise_mixed_quantization(self):
        print("Mixed layer quantization enabled, estimating sensitivities...")
        calib_batches = self._get_calibration_batches(self.args.mixed_quant_calib_batches)
        if not calib_batches:
            print("No calibration batch found, skip mixed layer quantization.")
            return

        base_loss = self._compute_avg_loss(calib_batches)

        candidates = []
        total_params = 0
        for name, param in self.model.named_parameters():
            if (not param.requires_grad) or param.dim() < 2 or (not name.endswith("weight")):
                continue

            layer_params = param.numel()
            risk = self._estimate_layer_sensitivity(name, param, base_loss, calib_batches)
            score_int4 = self._compute_layer_score(risk=risk, param_num=layer_params, bits=4)
            score_int8 = self._compute_layer_score(risk=risk, param_num=layer_params, bits=8)
            candidates.append((name, score_int4, score_int8, layer_params))
            total_params += layer_params

        if total_params == 0:
            print("No eligible weight layer found for mixed quantization.")
            return

        int4_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
        int8_candidates = sorted(candidates, key=lambda item: item[2], reverse=True)

        target_int4 = total_params * self.args.mixed_int4_ratio
        target_int8 = total_params * self.args.mixed_int8_ratio

        int4_params = 0
        int8_params = 0
        int4_full = False
        model_params = dict(self.model.named_parameters())

        for name, _, _, layer_params in int4_candidates:
            if int4_full:
                break
            if int4_params + layer_params < target_int4:
                param = model_params[name]
                param.data.copy_(self._pseudo_quantize_tensor(param.data, bits=4))
                param.requires_grad = False
                self.quantized_parameter_names.add(name)
                int4_params += layer_params
            else:
                int4_full = True

        for name, _, _, layer_params in int8_candidates:
            if name in self.quantized_parameter_names:
                continue
            if int8_params + layer_params < target_int8:
                param = model_params[name]
                param.data.copy_(self._pseudo_quantize_tensor(param.data, bits=8))
                param.requires_grad = False
                self.quantized_parameter_names.add(name)
                int8_params += layer_params

        print(
            f"Mixed quantization stats: total={total_params}, "
            f"int4={int4_params}, int8={int8_params}, "
            f"int4_ratio={int4_params / total_params:.4f}, int8_ratio={int8_params / total_params:.4f}"
        )

    def zero_shot_eval(self):
        self.global_step = 0
        self.total_step = 0
        self.max_accuracy = 0
        self.epoch = 0
        self.eval()

    def train(self):
        progress_bar = tqdm(total=min(self.args.max_iterations, self.args.num_train_epochs * len(self.train_dataloader)))
        self.global_step = 0
        self.total_step = 0
        self.max_accuracy = 0
        for self.epoch in range(self.args.num_train_epochs):
            for batch in self.train_dataloader:
                resample_count = 0
                while True:
                    if self.global_step >= self.args.max_iterations:
                        return
                    
                    self.total_step += 1
                    self.rand_seed = torch.randint(0, 10000000, (1,)).item()

                    updated = True
                    snr = None
                    if self.args.optimizer == "zo" and self.args.zo_mode == "single":
                        loss, updated, snr = self.one_step(batch)
                    else:
                        loss = self.one_step(batch)

                    postfix = {"loss": loss.item(), "updated": int(updated)}
                    if snr is not None:
                        postfix["snr"] = round(float(snr), 4)
                    progress_bar.set_postfix(postfix)

                    if updated:
                        self.global_step += 1
                        self._record_train_loss(loss.item())
                        progress_bar.update(1)
                        if self.global_step % self.args.logging_steps == 0:
                            self.eval()
                        if self.global_step >= self.args.max_iterations:
                            return
                        break

                    resample_count += 1
                    if resample_count >= self.args.max_resample_k:
                        break


    def one_step_fo(self, batch):
        self.model.train()
        outputs = self.model(**batch)
        loss = outputs.loss
        # loss.backward()
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss


    def one_step_zo_single(self, batch):
        self.model.train()
        with torch.no_grad():
            input_ids = batch['input_ids'].repeat(self.args.n, 1)
            attention_mask = batch['attention_mask'].repeat(self.args.n, 1)
            labels = batch['labels'].repeat(self.args.n, 1)

            st_rand_time = time.time()
            saved_z = {} 
            self.zo_generator.manual_seed(self.rand_seed)
            for name in self.zo_candidate_names:
                param = self.trainable_parameters[name]

                if self.args.lowrank and param.dim() == 2:
                    rows, cols = param.shape
                    u = torch.randn((rows, 1), dtype=param.dtype, device=param.device, generator=self.zo_generator)
                    v = torch.randn((1, cols), dtype=param.dtype, device=param.device, generator=self.zo_generator)
                    z = torch.matmul(u, v)
                else:
                    z = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=self.zo_generator)

                saved_z[name] = z
                param.add_(z, alpha=self.args.eps)
            
            ed_rand_time = time.time()
            self.rand_time += (ed_rand_time - st_rand_time)
            
            logits = self.model(input_ids, attention_mask, return_dict=False)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            del logits

            # Compute loss for each n
            shift_logits = shift_logits.view(self.args.n, -1, shift_logits.size(-1))
            shift_labels = shift_labels.view(self.args.n, -1)

            loss1 = torch.stack([self.criteria(shift_logits[i], shift_labels[i]) for i in range(self.args.n)])
            loss1_mean = loss1.mean()
            loss1_value = float(loss1_mean.item())

            del shift_logits, shift_labels

            early_exit_enabled = bool(getattr(self.args, "enable_early_exit", False))
            if early_exit_enabled:
                mu_l, sigma_l = self._get_loss_stats()
                snr = None
                should_skip_update = False
                if mu_l is not None and sigma_l is not None:
                    delta_l = abs(loss1_value - mu_l)
                    snr = delta_l / max(sigma_l, 1e-8)
                    should_skip_update = snr < self.args.gamma
                self._update_loss_ema(loss1_value)

                if should_skip_update:
                    st_rand_time = time.time()
                    for name, z in saved_z.items():
                        param = self.trainable_parameters[name]
                        param.sub_(z, alpha=self.args.eps)
                    del saved_z
                    ed_rand_time = time.time()
                    self.rand_time += (ed_rand_time - st_rand_time)
                    return loss1_mean, False, snr
            else:
                snr = None
            
            st_rand_time = time.time()
            for name, z in saved_z.items():
                param = self.trainable_parameters[name]
                param.sub_(z, alpha=2.0 * self.args.eps)
            ed_rand_time = time.time()
            self.rand_time += (ed_rand_time - st_rand_time)

            logits = self.model(input_ids, attention_mask, return_dict=False)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            del logits
            # Compute loss for each n
            shift_logits = shift_logits.view(self.args.n, -1, shift_logits.size(-1))
            shift_labels = shift_labels.view(self.args.n, -1)

            loss2 = torch.stack([self.criteria(shift_logits[i], shift_labels[i]) for i in range(self.args.n)])
            del shift_logits, shift_labels

            projected_grad = (loss1 - loss2) / (2.0 * self.args.eps)
            for name, z in saved_z.items():
                param = self.trainable_parameters[name]
                param.add_(z, alpha=self.args.eps)
                g = projected_grad.to(param.device)
                if self.args.n == 1:
                    delta = -self.args.learning_rate * g * z                                        
                else:                              
                    view_shape = [self.args.n] + [1] * (z.dim() - 1)
                    delta = -self.args.learning_rate * (g.view(*view_shape) * z).mean(dim=0, keepdim=True)
                if getattr(self.args, "zo_sign", False):
                    delta = -self.args.learning_rate * torch.sign(delta)
                param.data.add_(delta)

            del saved_z
            ed_rand_time = time.time()
            self.rand_time += (ed_rand_time - st_rand_time)

        return loss1_mean, True, snr
      
    def one_step_zo_dual(self, batch):
        self.model.train()
        with torch.no_grad():
            input_ids = batch['input_ids'].repeat(2*self.args.n, 1)
            attention_mask = batch['attention_mask'].repeat(2*self.args.n, 1)
            labels = batch['labels'].repeat(2*self.args.n, 1)

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                z = self.args.eps * torch.randn_like(param.data[:self.args.n])
                param.data[:self.args.n].add_(z)
                param.data[self.args.n:].sub_(z)

            logits = self.model(input_ids, attention_mask, return_dict=False)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            del logits

            # Compute loss for each n
            shift_logits = shift_logits.view(2, self.args.n, -1, shift_logits.size(-1))
            shift_labels = shift_labels.view(2, self.args.n, -1)

            loss1 = torch.stack([self.criteria(shift_logits[0, i], shift_labels[0, i]) for i in range(self.args.n)])
            loss2 = torch.stack([self.criteria(shift_logits[1, i], shift_labels[1, i]) for i in range(self.args.n)])

            del shift_logits, shift_labels

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                z = self.args.eps * torch.randn_like(param.data[:self.args.n])
                param.data[:self.args.n].sub_(z)
                param.data[self.args.n:].add_(z)
            
            projected_grad = (loss1 - loss2) / (2.0 * self.args.eps)

            torch.manual_seed(self.rand_seed)
            for name, param in self.trainable_parameters.items():
                projected_grad = projected_grad.to(param.device)
                if self.args.n == 1:
                    z = self.args.learning_rate * projected_grad * torch.randn_like(param.data[:self.args.n])
                else:
                    z = self.args.learning_rate * (projected_grad.view(-1, 1, 1) * torch.randn_like(param.data[:self.args.n])).mean(dim=0, keepdim=True)
                param.data[:self.args.n].sub_(z)
                param.data[self.args.n:].sub_(z)

        return loss1.mean()

    @fp16_zo_eval
    def eval_cls(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for step, batch in enumerate(self.eval_dataloader): 
                labels = batch["input_ids"][:, -1]
                batch.pop("labels")
                outputs = self.model(**batch)

                logits = outputs.logits

                cls_logits = logits[:, -2, self.cls_idx_list]  # shape: [batch_size, num_classes]

                pred_idx_in_list = cls_logits.argmax(dim=-1)  # returns index in cls_idx_list
                predictions = torch.tensor([self.cls_idx_list[idx] for idx in pred_idx_in_list], device=labels.device)

                correct += (predictions == labels).sum().item()
                total += len(labels)

            accuracy = correct / total
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy

            print(f"step {self.global_step} accuracy: {accuracy}")
            # wandb.log({"eval/accuracy": accuracy ,"epoch": self.epoch, "step": self.global_step, "eval/max_accuracy": self.max_accuracy})
            self._record_eval(accuracy=accuracy)

    @fp16_zo_eval
    def eval_mch(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for step, batch in enumerate(self.eval_dataloader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = self.model(input_ids, attention_mask=attention_mask)

                logits = outputs.logits

                log_probs = F.log_softmax(logits, dim=-1)  # Shape: [bsz, seq_len, vocab_size]

                # iterate over the batch every two examples
                for i in range(0, len(log_probs), 2):
                    valid_len1 = (labels[i] != -100).sum()
                    valid_len2 = (labels[i + 1] != -100).sum()

                    valid_log_probs1 = log_probs[i, -(valid_len1+1):-1]
                    valid_log_probs2 = log_probs[i + 1, -(valid_len2+1):-1]

                    valid_log_probs1 = valid_log_probs1[range(len(valid_log_probs1)), labels[i, -valid_len1:]]
                    valid_log_probs2 = valid_log_probs2[range(len(valid_log_probs2)), labels[i + 1, -valid_len2:]]

                    valid_log_probs1 = valid_log_probs1.mean()
                    valid_log_probs2 = valid_log_probs2.mean()

                    correct += (valid_log_probs1 > valid_log_probs2).item()
                    total += 1
            
            accuracy = correct / total
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy

            print(f"step {self.global_step} accuracy: {accuracy}")
            # wandb.log({"eval/accuracy": accuracy ,"epoch": self.epoch, "step": self.global_step, "eval/max_accuracy": self.max_accuracy})
            self._record_eval(accuracy=accuracy)

    @fp16_zo_eval    
    def eval_qa(self):
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            all_f1_scores = []
            samples = 0
            for step, batch in enumerate(self.eval_dataloader):
                for i in range(len(batch['input_ids'])):
                    input_ids = batch['input_ids'][i].unsqueeze(0)
                    labels = batch['labels'][i]

                    valid_len = (labels != -100).sum()
                    valid_labels = labels[-valid_len:]

                    valid_input_ids = input_ids[:, :-valid_len]

                    outputs = self.model.generate(
                        valid_input_ids, do_sample=False, temperature=1.0, max_new_tokens=50,
                        num_beams=1, top_p=0.95, top_k=None, num_return_sequences=1, 
                        eos_token_id=[self.tokenizer.encode("\n", add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
                    )

                    outputs = outputs[0][valid_input_ids.size(1):]

                    # Convert tensors to strings for F1 computation
                    pred_str = self.tokenizer.decode(outputs, skip_special_tokens=True)
                    label_str = self.tokenizer.decode(valid_labels, skip_special_tokens=True)

                    # print(pred_str, "#######", label_str)
                    all_f1_scores.append(compute_f1([pred_str], [label_str]))
                    samples += 1
                print(f"sample {samples}, avg f1: {np.mean(all_f1_scores)}")
            print("Evaluation time: ", time.time() - start_time)
            avg_f1 = np.mean(all_f1_scores)
            if avg_f1 > self.max_accuracy:
                self.max_accuracy = avg_f1

            print(f"step {self.global_step} f1: {avg_f1}")
            # wandb.log({"eval/f1": avg_f1, "epoch": self.epoch, "step": self.global_step, "eval/max_f1": self.max_accuracy})
            self._record_eval(metric_name="f1", metric_value=avg_f1)
            # Exit if the f1 score is 1.0
            if avg_f1 < 0.1:
                exit()
