# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from consistency_workflow.py. Keep workflow changes minimal; all SAART-specific
# logic lives in saart_trainer.py. This file only swaps in SAARTSeq2SeqTrainer.

from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .saart_trainer import SAARTSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_saart_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # SAART inserts a single trigger after BOS / at the prompt end; with sequence packing
    # one packed sequence holds several prompt-response pairs, which would corrupt the
    # insertion and the response mask. Disallow packing for SAART.
    if getattr(data_args, "packing", None):
        raise ValueError("SAART does not support sequence packing; set `packing: false`.")

    # Phase-2：开启在线 MLP 关联签名(L_assoc-reg)时，需要 adv 分支的 MLP 激活带梯度才能反传到 LoRA。
    # 做法：保持梯度检查点开启，但切到【非重入】模式(use_reentrant=False)——见 checkpointing.py 的说明。
    #   · 重入(reentrant)检查点：初始前向不建图 → forward-hook 抓到的激活不带梯度 → L_assoc-reg 断梯度；
    #   · 非重入(use_reentrant=False)：初始前向正常建图、仅丢弃中间张量待反向重算 → hook 激活仍带梯度，
    #     同时保留 GC 的省显存特性。这样兼顾"adv 激活带梯度"与"省显存"，解决两个实测问题：
    #     (1) gemma-9B 关 GC 时 ~42GB 超 40GB 卡装不下；(2) qwen2 关 GC 时 bf16 数值不稳→首步梯度范数为 0/NaN
    #         触发 DeepSpeed `assert all_groups_norm > 0`。
    # 通过运行期标记 model_args.saart_nonreentrant_gc 传给 checkpointing.prepare_model_for_training。
    # TODO(SAART-P2): 若个别模型在非重入 GC 下 assoc 梯度仍不稳，可回退到"完全关 GC"(disable_gradient_checkpointing=True)
    #   或"仅 adv 那次前向临时关 GC"的 per-forward toggle 方案。
    if getattr(finetuning_args, "use_saart", False) and getattr(finetuning_args, "use_assoc_reg", False):
        model_args.disable_gradient_checkpointing = False   # 不关 GC，改用非重入模式
        setattr(model_args, "saart_nonreentrant_gc", True)  # 运行期标记：checkpointing.py 据此用 use_reentrant=False
        logger.info("[SAART Phase-2] use_assoc_reg=True → 梯度检查点切到非重入(use_reentrant=False)，省显存且 adv 激活带梯度")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # Initialize our Trainer
    trainer = SAARTSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset, predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
