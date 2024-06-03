import os
import random
import numpy as np
from datetime import datetime

import torch

from transformers import EarlyStoppingCallback, TrainingArguments

from datasets import load_dataset
from evaluate import load
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import is_bfloat16_supported

from model.LLM import load_model
from Dataset.MedDialogueDataset import generate_prompt_in_batch
os.environ["WANDB_PROJECT"] = "llm_training"
os.environ["WANDB_LOG_MODEL"] = "checkpoints"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

bleu = load('bleu')
meteor = load('meteor')
rouge = load('rouge')


def compute_metrics(pred, tokenizer):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)

    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    bleu4_output = bleu.compute(predictions=pred_str, references=[[ref] for ref in label_str], max_order=4)
    bleu2_output = bleu.compute(predictions=pred_str, references=[[ref] for ref in label_str], max_order=2)
    meteor_output = meteor.compute(predictions=pred_str, references=label_str)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    return {'bleu4': round(bleu4_output['bleu'], 4), 
            'bleu2': round(bleu2_output['bleu'], 4),
            'meteor': round(meteor_output['meteor'], 4),
            'rouge1': round(rouge_output['rouge1'], 4),
            'rouge2': round(rouge_output['rouge2'], 4),
            'rougeL': round(rouge_output['rougeL'], 4),
            'rougeLsum': round(rouge_output['rougeLsum'], 4)
           }

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = logits.argmax(dim=-1)

    return pred_ids, labels

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_everything(args.seed)

    # Set the arguments
    model_name = args.model_name
    lr = args.learning_rate
    batch_size = args.batch_size
    accumulation_step = args.accumulation_step
    using_scheduler = args.using_scheduler
    sch_type = args.scheduler_type
    num_epochs = args.num_epochs
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    save_name = args.save_path + '/' + model_name.split("/")[-1]
    merged_model_name = 'full_model' + '/' + model_name.split("/")[-1]
    r = args.r
    lora_dropout = args.lora_dropout
    lora_alpha = args.lora_alpha

    # Load the model
    model, tokenizer = load_model(model_name, r=r, alpha=lora_alpha, dropout=lora_dropout)

    torch.cuda.empty_cache()

    # Load the dataset
    files = {'train': train_path, 'validation': val_path, 'test': test_path}
    dataset = load_dataset('json', data_files=files)
    dataset = dataset.map(lambda x: {'prompt': generate_prompt_in_batch(x)}, batched=True, load_from_cache_file=False,
                          remove_columns=['description', 'utterances'])

    print(f'Train Dataset: {len(dataset["train"])} | Valid Dataset: {len(dataset["validation"])} | Test Dataset: {len(dataset["test"])}')

    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
                                                    instruction_template='<INST>',
                                                    response_template='</INST>',
                                                    mlm=False,
                                                    return_tensors='pt'
                                                    )

    # Early stopping callback and scheduler
    if num_epochs <= 10:
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    else:
        early_stopping = EarlyStoppingCallback(early_stopping_patience=15)

    if using_scheduler and sch_type == 'cosine':
        warmup_step = args.warmup_step
        scheduler_type = 'cosine'
    elif using_scheduler and sch_type == 'reduce_on_plateau':
        warmup_step = 0
        scheduler_type = 'reduce_on_plateau'
    else:
        warmup_step = 0
        scheduler_type = 'constant'

    # Training the model
    training_args = TrainingArguments(per_device_train_batch_size=batch_size,
                                      gradient_accumulation_steps=accumulation_step,
                                      gradient_checkpointing=True,
                                      warmup_steps=warmup_step,
                                      num_train_epochs=num_epochs,
                                      learning_rate=lr,
                                      fp16=not is_bfloat16_supported(),
                                      bf16=is_bfloat16_supported(),
                                      logging_steps=5,
                                      optim="adamw_8bit",
                                      weight_decay=0.1,
                                      lr_scheduler_type=scheduler_type,
                                      do_eval=True,
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch',
                                      load_best_model_at_end=True,
                                      greater_is_better=False,
                                      metric_for_best_model='eval_loss',
                                      seed=3407,
                                      output_dir=args.save_path,
                                      logging_dir='./logs',
                                      run_name=f'{model_name.split("/")[-1]}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
                                      report_to='wandb',
                                      dataloader_num_workers=8,
                                      dataloader_pin_memory=True,
                                      )

    trainer = SFTTrainer(model=model,
                         tokenizer=tokenizer,
                         args=training_args,
                         max_seq_length=args.max_len,
                         dataset_num_proc=2,
                         train_dataset=dataset['train'],
                         eval_dataset=dataset['validation'],
                         dataset_text_field='prompt',
                         packing=False,
                         data_collator=data_collator,
                         compute_metrics=lambda x: compute_metrics(x, tokenizer),
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                         callbacks=[early_stopping]
                         )

    tester = SFTTrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        dataset_text_field='prompt',
                        max_seq_length=args.max_len,
                        dataset_num_proc=2,
                        eval_dataset=dataset['test'],
                        packing=False,
                        data_collator=data_collator,
                        compute_metrics=lambda x: compute_metrics(x, tokenizer),
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics
                        )

    model.config.use_cache = False
    trainer.train()
    result = tester.evaluate()
    print(result)

    # save the adapter weight
    model.save_pretrained_merged(save_name, tokenizer, save_method='lora')

    # Save the merged model (base model weights + QLoRA weights)
    model.save_pretrained_merged(merged_model_name, tokenizer, save_method = "merged_4bit")
