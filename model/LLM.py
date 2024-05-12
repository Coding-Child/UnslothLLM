from unsloth import FastLanguageModel


def load_model(model_path, max_len: int = 4096, r: int = 16, alpha: int = 32, dropout: float = 0.1):
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_path,
                                                         max_seq_length=max_len,
                                                         dtype=None,
                                                         load_in_4bit=True,
                                                         device_map='auto'
                                                         )
    
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens(["<INST>", "</INST>"])  
    model.resize_token_embeddings(len(tokenizer)) 
    
    model = FastLanguageModel.get_peft_model(model=model,
                                             r=r,
                                             target_modules=["q_proj", 
                                                             "k_proj", 
                                                             "v_proj", 
                                                             "o_proj", 
                                                             "gate_proj", 
                                                             "up_proj", 
                                                             "down_proj"
                                                             ],
                                             lora_alpha=alpha,
                                             lora_dropout=dropout,
                                             use_gradient_checkpointing='unsloth',
                                             random_state=42,
                                             use_rslora=False,
                                             loftq_config=None,
                                             )

    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = load_model('unsloth/llama-2-7b-bnb-4bit')
    print(model)
