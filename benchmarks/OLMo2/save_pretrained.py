from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")
    
    model.save_pretrained("OLMo-7B-local/")
    tokenizer.save_pretrained("OLMo-7B-local/")
