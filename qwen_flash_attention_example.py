import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Qwen2 model with Flash Attention 2 and correct dtype
model_name = "Qwen/Qwen2-7B-Instruct"  # Replace with your desired Qwen2 model

# Option 1: Load model with dtype=torch.float16
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example inference
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Option 2: Use torch.autocast for mixed precision during inference
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
if __name__ == "__main__":
    prompt = "Hello, how are you?"
    response = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")