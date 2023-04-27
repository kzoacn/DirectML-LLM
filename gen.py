from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import functional as F
import torch_directml
device = torch_directml.device(torch_directml.default_device()) 
# device = 'cpu'

# checkpoint = "Langboat/bloom-2b5-zh"
checkpoint = "Langboat/bloom-389m-zh"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float16).to(device)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map='auto', load_in_8bit=True).to(device)

#ctx=torch.amp.autocast(device_type='cuda',dtype=torch.float32)
model.eval()

inputs = tokenizer.encode("中国的首都是", return_tensors="pt").to(device)  

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[-1023:]
        out = model(idx_cond)
        logits=out['logits'] 
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1).float()
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx

outputs=generate(model,inputs,100,0.9,20) 
print(tokenizer.decode(outputs[0]))
