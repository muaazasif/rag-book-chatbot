from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32
).to("cpu")


def generate_answer(question: str, context: str) -> str:
    """
    Strict document-based answer generation.
    Returns 'Not found in document' if the context does not contain answer clues.
    """
    
    # Guard: check if question keywords exist in context
    question_keywords = [word.lower() for word in question.split()]
    context_lower = context.lower()
    if not any(kw in context_lower for kw in question_keywords):
        return "Not found in document"

    # Prompt strictly instructing LLM to use ONLY context
    prompt = f"""
You are a strict document-based assistant.
Answer ONLY using the context below.
Do NOT use prior knowledge or add any information not present in the context.
If the answer is not present in the context, reply exactly:
Not found in document.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,        # deterministic
        temperature=0.0,        # no randomness
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_text = answer.split("FINAL ANSWER:")[-1].strip()

    # Extra guard: if none of the answer words appear in context, return strict 'Not found'
    if not any(word.lower() in context_lower for word in answer_text.split()):
        return "Not found in document"

    return answer_text