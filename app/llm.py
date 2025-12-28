import requests

# HF Qwen Space endpoint
QWEN_API = "https://muaazasif-qwen-book-chatbot.hf.space/chat"

def generate_answer(question: str, context: str) -> str:
    """
    Strict document-based answer generation using HF Qwen Space API.
    Returns 'Not found in document' if context does not contain answer clues.
    Cleans output to remove prompt instructions or repeated lines.
    """

    # Build strict prompt
    prompt = f"""
You are a strict document-based assistant.
Answer ONLY using the context below.
Summarize in MAXIMUM 3 sentences.
Do NOT repeat any sentences.
Do NOT include headings or chapter numbers.
If the answer is not present in the context, reply exactly:
Not found in document.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        # Call Qwen Space API
        res = requests.post(QWEN_API, params={"prompt": prompt}, timeout=60)
        raw_answer = res.json().get("response", "No response from Qwen.")

        # --- Clean the response ---
        # Take text after "Answer:" if present
        if "Answer:" in raw_answer:
            answer_text = raw_answer.split("Answer:")[-1].strip()
        else:
            answer_text = raw_answer.strip()

        # Remove any leftover instruction phrases
        for phrase in [
            "You are a strict document-based assistant",
            "Do NOT repeat any sentences",
            "Summarize in MAXIMUM 3 sentences",
            "Do NOT include headings or chapter numbers",
            "If the answer is not present in the context, reply exactly"
        ]:
            answer_text = answer_text.replace(phrase, "").strip()

        # Fallback if empty
        if not answer_text:
            answer_text = "Not found in document"

        return answer_text

    except Exception as e:
        return f"Error contacting Qwen: {str(e)}"
