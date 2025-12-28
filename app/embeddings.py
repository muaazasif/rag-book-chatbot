from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-small")  # small & fast, CPU
def embed(text: str):
    return model.encode(text)
