from rag import build_index, generate_answer, verify_answer, is_refusal

question = "What counts as manipulating someone's behavior?"

build_index()

answer, context = generate_answer(question)

if is_refusal(answer):
    verdict = "REFUSAL"
    explanation = "Verification skipped for refusal."
else:
    verdict, explanation = verify_answer(answer, context)

print("Answer:")
print(answer)

print("\nVerdict:")
print(verdict)