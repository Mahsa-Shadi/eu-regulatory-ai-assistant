from eval_set import eval_set
from rag import build_index, retrieve, generate_answer, verify_answer, LLMError, LLM_MODEL, is_refusal
import json
from datetime import datetime
from pathlib import Path
import time



BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = RESULTS_DIR / f"eval_results_{timestamp}.json"



def save_evaluation_results(report):
    temp_path = results_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.flush()

    temp_path.replace(results_path)




def save_progress(retrieval_results, generation_results, retrieval_accuracy):
    progress_report = {
        "evaluation_version": "v2",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL,
        "status": "in_progress",
        "metrics": {
            "retrieval_accuracy": retrieval_accuracy,
        },
        "counts": {
            "total_cases": len(eval_set),
            "generation_results_saved": len(generation_results),
        },
        "retrieval_results": retrieval_results,
        "generation_results": generation_results,
    }

    save_evaluation_results(progress_report)

build_index()

retrieval_results = []

for case in eval_set:

    question = case['question']

    retrieved = retrieve(question)

    chunk_ids = [
        chunk_id
        for chunk_id, document, distance in retrieved
    ]

    if case["answerable"]:

        expected_chunk_id = case["expected_chunk_id"]

        expected_chunk = f"chunk_{expected_chunk_id}"

        retrieval_success = expected_chunk in chunk_ids

    else:

        expected_chunk = None

        retrieval_success = None

    retrieval_results.append({
        "question": question,
        "expected_chunk": expected_chunk,
        "retrieved_chunks": chunk_ids,
        "retrieval_success": retrieval_success,
        "answerable": case["answerable"],
    })



# Retrieval Accuracy
answerable_retrieval = [
    r
    for r in retrieval_results
    if r["answerable"]
]

successful_retrievals = sum(
    r["retrieval_success"] == True
    for r in answerable_retrieval
)

retrieval_total = len(answerable_retrieval)

retrieval_accuracy = (
    successful_retrievals / retrieval_total
) * 100 if retrieval_total else None





generation_results = []
consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 3
PACING_SECONDS = 7
stop_reason = None

for case_index, case in enumerate(eval_set):
    question = case["question"]

    # Step 1: Generate an answer
    try:
        print(f"Generating answer: {question}", flush=True)
        answer, context = generate_answer(question)

    except LLMError as e:
        print(
            f"Generation error for question: {question}: {e}",
            flush=True,
        )

        generation_results.append({
            "question": question,
            "answerable": case["answerable"],
            "answer": None,
            "verdict": None,
            "explanation": None,
            "error_stage": "generation",
            "error": str(e),
            "refusal": False,
        })

        save_progress(
            retrieval_results,
            generation_results,
            retrieval_accuracy,
        )

        consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            stop_reason = (
                f"Stopped after {consecutive_failures} "
                "consecutive failed cases."
            )
            break

        if case_index < len(eval_set) - 1:
            time.sleep(PACING_SECONDS)

        continue

    # Step 2: Skip verification for refusals
    if is_refusal(answer):
        print(
            f"Skipping verification for refusal: {question}",
            flush=True,
        )

        generation_results.append({
            "question": question,
            "answerable": case["answerable"],
            "answer": answer,
            "verdict": None,
            "explanation": None,
            "error_stage": None,
            "error": None,
            "refusal": True,
        })

        save_progress(
            retrieval_results,
            generation_results,
            retrieval_accuracy,
        )

        consecutive_failures = 0

        if case_index < len(eval_set) - 1:
            time.sleep(PACING_SECONDS)

        continue

    # Step 3: Verify the generated answer
    try:
        print(
            f"Verifying answer for question: {question}",
            flush=True,
        )

        verdict, explanation = verify_answer(answer, context)

        print(f"Finished. Verdict: {verdict}", flush=True)

        generation_results.append({
            "question": question,
            "answerable": case["answerable"],
            "answer": answer,
            "verdict": verdict,
            "explanation": explanation,
            "error_stage": None,
            "error": None,
            "refusal": False,
        })

        save_progress(
            retrieval_results,
            generation_results,
            retrieval_accuracy,
        )

        # Reset the counter after a fully successful case
        consecutive_failures = 0

        if case_index < len(eval_set) - 1:
            time.sleep(PACING_SECONDS)

    except LLMError as e:
        print(
            f"Verification error for question: {question}: {e}",
            flush=True,
        )

        generation_results.append({
            "question": question,
            "answerable": case["answerable"],
            "answer": answer,
            "verdict": None,
            "explanation": None,
            "error_stage": "verification",
            "error": str(e),
            "refusal": False,
        })

        save_progress(
            retrieval_results,
            generation_results,
            retrieval_accuracy,
        )

        consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            stop_reason = (
                f"Stopped after {consecutive_failures} "
                "consecutive failed cases."
            )
            break

        if case_index < len(eval_set) - 1:
            time.sleep(PACING_SECONDS)


if stop_reason:
    remaining = len(eval_set) - len(generation_results)
    print(
        f"{stop_reason} "
        f"{remaining} of {len(eval_set)} cases not attempted.",
        flush=True,
    )




# Refusal Accuracy

unanswerable_generation = [
    g
    for g in generation_results
    if not g["answerable"] and g["answer"] is not None
]

correct_refusals = sum(
    is_refusal(g["answer"])
    for g in unanswerable_generation
)

refusal_total = len(unanswerable_generation)

refusal_accuracy = (
    correct_refusals / refusal_total
) * 100 if refusal_total else None


# False Refusals
answerable_generation = [
    g
    for g in generation_results
    if g["answerable"] and g["answer"] is not None
]

false_refusal_cases = [
    g
    for g in answerable_generation
    if is_refusal(g["answer"])
]

false_refusal_count = len(false_refusal_cases)
false_refusal_total = len(answerable_generation)

false_refusal_rate = (
    false_refusal_count / false_refusal_total
) * 100 if false_refusal_total else None


# Groundedness Rate
grounded_results = [
    g
    for g in generation_results
    if g["verdict"] in ("GROUNDED", "NOT_GROUNDED")
]

grounded_count = sum(
    g["verdict"] == "GROUNDED"
    for g in grounded_results
)

grounded_total = len(grounded_results)

groundedness_rate = (
    grounded_count / grounded_total
) * 100 if grounded_total else None


# Print metrics
def print_metric(label, percentage, passed, total, description="completed"):
    if total == 0:
        print(f"{label}: N/A (no completed cases)")
    else:

        print(
            f"{label}: {percentage:.2f}% "
            f"({passed} of {total} {description})"
        )


print_metric(
    "Retrieval Accuracy",
    retrieval_accuracy,
    successful_retrievals,
    retrieval_total,
    "answerable cases",
)

print_metric(
    "Refusal Accuracy",
    refusal_accuracy,
    correct_refusals,
    refusal_total,
    "completed",
)


if false_refusal_total == 0:
    print("False Refusals: N/A (no answerable cases with an answer)")
else:
    print(
        f"False Refusals: {false_refusal_count} of "
        f"{false_refusal_total} answerable cases with an answer "
        f"({false_refusal_rate:.2f}%) — lower is better"
    )

print_metric(
    "Groundedness Rate",
    groundedness_rate,
    grounded_count,
    grounded_total,
    "answers that make claims",
)




# Error summary
generation_error_cases = [
    g for g in generation_results
    if g["error_stage"] == "generation"
]

verification_error_cases = [
    g for g in generation_results
    if g["error_stage"] == "verification"
]

unclear_verdict_cases = [
    g for g in generation_results
    if g["verdict"] is not None
    and g["verdict"] not in ("GROUNDED", "NOT_GROUNDED")
]

not_attempted = len(eval_set) - len(generation_results)

refusal_count = sum(
    1 for g in generation_results
    if g["refusal"]
)

print(
    f"\nCases: {len(eval_set)} total | "
    f"attempted: {len(generation_results)} | "
    f"not attempted: {not_attempted} | "
    f"generation errors: {len(generation_error_cases)} | "
    f"verification errors: {len(verification_error_cases)} | "
    f"unclear verdicts: {len(unclear_verdict_cases)} | "
    f"refusals (not judged): {refusal_count}"
)

# Show failed cases
failed_cases = generation_error_cases + verification_error_cases

if failed_cases:
    print("\nFailed cases:")

    for case in failed_cases:
        print(
            f'- Question: {case["question"]} | '
            f'Stage: {case["error_stage"]} | '
            f'Error: {case["error"]}'
        )

# Show unclear verdicts
if unclear_verdict_cases:
    print("\nUnclear verdicts:")

    for case in unclear_verdict_cases:
        print(
            f'- Question: {case["question"]} | '
            f'Verdict: {case["verdict"][:60]}'
        )


# Save evaluation results

evaluation_report = {
    "evaluation_version": "v2",
    "run_at": datetime.now().isoformat(timespec="seconds"),
    "model": LLM_MODEL,
    "status": "stopped_early" if stop_reason else "complete",
    "stop_reason": stop_reason,
    "cases_attempted": len(generation_results),

    "metrics": {
        "retrieval_accuracy": retrieval_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "false_refusal_rate": false_refusal_rate,
        "groundedness_rate": groundedness_rate,
    },
    "counts": {
        "total_cases": len(eval_set),
        "retrieval_successes": successful_retrievals,
        "retrieval_total": retrieval_total,
        "correct_refusals": correct_refusals,
        "refusal_total": refusal_total,
        "false_refusal_count": false_refusal_count,
        "false_refusal_total": false_refusal_total,
        "grounded_count": grounded_count,
        "grounded_total": grounded_total,
        "generation_errors": len(generation_error_cases),
        "verification_errors": len(verification_error_cases),
        "unclear_verdicts": len(unclear_verdict_cases),
    },
    "retrieval_results": retrieval_results,
    "generation_results": generation_results,
    "false_refusal_questions": [
    g["question"] for g in false_refusal_cases
    ],
}



save_evaluation_results(evaluation_report)
print(f"\nEvaluation results saved to: {results_path}")