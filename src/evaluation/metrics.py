import evaluate

def compute_translation_metrics(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")

    sacrebleu_score = sacrebleu.compute(
        predictions=predictions,
        references=references
    )

    return sacrebleu_score
