import evaluate

def compute_translation_metrics(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    ter = evaluate.load("ter")

    flat_references = [reference[0] for reference in references]

    return {
        "sacrebleu": sacrebleu.compute(
            predictions=predictions,
            references=flat_references,
        )["score"],
        "chrf": chrf.compute(
            predictions=predictions,
            references=flat_references,
        )["score"],
        "ter": ter.compute(
            predictions=predictions,
            references=flat_references,
        )["score"],
    }
