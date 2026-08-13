import evaluate


def compute_all_metrics(sources: list, predictions: list, references: list) -> dict:
    scores = {}

    try:
        sacrebleu = evaluate.load("sacrebleu")
        scores["bleu"] = sacrebleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )["score"]
    except Exception as e:
        print(f"[metrics] skipping bleu: {e}")
        scores["bleu"] = None

    try:
        import nltk
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception:
            pass
        meteor = evaluate.load("meteor")
        scores["meteor"] = meteor.compute(
            predictions=predictions,
            references=references
        )["meteor"]
    except Exception as e:
        print(f"[metrics] skipping meteor: {e}")
        scores["meteor"] = None

    try:
        ter = evaluate.load("ter")
        scores["ter"] = ter.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )["score"]
    except Exception as e:
        print(f"[metrics] skipping ter: {e}")
        scores["ter"] = None

    try:
        chrf = evaluate.load("chrf")
        scores["chrf"] = chrf.compute(
            predictions=predictions,
            references=[[r] for r in references],
            word_order=0
        )["score"]
    except Exception as e:
        print(f"[metrics] skipping chrf: {e}")
        scores["chrf"] = None

    try:
        chrf = evaluate.load("chrf")
        scores["chrf++"] = chrf.compute(
            predictions=predictions,
            references=[[r] for r in references],
            word_order=2
        )["score"]
    except Exception as e:
        print(f"[metrics] skipping chrf++: {e}")
        scores["chrf++"] = None

    try:
        comet = evaluate.load("comet")
        scores["comet"] = comet.compute(
            predictions=predictions,
            references=references,
            sources=sources
        )["mean_score"]
    except Exception as e:
        print(f"[metrics] skipping comet: {e}")
        scores["comet"] = None

    try:
        bertscore = evaluate.load("bertscore")
        result = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type="bert-base-multilingual-cased"
        )
        scores["bertscore"] = sum(result["f1"]) / len(result["f1"])
    except Exception as e:
        print(f"[metrics] skipping bertscore: {e}")
        scores["bertscore"] = None

    return scores
