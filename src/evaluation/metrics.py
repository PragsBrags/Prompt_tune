import importlib
import os
import subprocess
import sys

import evaluate

_COMET_MODEL = None


def _ensure_package(module_name: str, pip_name: str):
    try:
        importlib.import_module(module_name)
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "--no-build-isolation", "--no-deps"]
        )


def _load_comet_model():
    global _COMET_MODEL

    if _COMET_MODEL is not None:
        return _COMET_MODEL

    _ensure_package("comet", "unbabel-comet>=2.2.2")

    from comet import download_model, load_from_checkpoint

    model_path = os.environ.get("COMET_MODEL_PATH")
    if not model_path:
        model_path = download_model("Unbabel/wmt22-comet-da")

    _COMET_MODEL = load_from_checkpoint(model_path)
    return _COMET_MODEL


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
        comet_model = _load_comet_model()
        import torch
        comet_input = [
            {"src": source, "mt": prediction, "ref": reference}
            for source, prediction, reference in zip(sources, predictions, references)
        ]
        comet_result = comet_model.predict(
            comet_input,
            batch_size=8,
            gpus=1 if torch.cuda.is_available() else 0,
            progress_bar=False,
        )
        scores["comet"] = float(comet_result.system_score)
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