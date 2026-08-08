from inference.generator import translate
from evaluation.metrics import compute_translation_metrics
from prompting.shot_prompts import build_messages_zero, build_messages_3


def run_evaluation(cfg,dataset,tokenizer,model):

    prediction = []
    references = []
    messages = None

    for i in range(cfg.data.max_samples):
        sample = dataset[i]
        english = sample[cfg.data.source_column]
        target = sample[cfg.data.target_column]

        if cfg.prompt.strategy == "zero_shot":
            messages = build_messages_zero(english)

        elif cfg.prompt.strategy == "3_shot":
            messages = build_messages_3(english)

        generated = translate(model,tokenizer,messages,cfg.model)
        print(generated)
        prediction.append(generated)
        references.append([target])

    score = compute_translation_metrics(prediction,references)

    return score