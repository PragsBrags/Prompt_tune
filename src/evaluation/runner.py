from inference.generator import translate
from evaluation.metrics import compute_translation_metrics
from data.data_loader import load_translation_data
from inference.model_loader import load_model
from prompting.shot_prompts import build_messages_zero, build_messages_3


def run_evaluation(cfg):

    prediction = []
    references = []
    messages = None

    dataset = load_translation_data(cfg.data, cfg.run.seed)
    tokenizer, model = load_model(cfg.model)

    for i in range(len(dataset)):
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