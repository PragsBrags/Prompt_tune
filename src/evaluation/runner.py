import wandb
import csv
from pathlib import Path

from inference.generator import translate
from inference.model_loader import load_model
from evaluation.metrics import compute_all_metrics

from data.data_loader import load_translation_data
from inference.model_loader import load_model
from retrieval.retriever import TranslationRetriever
from prompting.shot_prompts import (
    build_messages_3,
    build_messages_rag,
    build_messages_back_translation,
    build_messages_consistency_review,
    build_messages_cot_translation,
    build_messages_zero,
    extract_final_translation,
)

def run_evaluation(cfg):

    prediction = []
    references = []
    sources = []
    source_languages = []
    target_languages = []

    dataset = load_translation_data(
        cfg.eval_data,
        cfg.run.seed
        )

    tokenizer, model = load_model(cfg.model)

    batch_size = cfg.eval_data.batch_size
    retriever = None
    if cfg.prompt.strategy == "rag_few_shot":
        retriever = TranslationRetriever(cfg.rag)

    for i in range(0, len(dataset), batch_size):

        message_batch = []
        batch_sources = []
        batch_source_langs = []
        batch_target_langs = []

        for j in range(i, min(i + batch_size, len(dataset))):

            sample = dataset[j]

            source = sample["source"]
            target = sample["target"]
            
            source_languages.append(sample["source_language"])
            target_languages.append(sample["target_language"])
            batch_sources.append(source)
            batch_source_langs.append(sample["source_language"])
            batch_target_langs.append(sample["target_language"])

            if cfg.prompt.strategy == "zero_shot":
                messages = build_messages_zero(
                    source,
                    sample["source_language"],
                    sample["target_language"]
                )
            elif cfg.prompt.strategy == "few_shot":
                pair = cfg.prompt.direction
                examples = cfg.prompt.examples[pair]
                messages = build_messages_3(
                    examples, 
                    sample["source_language"], 
                    sample["target_language"], 
                    source
                )

            elif cfg.prompt.strategy == "rag_few_shot":
                if retriever is None:
                    raise RuntimeError("Retriever is required for rag_few_shot strategy")

                examples = retriever.retrieve(
                source_text=source,
                source_lang=sample["source_language"],
                target_lang=sample["target_language"],
                )

                messages = build_messages_rag(
                    examples,
                    sample["source_language"],
                    sample["target_language"],
                    source,
                )

            elif cfg.prompt.strategy == "cot_translation":
                messages = build_messages_cot_translation(
                    source,
                    sample["source_language"],
                    sample["target_language"]
                )
            elif cfg.prompt.strategy == "back_translation":
                # forward pass uses a plain zero-shot prompt; the back-translation
                # + consistency review happens after generation, below
                messages = build_messages_zero(
                    source,
                    sample["source_language"],
                    sample["target_language"]
                )
                
            else:
                raise ValueError(
                    f"Unknown prompt strategy: {cfg.prompt.strategy}"
                )

            message_batch.append(messages)
            references.append(target)

        generated = translate(
            model,
            tokenizer,
            message_batch,
            cfg.model
            )

        if cfg.prompt.strategy == "cot_translation":
            generated = [extract_final_translation(g) for g in generated]

        elif cfg.prompt.strategy == "back_translation":
            # for back-translation, we need to do a second pass to check
            # consistency of the generated translation with the original source
            reviewed = []

            for source, src_lang, tgt_lang, candidate in zip(batch_sources, batch_source_langs, batch_target_langs, generated):
               back_messages = build_messages_back_translation(candidate, src_lang, tgt_lang)
               back_translation = translate(model, tokenizer, [back_messages], cfg.model)[0]

               review_messages = build_messages_consistency_review(source, candidate, back_translation, src_lang, tgt_lang)
               reviewed_translation = translate(model, tokenizer, [review_messages], cfg.model)[0]
               reviewed.append(extract_final_translation(reviewed_translation))

            generated = reviewed

        prediction.extend(generated)
        sources.extend(batch_sources)
        print(f"Processed {min(i + batch_size, len(dataset))}/{len(dataset)}")

    
    output_file = cfg.model.name + "_" + cfg.model.source + "_"
    safe_model_name = cfg.model.name.replace("/", "__")
    full_path = Path(cfg.eval_data.model_output) / f"{safe_model_name}_{cfg.model.source}.csv"
    full_path.parent.mkdir(parents=True, exist_ok=True)


    with open(full_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "prediction", "reference"])
        for src, pred, ref in zip(sources, prediction, references):
            writer.writerow([src, pred, ref])

    score = compute_all_metrics(sources, prediction, references)
    
    wandb.log({
        f"eval/{name}": value
        for name, value in score.items()
        if value is not None
    })

    return score
