import wandb

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
    build_messages_expert_language_specific,
    build_messages_self_refinement_initial,
    build_messages_self_refinement_refine,
    build_messages_pivot,
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
            elif cfg.prompt.strategy == "expert_language_specific":
                # optional domain and formality may be present in cfg.prompt
                domain = getattr(cfg.prompt, "domain", None)
                formality = getattr(cfg.prompt, "formality", None)
                messages = build_messages_expert_language_specific(
                    source,
                    sample["source_language"],
                    sample["target_language"],
                    domain=domain,
                    formality=formality,
                )
            elif cfg.prompt.strategy == "self_refinement":
                messages = build_messages_self_refinement_initial(
                    source,
                    sample["source_language"],
                    sample["target_language"],
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

                print(messages)

            elif cfg.prompt.strategy == "cot_translation":
                messages = build_messages_cot_translation(
                    source,
                    sample["source_language"],
                    sample["target_language"]
                )
            elif cfg.prompt.strategy == "pivot":
                pivot_lang = getattr(cfg.prompt, "pivot_language", "English")
                messages = build_messages_pivot(
                    source,
                    sample["source_language"],
                    sample["target_language"],
                    pivot_lang=pivot_lang,
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

        elif cfg.prompt.strategy == "self_refinement":
            # perform a second pass where the model critiques and refines
            refined = []
            for source, src_lang, tgt_lang, candidate in zip(batch_sources, batch_source_langs, batch_target_langs, generated):
                review_messages = build_messages_self_refinement_refine(candidate, source, src_lang, tgt_lang)
                reviewed_translation = translate(model, tokenizer, [review_messages], cfg.model)[0]
                refined.append(reviewed_translation.strip())

            generated = refined

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

    score = compute_all_metrics(sources, prediction, references)
    
    wandb.log({
        f"eval/{name}": value
        for name, value in score.items()
        if value is not None
    })

    n = min(cfg.wandb.sample_prediction_rows, len(prediction))

    rows = [
        [
            source_languages[i],
            target_languages[i],
            sources[i],
            references[i],
            prediction[i],
        ]
        for i in range(n)
    ]

    table = wandb.Table(
        columns=[
            "source_language",
            "target_language",
            "source",
            "reference",
            "prediction",
        ],
        data=rows,
    )

    wandb.log({"eval/predictions": table})

    return score
