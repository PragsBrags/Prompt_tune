from training.prompt_format import render_prompt, render_completion


def train_message(data, family):

    def to_message(row):
        return {
            "prompt": render_prompt(
                family, row["source_language"], row["target_language"], row["source"]
            ),
            "completion": render_completion(family, row["target"]),
        }

    return data.map(to_message, remove_columns=data.column_names)