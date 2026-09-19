import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

class ExpLogger:
    def __init__(self, log_dir: str = "experiment_logs"):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.log_dir, exist_ok = True)

    def log_run(
            self,
            model_name: str,
            mode: str,
            dataset:str,
            seed: int,
            evaluation_direction: Optional[str] = None,
            lora_config: Optional[Any] = None
    ):
        if hasattr(lora_config, "to_dict"):
            lora_params = lora_config.to_dict()
        else:
            lora_params = lora_config or {}

        record = {
            "run_id": f"run_{self.timestamp}",
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "model_name": model_name,
            "dataset": dataset,
            "seed": seed,
            "evaluation direction": evaluation_direction,
            "lora_parameters": lora_params,
        }

        filepath = os.path.join(self.log_dir, f"{record['run_id']}.json")
        with open(filepath, "w") as f:
            json.dump(record, f, indent = 4)

    def log_eval(
            self,
            model_name: str,
            source_col: str,
            target_col: str,
            technique: str,
            model_type: str,
            eval_config: Optional[Dict[str, float]] = None,
    ):
        eval_record = {
            "run_id": f"run_{self.timestamp}",
            "eval_model": model_name,
            "timestamp": datetime.now().isoformat(),
            "source_column": source_col,
            "target_column": target_col,
            "technique": technique,
            "model_type": model_type,
            "evaluation_scores": eval_config or {},
        }

        filepath = os.path.join(self.log_dir, f"eval{eval_record['run_id']}.json")
        with open(filepath, "w") as f:
            json.dump(eval_record, f, indent = 4)