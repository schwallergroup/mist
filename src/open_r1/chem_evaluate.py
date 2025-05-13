"""Custom evaluation tasks for Chemical Reasoning."""

from lighteval.tasks.requests import Doc
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric, SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig, LightevalTask

import numpy as np
from open_r1.tasks.task_utils import compute_tanimoto_similarity
from open_r1.tasks import Iupac2Smiles

task_mode = "tagged"
test_data_path = "test_data.iupacsm.jsonl"
generation_size = 4096
num_samples = 10

task = Iupac2Smiles(task_mode=task_mode)

def prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=task.question_template.format(line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )

def mol_exact_match(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
# def mol_exact_match(golds: list[str], predictions: list[str], **kwargs) -> bool:
    """
    Check if the predicted SMILES is exactly the same as the gold standard.
    """
    pred = predictions[0] # top-1 only
    pred = task.extract_smiles_from_answer(task.preprocess_response(pred))
    tanimoto_sim = compute_tanimoto_similarity(formatted_doc.get_golds()[0], pred)
    if tanimoto_sim is None:
        acc = 0
    elif tanimoto_sim == 1:
        acc = 1
    else:
        acc = 0
    tanimoto_sim = 0 if tanimoto_sim is None else tanimoto_sim
    return {"acc": acc, "tanimoto_sim": tanimoto_sim}

# def aggregate_fn(results: list[dict], **kwargs) -> dict:
#     """
#     Aggregate the results from multiple samples.
#     """
#     acc = np.mean([result["acc"] for result in results])
#     tanimoto_sim = np.mean([result["tanimoto_sim"] for result in results])
#     return {"acc": acc, "tanimoto_sim": tanimoto_sim}

sample_level_metric = SampleLevelMetricGrouping(
    metric_name="mol_exact_match",
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=mol_exact_match,
    corpus_level_fn={"acc": np.mean, "taniomoto_sim": np.mean},
    higher_is_better={"acc": True, "tanimoto_sim": True},
)

iupacsm_task = LightevalTaskConfig(
    name="iupacsm",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo=test_data_path,
    hf_subset="default",
    hf_avail_splits=['test'],
    evaluation_splits=['test'],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=generation_size,
    num_samples=num_samples,
    metric=[sample_level_metric], 
)

TASKS_TABLE = [
    iupacsm_task,
]

if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))