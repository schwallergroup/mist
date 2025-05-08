from .kinetic_data.kinetic_data_category_classification_with_metrics import (
    KineticDataCategoryClassificationWithMetrics,
    KineticDataCategoryClassificationWithRawDataMetrics
)
from .kinetic_data.kinetic_data_classification import KineticDataClassification
from .kinetic_data.kinetic_data_classification_with_metrics import (
    KineticDataClassificationWithMetrics,
    KineticDataClassificationWithRawDataMetrics,
)
from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.canonical import CanonicalizeSmiles
from .reactions.forward import ForwardReaction, ForwardReactionWithTags
from .reactions.iupac2smi import Iupac2Smiles, Iupac2SmilesWithTags
from .reactions.mcqa_inversion import SmilesInversion
from .reactions.mcqa_reaction_diff import SmilesReplacement
from .reactions.reaction2name import Smiles2Name
from .reactions.smi_permute import PermuteSmiles
from .smiles_understanding.smiles_hydrogen import SmilesHydrogen

# Task keys as specified in the task recipes and documentation
CHEMTASKS = {
    "rxnpred_with_tags": ForwardReactionWithTags,
    "iupacsm_with_tags": Iupac2SmilesWithTags,
    "smi_permute": PermuteSmiles,
    "rxnpred": ForwardReaction,
    "canonic": CanonicalizeSmiles,
    "canonmc": CanonicalizeSmilesMCQA,
    "iupacsm": Iupac2Smiles,
    "smhydrogen": SmilesHydrogen,
    "kinetic": KineticDataClassification,
    "kinetic_metrics": KineticDataClassificationWithMetrics,
    "kinetic_metrics_category": KineticDataCategoryClassificationWithMetrics,
    "kinetic_metrics_raw_data_category": KineticDataCategoryClassificationWithRawDataMetrics,
    "rxn_inversion": SmilesInversion,
    "rxn_replacement": SmilesReplacement,
    "rxn_naming": Smiles2Name,
}
