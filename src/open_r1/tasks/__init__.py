try:
    from .kinetic_data.kinetic_data_category_classification_with_metrics import (
        KineticDataCategoryClassificationWithMetrics,
        KineticDataCategoryClassificationWithRawDataMetrics,
    )
except ImportError:
    KineticDataCategoryClassificationWithMetrics = None
    KineticDataCategoryClassificationWithRawDataMetrics = None

try:
    from .kinetic_data.kinetic_data_classification import KineticDataClassification
except ImportError:
    KineticDataClassification = None

try:
    from .kinetic_data.kinetic_data_classification_with_metrics import KineticDataClassificationWithMetrics
except ImportError:
    KineticDataClassificationWithMetrics = None

try:
    from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
except ImportError:
    CanonicalizeSmilesMCQA = None

try:
    from .reactions.canonical import CanonicalizeSmiles
except ImportError:
    CanonicalizeSmiles = None

try:
    from .reactions.forward import ForwardReaction, ForwardReactionWithTags
except ImportError:
    ForwardReaction = None
    ForwardReactionWithTags = None

try:
    from .reactions.iupac2smi import Iupac2Smiles, Iupac2SmilesWithTags
except ImportError:
    Iupac2Smiles = None
    Iupac2SmilesWithTags = None

try:
    from .reactions.mcqa_inversion import SmilesInversion
except ImportError:
    SmilesInversion = None

try:
    from .reactions.mcqa_reaction_diff import SmilesReplacement
except ImportError:
    SmilesReplacement = None

try:
    from .reactions.reaction2name import Smiles2Name
except ImportError:
    Smiles2Name = None

try:
    from .reactions.smi_permute import PermuteSmiles
except ImportError:
    PermuteSmiles = None

try:
    from .smiles_understanding.smiles_hydrogen import SmilesHydrogen
except ImportError:
    SmilesHydrogen = None

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
