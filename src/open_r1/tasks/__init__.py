from .kinetic_data.kinetic_data_classification import KineticDataClassification
from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.canonical import CanonicalizeSmiles
from .reactions.forward import ForwardReaction, ForwardReactionWithTags
from .reactions.iupac2smi import Iupac2Smiles, Iupac2SmilesWithTags
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
}

try:
    from .condmatgen.condmatgen import ConditionalMaterialGeneration
    CHEMTASKS["condmatgen"] = ConditionalMaterialGeneration
except ImportError:
    pass
