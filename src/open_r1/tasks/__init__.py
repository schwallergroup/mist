from .reactions.forward import ForwardReaction, ForwardReactionWithTags
from .reactions.canonical import CanonicalizeSmiles
from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.iupac2smi import Iupac2Smiles, Iupac2SmilesWithTags
from .reactions.smi_permute import PermuteSmiles
from .smiles_understanding.smiles_hydrogen import SmilesHydrogen
from .kinetic_data.kinetic_data_classification import KineticDataClassification

# Task keys as specified in the task recipes and documentation
CHEMTASKS = {
    "rxnpred": ForwardReaction,
    "canonic": CanonicalizeSmiles,
    "canonmc": CanonicalizeSmilesMCQA,
    "iupacsm": Iupac2Smiles,
    "smhydrogen": SmilesHydrogen,
    "ForwardReactionWithTags": ForwardReactionWithTags,
    "Iupac2SmilesWithTags": Iupac2SmilesWithTags,
    "PermuteSmiles": PermuteSmiles,
    "kinetic": KineticDataClassification,
}
