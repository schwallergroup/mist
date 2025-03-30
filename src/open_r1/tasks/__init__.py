from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.canonical import CanonicalizeSmiles
from .reactions.forward import ForwardReaction
from .reactions.iupac2smi import Iupac2Smiles
from .kinetic_data.kinetic_data_classification import KineticDataClassification

# Task keys as specified in the task recipes and documentation
CHEMTASKS = {
    "rxnpred": ForwardReaction,
    "canonic": CanonicalizeSmiles,
    "iupacsm": Iupac2Smiles,
    "canonmc": CanonicalizeSmilesMCQA,
    "kinetic": KineticDataClassification,
}
