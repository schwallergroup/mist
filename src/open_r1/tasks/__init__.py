from .reactions.forward import ForwardReaction
from .reactions.canonical import CanonicalizeSmiles
from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.iupac2smi import Iupac2Smiles
from .smiles_understanding.smiles_hydrogen import SmilesHydrogen
from .kinetic_data.kinetic_data_classification import KineticDataClassification

# Task keys as specified in the task recipes and documentation
CHEMTASKS = {
    "rxnpred": ForwardReaction,
    "canonic": CanonicalizeSmiles,
    "canonmc": CanonicalizeSmilesMCQA,
    "iupacsm": Iupac2Smiles,
    "smhydrogen": SmilesHydrogen,
    "kinetic": KineticDataClassification,
}