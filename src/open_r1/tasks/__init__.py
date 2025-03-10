
from .countdown.cd_task import CountdownTask
from .reactions.forward import ForwardReaction
from .reactions.canonical import CanonicalizeSmiles
from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.iupac2smi import Iupac2Smiles
from .smiles_understanding.smiles_hydrogen import SmilesHydrogen


CHEMTASKS = {
    "CountdownTask": CountdownTask,
    "ForwardReaction": ForwardReaction,
    "CanonicalizeSmiles": CanonicalizeSmiles,
    "Iupac2Smiles": Iupac2Smiles,
    "CanonicalizeSmilesMCQA": CanonicalizeSmilesMCQA,
    "SmilesHydrogen": SmilesHydrogen,
}