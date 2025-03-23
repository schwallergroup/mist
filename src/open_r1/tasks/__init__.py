
from .countdown.cd_task import CountdownTask
# from .reactions.forward import ForwardReaction
from .reactions.canonical import CanonicalizeSmiles
from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.iupac2smi import Iupac2Smiles, Iupac2SmilesWithTags
from .reactions.smi_permute import PermuteSmiles


CHEMTASKS = {
    "CountdownTask": CountdownTask,
    # "ForwardReaction": ForwardReaction,
    "CanonicalizeSmiles": CanonicalizeSmiles,
    "Iupac2Smiles": Iupac2Smiles,
    "Iupac2SmilesWithTags": Iupac2SmilesWithTags,
    "CanonicalizeSmilesMCQA": CanonicalizeSmilesMCQA,
    "PermuteSmiles": PermuteSmiles
}