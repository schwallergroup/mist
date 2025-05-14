from .kinetic_data.kinetic_data_classification import KineticDataClassification
from .reactions.canon_mcqa import CanonicalizeSmilesMCQA
from .reactions.canonical import CanonicalizeSmiles
from .reactions.forward import ForwardReaction, ForwardReactionWithTags
from .reactions.iupac2smi import Iupac2Smiles, Iupac2SmilesWithTags, Iupac2SmilesV2
from .reactions.smi_permute import PermuteSmiles
from .smiles_understanding.smiles_hydrogen import SmilesHydrogen
from .reactions.mcqa_inversion import SmilesInversion, SmilesInversionV2
from .reactions.mcqa_reaction_diff import SmilesReplacement, SmilesReplacementV2
from .reactions.reaction2name import Smiles2Name, Smiles2NameV2
from .reactions.reaction_truefalse import ReactionTrueFalse

# Task keys as specified in the task recipes and documentation
CHEMTASKS = {
    "rxnpred_with_tags": ForwardReactionWithTags,
    "iupacsm_with_tags": Iupac2SmilesWithTags,
    "smi_permute": PermuteSmiles,
    "rxnpred": ForwardReaction,
    "canonic": CanonicalizeSmiles,
    "canonmc": CanonicalizeSmilesMCQA,
    "iupacsm": Iupac2Smiles,
    "iupacsm_v2": Iupac2SmilesV2,
    "smhydrogen": SmilesHydrogen,
    "kinetic": KineticDataClassification,
    "rxn_inversion": SmilesInversion,
    "rxn_inversion_v2": SmilesInversionV2,
    "rxn_replacement": SmilesReplacement,
    "rxn_replacement_v2": SmilesReplacementV2,
    "rxn_naming": Smiles2Name,
    "rxn_naming_v2": Smiles2NameV2,
    "rxn_truefalse": ReactionTrueFalse,
}
