"""
MathDynamicMultiNets -- a Ren machine: a non-Turing architecture that forms
multiple mapping rules (neural and symbolic) dynamically, under an LLM
controller.

Implements the architecture of "A Non-Turing Computer Architecture for
Artificial Intelligence Forming Multiple Dynamic Rules and Its Halting Problem"
(Ren, 2026): two tapes with two alphabets (symbols and images), mapping rules
between and within them, reasoning as the transfer of those rules, the
base-plus-specialists construction for driving a rule's accuracy up, the
statistical anytime algorithm for the halting problem, and the sketch-based
robotic loop of Appendix A.

Quick start:

    from dynamicmultinet import RenMachine, ScriptedController

    m = RenMachine(goal="learn the distributive rule and use it")
    m.write("abstract", "12*30")
    m.apply_rule("decimal_split")
    print(m.state())

Everything symbolic works with numpy alone; torch is imported only when a
learned rule is actually built or trained.
"""

from .compose import Task, distill, library_report, simplify
from .controller import (ControllerRun, LLMController, ScriptedController,
                         make_controller)
from .dataset import Example, ExampleSet
from .halting import calibrate, halting_budget_for_library
from .machine import RenMachine
from .proof import Proof, proof_to_rule, search
from .rules import (CompositeRule, EnsembleRule, NeuralRule, PythonRule, Recipe,
                    Rule, RuleLibrary, TableRule)
from .tapes import ABSTRACT, SPECIFIC, AbstractTape, Content, SpecificTape
from .tools import build_tools
from .verify import equivalent, verify_against_rules, verify_rule

__version__ = "0.1.0"

__all__ = [
    "RenMachine",
    "Content", "AbstractTape", "SpecificTape", "ABSTRACT", "SPECIFIC",
    "Rule", "PythonRule", "TableRule", "NeuralRule", "CompositeRule",
    "EnsembleRule", "RuleLibrary", "Recipe",
    "Example", "ExampleSet",
    "search", "Proof", "proof_to_rule",
    "verify_rule", "verify_against_rules", "equivalent",
    "Task", "library_report", "simplify", "distill",
    "calibrate", "halting_budget_for_library",
    "LLMController", "ScriptedController", "ControllerRun", "make_controller",
    "build_tools",
    "__version__",
]
