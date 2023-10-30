from .lego_evaluator import LegoEvaluator
from .ni_evaluator import NIEvaluator
from .addition_evaluator import AdditionEvaluator 
from .alpaca_evaluator import AlpacaEvaluator
from .hf_evaluator import HFEvaluator
EVALUATOR2CLASS = {
    "addition": AdditionEvaluator,
    "alpaca": AlpacaEvaluator,
    "lego": LegoEvaluator,
    "ni": NIEvaluator,
    "law": HFEvaluator
}

def Evaluator2Class(task_name):
    if task_name not in EVALUATOR2CLASS:
        raise NotImplementedError
    else:
        return EVALUATOR2CLASS[task_name]