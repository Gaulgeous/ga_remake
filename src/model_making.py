from typing import Any, Dict
from random import choices, randint
import numpy as np

from models import cleaners

def create_genome(model: Dict[str, any], cols: int) -> Dict[str, Any]:
    """
        Create a random genome for the genetic algorithm
        Randomly select drop margins, cleaners and model parameters

        Args:
            None

        Returns:
            genome (Dict[str, Any]): Randomised genome from the available models and cleaning
    """
    genome: Dict[str, Any] = {}
    clean: Dict[str, Any] = {}

    for key in model:
        value: Any = choices(model[key], k=1)[0]
        genome[key] = value

    for cleaner in cleaners:
        value: Any = choices(cleaners[cleaner], k=1)[0]
        clean[cleaner] = value

    drop_margins: np.ndarray = np.array([randint(0, 1) for _ in range(cols)])
    if sum(drop_margins) < 2:
        a: int = randint(0, len(drop_margins) - 1)
        b: int = randint(0, len(drop_margins) - 1)
        while b == a:
            b = randint(0, len(drop_margins) - 1)
        drop_margins[a] = 1
        drop_margins[b] = 1
        drop_margins = np.asarray(drop_margins)

    filters: Dict[str, any] = {"drop_margins": drop_margins}

    return {"filters": filters, "cleaners": clean, "model": genome}