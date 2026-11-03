"""ALFWorld (TextWorld) agent RL with Hindsight Privileged Information (HPI).

Layout note: this package lives at ``$PROJ/tasks.alfworld.v1`` rather than the
originally suggested ``$PROJ/examples/tasks.alfworld.v1``. ``$PROJ/examples`` does not
exist and, more importantly, ``$PROJ/frameworks/slime-ascend`` precedes ``$PROJ``
on ``PYTHONPATH``, so ``import examples`` already resolves to the vendored slime
examples tree and a sibling ``examples`` package would be shadowed. A top-level
package mirrors ``math_opsd`` and is importable as ``tasks.alfworld.v1.*``.

The ALFWorld Python dependencies (textworld / alfworld / jericho) are installed
into an isolated overlay venv and exposed to the training interpreter by
*appending* its site-packages to ``sys.path`` (see ``paths.py``). Appending -- not
prepending -- guarantees the overlay can never shadow a cua-rl package; the two
trees share no module names other than pip/setuptools.
"""

__all__ = [
    "action_parser",
    "env_adapter",
    "eval",
    "loss",
    "paths",
    "privilege",
    "prompts",
    "reward",
    "rollout",
    "train_data",
    "trajectory",
]
