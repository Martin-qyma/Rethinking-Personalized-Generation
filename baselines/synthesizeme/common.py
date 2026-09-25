"""Shared plumbing for the SynthesizeMe baseline (Ryan et al., 2025).

Uses the released package (``pip install SynthesizeMe``, built against dspy 2.6.12)
with a Llama-3.1-8B-Instruct judge served by vLLM. Three issues of the package under
dspy 3.x / with an 8B judge are worked around here, deviating as little as possible:

1. dspy>=3 defaults the LM temperature to None; the package's repeat_dspy_call
   computes ``lm.kwargs["temperature"] * x``, so every judge call raises TypeError and
   silently degrades to "Tie". Fix: build the LM explicitly with temperature=0.0 (the
   dspy 2.x default the paper ran with).
2. GeneratePersonaProgram.forward computes a ``demos_list`` fallback when bootstrap
   finds no verified demos ("Using all examples.") but then iterates the empty
   optimized demo list anyway, so the persona is synthesized from an empty judgement
   history (hallucinated from the packaged few-shot demos). Fix: iterate demos_list,
   as the code clearly intends (``apply_patches``).
3. With the packaged llama8b.json few-shot demos, Llama-3.1-8B at temperature 0
   copies the last demo's persona verbatim instead of synthesizing one from the
   user's history (out-of-domain judgements, ~20k-token prompt). Fix: keep the
   MIPRO-optimized instruction but strip the demos (llama8b_nodemos.json, shipped
   next to this file; ``nodemos_prompt_path`` regenerates it from the package).

The package's predict_pairwise calls a non-existent ``program.predict(context=...)``,
so the judge program is always invoked via ``forward(conversation=..., ...)`` with
the package's own ``format_conversation`` (see bon.py).
"""
import importlib.resources
import json
import os

import dspy
from synthesizeme.utils import dspy_methods as _dm

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))


def make_lm(port, host="localhost"):
    """dspy LM backed by an OpenAI-compatible vLLM server of the judge."""
    os.environ.setdefault("DSPY_CACHEDIR", os.path.join("checkpoints", "synthesizeme", "dspy_cache"))
    lm = dspy.LM(f"litellm_proxy/{JUDGE_MODEL}", api_base=f"http://{host}:{port}/v1", api_key="local",
                 model_type="chat", max_tokens=4096, temperature=0.0)
    dspy.configure(lm=lm)
    return lm


def nodemos_prompt_path():
    """The package's llama8b.json persona-synthesis program without its few-shot demos."""
    out = os.path.join(HERE, "llama8b_nodemos.json")
    if not os.path.exists(out):
        src = str(importlib.resources.files("synthesizeme").joinpath("prompts/llama8b.json"))
        with open(src) as f:
            d = json.load(f)
        d["synthesize.predict"]["demos"] = []
        with open(out, "w") as f:
            json.dump(d, f, indent=1)
    return out


def _patched_forward(self, user_train, user_val, user_id):
    """GeneratePersonaProgram.forward with fix 2 (iterate the fallback demos_list)."""
    from dspy.teleprompt import BootstrapFewShotWithRandomSearch
    from synthesizeme.utils.utils import exact_match
    optimizer = BootstrapFewShotWithRandomSearch(
        max_bootstrapped_demos=min(self.max_bootstrapped_demos, len(user_train)),
        max_labeled_demos=min(self.max_labeled_demos, len(user_train)),
        num_candidate_programs=self.num_candidates,
        num_threads=self.num_threads_inner,
        stop_at_score=self.stop_at_score,
        metric=exact_match,
    )
    optimized = optimizer.compile(_dm.LLMAsAJudgeProgram(), trainset=user_train, valset=user_val,
                                  restrict=range(-2, self.num_candidates))
    demos_list = optimized.judge.predict.demos
    if len(demos_list) == 0:
        print(f"User: {user_id} found no useful demos.  Using all examples.")
        demos_list = user_train
    history = []
    for demo in demos_list:
        if "reasoning" in demo:
            conversation = f"""================================================\n**Conversation**: {demo['conversation']}\n===
**Completion One (FIRST COMPLETION)**: «{demo['first_completion']}»\n===
**Completion Two (SECOND COMPLETION)**: «{demo['second_completion']}»\n===
**Reasoning**: «{demo['reasoning']}»\n===
**TRUE USER PREFERENCE**: {demo['preference']}\n================================================\n\n"""
        else:
            conversation = f"""================================================\n**Conversation**: {demo['conversation']}\n===
**Completion One (FIRST COMPLETION)**: «{demo['completion_one']}»\n===
**Completion Two (SECOND COMPLETION)**: «{demo['completion_two']}»\n===
**TRUE USER PREFERENCE**: {demo['chosen']}\n================================================\n\n"""
        history.append(conversation)
    history_str = "\n\n".join(history)
    if self.output_dir is not None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_dir + f"{user_id}.history", "w") as f:
            f.write(history_str)
    persona = self.synthesize(past_judgements=history_str)
    return dspy.Prediction(persona=persona.synthesized_persona, reasoning=persona.reasoning)


def apply_patches():
    _dm.GeneratePersonaProgram.forward = _patched_forward


def load_judge_program(ckpt_dir, user_id):
    """A fitted persona judge (``<uid>_persona.txt`` + optimized program ``<uid>.json``)."""
    with open(os.path.join(ckpt_dir, f"{user_id}_persona.txt")) as f:
        persona = f.read()
    program = _dm.LLMAsAJudgeProgramPersona(persona)
    program.load(os.path.join(ckpt_dir, f"{user_id}.json"))
    return program


def default_judge_program():
    """The package's judge without a persona (the 'generic judge' control)."""
    return _dm.LLMAsAJudgeProgram()
