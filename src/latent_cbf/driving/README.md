# driving: latent reachability safety filter

Implementation of `docs/DRIVING_ADAPTATION_PLAN.md` (WOMD vectorized route).
The package depends only on torch; CAT-K and CtRL-Sim are reached through
`adapters/` and are not imported at package import time.

| Module | Content | Plan stage | Status |
|---|---|---|---|
| `stl/` | STL robustness semantics, driving signals (clearance, TTC, road edge), rule monitor | 2 | unit tested |
| `hj/margin.py` | boundary function l(z), regression or classification labels, gradient penalty | 3 | unit tested |
| `hj/avoid_q.py` | discrete avoid Q-learning in imagination (PyHJ time-discounted backup) | 3 | unit tested against a closed-form problem |
| `safety_filter.py` | top-K token filter, `cbf` and `lr` modes | 4 | unit tested |
| `conformal.py` | split-conformal threshold for the false-safe rate | 5 | unit tested |
| `rollouts.py` | closed-loop evaluation and calibration rollouts in an imagination | 4, 5 | tested on the toy problem |
| `toy/car_following.py` | stochastic car-following problem for the end-to-end test | - | tested |
| `adapters/ctrlsim.py` | CtRL-Sim episode record to STL labels, planner wrapper with a safety hook | 2, 4 | unit tested with synthetic records, not run inside Nocturne |
| `adapters/smart_imagination.py` | SMART (CAT-K) as a step-wise imagination | 1, 3 | **not executed yet**, run the check script first |

## Tests

    python -m pytest tests/driving

## Running on SMART

All commands from `src/latent_cbf/`, in the CAT-K environment:

    # 0. the step-wise rollout must reproduce CAT-K inference
    python scripts/driving/check_smart_imagination.py --catk-root $CATK --ckpt $CKPT --data-dir $DATA

    # 1. feasibility gate: does the ego latent predict the clearance margin?
    python scripts/driving/train_smart_hj.py --stage probe --catk-root $CATK --ckpt $CKPT --data-dir $DATA

    # 2. latent HJ critic, then filter and conformal sanity check inside the imagination
    python scripts/driving/train_smart_hj.py --stage train --rollouts 2000 --catk-root $CATK --ckpt $CKPT --data-dir $DATA
    python scripts/driving/train_smart_hj.py --stage eval --catk-root $CATK --ckpt $CKPT --data-dir $DATA

## Notes for the next steps

* In SMART the agents' poses are explicit, so the collision margin is computed
  analytically during imagination (`SmartImagination.true_margin`). A learned
  l(z) is only needed for rules that cannot be evaluated from the imagined
  state. `--learned-margin` switches to the learned head.
* The fixed gradient-penalty level of latent_cbf (0.1) does not transfer to a
  new latent space; `fit_margin_head` estimates the level from the data.
* The critic is an expectation over the other agents' sampled behaviour, so
  `Q > 0` is optimistic. On the toy problem the uncalibrated filter leaves many
  violations and the conformal threshold is clearly above 0.
* Still open: converting live Nocturne state into SMART's tokenized input, so
  that the filter can run inside `PlannerAdversaryEvaluator` through
  `FilteredPlanner`'s `safety_hook`.
