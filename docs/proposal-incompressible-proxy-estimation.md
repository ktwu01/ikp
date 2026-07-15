# Research proposal: a general theory of "incompressible proxy estimation"

*Status: strategy / discussion proposal — no code change requested here.
Filed as a repo doc because GitHub Issues are disabled on this repository;
move to an Issue/Discussion if those get enabled.*

## Summary

IKP works because factual capacity is an **incompressible proxy** for
parameter count: you can't score high without actually having the
parameters. This proposes generalizing that single insight into a
**framework** — a position + method paper — of which IKP is one instance.
Ambitious and higher-risk than an incremental IKP extension, but
potentially the more influential contribution.

**One-line thesis:** *many latent properties of a black box can be
estimated from behavioral proxies that are costly to fake, and the
trustworthiness of such an estimate is governed by a formal
gaming-resistance bound.*

## The framework

A four-part recipe:

1. **Latent quantity** `θ` we can't observe directly (LLM parameter count;
   a developer's genuine contribution; the size of a training corpus).
2. **Costly-to-fake proxy** `f(·)` — observable behavior an adversary
   cannot cheaply forge in the direction that flatters them.
3. **Calibration** `θ̂ = g(f)` fit against a ground-truth cohort, with
   honest error bars (R², LOO-CV, prediction intervals).
4. **Gaming-resistance bound** — the minimum adversary effort `ε` to move
   `θ̂` by a factor `k`. An estimator is *credible* only where `ε` is large
   or the manipulation is *detectable*.

The novel object is part (4). Calibration is standard; a formal, per-instance
**cost-of-gaming curve** is not, and it's what separates a measurement
instrument from a leaderboard.

## Case studies

- **Instance A — LLM scale (IKP).** Already done, and we now have the
  gaming-resistance analysis to seed part (4): see `../ADVERSARIAL_IKP.md`
  and `../scripts/17_adversarial_robustness.py`. Headline: sandbagging is
  cheap (median model looks 2× smaller by refusing ~7% of what it knows)
  and *cheaper with scale*; contamination needs probe-set access.
- **Instance B — developer contribution.** Deterministic GitHub-account
  scoring (e.g. ghfind's six-dimension engine) is the same shape: estimate
  genuine contribution from proxies, with explicit anti-gaming penalties
  (PR-farming, star inflation, template-spam). The open problem — no ground
  truth for "true developer value" — is itself an interesting part of the
  framework (calibrate against a *proxy* ground truth such as sustained
  downstream reuse).
- **Instance C — a third, to prove generality.** Candidates: estimating
  **training-corpus size / data freshness** from memorization signals;
  estimating **quantization/precision** from behavioral degradation;
  estimating **RLHF intensity** from refusal geometry. Pick one with a
  ground-truth cohort so part (3) is real.

## Why now / why it could land

- The "measure black-box AI systems you don't control" problem is timely
  (audits, capability disclosure, procurement).
- A framework paper needs ≥ 2 worked instances + 1 real formalism; IKP
  gives us a strong, validated instance A for free.
- The honest risk: without a genuine formalism in part (4), this reads as
  two case studies stapled together. The gaming-resistance bound has to be
  the real contribution, not decoration.

## Proposed next steps

- [ ] Formalize the gaming-resistance bound (adversary action space, effort
      metric, detectability, estimator-credibility region).
- [ ] Generalize `scripts/17_adversarial_robustness.py` into an
      instance-agnostic `cost_of_gaming(estimator, attacks)` harness.
- [ ] Stand up instance C on a dataset with ground truth.
- [ ] Draft the position framing; target a NeurIPS/ICLR-style venue.
- [ ] **Collaboration:** align with the IKP author (Bojie Li / Pine AI) —
      instance A is his work and the natural anchor. The credible way in is
      a concrete contribution to IKP first (the adversarial analysis in
      this branch), then the generalization.

## Relationship to the adversarial-robustness work

The adversarial-robustness analysis (this branch / PR) is the tractable,
standalone contribution and directly supplies part (4) for instance A.
This proposal is the ambitious umbrella it could roll up into.
Recommendation: land the adversarial analysis first, then decide whether to
pursue this framework.
