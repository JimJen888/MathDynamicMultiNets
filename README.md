# MathDynamicMultiNets

An implementation of the non-Turing computer architecture in *"A Non-Turing
Computer Architecture for Artificial Intelligence Forming Multiple Dynamic
Rules and Its Halting Problem"* (Jineng Ren, 2026): two tapes with two
different alphabets, mapping rules between them formed dynamically as neural
networks, an LLM as the controller, and a conciseness objective that decides
which rules are worth keeping.

The base network for a learned rule is
[`DetourNet`](../RL_training/rl_training/detourNet.py) generalised by one
parameter — same two views, same shared encoder, same semantic-palette front
end, same `fa - fb` fusion; the head goes from `num_classes` logits to
`num_slots × num_classes`, and `num_slots == 1` reproduces DetourNet exactly.

```
                     ┌───────────────────────────────┐
   symbols  ────────►│  ABSTRACT tape                │  exact arithmetic,
   "12*30"           │  read/write head              │  rigid definitions,
                     └───────────────┬───────────────┘  formal logic
                                     │
                            ┌────────┴────────┐
                            │   CONTROLLER    │◄── an LLM chooses the next
                            │  (Claude, or a  │    operation from a fixed
                            │  scripted plan) │    instruction set
                            └────────┬────────┘
                                     │
                     ┌───────────────┴───────────────┐
   images   ────────►│  SPECIFIC tape                │  structure as layout,
   (64,384,3)        │  read/write head              │  perception required
                     └───────────────────────────────┘
   the two are connected ONLY by mapping rules:
     abstract→abstract  calculation, algebraic identities, substitution
     abstract→specific  rendering (built in)
     specific→specific  a structural rewrite, learned from experiments
     specific→abstract  reading (learned) — the paper's YOLO step
```

## Install and run

```bash
conda env create -f environment.yml      # python 3.10, numpy, torch+CUDA, pytest
conda activate dynamicmultinet
python -m pytest tests/ -q               # 135 tests, ~80 s

python examples/run_navier_stokes.py     # experiment 1: the construction as rules
python examples/run_complete.py          #   the assembled chain to the theorem
python examples/run_census.py            #   every rule, how it was formed

#   the eleven intermediate results, decomposed one at a time
python examples/run_construction.py      #   Thm 4.6, Props 5.5, 7.5, 9.6
python examples/run_initialize.py        #   Prop 9.5, on the order calculus
python examples/run_summation.py         #   Prop 9.9
python examples/run_theorem.py           #   Prop 10.1 and Theorem 1.1
python examples/run_force_extension.py   #   Lemma 10.3
python examples/run_energy_bound.py      #   Lemma 10.4
python examples/run_comparison.py        #   Lemma 10.5

#   and the experiments that found out how to do it
python examples/run_decomposition.py     #   the first attempt, and what it missed
python examples/run_discharge.py         #   replacing an import by its proof
python examples/run_estimate.py          #   one granted estimate, taken apart
python examples/run_estimates.py         #   the rest of them
python examples/run_deeper.py            #   the computations one level further
python examples/run_leaf_census.py       #   how much of it is named mathematics
python examples/run_apparatus.py         #   why the front half needed a prerequisite
python examples/run_tail.py              #   how much closed before it did
python examples/run_multiplication.py    # experiment 2
python examples/run_geometry.py          # experiment 3
python examples/run_robotics.py          # appendix A
python examples/run_riemann.py           # experiment 4
python examples/run_montgomery.py        # experiment 5
python examples/run_finite_height.py     # experiment 6
#   add --quick for a 30 s smoke run, --llm to let Claude drive
```

**Rendered images.** Every example ends by writing what it drew to
`renders/<experiment>/`, because an accuracy number over pictures is not
checkable and the pictures are: held-out inputs captioned with what the rule
answered and what the oracle wanted, the drawings a specific→specific rule
produced, the tape cells, and — in the geometry run — the proof replayed one
image per step. Filenames are truncated, so the full captions live next to them
in `index.txt`:

```
renders/geometry/index.txt
  step03_construct_aux_line_triangle0_move_up_mo.png   construct_aux_line_triangle0|move_up|move_up|rotate_cw
  read_angle_facts_02_WRONG_want-no_facts_got-A1.png   WRONG_want-no_facts_got-A1=B1,A2=B2,A1+A3+A2=180
```

Pass `--dump DIR` to put them somewhere else, `--no-dump` to skip them. A
`--quick` run writes the same images from badly trained rules, which is the
fastest way to see what the experiment is doing before paying for a real run.

`requirements.txt` is the pip equivalent if you already have an environment.
Only numpy is strictly required — the tapes, renderer, prior rules, proof
search and the objective all run without torch, and declaring a learned rule
without it fails immediately with an install hint rather than several tool
calls later.

**Device.** Learned rules run on the GPU when there is one — `device=None`
detects, the same convention as `DetourPredictor`. Pass `--device cpu` (or
`RenMachine(device="cpu")`) to pin it, which is worth doing when two runs have
to match exactly: cuDNN kernels do not reproduce CPU kernels bit-for-bit even
under the same seed, so the two devices are different random draws of the same
procedure. Every run prints which device it chose. The speedup is real but
modest — experiment 2 is 2m04s on a 4090 against ~8 min on CPU — because the
nets are small and scene generation, glyph rendering and pixel quantization
stay on the CPU in numpy. On `--quick` runs the two are indistinguishable.

The examples run offline through a `ScriptedController`, which drives *the
identical tool table* an LLM controller uses — a demo without an API key is not
a different program. With `--llm` (needs `ANTHROPIC_API_KEY` or an
`ant auth login` profile) Claude decides what to do next instead.

```bash
python -m dynamicmultinets.cli tools       # the controller's instruction set
python -m dynamicmultinets.cli catalogue   # generators and oracles it may pick
python -m dynamicmultinets.cli run --goal "..." --llm
```

## The objective

Rules "as concise and useful as possible" is a single number, priced as a
two-part code in one currency:

```
J(library) = bits to write the rules down  +  1000 × rule applications needed
                                              to solve the benchmark tasks
```

A rule earns its place when it shortens more derivation than it costs to state.
A learned rule is charged for its **recipe** — generator, oracle, architecture,
seed — not its weights, because the machine can regenerate the weights from the
recipe, so the recipe is the actual description. Three consequences, all of
them behaviours the paper describes:

* a rule nothing uses is deleted (pure first term);
* a recurring six-step chain is worth replacing with one composite or one
  distilled net (six applications cost more than one recipe);
* a net that duplicates a two-symbol identity is not worth keeping at any
  accuracy, because the identity is cheaper to state.

## What the machine can do

Every operation is a tool the controller picks by name; the LLM never writes or
executes code. `python -m dynamicmultinets.cli tools` prints the full list.

| | |
|---|---|
| `propose_rules` | decide *what* to learn: compare unsolved cases with solved ones **in the specific domain** and summarise what they share |
| `generate_data` / `label_data` | run an experiment, then ask an oracle what is true |
| `declare_rule` / `train_rule` | decide what mapping a rule performs, and fit it |
| `verify_rule` / `verify_against_rules` | check on fresh data, against an oracle **or** against a chain of already-trusted rules |
| `grow_ensemble` | train a specialist on what the base rule gets wrong and let it override (paper §3) |
| `compose_rules` / `distill_rule` | name a chain, or collapse it into one net |
| `prove` / `prove_from_tape` / `keep_proof` | search for a rule chain; keep the one you find |
| `library_report` / `simplify_library` | price the library; drop what stopped paying |
| `halting_budget` | a search budget with a stated error rate (paper §5) |

### Where a hypothesis comes from

Verifying a rule presupposes somebody chose it. `propose_rules` is the step
before that ([`propose.py`](dynamicmultinets/propose.py)): cases the machine
cannot derive are put beside cases it can, **both drawn onto the specific
tape**, and what they share comes back as rules worth forming.

Drawing them is the point, not presentation. `12*30` and `10*30+2*30` are
different strings; drawn, they are two arrangements of the same marks, and the
regrouping is a fact about the layout — which is the architecture's claim, so a
proposer that pattern-matched the abstract strings would be answering an easier
question.

By default the cell reaches the controller **as text**: what the drawing lays
out, with every glyph decoded to the characters it came from.

```
UNSOLVED 12*30 => 10*30+2*30
    reads as:  12*30
    layout:    1 box
      box 1: 12
             *30
    must become:
    reads as:  10*30+2*30
    layout:    2 boxes, joined by '+'
      box 1: 10
             *30
      box 2: 2
             *30
```

That is the structured view described rather than pictured — the boxes it
draws, the separator between them, the factors stacked inside each — and it is
where the analogy lives: **one box becomes two, split at the place-value
boundary**, while the contrast case `9*7 => 63` stays one box throughout. The
regrouping is a layout fact, not a string edit, which is the claim; but it
arrives as text a model can read. `form="image"` sends the rendered pixels
instead, worth it when layout is genuinely pictorial — a geometry sketch says
more as an image than any description of it does. Any example script takes
`--form image` to start a whole run that way:

```bash
./run.sh examples/run_geometry.py --llm --form image
```

which moves the default the `propose_rules` tool falls back to, so it holds for
the calls the LLM controller makes rather than only the ones written into a
scripted plan. Two things it does *not* touch. `--dump` is separate: that
writes PNGs to disk for a human to check, and has nothing to do with what the
model is sent. And the form is only read on the `--llm` path — with no LLM,
`heuristic_proposals` sees neither the pixels nor the layout text, it runs each
oracle over the case's plain string and keeps whichever produce an answer, so
the same proposals come back under either form.

The layout comes from the same `split_top_level` and `_term_lines` the renderer
uses, so it is a faithful transcript of what was drawn. It is a transcript,
not a perceptual reading: no net looks at the pixels here, and a rule that must
read an *unseen* drawing still has to be learned.

Posing the cases matters for a second reason the tool makes explicit:

```
propose_rules(unsolved=["12*30 => 10*30+2*30"])                    # 0 unsolved
propose_rules(unsolved=[...], domain="specific", observed=True)    # 1 unsolved
```

As symbols the case is *already solved* — `decimal_split → distribute_symbolic`
is prior knowledge, and there is nothing to explain. As a drawing nothing
touches it, and `observed` stops `transcribe_unsafe` answering by copying the
caption. Solvedness is measured by proof search, not taken on the caller's
word, so a case handed in as unsolved moves lists if the machine can in fact
derive it.

A proposal is a **claim that can be wrong**, and it comes in two shapes with
two different tests.

**Shared pattern** — *"the pattern established on the known instances also
holds on the unknown ones, under `<condition>`."* Same problem with new
instances (a rewrite verified for two-digit products, claimed for three-digit
ones) or two problems sharing a structure (what is proven of the 2-D case,
claimed in 3-D under the right hypotheses). The proposal names **two** families
— where the pattern holds, and where it is being claimed — because a single
family cannot state a transfer.

A pattern can also be carried to a different **form** of itself, which is
often the sharper claim: set `known_oracle` to the form it already holds in and
`oracle` to the form being claimed. The distributive law established as a split
of the left factor, claimed as a split of the right one, over the very same
numbers — a transfer that varying generator parameters cannot express.

Both halves are checked, because "established" is half the claim: a pattern
that never held on the known family has nothing to carry across, and saying so
beats training a net to find out. A transfer that fails has produced
counterexamples, which is the more useful outcome. From a live run:

```
place_value_split_of_left_factor  [shared_pattern]
  claim: 'distributive_rewrite', established on {"round_b": true}, also holds
         on {"round_b": false}, provided the rewrite splits only the left factor
         at the tens/units boundary and copies the right factor unchanged
  checked: applies to 54/60 of the unknown family
```

**Interconversion** — *"the unproven case maps to a solved one, so it is
established by transport."* Fermat's Last Theorem via the semistable case of
Taniyama–Shimura is the shape: the work is building the correspondence, not
spotting a shared pattern. So the proposal is a **search** — find a chain of
mapping rules from the unproven statement to the established one, or back — and
`proof.search` already does exactly that:

```
triangle_angle_sum_transported  [interconversion]
  claim: the unproven 'B1+A3+B2=180' maps -> the established
         'A1=B1,A2=B2,A1+A3+A2=180', so establishing the second carries the first
  checked: no chain within depth 8 -- the correspondence is unbuilt, which is a
           bound on the attempt, not a refutation
```

Neither is a rule, and neither is code: every part is a name the machine
already has, validated before return, because the controller is a language
model and must select rather than emit. Validation **executes** the recipe on a
handful of examples rather than spell-checking it — the first live run proposed
`tail_digits=0`, which raises inside the generator, and `domain="integers"`,
which is not a value `mul_pairs` knows. Proposals come back untrusted and
undeclared: a shared pattern goes through the ordinary `generate_data →
label_data → declare_rule → train_rule → verify_rule` path, an interconversion
is settled by `prove`. Without credentials a much weaker offline heuristic runs
instead, which can only notice that an oracle applies and never why, and cannot
tell the established family from the conjectured one at all; the gap between
the two is the honest measure of what the controller contributes.

**Only trusted rules may appear in a proof**, and trust comes from verification
on data the rule was not trained on. The strength of the evidence is tracked,
not just the accuracy:

| grounding | worth | what it means |
|---|---|---|
| `definitional` | 1.0 | bottoms out in a definition (multiplication as repeated addition) |
| `independent_chain` | 0.95 | a route sharing no rule and no teacher with what it checks |
| `measured` | 0.9 | came from outside (collision geometry) |
| `derived` / `rule_chain` | 0.7 | inherits the errors of the rules it used |
| `constructed` | 0.2 | the machine checked its own handwriting |

`independent_chain` is the interesting one, because grounding is a property of
the *pair*, not of the reference alone. Two routes that share no rule and no
training oracle cannot agree on the same *wrong* answer except by coincidence,
and `verify.collision_probability` measures how likely that coincidence is from
the reference's own answers: rewriting `37*32` into a sixteen-character
expression collides at 0.0001, so agreement is confirmation rather than
evidence, and the reported accuracy becomes a **lower bound** — disagreements
are unattributed, since an unrelated reference is exactly the kind that can be
wrong by itself. The same argument is worthless for a rule choosing one of four
actions, where a quarter of agreements are luck, so the test is measured rather
than assumed. The converse guard matters more: a reference that *runs* the rule
under test scores a perfect 1.000 and means nothing, so it is refused outright.

Sharing a component is only fatal when its **errors** are shared, which is why
the guard asks whether each shared rule *can be wrong* rather than merely
whether it is shared. Every rule the machine starts with is exact — reading,
writing, transcribing, the symbolic rewrites, the memorised table — so two
routes may share any number of them and still be independent. A learned route
ending in `eval_arith` may be checked against a symbolic route ending in
`eval_arith`, and a route that draws a cell and reads it back may be checked
against another that does the same: the two still disagree wherever the work
they do *not* share differs, which is exactly what the check is measuring.
Refusing that would leave every rule that finishes by computing something
unverifiable by any route. Only a shared *fallible* rule is refused, because
its mistakes really do appear on both sides.

One case turns on something other than mistakes, and it is a case of *not*
sharing. Checking a reader against `transcribe_unsafe` is the `read_back`
check written as a chain: the reference is the reader's own supervision signal
rather than a second route, so it is graded `constructed`, the weakest
grounding on the scale, and never reported as independent. That applies only
when the reference reads the caption and the rule under test does not. When
both sides read it, it is common ground like any other basic rule and the
comparison is between what they do afterwards.

`simplify_library` needs its own guard for the same rule. Two rules that agree
on a probe set are candidates for merging, and `transcribe_unsafe` agrees with
a *real* reader on every cell the machine drew itself while costing a tenth as
many bits. Dropping the reader in its favour would leave a machine that cannot
read an observed cell at all, and neither the bit count nor the agreement rate
can see that, because the probe set is made of cells the machine captioned. So
the two rules are asked again with the probe marked observed, and a rule is
never displaced by one that answers strictly fewer kinds of cell — nor by one
the machine trusts less. Every applied drop is then re-priced on its own and
put back if the benchmark loses a task.

Confidence multiplies along a chain, so ten steps at 0.99 is a 0.90 proof —
the reason `verify_rule` defaults to a 0.99 threshold rather than something
that "looks fine".

## Results

Default settings, single runs, no cherry-picking. Experiment 1 trains nothing
and asks for no GPU: it is 2m32s of oracle time on CPU. The rest are on one
RTX 4090 (2m04s for experiment 2, 4m01s for experiment 3, 4m15s for appendix
A); the same runs on CPU take a few times longer and land in the same place.

**Experiment 1 — a published construction, rebuilt as rules**
(`run_navier_stokes.py`)

The OpenAI preprint constructs a forced Navier–Stokes solution that starts from
rest, keeps bounded kinetic energy, and becomes unbounded at t = 1 —
alternatives (C) and (D) of Fefferman's statement, not the unforced (A)/(B).
This experiment takes the construction rather than its conclusion and rebuilds
it here, with every named result as a rule.

**Where it ends up, since the sections below were written as the work went
and several record a conclusion a later one overturns:** Theorem 1.1 is
derived. The machine starts at one cell and reaches it in 129 rule
applications, through all eleven intermediate results, each decomposed to
named theorems and the paper's own estimates cited on the record. Jump to
[Assembling the whole argument](#assembling-the-whole-argument) for the end
state. The sections in between are the route, including the six times I
predicted the method would break and was wrong.

The results split, and the machine keeps them apart by itself. Six reduce to
something decidable on an instance, so each is declared **unverified** and then
checked against an oracle that answers the same question by another route:

```
result             oracle                         n     acc    base  answers
energy_budget_3_5  numerical quadrature        1200   1.000   0.003  967 classes
lemma_4_5_cone     the wave amplitudes solved  1200   1.000   0.832    2 classes
lemma_4_5_cone     the definition (4.21)       1200   1.000   0.835    2 classes
lemma_7_4_pulse    integrating the amplitude    400   1.000   0.782    3 classes
lemma_A6_heat      differencing the field      1200   1.000   0.508    2 classes
lemma_A1_moments   the determinant, built      1200   1.000   0.531    2 classes
prop_9_6_decay     iterating the recursion     1200   1.000   0.003 1039 classes
```

Those counts are not one number applied to everything. Each check gets about
ten seconds of oracle time, so the count follows the cost — `lemma_10_4`'s
oracle integrates a stiff system at 0.37 s an instance and gets 160, the
decay recursion is `Fraction` arithmetic and gets 1200 — and is then capped
where the generator would start repeating itself. Confidence is
Laplace-smoothed, so 1200 clean checks report 0.9992 where 160 report 0.9938,
and that difference is only real if the extra instances were extra
*questions*.

The cone rule is the one worth reading. Lemma 4.5 says a quadratic inequality
decides the admissible stress cone; the oracle does not evaluate that
inequality, it **builds the two pulse families and solves for their squared
amplitudes**, which is the only reason the cone condition exists. Agreement is
the lemma, measured.

Perfect agreement is also what a check that cannot fail would print, so the
run breaks each rule on purpose and reports whether the oracle objects:

```
rule                one thing made wrong                   caught
energy_budget_3_5   dissipation threshold -1 -> -3/2        50/1200
lemma_4_5_cone      quadratic clause dropped               227/1200
lemma_7_4_pulse     closing rate read mid-slot              92/400
lemma_A6_heat       exponent allowed to miss by 1/40       486/1200
lemma_A1_moments    distinctness ignored                   637/1200
prop_9_6_decay      cycle gain 1/10 -> 1/9                1200/1200
eq_4_1_exponents    A + D = 1 relaxed to within 1/10       300/600
increment_identity  divergence-free condition dropped      600/1200
```

Raising those counts turned up a worse hole than the mutation audit did.
Confidence is charged per check, and nothing was checking that the checks
were *different*. The increment identity's generator emitted exactly two
cells — divergence-free and not — however many instances were asked for, so
160 checks were two facts and a reported 0.9938. Four more generators had
saturated less dramatically: 35 summation schedules, 50 Borel schedules, 61
correction stages, 323 moment families. `verify` now counts distinct inputs
(pixels where there are pixels, since two renderings of the same caption are
two questions for a rule that reads them) and charges the rule once per
question, printing a `NARROW` warning when the repeats outnumber the
questions four to one. The multiplication experiment below was affected
too: 120 rendered products are 62 distinct expressions.

The generators were then widened until more instances meant more questions —
the increment oracle now builds its potentials from an index, so each cell is
a different background, increment and pressure — and the mutation audit was
re-run to confirm the wider ranges had not thinned out the cases that catch a
broken rule. One had: spreading the moment weights made accidental repeats
rare, and repeats are the only instances the "distinctness ignored" mutant
gets wrong, so the deliberate-collision rate was raised to one half to buy
the contrast back. The widened check is better on both axes than the original
(637/1200 caught against 204/400, 1087 distinct against 323).

Parameterizing those potentials also produced, briefly, exactly the kind of
void check this experiment exists to catch. The first version wrote component
*i* of each potential without its own coordinate, which makes the field
divergence-free whatever its frequencies — so the half of the instances that
are supposed to violate the identity did not, and the rule scored 0.50 with
every failure a false one. The generator is now explicit about why each
component depends on all three coordinates.

Writing that audit found a real hole. The heat generator only ever missed the
exponent by 1/20 or more, so the check could not tell "exactly 1/2+h" from
"within 1/40" and passed the mutant 160/160. Differencing the field turns out
to resolve the exponent to about one part in 10⁵ — the relative residual is
linear in the miss and its noise floor is 4×10⁻⁸ — so the blind spot was in
the data, not the oracle. With mismatches sampled across decades the same
mutant now fails 64 times in 160.

Five limits survive the audit and are stated rather than patched: the pulse
rule and its oracle compute rates from the same function, so an error in the
rates is invisible to both; the covariance oracle drops the error terms of
(7.28); the cone's *second* oracle shares the coordinate map (4.20) with the
rule, which is why the covariance route is the primary one; `prop_9_6_decay`
checks that a closed form matches its recursion, not that a cycle gains 1/10;
and the core exponents are definitions written in the module rather than
measurements of a field.

The other nine results — Theorem 4.6, Propositions 5.5, 7.5, 9.6, 9.9, and the
localization and comparison of Section 10 — are estimates on function spaces.
Rather than leave them as citations, `nsmechanisms.py` takes the **method**
behind each and runs it:

```
imported step        mechanism carried out here
thm_4_6_profiles     the residual really does collapse to one profile, on a
                     divergence-free field built from the exponents (A+D=1)
prop_5_5_background  the cutoff scales from the paper's own recursion, and
prop_9_9_summation   the resulting tail measured against (5.35)
prop_7_5_stress      the covariance integrated from the pulses themselves and
                     solved for positive squared amplitudes
prop_9_6_cycle       the cycle's thirteen-term exponent table, enumerated,
                     against the four closing inequalities the proof states
prop_10_1_localize   the cutoff applied to the potential before the curl, with
                     the divergence measured on the transition region
lemma_10_3_force     the Borel-type extension built and differentiated
lemma_10_4_energy    the energy identity on a model with the same structure
lemma_10_5_unique    no mechanism built; see below, where the lemma is
                     decomposed instead
```

That table is about *mechanisms run on instances*, which is what this run
does. Every one of the eleven is separately **decomposed** further down,
including Lemma 10.5, for which no mechanism was ever built.

Each mechanism agrees with its oracle on every instance, at base rates near
one half because each generator emits both answers — a cutoff schedule that
works and one that does not, a nonlinearity that cancels in the energy
identity and one that does not, a curl taken before the cutoff and after.

Two things turned up while writing these. The exponent check first read the
residual at a single similarity point and passed a mismatched pair, because
the terms whose exponent depends on `D` cancel against each other near
η = 1/10 for that profile; it now reads four points and takes the worst, the
same repair the heat check needed. And the paper's
closing summary of Proposition 9.6 and its own term table are not
equivalent:
for radial derivative losses between 1/29 and 1/25 the summary fails while the
table still gives the 1/10 gain. The summary is sufficient, not necessary, and
the run skips that window rather than scoring the difference as an error.

The derivation itself is now chained rather than described. `nsderivation.py`
writes the correction cycle of Proposition 9.6 as four rules over a cell
holding the decay orders, one per numbered step of its proof, and the machine
derives a cycle the way it derives anything:

```
search over the four steps: found in 4 moves
  cycle_op1_harmonics  cycle_op2_stress  cycle_op3_mean  cycle_op4_moments
kept as one rule: 4 primitive steps, confidence 0.0625, UNTRUSTED
cycle_once: 1.0000 over 1200 checks vs cycle_state_by_closed_form
after checking: confidence 0.9992, trusted
ten cycles from the same start: found, 10 moves, confidence 0.9917
```

That is the architecture's form of transfer. A path is searched for, kept
under one name by `keep_proof`, checked as a chain, and then reused as a
single move — so ten cycles is a ten-step proof rather than a forty-step one.
It also forced a correction to the core: a composite's confidence was the
product of its members', which is right for a chain nobody has checked and
wrong once the chain itself has been measured. Four unverified steps price a
cycle at 1/16; twelve hundred agreements with an oracle outside it say
something much stronger. `CompositeRule.confidence` now takes the better supported of the
two, and cannot launder trust, which is a separate flag.

With the cycle and the increment identity in place the whole derivation is a
chain the machine finds for itself, seventeen moves from the exponent to the
theorem, and it reports which moves are its own:

```
the exponents and the energy budget         4 moves, 0 imported
                                            confidence 0.9992
profiles, background, the wave increment    3 moves, 2 imported
                                            1.0000 over 1 measured step, 2 unmeasured
the stress, then one correction cycle       4 moves, 3 imported
                                            0.9992 over 1 measured step, 3 unmeasured
summation, localization, force, comparison  6 moves, 6 imported
                                            nothing measured, 6 unmeasured steps
```

The second column is the change that makes those numbers readable. A chain
used to report one figure, the product over its steps, which for a leg of six
never-checked estimates is 0.0156 and for the whole derivation 0.0005 — a
number that looks like a verdict of near-certain failure when what it records
is that nothing here has read six proofs. `Proof` now carries both: a
confidence over the steps that have evidence behind them, and a **count** of
the steps that do not. Ignorance is reported as ignorance instead of being
compounded into a probability.

The increment identity of Section 3.3 — that adding a divergence-free `w`
splits the residual into `L_u(w, pi)` and `div(w ⊗ w)` and nothing else — is
one of the checked moves, verified by building fields, differentiating them,
and confirming that an increment which is *not* divergence-free leaves the
`w div(w)` term behind.

### Three steps that are proved, and why only three

Everything above is sampling. A rule is asked questions and compared with an
oracle, which settles it on the questions asked and says nothing about the
rest — right for a rule that reads pixels or integrates a field, and wrong
for a rule whose content is a statement about every case. That is why the
imported steps stay untrusted however many instances agree.

Three of the paper's derivations are not estimates at all, and all three
are plain enough to carry out here rather than import. One is an identity,
one is a recursion, and one is an elementary inequality over a continuum.

**The increment identity of Section 3.3.** Everything in Sections 7 and 9
rests on

    R(u+w, p+pi) = R(u, p) + L_u(w, pi) + div(w ⊗ w),

and the derivation is four lines. The time derivative, the Laplacian and
the pressure gradient are linear and pass through the sum without comment,
so the whole content is the advection term: expanding
`(u+w)·∇(u+w)` gives the four products, and the last of them is the
quadratic flux only up to `w div w`. The two sides therefore differ by
exactly `-w div w`, which vanishes precisely when the increment is
divergence-free.

Every step of that is algebra among the values of the field and its first
derivatives *at a point*. Nothing in a pointwise identity knows that
`dw0_1` came from differentiating anything — so take those values as
independent symbols, expand both sides as polynomials, and subtract.
`symalg.py` does that, and the difference is the zero polynomial, which
settles the identity at every point of every smooth field with no field
ever chosen. The defect comes out in closed form rather than as a residual
that happened to be small, which is why the divergence-free hypothesis is
visible in the answer instead of assumed. The 1200 differentiated fields
still run; they now confirm a theorem instead of standing in for one.

**The induction of Proposition 9.6.** This is the exception among the
uniform claims, and it is worth being exact about why. Its
content is an *induction*: one correction cycle gains 1/10 in the decay
order, so σ_j = 1/5 + j/10 at every stage. "Every stage" ranges over the
non-negative integers, every order the cycle touches is linear in the stage
index with rational coefficients, and a universally quantified linear
inequality over such a variable is decidable. So this one claim can be
decided rather than sampled.

`linarith.py` is that decision procedure. Together with `symalg.py` it sits
behind `provers.py`, which is the third way a rule can earn its standing,
alongside oracles and reference chains. The
cycle is run **once**, with the stage index *and* the radial derivative loss
both left as variables. Its four moves are the same Python the rules run —
they call `least` and `short` instead of `min` and `<`, which behave
identically on numbers and defer on linear forms — so what is proved is the
rule and not a transcription of it. Eleven inequalities fall out and each is
decided by reading its coefficients:

```
PROVED for every stage n >= 0 and every kappa with kappa <= 1/10:
  11 inequalities, all decided by inspection
prop_9_6_all_stages: PROVED, exact True, confidence 1.0000
```

The budget is **derived, not quoted**. Leaving the loss symbolic turns the
conjunction into an interval, and the interval is κ ≤ 1/10; the binding
constraint is step 2's stress correction, which costs four radial
derivatives against the 1/10 the stage must gain. The construction takes
κ = 1/100000, four orders of magnitude inside it. The closed form (9.8) is
then a corollary rather than an import.

That confidence of 1.0000 is not a thousand agreements rounded up. There are
no instances in it, and the library prints such a rule as `PROVED` rather
than `trusted` so the two cannot be confused.

Everything below this point was written later, after the eleven were
decomposed. The main run still reports them as untrusted imports, because
that is what they are *to it*: the decompositions live in their own files
and reach the same cells by a different route. Both readings are in the
repository on purpose.

The guard against the expensive mistake is that the prover has to be able to
fail, and the tests break both halves to show it does. Move the rule's
threshold to 1/5 and the proof is refused, naming a loss where rule and
procedure disagree. Make the first move gain nothing and no budget exists at
all. Make step 2 cost one derivative instead of four and the derived budget
widens accordingly — the number follows the moves, not the docstring.

### Statements about function space norms, reached by chaining

Calling the remaining quantifiers "out of reach because they are about
norms" was too coarse, and `normcalc.py` is the correction. An estimate

    ||∇^s P_M u||_{L^p(R³)}  ≤  C · M^a · q^b

has two halves that behave completely differently. That *some* finite
constant works is analysis and is not decidable here. What `a`, `b`, `s`
and `1/p` have to be, and what happens to them when two estimates are
combined, is linear arithmetic, and it is what the construction actually
manipulates. So the module splits them and says which is which on every
line it prints.

The exponents are **derived, not quoted**. An inequality between norms has
to survive replacing `f` by `f(λ·)`, and that requirement forces the
exponent: rescaling moves the frequency and both norms, and a constant that
does not depend on λ leaves `a = d(1/p − 1/r)`. The prover rearranges that
equation symbolically with the indices left as variables, then runs the
rule's own arithmetic on the same variables and compares. Writing it that
way caught a real hole in the first version, which compared against a
module helper instead of the rule and certified a rule with `2/p` in place
of `3/p`.

Two rules are where a family of estimates becomes one statement, and they
are exactly the quantifiers the run kept deferring:

```
sum_over_bands            converges iff the frequency exponent < 0
limit_at_the_singularity  vanishes  iff the scale exponent > 0
```

Both decline otherwise, so a divergent sum or a bound that grows stops the
search rather than passing through it. A derivation then looks like this,
starting from an L² band estimate for the residual of the kind Proposition
7.5 delivers, which is **assumed**:

```
est(d=3,dv=0,ip=1/2,fr=-3,sc=1/5)
  --bernstein_uniform-->        est(d=3,dv=0,ip=0,fr=-3/2,sc=1/5)
  --sum_over_bands-->           glob(d=3,dv=0,ip=0,sc=1/5)
  --limit_at_the_singularity--> vanishes(d=3,dv=0,ip=0)
```

The residual's L^∞ norm tends to zero at the singularity, over three moves,
with every exponent exact. The quadratic term goes the same way through
`holder_product`, where the integrability indices add rather than being
chosen, and the rule declines once their sum passes one.

The accounting that makes this honest is `Rule.assumes`. Each rule names
the classical fact it leans on but does not establish, and
`assumptions_behind` collects them back out of a finished chain: Bernstein,
Hölder, the geometric series, the convergence of the Littlewood-Paley
decomposition, and the starting estimate itself. An exponent computed
exactly on top of an unstated assumption is the most misleading thing this
package could print.

So the gain is real and narrower than it looks. Bernstein and Hölder are
not proved here and are not in question. What this section establishes is
that everything the construction does *with* an estimate is machine-checked
once it has one.

What it does not establish is the estimate, and at the time this was
written I took that to be the end of the matter: Theorem 4.6 and
Propositions 5.5, 7.5 and 9.9 are not corollaries of Bernstein and Hölder,
and no chain of them produces one. Both halves of that are still true and
the conclusion I drew from them was wrong. Those four are not reached by
chaining general inequalities; they are reached by following their own
proofs, which is what the later sections do.

**The viscous balance of Section 7.2.** The carrier frequency is
`k = ⌈ε^{-1/2}⌉`, and the construction needs `1 ≤ εk² ≤ 4` so that
viscosity neither drops out of the amplitude equation nor swamps the shear
amplification. That has to hold at *any* ε the construction picks, which is
a continuum, so instances were never going to reach it. The argument is
three lines. The rule now enforces the two conditions that define the
ceiling before it answers, so the lower bound is immediate; the upper one
follows from

    4(k−1)² − k² = (3k−2)(k−2),

expanded here as a polynomial identity rather than quoted, with both
factors non-negative for `k ≥ 2` by linear arithmetic; and `k = 1` pins
`ε = 1`.

That proof also settles something no amount of running could have. The
rule's `viscous_balance_lost` branch is **unreachable**, so its base rate
was never going to be informative, and only a proof could say so.

Bounding the coefficient is only half of deciding the rule, and the first
version of this prover missed the other half: a rule that had narrowed its
acceptance window would still have been sitting on a true theorem while
giving the wrong verdict. The coefficient is `4ε` across `[1/4, 1)`, so its
achievable values fill `[1, 4)`, and the prover now probes both ends of
that. A window narrowed to `[1, 2]` is refused, and that is a test.

### The one door in the closed registry

The catalogue tells a controller, in as many words, that it must pick
generators and oracles by name and cannot write new ones. That is there to
stop a specific failure: an oracle written by whoever wrote the rule is not
a second opinion, and verification would become the machine reading its own
handwriting.

But "written by a human" was never the requirement. The requirement is that
the thing be checkable, and for one class of rule it is. A band
inequality's frequency exponent is *forced* by the inequality surviving a
rescaling of the function, so a proposal either has that exponent or it
does not, and the machine can tell which. `propose_band_rule` is that door.

A proposal arrives as data, never as code: where the integrability index
lands, what the proposer claims the frequency costs as coefficients, and
the classical fact being leaned on. Three kinds of bad proposal are refused
and none of the refusals trusts the proposer.

```
bernstein_to_L4    ADMITTED  the honest one
wrong_exponent     REFUSED   two powers of 1/p where scaling forces three
smuggles_a_scale   REFUSED   a claim about the concentration scale
                             rescaling cannot support
no_assumption      REFUSED   declines to say what it leans on
```

An admitted rule works immediately. A target in L⁴ that the shipped
calculus cannot reach is reached through it in three moves, and the
assumption it rests on comes back out of the chain rather than disappearing
into it.

What this settles is narrow and worth being exact about. A controller can
now extend the calculus without being trusted, because the part that could
be wrong is the part the machine decides. Generators and oracles stay
closed, where correctness is not decidable and the registry is still doing
real work. And scaling fixes the exponent of an inequality that is true; it
does not make one true. An admitted rule is a proved exponent sitting on a
named assumption, which is exactly the standing of the rules that shipped.

### Lemma 10.5, the one I said had no representation

I classified the comparison lemma as out of reach because it "turns on a
pressure flux controlled by Riesz transforms on the whole space, and
nothing here models that". That was a claim about the machine made without
reading the proof. `run_comparison.py` follows the paper's own argument,
section by section, using its section names.

```
derived steps: 15, validated: 15
leaves that are assertions about the construction: 0
inputs: 4  -- the lemma's stated hypotheses
```

The Riesz transforms appear inside a registered fact, exactly the way
Cauchy-Schwarz did in the energy bound, and the argument around them is an
implication like any other: subtract the equations, identify the pressure
gradient by a Liouville argument on a harmonic tempered distribution,
bound the pressure flux across an expanding sphere, pair with the cutoff,
absorb the fluxes by Young, Gronwall from rest, let the cutoff exhaust the
space.

**This one is weaker than the previous two and the difference is worth
naming.** Lemma 10.4 rested on six facts any analysis course covers. This
rests on fifteen, and several are the paper's own computations rather than
textbook theorems — the commutator kernel bound, and the claim that every
power of `A_R` in the flux bounds is below two so Young absorbs them. Both
are on the assumption list under their own names, and a reader should
check those in the paper rather than take them from the file. A finer
decomposition would push them further down; this one stops where the
paper's prose stops.

### The "quantitative" steps are transforms too

I claimed five of the eleven were out of reach because they need a
quantitative hypothesis, and that granting it would amount to granting the
estimate. That was wrong, and the objection is simple: those five are
**implications**. Each takes an input estimate and produces an output one,
and its hypothesis is the preceding proposition's conclusion rather than a
number smuggled in. Using the previous link is what a proof is.

`run_force_extension.py` tests it on Lemma 10.3, the one I called hardest
of the five.

```
derived steps: 8, validated: 8
leaves that are assertions about the construction: 0
inputs: 3  -- Lemma 10.2's conclusion and a choice of widths
prior facts: 8
```

Eight facts, every one at home in an analysis course: a bump is smooth, a
cutoff of width w has kth derivative of size w^-k, Leibniz, the
Weierstrass M-test extended to derivative series, a C^k-convergent series
of smooth functions has a smooth sum, Borel's matching construction. None
names the construction, and a test asserts that.

The step worth being suspicious of is the one where the schedule does the
work, because it would be easy for that single rule to be the whole lemma
in disguise. It is not taken on trust: the package already runs that
schedule on instances and measures it against a naive alternative, at
1.0000 over 400 checks with 377 distinct.

So the category was wrong, and the same reading applies to the other four
grouped with it. What survives is narrower and worth keeping separate: a
chain of transforms is only as good as where it starts, and this chain
starts at an existential about a function. The transforms were never the
obstacle.

### Proving an existential by checking the candidate

The right way to prove "there exists a profile with these properties" is
not to deduce it, which no chain of implications does, but to exhibit the
paper's candidate and verify the conditions. Two of Theorem 4.6's four are
now attacked that way.

**The moment identities, exactly.** They are a finite linear system over
the rationals, so `witness.py` solves it and checks the solution. The
coefficients come out as `1/40, -11/24, 9/5` and every condition holds on
the nose. The arithmetic stays exact because of a substitution the
construction already uses: half-integer powers of rationals are
irrational, and under `x = t²` every moment is a closed-form rational.
That matters, because a determinant near zero and a singular matrix are
the same thing to a float and different things to the theorem.

**The heat exterior, symbolically.** This was the extension I flagged
twice and kept not building. Substituting the similarity field
`s^-A H(2τ/s)` into the exterior equation and differentiating reduces it
to a **single** power of s times an equation in `Z, H, H', H''`:

```
(-1/2 + 2A²) H  +  2 H'  +  (2 + 4A) Z H'  +  2 Z² H''  =  0
```

That collapse onto one power of s *is* the similarity structure closing,
and `symdiff.py` checks it rather than assuming it. The equation is
derived with the exponent left symbolic, so it is not specific to the
paper's choice, and then the candidate is measured against it: the
paper's `A = 1/2 + h` gives a relative residual of 3×10⁻⁵, while `A = 1/2`
gives 0.44 and `A = 1/2 + 2h` gives 0.60.

The second half is quadrature, so this is a partial result and the prover
reports `whole_domain=False`, which means the rule is **not** marked
exact. That flag existed since the provers were added and this is the
first thing to use it, which is the sort of detail that tells you whether
a distinction was real or decorative.

### One of the eleven, decomposed into prior facts

I claimed that however finely you decompose one of the eleven, the leaves
carrying the analytic content would be assertions about the construction
rather than rule applications. That is falsifiable, and
`run_energy_bound.py` falsifies it for Lemma 10.4.

The energy bound has the most elementary proof of the eleven. Pair the
equation with the velocity, watch the nonlinearity drop out because the
field is divergence-free, discard the dissipation because it has a sign,
apply Cauchy-Schwarz, divide by the norm, integrate from rest. Six facts
are registered as prior rules and each is general, reusable and mentions
nothing about profiles, pulses, cones or the correction cycle.

```
derived steps:                         6
validated by a rule the machine holds: 6
leaves that are assertions about this construction: 0
leaves that are the lemma's own hypotheses: 3
prior facts the whole thing rests on: 6
```

Zero. Every derived step went through a registered textbook fact, and the
only inputs are the lemma's own hypotheses, which is what a lemma is
supposed to have. In the main run this lemma is an untrusted import; here
it is derived, and that is a real change in its standing.

Two caveats, and the second is a genuine weakness. The six facts are
registered rather than proved here, so the honest status is *complete
modulo six classical facts* — which is what mathematics does with a
standard library, and is legitimate provided the list is printed, which it
is. And I wrote both the decomposition and the prior rules, so I could
have shaped the rules to fit the steps. The defence is that each rule is
checkable by eye as a general statement, and a test asserts that none of
them names anything from the construction. A stronger test would register
the library first and then decompose a lemma nobody chose it for.

**Instantiating it.** A proved lemma is inert: it is an implication and
nothing here establishes its antecedents for the construction's own field.
The paper supplies those, and so does any reader, so they are granted
through `assume` with their standing declared. For this lemma the three
are unusually cheap and a reader can check that in a minute: the force is
*defined* as the residual, the field is a curl so its divergence vanishes,
and starting from rest is stipulated by the theorem. With them the energy
bound holds for the constructed field, all six steps validated, resting on
six textbook facts and three instantiations.

Doing that exposed a design flaw the arguments had not. `JoinRule` was
built with **exact** premises, on the reasoning that a conjunction names
what it collects. A join written for `u` would not fire on `U`, so every
general lemma with a conjunction in it was single-use and could not be
instantiated at all, while the unary rules were patterns and instantiated
without trouble. Premises now carry `?name` placeholders bound
consistently across a rule, and the instantiated lemma goes from three of
six steps to six of six.

One gap remains and it is the last honest one here. Identifying the cell
this reaches with the imported step's cell is a judgement that two
sentences in two different languages say the same thing. No machinery here
makes that judgement, and asserting it would be exactly the unaudited
claim this package exists to catch.

What this does not generalise to is the other ten, and the experiment is
more useful for showing that than for the lemma it settles. The three
existentials assert an object into being and no chain of textbook facts
produces one. The remaining bounds concern objects the construction
builds, so their leaves reach for properties of those objects rather than
for general facts. The eleven are not alike, and I had been treating them
as though they were.

### Auditing a proof someone else wrote

The direction this is built toward. A proof assistant checks that each step
follows and then says `theorem proved`, which is the right answer to the
question it is asked and not the only question worth asking. A proof can be
perfectly valid and rest on a citation nobody looked up, an estimate quoted
from elsewhere, or a hypothesis the author supplied because they knew it
was true. Validity does not separate those from a step checked against a
computation, and this package tracks exactly that separation everywhere
else, so `audit.py` makes it the product.

A proof arrives as a list of steps, each naming the steps it follows from.
The submitter's justification is recorded and **never consulted**: the
machine finds the rule itself or reports that it cannot, so a justification
that sounds right and matches no rule comes back unvalidated. What you get
is a ledger.

```
verdict: INCOMPLETE
  steps derived: 9, validated: 8
  steps with no justification in this library:
    7: NOT EXPRESSIBLE -- the statement has no cell the machine can read
  inputs the proof assumes rather than derives: 5
  steps that validated THROUGH an untrusted rule:
    2: via lemma_5_4_summation
  how each rule it used earned its place:
    proved on its whole domain  bernstein_uniform, increment_identity, ...
    exact arithmetic            assemble_local_field_theorem, ...
    leans on an assumption      bernstein_uniform, decay_order_to_band_estimate
    untrusted import            lemma_5_4_summation
  the conclusion rests on 4 assumption(s): ...
```

The line about untrusted rules was added because the report earned it. On
its first run over a real decomposition it showed a step counted as
validated that had got there through an untrusted import, and "validated"
was hiding the difference. A step is validated when the machine finds a
rule joining two cells. Whether that rule is any good is a separate
question and is now a separate line, with its own verdict.

`check_proof` is a controller tool, so a model can submit a proof and get
the ledger back. The rule facets are deliberately not exclusive buckets: a
band inequality is exact arithmetic **and** proved on its whole domain
**and** leaning on Bernstein, and an auditor needs all three rather than
whichever one a classifier picked first.

### Conjunction, and the experiment that found it missing

Taking one of the eleven apart turned up a gap I had not predicted, which
is the main reason the experiment was worth running. `run_decomposition.py`
writes Proposition 9.9 as the steps a reader would write and asks the
machine to find a rule justifying each link.

The first version scored three of ten and blamed missing inference rules.
It was wrong: it had the decomposition as a **list**, which silently gave
one step the wrong premise. Proofs are graphs. Fixing the shape raised the
count to six of eight and exposed the actual obstacle.

The obstacle was that a proof here was a **chain**: one rule, one cell, one
successor. The final step of Proposition 9.9 follows from four facts
established separately, and there was nowhere to put a collecting step.
Nearly every real argument ends in one, so a single failing link badly
understated it.

`JoinRule` states a rule with several premises, and `saturate` derives a
**set** of cells instead of walking a path, applying every rule that fires
until the target appears or the budget runs out. The accounting survives
the change of search: a conclusion collected from assumed premises still
reports them. With it the decomposition reaches **seven of eight**, and the
one remaining failure is that smoothness of the summed field has no cell.

That is the only gap in this conversation that went from named to fixed.
The rest of what the experiment found is unchanged: of the five things the
decomposition assumes, four are setup and one is an L² band estimate for a
correction, which is the analytic content and which nothing here derives.

### Granting a hypothesis, on the record

A general theorem in the library is inert until something supplies its
hypothesis about a specific object. Register "a continuous function on a
closed interval is bounded" and it chains correctly to integrability in two
steps, and never fires, because nothing here derives `continuous(u,[0,1])`
for any object in the construction. Classical theorems are cheap to hold
and useless without instantiation.

A reader supplies that instantly, and so does a language model: that a curl
of a smooth potential is smooth is not something anyone rederives. So
`assume` grants a hypothesis and the controller has a tool for it.

The danger is obvious and the design is entirely about it. An assertion is
epistemically identical to an import; nothing becomes proved because a
model said it. And the machinery cannot tell a textbook fact from a paper's
central estimate, because both arrive as a sentence. So three things
happen. The asserter says whether the fact is `standard` or
`construction-specific`. The provenance is recorded, so an LLM-driven run
can be audited against a scripted one. And every proof that passes through
one counts it:

```
PROVED: 'u' => 'integrable(u,[0,1])'
  (3 steps, confidence 1.0000, resting on 3 assumed steps)
    assume:continuous(u,[0,1]) -> continuous(u,[0,1])
    weierstrass                -> bounded(u,[0,1])
    bounded_integrable         -> integrable(u,[0,1])
```

`Proof.assumed` counts these separately from `unmeasured`, because a rule
can be perfectly reliable as a rewrite and still carry a premise nobody
checked. A conclusion reached this way is a proof modulo its assumptions
and is never printed as anything else. Granting the thing you were asked to
prove stays possible and becomes visible, which is the most the machinery
can do about it.

This also made the existing reports more honest. The norm calculus chain
used to print `confidence 1.0000`; it now prints `confidence 1.0000,
resting on 2 assumed steps`, because Bernstein and the geometric series
were always doing part of the work.

### Two quantities the machine worked out for itself

Everything above checks numbers the paper supplies. These two the machine
produces, and the distinction is the one that matters for whether an
architecture like this is doing anything.

**The step size of Proposition 9.6.** The cycle gains 1/10 in the decay
order per pass, and that 1/10 was written into step 4 with everything
downstream taking it on faith. It does not have to be. The first three
moves never mention the gain — they say where the orders land — and step 4
only compares. Ask instead for the largest gain those three would clear,
and the number comes out:

```
the four moves support a gain of up to 17/100 per cycle at kappa=1/100000;
the construction takes 1/10, which is inside it.
The cap is the mean order, at 17/100.
```

Eleven margins, each a linear form in the radial derivative loss. None
shrinks as the stage grows, so stage zero is the binding case and the gain
derived there holds at every stage. The paper's choice is conservative, the
term that caps it is named, and weakening step 3's estimate drags the
derived gain down with it rather than leaving the docstring's number
standing. That last part is a test.

**The estimate the derivation would need.** Every chain in the norm
calculus takes an estimate and carries it forward. Run the same chain with
the estimate left as variables and the guards report what it would have to
be:

```
from est(d=3,dv=0,ip=1/2,fr=<fr>,sc=<sc>)
  needs -3/2 - fr > 0   [the dyadic sum converges]
  needs sc > 0          [the bound vanishes at q = 0]
```

That is `propose_rules` applied to analysis. What comes back is the thing
worth proving, derived by running the rules rather than by reading them,
and the −3/2 is Bernstein's `d/p` rather than a constant anyone typed.

Neither is a proof, and neither is the paper's insight. Which construction
to try, which profile, which pair of pulse families: none of that was found
here and this experiment does not claim otherwise. What the two show is the
narrower thing. Where a constant or a hypothesis is already implied by
rules the machine holds, it can be made to produce it instead of being told
it.

What this does not reach is the whole of the rest. The construction
quantifies over five things:

| quantifier | status |
|---|---|
| every correction stage j | **proved** — a rational recursion on an integer index |
| every point of every smooth field, for the increment identity | **proved** — a polynomial identity in the one-jet |
| every ε in (0, 1], for the viscous balance | **proved** — a factorisation whose sign linear arithmetic decides |
| every order of the background expansion | **decomposed later** — Lemma 5.4's schedule in `run_estimates.py`, its convergence resting on comparison with a geometric series |
| every dyadic band | **reached by derivation**, given Bernstein: the dyadic sum is decided by its exponent |
| every concentration scale q as q → 0 | **reached by derivation**: a limit decided by the sign of one exponent |
| every slow label | **not established here** — the phrase is my own shorthand rather than the paper's, and no decision procedure touches it. The statements it stood for are decomposed later along with the rest |

So the answer to "can the proof be finished here" is still no, and the
reason has moved twice. It is not that more instances are needed. Where the
paper's derivation is plain, an identity in the one-jet or a rational
recursion on an integer index, it has been carried out and the result is a
proof rather than an accuracy. Where the claim is about norms, the exponent
half is now derived and chained and the analytic half is assumed by name.
What remains is the analytic half itself: the specific estimates the
construction needs are not consequences of the general inequalities, and
nothing here produces one. CHECKING those estimates would mean formalising
the analysis, which is what the Lean development does and what this
architecture does not attempt.

That is a different thing from finishing the proof, and conflating the two
is the mistake this paragraph originally made. Citing an estimate is how
every proof uses a result it does not reprove, and the argument assembles
on cited estimates exactly as it would on formalised ones. It is assembled
below, end to end, and what stays outside the machine is the analysis
itself rather than the proof.

Two further derivations looked plain enough to be worth the same
treatment, and I wrote here that both needed machinery this package did not
have. **Both were done afterwards.** The exterior heat solution of Lemma
A.6 is symbolic differentiation, and `symdiff.py` now substitutes the
similarity ansatz, reduces the exterior equation to one ordinary
differential equation, and gets the paper's own exponent out. The cutoff
schedule of Lemma 5.4 is decomposed in `run_estimates.py`, where its
convergence bottoms out in comparison with a geometric series.

None of this makes any of the nine trusted, and *within this run* that is
the finding rather than a shortfall. Every one asserts something uniform in
the concentration scale, the dyadic band, the slow label and the correction
stage, and a mechanism run on instances is not an argument of that shape.
So in `run_navier_stokes.py` they stay untrusted, at the Laplace prior of
1/2 that an unchecked rule deserves. What that run reports about them is
how much of its chain is imported, and what it reports about the theorem is
the derivation:

```
the chain through the imported steps: 17 steps, confidence 0.9983 over 6
                                      measured steps, 11 unmeasured
Theorem 1.1 with the prerequisites present: DERIVED
  stages walked: 11 of 11, starting at moment_family(B)
  rule applications end to end:        129
```

Eleven unmeasured steps is the machine reporting that it has not read eleven
proofs — a statement about those eleven imports, not about the theorem.

The run ends by demonstrating the two real gaps rather than asserting them:
the cone rule answers confidently far outside any shear a profile can
produce, and 11160 decided instances do not reach a statement quantified
over every scale, band, label and correction stage.

The perception half is the same story it is everywhere in this package.
Drawn on the specific tape, the exponent cell is read by `transcribe_unsafe` and the
five-step chain to the energy verdict goes through; marked `observed`, nothing
in the library can start, and the reader that would close it is exactly what
this run does not build.

### A note on reading order

The sections above were written as the work went, and several of them
record a conclusion that a later section overturns. That is deliberate and
the corrections are marked where they occur, because the sequence is the
most useful thing here: six times I predicted where the method would
break, and six times the prediction was wrong in a way that took an
experiment rather than an argument to settle.

If you want the end state rather than the route, read from here.

### Assembling the whole argument

The sections above each settle one question. This one is what they add up
to, and it is the part I did not expect to reach.

**All eleven intermediate results are decomposed.** Every one was a single
untrusted label when this began — a rule that read one cell name and wrote
another, with the mathematics living entirely in the docstring. Each now
has a decomposition underneath it, following the paper's own proof, whose
leaves are named theorems or the construction's definitions.

The front half needed one prerequisite the back half did not. Sections 4
to 9 reason in the construction's own order calculus and name no public
theorem at all, which is where I stopped once and said so. That was half
wrong: the calculus is **Proposition 6.6**, a stated result with a proof,
so registering it cites a proposition rather than paraphrasing prose.
`apparatus.py` holds it — the product laws, both operator tables, the
sum-across-orders rule — as exact rational arithmetic. With it in place
Proposition 9.5 reproduces the paper's own exponents:

```
W(1/2,1) squared is W(1,2), radial derivative gives W(99999/100000, 2)
the paper states 1 - kappa_s                              matches
```

**The chain closes, and it runs as one derivation.** Not eleven results
checked separately and observed to line up: the machine starts at
`moment_family(B)`, the construction's finite family of radial powers, and
derives `theorem_1_1_forced_blowup` with nothing but its trusted library.

```
stages derived: 11 of 11
rule applications from moment_family(B) to Theorem 1.1: 129
  granted hypotheses pulled in:  29
  correspondences between languages:  20
  applications of a named theorem:  80
```

Getting that number honest took two fixes worth naming, because both
looked like success. Every link's entry cell was granted as an axiom, and
a granted cell is reachable from *anything* in one step, so each
decomposition already held its own inputs and the bridges between links
never fired. `chain_supplied_cells` now withholds any cell the chain is
supposed to produce, so a link that skipped its predecessor fails instead
of passing. And a conjunction recorded its premise *patterns* as the
origins of its conclusion, which no derived cell ever matches, so the walk
back from the target stopped at the first join and every link reported two
steps whatever was underneath it. Joins now report the cells they actually
fired on. The 129 is what was there all along and could not be seen.

The three counts are the ledger and the split is the part to read. Eighty
of the steps apply a named theorem. Twenty-nine pull in something the
paper asserts and this repository grants. Twenty are my reading that a
cell in one decomposition's language says what a cell in the next one's
language says, and those are the weakest items in the whole argument.

**And the 25 granted estimates are discharged.** Nine decomposed, four
duplicates, two pointing back into the chain, one that turned out to be a
*definition* rather than a claim (the scale at which the pulses are
introduced is chosen), and one already covered by another entry. Nothing
netted away; the ledger prints all of it. Two of them the machine does
better than cite: Lemma A.6's profile equation comes out of `symdiff`
substituting the ansatz and differentiating, and the increment identity
the cycle gains rest on was proved by expanding it in the one-jet.

### Every rule, and how it was formed

This machine forms rules four ways, and the census reads each rule's own
state rather than a label. `run_census.py` prints this.

| type | how it is formed | rules |
|---|---|---|
| **PROVED** (3) | decided on its whole domain by a decision procedure; no instances in it | `increment_identity`, `ns_carrier_frequency`, `prop_9_6_all_stages` |
| **CHAINING** (1) | a found path kept as one rule, then checked *as* a chain | `cycle_once` |
| **DISCOVERY** (12) | instances validated against an oracle taking an independent route | `energy_budget_3_5`, `eq_4_1_exponents`, `lemma_4_5_cone`, `lemma_5_4_summation`, `lemma_7_4_pulse`, `lemma_A1_moments`, `lemma_A6_heat`, `lemma_10_3_borel`, `lemma_10_4_energy_bound`, `prop_9_6_decay`, `prop_9_6_table`, `prop_10_1_curl_order` |
| **KNOWN** (144) | registered from outside: a named theorem, or a definition the construction makes | listed below by source |
| *superseded* (11) | the original imports, still untrusted, each now with a decomposition reaching the same cell | `thm_4_6_profiles`, `prop_5_5_background`, `prop_7_5_stress`, `prop_9_5_initialize`, `prop_9_6_induction`, `prop_9_9_summation`, `prop_10_1_localize`, `lemma_10_3_force`, `lemma_10_4_energy`, `lemma_10_5_unique`, `thm_1_1_blowup` |

Folding the superseded eleven into KNOWN would suggest the argument still
runs through them, so they are counted apart and a test asserts they stay
untrusted.

**The 144 KNOWN rules, by which decomposition registers them.** Read the
names: none of them is about this paper.

| source | n | rules |
|---|---|---|
| `run_construction` | 22 | `determinant_multilinear`, `rolle_counts_zeros`, `nonzero_determinant_inverts`, `self_map`, `contraction`, `banach_fixed_point`, `same_in_both_spaces`, `implicit_function`, `profiles_exist`, `axis_formula`, `bounded_reduction`, `scaled_cutoffs`, `background_hypotheses`, `background_sums`, `angular_average`, `covariance_columns`, `positive_diagonal_inverts`, `positive_weights`, `componentwise_roots`, `stress_realized`, `cycle_gains`, `closes_forever` |
| `run_estimates` | 34 | `chain_rule_on_cutoffs`, `choose_the_scales`, `locally_finite_sum`, `curl_keeps_divergence`, `compare_with_a_partial_sum`, `summation_lemma`, `similarity_reduction`, `integral_solves_it`, `exterior_solves_heat`, `losses_add`, `stage_bound`, `gain_grows`, `stagewise_gains`, `supports_shrink`, `common_domain`, `increment_splits`, `cancel_the_source`, `remainders_are_higher_order`, `moments_preserved`, `cycle_gain`, `change_of_variables`, `half_line_representation`, `edge_weight_monotone`, `torus_fourier`, `temporal_inverse`, `moment_system_square`, `solve_the_system`, `compact_support_kept`, `average_the_equation`, `leibniz_algebra`, `operator_norm`, `small_ball_absorbs`, `pressure_up_to_a_constant`, `prepare_exterior` |
| `run_comparison` | 15 | `subtract_the_equations`, `flux_is_integrable`, `fourier_of_L1_is_bounded`, `multiplier_identity`, `divergence_of_the_difference`, `harmonic_and_tempered_is_zero`, `sobolev_on_the_cutoff`, `riesz_is_bounded_on_Lp`, `commutator_kernel`, `pressure_flux_bound`, `energy_pairing_with_cutoff`, `transport_flux`, `young_absorbs_the_fluxes`, `gronwall_from_rest`, `exhaust_the_space` |
| `run_deeper` | 15 | `calderon_zygmund_kernel`, `commutator_drops_the_identity`, `cutoff_is_lipschitz`, `kernel_bound`, `polar_coordinates`, `power_integral`, `kernel_norm`, `young_convolution`, `similarity_coordinates`, `first_case`, `second_case`, `combine_the_cases`, `young_with_epsilon`, `compare_exponents`, `absorb` |
| `run_summation` | 13 | `cut_the_finite_block`, `azimuthal_stays_solenoidal`, `summation_hypotheses`, `summation_lemma`, `axis_representatives`, `finitely_many_terms`, `derivative_bounds`, `fundamental_theorem`, `corrections_vanish_outside`, `rising_factorial_bound`, `exterior_solves_heat`, `inner_expansion`, `theorem_3_1` |
| `run_theorem` | 12 | `coordinate_bound`, `cutoffs_exist`, `cut_the_potential`, `curl_is_solenoidal`, `axis_regularity`, `support_is_compact`, `localized_fields_exist`, `solves_on_the_interval`, `regularity_class`, `velocity_diverges`, `no_classical_continuation`, `contradiction_with_a_competitor` |
| `run_force_extension` | 8 | `cutoff_is_smooth`, `cutoff_derivative_scaling`, `terms_are_smooth`, `term_bounds`, `widths_beat_the_growth`, `weierstrass_for_derivatives`, `limit_of_smooth_is_smooth`, `borel_matching` |
| `run_estimate` | 8 | `riccati_form`, `invariant_region`, `log_derivative_bounded`, `integrate_the_derivative`, `cannot_cross_zero`, `frame_conversion`, `differentiate_the_ode`, `pulse_estimate` |
| `run_energy_bound` | 6 | `trilinear_antisymmetry`, `dissipation_has_a_sign`, `cauchy_schwarz`, `energy_pairing`, `drop_the_dissipation`, `divide_and_integrate` |
| `run_initialize` | 1 | `clears_stage_zero` |

The named theorems in that list are the ones you would expect: Banach's
fixed point theorem, Rolle, the implicit function theorem, Cauchy–Schwarz,
Gronwall, Sobolev, Riesz boundedness, Calderón–Zygmund, Weierstrass,
Borel, Leibniz, Young twice, polar coordinates, the fundamental theorem of
calculus.

### The graph

The spine. Each cell is produced by one intermediate result's
decomposition and consumed by the next.

```
h ─▶ ns_exponents ─▶ ns_core_scales ─▶ ns_core_energy ─▶ energy_budget_3_5
 │
 ├─ Theorem 4.6      ─▶ leading_profiles
 ├─ Proposition 5.5  ─▶ background_annular_stress
 ├─ Proposition 7.5  ─▶ stress_realized_by_waves
 ├─ Proposition 9.5  ─▶ state_at_stage_zero
 ├─ Proposition 9.6  ─▶ residual_flat_at_singularity
 ├─ Proposition 9.9  ─▶ local_field_theorem_3_1
 ├─ Proposition 10.1 ─▶ localized_fields
 ├─ Lemma 10.3       ─▶ compact_smooth_force
 ├─ Lemma 10.4       ─▶ uniform_energy_bound
 ├─ Lemma 10.5       ─▶ no_global_smooth_competitor
 └─ Theorem 1.1      ─▶ theorem_1_1_forced_blowup
```

Each link hangs a decomposition off the spine. Theorem 4.6's, which is the
one whose shape surprised me most, since the existential I spent several
sections arguing was unreachable turns out to be a fixed point:

```
moment_family ─determinant_multilinear─▶ determinant_splits
              ─rolle_counts_zeros─────▶ determinant_nonzero
              ─nonzero_determinant_inverts─▶ matrix_invertible ─┐
                                                               ├─self_map─▶ maps_ball_to_itself ─┐
quadratic_bounds ──────────────────────────────────────────────┘                                 ├─contraction─▶ is_a_contraction
                                                                                                 │
   ─banach_fixed_point─▶ solution_exists ─same_in_both_spaces─▶ solution_regular
   ─implicit_function─▶ solution_smooth ─┐
   exterior_prepared ──────────────────── ├─profiles_exist─▶ leading_profiles
   cone_margin ─────────────────────────┘
```

And Lemma 10.4's, the shortest, which is six textbook facts end to end:

```
forced(u,f) ─cauchy_schwarz─▶ pairing_bounded ────────────┐
divfree(u)  ─trilinear_antisymmetry─▶ trilinear_vanishes ─┼─energy_pairing─▶ energy_identity
            ─dissipation_has_a_sign─▶ dissipation_nonneg ─┤                        │
                                                          │   ─drop_the_dissipation─▶ energy_inequality
starts_from_rest(u) ──────────────────────────────────────┴─divide_and_integrate─▶ energy_bounded
```

The files print every link of every decomposition; these two are here to
show the shape.

That table undersells two of the four, and the correction matters. It
counts rules *created* by keeping a found path, which happened once. What
search actually did:

```
links the machine FOUND rather than was told:  137
rule applications selected by those searches:  136
of those, conjunctions found by saturation:     57
```

Every one was a search over 160 rules, handed two cells and asked for a
path. The prose justification beside each step is recorded for the reader
and **never consulted**, which is what makes the number mean anything: a
step whose reason sounds right and matches no rule comes back unvalidated,
and several did. Fifty-seven are conjunctions found by saturation rather
than by walking a path — when the decomposition experiment first flagged
the missing conjunction I read it as one failing link out of eight, and it
turns out to be over forty percent of every link in the finished argument.

Search also chose things nobody specified: the order of the four
corrections in the cycle, and the seventeen-step path to the theorem.

### What this is, and what it is not

**Theorem 1.1 is derived here.** The machine starts at one cell and reaches
it in 129 rule applications, every link found by search rather than
asserted, every leaf named, with a printed ledger of what is granted and a
test that fails if that ledger quietly grows. In the sense the word has
inside this architecture — a chain of mapping rules from established
prerequisites to the target — it is proved, and saying otherwise would be
holding it to a standard no proof meets.

The standard no proof meets is reproving its citations. The leaves here are
named theorems a reader checks against a textbook, the construction's own
definitions, which have no proofs because they are choices, and the paper's
own analytic estimates, granted by citation. Nothing in this repository
CHECKS that analysis, and that is the real limit: this is the paper's
argument, not an independent confirmation of it. Those are different
claims and only the second is out of reach.

The weakest items are neither the theorems nor the estimates. They are the
**twenty-one correspondences** — eleven saying a decomposition's conclusion
is the statement the chain names, ten saying one link's conclusion is the
next link's hypothesis. Each is my judgement that two cells in two
languages make the same statement. They are recorded with my name on them
so a second reader can reject any of them, and a rejected one breaks the
chain at a named place.

The honest summary of the arc: I predicted where this method would break
at least six times, and was wrong every time. Instances cannot reach
uniform claims — right, and beside the point. Implications cannot produce
existentials — right, and the paper produces one with Banach's fixed point
theorem. The front half uses unknown mathematics — wrong, it uses
Proposition 6.6. The remaining four are where the estimates live — wrong,
they were simply unattempted. Each correction came from the reader rather
than from me noticing.

**Experiment 2 — the distributive rule** (`run_multiplication.py`)

![Experiment 2: the final library and the conciseness objective](docs/experiment1.png)

```
--- objective ---
library: 11 rules, 9816 bits
benchmark: 4/4 solved, 6 rule applications
objective J = 15816 bits
  read_screen:      ok 1 steps via read_expression
  rewrite:          ok 2 steps via decimal_split, distribute_symbolic
  screen_to_value:  ok 2 steps via read_expression, eval_arith
  value:            ok 1 steps via eval_arith
```

| rule | holdout | fresh data | trusted |
|---|---|---|---|
| `read_expression` (specific→abstract) | 0.990 | **0.9867** (150 checks) | yes |
| `distributive_learned` (specific→specific) | 0.983 | **0.9609** (230 checks) | yes |

The learned rewrite was checked twice on the same fresh set: **0.9609** against
the oracle, and **0.9609** against the independent chain
`read_expression → decimal_split → distribute_symbolic → render`. Two routes
that share no machinery agreeing to four digits is what verification is for.

All four benchmark tasks solved, `J = 15816 bits` over 11 rules and 6 rule
applications, including `'12*30' (drawn, unlabelled) → '360'` in two steps at
confidence **0.9803** — a chain that leaves the specific domain and comes back.
Unlike the construction loop in experiment 3, nothing here is applied to its
own output more than once, so the chain is two links long and the reader's
0.9867 is most of what the confidence is made of.

Then the objective does something worth noticing: the learned distributive rule
is verified and trusted, and still reported **unused**. Going through the
picture costs two extra domain crossings to reproduce an identity the machine
already had, so `simplify_library` proposes dropping it — while keeping the
reader, which has no symbolic substitute. The rule is not wrong; it just does
not pay for itself. That is the conciseness objective doing its job, and it is
left in the example rather than tuned away.

The `grow_ensemble` step usually **discards** its specialist here, and says so:
in this run it mined 55 hard cases, trained on them, measured base 0.985 →
ensemble 0.960 on 454 held-out examples, and left the base rule alone. §3's
construction is treated as a claim to be checked, not an assumption.

**Experiment 3 — interior angles of a triangle** (`run_geometry.py`)

![Experiment 3: the final library after the proof is kept as a rule](docs/experiment2.png)

```
--- objective ---
library: 13 rules, 10736 bits
benchmark: 1/1 solved, 1 rule applications
objective J = 11736 bits
  triangle_180: ok 1 steps via triangle_angle_sum
```

Two learned rules of different kinds: `construct_aux_line` looks at the drawing
and *edits* it (output is another drawing, so it can be applied again — the
construction loop is proof search inside the specific domain), and
`read_angle_facts` says what the finished figure licenses. `iterate_rule` then
folds the loop into `construct_until_parallel`, which runs the constructor to
its own stopping point and keeps whichever drawing the reader most calls the
proof configuration — so a wrong step costs a candidate rather than the proof.
From a drawing whose auxiliary line starts off the apex and at the wrong angle:

```
PROVED: 'triangle0' => 'B1+A3+B2=180'   (3 steps, confidence 0.877)
  --construct_until_parallel [spec->spec]-->  triangle0|move_up|move_up|rotate_cw x5
  --read_angle_facts        [spec->abst]-->  'A1=B1,A2=B2,A1+A3+A2=180'
  --substitute_equalities   [abst->abst]-->  'B1+A3+B2=180'
```

Confidence is the product along the chain, so a rule that is merely *above
threshold* is not good enough to be applied six times: six links at 0.95 is
0.74 before the reader is even consulted. That is the conclusions' 0.99999¹⁰⁰⁰
argument seen from the wrong end, and it is why this example trains to
convergence rather than to "verified".

Underneath sat a gap verification could not see, and it is worth reading as the
cautionary result of the repository. A specific→specific rule is applied to its
own outputs, so a held-out set of generated *inputs* is the weaker of the two
checks available. The construction moves the line in steps of 0.12 and rotates
it in steps of 0.15, into tolerances of 0.06 and 0.12 — it therefore finishes
*near* the proof configuration and essentially never *on* it, while
`triangle_scenes` drew nothing but the exact configuration. Probed directly,
`read_angle_facts` called an exactly parallel line correct 40 times out of 40
and the line its own construction produces 21 times out of 40. A rule
verifying at **0.995** that cannot recognise a finished proof — and the search
duly reported "space exhausted" after eleven nodes while every rule in the
table read as trusted.

The repair takes two halves that do nothing apart: the policy now stops once
another rotation would not get *closer* rather than the instant the tolerance
is met, and the generator draws solved scenes across exactly that landing zone.
Changing only the policy leaves every training label identical, because no
generated scene lies in the window it affects; changing only the generator, if
it fills the whole tolerance band, restores the proof and triples the rate of
proofs licensed on unfinished constructions, because positives at 0.119 and
negatives at 0.121 are the same picture with opposite labels.

`keep_proof` then stores the whole thing as one rule — `triangle_angle_sum`,
specific→abstract at 0.87 — and the benchmark drops to a single rule
application, which is the `1/1 solved, 1 rule applications` in the screenshot
above. The proof is replayed image by image into
`renders/geometry/` (see below), which is the only way to check that the
auxiliary line really did end up through the apex and parallel to the opposite
edge.

**Appendix A — sketch to escape direction** (`run_robotics.py`)

`RuleNet(num_classes=7, num_slots=1)` — DetourNet with a smaller action set —
on 6000 generated sketches, verified against the collision geometry on 290
fresh ones:

```
top-1 0.683    top-3 0.883    'direct' 142/144
```

Top-3 is the operational number, for the reason `detourNet.evaluate` gives:
the planner walks the ranked candidates and takes the first one a collision
check clears. `direct` — the class whose failure drives the arm into an
obstacle — is at **0.986**; the six detour classes are what the other 0.32
is made of, and they sit between 0.36 and 0.50.

This is the one experiment where training to convergence does **not** rescue
the rule. At 1500 sketches it scored 0.72 on its holdout and 0.603 on fresh
scenes; that 0.12 gap is a rule short of data, and 6000 sketches close it —
0.696 and 0.683, within a point of each other, top-3 up from 0.800 to 0.883.
What is left is not overfitting but the task: picking one of six detours from
a sketch is genuinely harder than deciding whether the path is blocked at all,
which is the part the net does learn. So the rule stays **below its own 0.85
threshold and is never trusted**, and the machine will not let it into a
proof. That is the intended behaviour of `verify_rule`, and it is the reason
this appendix reports a ranking rather than a chain.

**Experiment 4 — the Riemann hypothesis, and where it stops** (`run_riemann.py`)

RH is not a cell: it quantifies over the zeros of an analytic function and
nothing here can write that down. Robin's criterion is, though — RH holds iff
`sigma(n) < e^gamma n ln ln n` for every `n > 5040` — and that is a predicate
over the integers, exactly decidable one `n` at a time. So the machine gets a
reader (`read_integer`, the only learned rule), composes it with four exact
divisor-sum rules, and decides the criterion from a *drawing*:

```
dataset         rule                      n     acc    base  reading
robin_balanced  robin_from_drawing       60   1.000   0.567  informative: +0.433
robin_fresh     robin_from_drawing      300   1.000   1.000  VACUOUS
robin_tail      robin_from_drawing      200   1.000   1.000  VACUOUS
read_fresh      read_integer            300   0.997   0.030  informative: +0.967
robin_tail      read_integer            200   0.000   0.005  informative: -0.005
```

Every accuracy is printed against the **base rate**, and that column is the
experiment. Robin's inequality holds at every `n > 5040` anyone can enumerate,
so a test set drawn from that range carries one label and a rule that answers
`robin_holds` without reading anything scores 1.000 on it. The last two rows
are the same 200 drawings scored twice: the composite is **perfect** on the
6-digit tail while its own reader is at **0.000** there, having never seen a
drawing that wide. A verification can be passed perfectly by a rule whose
perception has completely failed, and only the base-rate column shows it.

`robin_cases` is the honest test — Robin's 26 exceptional integers, which fail,
balanced against integers above 5040, which pass — and on it the composite
genuinely reads, at 1.000 against a 0.567 base rate. Proofs come out as mixed
chains, `'5040' => 'robin_fails'` in one step from the drawing.

Two things this does **not** do, both demonstrated in the run rather than
asserted. The chain has no notion of what it is looking at: hand it a cell
reading `zeta(s)=0` and it returns `robin_holds` with the same confidence it
reports for a real integer, which is why the universal claim is kept out of the
prover — posed as a cell, it gets "proved" the same way. And 560 exact
instances do not reach a statement about infinitely many; independent
computation has checked RH-equivalents vastly further without that ever
becoming a proof. The machine states the remaining step as an untested transfer
instead of folding it into a chain with a confidence, which is the behaviour
the architecture is for.

**Experiment 5 — forming a conjecture instead of proving one** (`run_montgomery.py`)

The other thing the architecture is for. Experiment 4 walks up to RH and stops
at the quantifier; this one does what experimental mathematics does — forms a
rule where truth is known, applies it where it is not, and reports the result
as a conjecture.

Montgomery's question: normalise the gaps between consecutive zeta zeros to
mean 1, and ask what the distribution looks like. The machine samples level
spacings from three ensembles it can generate and therefore grade itself on
(GUE, GOE, Poisson), draws each as a histogram + CDF on the specific tape,
learns a `specific → abstract` **choice** rule naming the ensemble, verifies it,
then computes the actual zeros by Riemann–Siegel and applies the rule there.

```
spacing_fresh:   n=300  accuracy 1.000  base rate 0.333  (+0.667)
spacing_bigdim:  n=150  accuracy 1.000  base rate 0.333  (+0.667)   # 120×120, unseen size
zeta_blocks:     0 of 40 cells could be labelled — as intended
                 gue 40/40 (100%),  t ∈ [14, 18047],  20000 spacings
```

`spacing_ensemble` **declines every zeta cell**, so no verification number can
be produced for the open question even by accident — the refusal is the feature.
The rule is verified only where the machine drew the data, and the zeta verdict
is an application, not a check.

A three-way choice cannot say "none of these", so the run tests typicality
separately — and the first version of that test was wrong in an instructive
way. Comparing zeta's 40-block average against *single* 500-spacing GUE draws
made it look more typically GUE than GUE, which is a sample-size mismatch, not
a finding. Against the correct null — GUE **group means** over the same 40
blocks:

```
zeta vs mean gue 0.686   goe 2.509   poisson 6.560
gue group means sit 0.161 ± 0.033 from the gue mean (max of 400 draws: 0.255)
the zeta group mean sits at 0.686 — OUTSIDE, by 2.7× the largest of 400 draws
lower half  t from    14   L1 to gue mean 0.740
upper half  t from  9879   L1 to gue mean 0.637
```

So: far nearer GUE than the alternatives, **and** statistically distinguishable
from it at these heights — with the deviation shrinking as the zeros climb.
That is the known slow convergence (Odlyzko needed the 10²⁰-th zero for close
agreement), recovered from the machine's own data rather than assumed.

What this earns is the verified rule; what it conjectures is the zeta verdict,
about the nearest-neighbour spacing distribution only — Montgomery's conjecture
concerns pair correlation, a finer statistic the rule never sees. And the
conjecture is not new: it is Montgomery–Odlyzko, reached from data by a machine
told nothing about it, which is the point. A pipeline for generating conjectures
is worth exactly what it scores on the ones whose answer is already believed.

**Experiment 6 — a rule from data, and a transfer that can be tested**
(`run_finite_height.py`)

Experiment 5 measured the zeta spacings' deviation from GUE and set it aside.
This asks whether that leftover is lawful enough to state as a rule:

```
p_zeta(s; t)  =  p_GUE(s)  +  g(s) / L  +  o(1/L),      L = log(t / 2π)
```

a fixed shape, amplitude falling as one over the log of the height. The point
is the contrast with experiment 4: Robin's criterion generalises to an infinite
family and the transfer can never be run, while this generalises to *higher t*,
and higher t is reachable. Fit `g` on low bands, predict bands the fit never
saw. On 80000 zeros to t=61394, fitting below t=33190:

```
band   L    vs GUE   vs rule   improvement
4    8.67   0.0748   0.0372       50.3%
5    8.85   0.0703   0.0466       33.8%
6    9.00   0.0715   0.0449       37.2%
7    9.13   0.0682   0.0429       37.1%
mean                              39.6%
shuffled-shape null  -46.4% ± 10.5%  (max -14.6%)
```

The transfer survives, and scrambling `g` across bins makes the fit *worse* —
so the structure carries the prediction, not the magnitude. The shape says the
zeros are **more rigid than GUE** at these heights: a deficit of small gaps
(repulsion stronger than the random-matrix law), an excess near the mean
spacing, a deficit again in the tail.

Two controls that mattered. The zero scan originally used a fixed step and lost
~9 zeros in 80000 by t≈60000 — skipped zeros come in *pairs*, so Z's sign
pattern stays consistent and nothing complains, and what gets skipped is the
*closest* pairs, which depletes small spacings and imitates the very repulsion
being measured. `scan_step` now adapts to the local mean gap. And the effect
needs ~3000 spacings per band to be visible at all (1500/band: +3%, 3000: +37%,
5000: +47%), so a run that splits too finely reports nothing and means nothing
by it.

This is very probably **not new** — a 1/log(t) correction is the standard scale
for finite-height effects here, and published computations reach 10²² where
this reaches 6×10⁴. What is offered is the measurement, its controls, and a
pipeline that produced it without being told what to look for.

## Layout

```
dynamicmultinets/
  tapes.py       the two tapes, their alphabets, and the head operations
  render.py      abstract → pixels: the built-in write function (no font files,
                 no image library — exact palette colours, byte-reproducible)
  palette.py     semantic classes; colour means what a thing IS
  nets.py        RuleNet — DetourNet with a slot head
  codec.py       how a rule sees a cell and what its logits mean
  rules.py       Python / Table / Neural / Composite / Ensemble rules + library
  prior.py       what the machine already knows (exact, symbolic)
  generators.py  experiments it can run          } a closed registry: the LLM
  oracles.py     what is true, and how strongly   } picks by name, never by code
  propose.py     which rule is worth forming — solved and unsolved cases
                 compared in the specific domain
  train.py       fitting a rule; hard-case mining; specialists
  verify.py      trust, grounding strength, counterexamples
  audit.py       check a proof someone else wrote and ledger what it rests
                 on: proved, exact, checked, assumed, or merely untrusted
  provers.py     the third way a rule earns its standing: decided on its
                 whole domain rather than sampled, so `exact` and not
                 `accurate`. Narrow on purpose
  linarith.py    linear arithmetic over an index -- decides the correction
                 cycle's induction and derives its budget
  symalg.py      exact polynomial algebra -- settles pointwise identities by
                 expanding them in the one-jet
  symdiff.py     symbolic differentiation of the similarity ansatz, which
                 derives the profile equation instead of quoting it
  apparatus.py   the construction's order calculus, from Proposition 6.6:
                 how the coefficient classes compose and how the operators
                 move between them, as exact rational arithmetic
  witness.py     builds a witness for an existential and checks its finite
                 conditions exactly, naming the ones it cannot reach
  normcalc.py    function space norms as rules: exponents derived from
                 scaling, analytic content assumed by name
  proof.py       best-first search over rule chains, plus `saturate`:
                 derives a SET of cells, which is what a conjunction needs
  compose.py     composition, distillation, the objective J, simplification
  halting.py     the statistical anytime algorithm (§5)
  machine.py     RenMachine: state + operations
  tools.py       the controller's instruction set
  controller.py  LLM controller and its offline twin

  navierstokes.py  experiment 1: the construction's decidable content
  nsmechanisms.py  the imported estimates, with their methods carried out
  nsderivation.py  the derivation as chaining rules, and the stage induction
```

## Notes on faithfulness

Things implemented as described, and things where a choice had to be made:

* **The specific tape really is opaque.** A cell holds pixels; `Content.text`
  is provenance for logging and for supervising a reader. `transcribe_unsafe`
  (copy the caption) exists as a baseline reader and is an ordinary rule in
  every respect: copying is lossless, so it is exact, its confidence is 1.0 and
  it is trusted, and it may appear in a proof. What keeps it from standing in
  for perception is not a label but what it can do — it *declines any cell
  marked observed*, and every benchmark task that is really a perception task
  starts from one. A camera frame carries no caption to copy, so a machine
  holding this rule and no reader still fails `read_screen`, `screen_to_value`
  and `robin_from_screen`. Two guards in `verify` keep the accounting honest
  alongside it: a check that leans on this rule is graded as the machine
  reading back its own handwriting, and two routes that share it are refused.
* **The distributive rule is discovered, not asserted.** Its oracle checks each
  instance numerically before emitting it, and rejects any that does not hold.
* **A rule must be trained where it will be USED, and the geometry run is a
  case study in what happens otherwise.** The construction moves the auxiliary
  line in steps of 0.12 and rotates it in steps of 0.15, into tolerances of
  0.06 and 0.12 — so it can never finish on the exact configuration, and
  `triangle_scenes` drew nothing else. Measured, the reader called an exactly
  parallel line correct 40 times out of 40 and the line its own construction
  actually produces 21 times out of 40. That is a rule verifying at 0.995 which
  cannot recognise a finished proof, and it left `triangle_180` unproved with
  the search reporting "space exhausted" after eleven nodes. Verification could
  not see it: it draws from the same generator. The fix is in two halves that
  only work together — the policy stops once another move would not get closer
  rather than the instant the tolerance is met, and the generator draws solved
  scenes across exactly that landing zone, staying clear of the 0.12 boundary
  so no two near-identical drawings carry opposite labels.
* **A found proof is still not automatically a sound one.** Replaying each
  proof and asking the oracle whether the drawing it ends on genuinely licenses
  the angle facts, a real fraction is not, and those proofs terminate, cross
  domains and report a confidence like any other. Over 120 triangles the fix
  above takes the benchmark scene from unproved to a genuine three-step proof
  and removes every outright failure (14 scenes with no proof, now none), at
  92 genuine proofs against 95 before — but the unsound ones go from 11 to 28,
  because a reader that accepts a range of finished constructions accepts more
  of everything. Widening the search is worse still (see `proof.search`'s
  `beam`), and folding the loop into `iterate_rule` shortens the proof from
  eight steps to three without helping soundness, since the rule choosing among
  candidate drawings is the same reader that misjudges them — selection cannot
  repair its own judge. A benchmark counts a task solved on `proof.found`
  alone, so `library_report` prices these in. Auditing the final cell against
  something independent of the reader is the fix, and it is not implemented.
* **§3's base-plus-specialists construction is checked, not assumed.** The
  specialist is mined from one slice and the ensemble is judged on another; if
  it does not beat the base on held-out data, it is discarded and the base is
  left alone. In the robotics run it usually is discarded, which is information.
  Mine from a set the base did NOT train on: run it over its own training data
  and a rule that verifies at 0.61 on fresh scenes can show zero failures, so
  the construction silently does nothing. `grow_ensemble` compares the mined
  failure rate against what verification measured and says which case it hit,
  rather than reporting "no hard cases" either way.
* **§5's algorithm** computes `N = ⌈ln(1/δ)/(2λ²)⌉` and
  `k = ⌈N(1-ε+λ)/(1-2Nσ(1-ε))⌉`, and refuses when σ is not
  `o(min(1/t₍N₎, 1/N²))` rather than returning a threshold whose guarantee does
  not hold. Running time is measured in rule applications, so it calibrates a
  real search budget. N is capped: the formula grows as 1/λ², so λ=1e-4 asks
  for 150 million sampled programs, and an uncapped N turns a tightened
  parameter into a hang. When the cap binds, the calibration reports the λ the
  sample really buys rather than the one requested — capping the cost must not
  inflate the claim. `halting_budget` fits λ to however many proofs exist
  instead of demanding 3745 of them for a fixed λ=0.02, which no library has.
* **Image-to-image generation is not a generative model here.** Figure 3's
  "move the line up / rotate it" is a *sketch-space update*: the decision is
  learned perception, the update is arithmetic on scene parameters, and the
  output is a redrawn cell. The paper allows exactly this ("a generative model
  **or a sketch space updating module**").
* **The controller cannot execute code.** Generators and oracles are a closed
  registry chosen by name. This is a deliberate narrowing: it costs nothing the
  paper's experiments need, and an LLM with an `exec()` would add a failure mode
  the architecture does not call for.
