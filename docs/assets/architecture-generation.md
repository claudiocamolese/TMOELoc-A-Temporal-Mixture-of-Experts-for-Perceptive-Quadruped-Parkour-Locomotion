# Architecture figure provenance

The final asset is `expertparkour-architecture.png`, generated with the built-in `image_gen` tool and reviewed against the source code. No training code was changed to match the illustration.

Reference: Figure 2 in **Extreme Parkour with Legged Robots**, Xuxin Cheng, Kexin Shi, Ananye Agarwal, and Deepak Pathak, [paper](https://extreme-parkour.github.io/resources/parkour.pdf), [project](https://extreme-parkour.github.io/). The teacher–student presentation is the visual reference; the Transformer, MoE, and refinement panels describe this extension. The downloaded reference PDF is not redistributed here.

The figure is a conceptual overview. The README and linked configuration files specify tensor dimensions, clock rates, and evaluation assumptions. The PPO panel describes an experimental stage whose measured outcome was negative.

## Generation prompt

```text
Use case: scientific-educational
Asset type: publication-quality research architecture figure for ExpertParkour GitHub README, landscape about 2400 by 1400.
Primary request: Create an original, rigorous CVPR/IROS-style neural robotics diagram. Reference image is a page from Extreme Parkour; use only Figure 2 upper left as inspiration for teacher above, student below, visual inputs and dashed supervision. Do not copy the figure or paper text. Build the new architecture described here.
White background, crisp dark outlines, precise arrowheads, highly legible sans-serif labels, ample whitespace, restrained navy, teal, amber and muted violet. Tiny scientific depth-map and quadruped-on-steps illustrations, not photorealistic art. No conference logos, no invented performance claims.
Layout:
Top thin band headed "(a) Privileged teacher — RL pretraining". Inputs "Scandots", "Oracle heading", "Proprioception + privileged state" flow to "Teacher policy" then "Teacher actions". Teacher is frozen during student training. Dashed amber supervision arrow from teacher actions to student action output, labeled "Action imitation". Keep teacher narrow so student dominates.
Main large band headed "(b) Temporal Transformer + Mixture of Experts". Left: "Depth image · 10 Hz" small purple depth image -> "Depth CNN" -> "32-d". Separate "Proprioception · 50 Hz" -> "5-sample window" -> "MLP" -> "64-d". Two branches merge at "Token projection · 128-d". Next show eight little token tiles -> "Causal Transformer" with subtitle "2 layers · 4 heads · 8 tokens" and "Pre-LN · width 128"; output "Context c_t · 128-d". Label temporal strip "10 Hz token history". Below context, small auxiliary head box with "Terrain / Dynamics / Velocity / Heading" and dashed amber "Auxiliary supervision", not on mandatory action path. Heading prediction connects with a thin teal arrow to a small "Current proprioception + predicted heading" input in the actor band.
Right of context, merge c_t with "Current proprioception" and separate "Velocity*" into a round "Concatenate" port. Then split SAME fused input to both "Router MLP" (top) and four parallel expert blocks (below), labelled "Expert 0", "Expert 1", "Expert 2", "Expert 3". Above expert blocks router outputs four bars labelled "Top-2 / Softmax"; two active colored bars and two faint for top-2. Router connects via thin weighted gate lines to a "Weighted sum" node; each expert output also flows to that same node. All expert blocks subtitle collectively "MLPs · 256 → 128 → 12". Final output "12 joint targets · 50 Hz" followed by tasteful line drawing of Go2 quadruped stepping onto blocks. Add succinct note "4 experts evaluated; gates weight their outputs".
Bottom slim band headed "(c) Optional PPO refinement". "Frozen depth CNN + Transformer" -> "Train router + experts" and separate dashed box "Privileged critic (training only)" with reward arrow. Small neutral text "Experimental stage; no improvement established". Do not suggest PPO is a proven gain.
Footer small notes, legible: "* Recorded experiments use simulator velocity; predicted velocity is configurable." and "Solid: inference flow     Dashed: training supervision". Another short note "Default: context and gates at 10 Hz; actor at 50 Hz. Dense and control-rate variants supported."
Constraints: arrows must have the exact connections described. Auxiliary heads are not experts. No GRU in student. No hard terrain-to-expert labels. No actor deepcopy arrow. English labels only. Keep design quiet, polished and technically readable.
```

## Revision prompt

```text
Use case: scientific-educational. Edit the supplied ExpertParkour architecture figure. Preserve the polished publication style, three bands, whitespace, main topology, and illustrations. Correct technical labels and arrows precisely:
1. The four expert MLPs are PARALLEL. Remove EVERY horizontal arrow between adjacent experts. Each expert receives the same concatenated input from a shared input bus along its top; each expert sends a distinct output line down to a separate output collector bus, which feeds Weighted sum. Router gate dashed lines point to Weighted sum, not the expert input bus. The concatenated input must feed Router MLP AND the shared expert input bus.
2. Change "Auxiliary prediction heads (training only)" to "Context prediction heads". Change "Terrain type" to "Terrain latent" (not classification). Keep Dynamics, Velocity, Heading. On the three non-heading tiles label small "supervision"; Heading is active during inference.
3. The predicted Heading arrow MUST lead up to Current proprioception input (which has heading substituted). It MUST NOT connect directly to expert 0 or bypass the concatenate port.
4. Change velocity input box content from vx vy vz to "9-d explicit state" and keep simulator-velocity footnote.
5. Amber dashed Action imitation arrow from Teacher actions should end at the student "12 joint targets" output (a supervision connection spanning the right side), not in empty space or at student panel border.
6. The causal Transformer output Context c_t must clearly connect into Concatenate. The Transformer also has a SOLID feature arrow to context prediction heads; use amber dashed auxiliary supervision only as a training target mark into heads (not as an inference arrow out of Transformer).
7. In bottom band change "Train router + experts (via RL, e.g., PPO)" to "Train router + experts (PPO)". Replace arrow labelled reward from critic with "Value / advantages"; show small "Task reward" input separately if room, otherwise leave just Value / advantages. Preserve experimental no-improvement note.
No other changes. Keep every label legible. Avoid serial expert arrows. Avoid gates mixed into expert inputs. All four experts consume u_t independently, outputs and gate weights meet only at Weighted sum.
```

## Final label corrections

```text
Edit this scientific figure with ONLY these two text corrections. Preserve all geometry, art, colors, lines, arrows, panels and other labels exactly. 1. In the main student panel, the dashed prediction-head box currently says 'Context prediction heads (training only)'. Replace that ENTIRE heading with exactly 'Context prediction heads'. Do NOT retain '(training only)' because Heading is used at inference. 2. The top middle student input box currently says 'Current proprioception · 50 Hz'. Replace this heading with exactly 'Proprioception + velocity* · 50 Hz'. Keep '9-d explicit state' in that box. No other changes. English text must be exact and readable.
```

## Rollout GIFs and thumbnails

`rollout-env{1,2,3,4,6}.gif` are FFmpeg conversions of the complete corresponding committed `thirdperson_env*.mp4` files: 640 pixels wide, 12.5 fps, 128-color palettes, infinite looping. The README displays these animations inline and links to the original MP4 recordings.

The retained `rollout-env{1,2,3,4,6}.jpg` files are frame extractions at 3 seconds. Both formats come from recorded experiments, not generated locomotion imagery.
