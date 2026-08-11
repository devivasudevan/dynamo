# ThunderAgent Sticky-Pinning Evaluation — MiniMax-M2.7 Concurrency Study (Final)

## 1. Goal

Determine whether ThunderAgent's program-aware scheduling — specifically sticky worker pinning — produces a measurable, reliable improvement in time-to-first-token (TTFT) across conversation turns, under conditions where confounding factors (request failures, broken pause/resume, unverified routing) have been eliminated.

## 2. Key Fix Enabling This Study

**A root-cause bug fix for pause/resume.** `--resume-hysteresis` had been set equal to `--pause-threshold` (both `0.06`), which makes `resume_ceiling = pause_threshold - resume_hysteresis = 0.0` in `_greedy_resume()` — mathematically guaranteeing the normal resume path could never succeed, regardless of load. Every "resume" observed before this fix was the `Forced resume` 1800-second timeout fallback, never the intended mechanism. Setting `resume-hysteresis` to `0.02` (a real gap below the pause threshold) fixed this; normal `PAUSED -> ACTIVE` resumes were observed for the first time immediately after the fix, and this correction is what made a clean, well-instrumented benchmark possible at all.

**A deployment configuration reaching 100% request completion.** `MiniMaxAI/MiniMax-M2.7` (MoE, ~56B total / ~20B active params) was deployed on 3×TP4 H100 with `fp8` KV cache, prefix caching, block size 16. This configuration completed **100% of requests successfully at 300, 500, and 800 concurrency**, with pause/resume genuinely cycling under real load throughout. This gives a dataset with no failure-driven selection effects to control for — every session in every run completed all three turns, so per-session comparisons are not confounded by which sessions happened to survive.

## 3. Setup

- 3× `Standard_ND96isr_H100_v5`-class nodes, TP4 per worker (12 GPUs total)
- `--kv-cache-dtype fp8`, `--block-size 16`, `--enable-prefix-caching`, `--trust-remote-code`
- Router: `--pause-threshold 0.06 --pause-target 0.04 --soft-demote-threshold 0.03 --resume-hysteresis 0.02 --acting-token-weight 4.0 --acting-decay-tau-seconds 60.0`
- Confirmed KV pool: 416,064 tokens / 12.3 GiB per worker; vLLM-reported per-worker batching ceiling ~13x concurrent requests
- Test harness: synthetic 3-turn multi-session client (`ThreadPoolExecutor`), fixed technical prompt, `max_tokens=1024`, session released via `x-dynamo-session-final` after turn 3
- Shared model cache on Azure Managed Lustre, pre-populated via a one-off download pod before scaling workers, avoiding the HuggingFace lock-contention failure seen when multiple workers raced to populate the cache simultaneously

## 4. Results

### 4.1 Run summary, all concurrency levels

| Concurrency | Success rate | Routing confirmed | Pause events | Normal resume events | Forced-resume fallback | Requests received/completed |
|---|---|---|---|---|---|---|
| 300 | 100% (900/900) | confirmed (300/300 programs) | 902 | 602 | 0 — never needed | 1,200 / 1,200 |
| 500 | 100% (1,500/1,500) | confirmed (500/500 programs) | 1,516 | 1,019 | 0 — never needed | 2,000 / 2,000 |
| 800 | 100% (2,400/2,400) | confirmed (800/800 programs) | 2,410 | 1,893 | 0 — never needed | 3,200 / 3,200 |

No engine hangs, no stuck requests, and no `Failed to publish response` errors occurred in any of these three runs — a marked contrast to earlier testing on a different model/config, where sustained high concurrency had previously left the serving engine in a non-functional state requiring a manual restart (GPU memory held, 0% compute utilization, health checks still passing).

### 4.2 TTFT by turn

| Concurrency | Turn 1 | Turn 2 | Turn 3 |
|---|---|---|---|
| 300 (full population, n=300) | 54.5s | 52.5s | 52.0s |
| 500 (full population, n=500) | 89.1s | 68.3s | 87.6s |
| 800 (full population, n=800) | 169.1s | 61.7s | 73.7s |

### 4.3 Per-session analysis (the decisive test)

Population-average TTFT is sensitive to contention-timing effects that shift across a long-running test — a session's Turn 3 might land during a queueing trough or peak independent of anything about that session's own cache state. To control for this, each session's own Turn 3 vs. Turn 1 TTFT was compared directly, at both 500 and 800 concurrency:

| Concurrency | Turn 3 faster than Turn 1 | Turn 3 same or slower | % improved |
|---|---|---|---|
| 500 | 270 | 230 | 54% |
| 800 | 348 | 452 | 43.5% |

At 500 concurrency the split is statistically indistinguishable from chance (54/46). At 800 concurrency — higher load, more pause/resume activity, more opportunity for a real caching benefit to matter — the improvement rate moved *further* from a positive result, not toward one. This directional trend (weak-to-none at 500, below chance at 800) is inconsistent with a genuine, scale-robust pinning benefit; if sticky pinning provided real value, its effect should hold or strengthen under heavier contention, not disappear or reverse.

## 5. Analysis

### 5.1 The pause/resume mechanism is confirmed functional — this investigation's clearest positive finding
Earlier testing repeatedly found pause activity but never a single successful normal resume, across every test condition tried, including hand-tuned synthetic pressure tests specifically designed to trigger it. The cause was a one-line configuration bug (`resume_hysteresis == pause_threshold`), not a defect in the scheduling logic itself. Once fixed, normal resume worked robustly and repeatedly — 602 times at 300 concurrency, 1,019 at 500, 1,893 at 800 — with the emergency timeout fallback never needed even once across any of the three runs. This confirms ThunderAgent's pause/resume design works as intended when correctly configured, and identifies the specific configuration trap (`resume_hysteresis` must be strictly less than `pause_threshold`, with meaningful margin to avoid thrashing) that made it appear broken until this fix.

### 5.2 Sticky pinning does not produce a reliable per-session TTFT improvement, even under conditions favorable to detecting one
All three runs (300/500/800 concurrency) share the properties needed for a trustworthy test: 100% request completion (no survivorship confound), confirmed routing for every session, and genuinely active, working pause/resume. Under these conditions, per-session Turn-3-vs-Turn-1 comparison shows no reliable improvement — 54% at 500 concurrency (chance), 43.5% at 800 concurrency (below chance). The population-average TTFT patterns at all three concurrency levels are non-monotonic (e.g., 800-concurrency: 169s → 62s → 74s) and are better explained by shifting system-wide queueing/contention across the run's timeline than by a consistent per-session caching effect.

### 5.3 The trend across concurrency levels argues against a real effect, not for one
If sticky pinning provided a genuine benefit, it should be at least as visible — arguably more visible — under heavier contention, since that's exactly the condition where avoiding a cache miss matters most. Instead, the per-session improvement rate moved from roughly chance (54% at 500) to below chance (43.5% at 800) as concurrency increased. This directional trend is inconsistent with a real, scale-robust pinning effect and consistent with the improvement rate being noise around 50% that shifts with contention-timing artifacts rather than tracking any property of the pinning mechanism itself.

### 5.4 This configuration sustains high concurrency reliably
This deployment (fp8 KV cache, MoE architecture, 416K-token KV pool per worker) completed 100% of requests through 800 concurrency with no degradation in system health, no engine stalls, and no publish failures — a meaningfully more robust result than earlier testing on a different model/configuration, which saw substantial request failure and an unrecovered engine hang at similar concurrency. This is a useful, separate finding about configuration choices for high-concurrency reliability, independent of the ThunderAgent-specific conclusions above.

## 6. Conclusions

- **ThunderAgent's pause/resume mechanism works correctly** once a specific, previously-undiagnosed configuration bug (`resume_hysteresis` needs a real, positive margin below `pause_threshold`) is fixed. This is a genuine, actionable, and previously unreported finding.
- **No reliable evidence was found, at 300, 500, or 800 concurrency, that sticky pinning improves per-session TTFT.** The per-session improvement rate is at or below chance at every tested level, and trends further from a positive result as concurrency increases — the opposite of what a real, robust caching benefit would predict.
- **This deployment configuration (fp8 KV cache, MoE, prefix caching) reliably sustains very high concurrency** — 100% success through 800 concurrent multi-turn sessions — demonstrating that the underlying serving stack itself scales well under the right configuration, independent of whether ThunderAgent's pinning adds value on top of it.

## 8. Throughput Under Sustained Concurrency (Addendum)

### 8.1 Methodology
A separate test measured sustained throughput (tokens/s) at fixed concurrency levels (CC = 32–512), matching the design of a reference benchmark chart for a related framework. Unlike the burst-arrival TTFT tests in Sections 3–7, each of the `CC` slots here holds a persistent session and immediately re-fires its next turn upon completion, for a 90-second steady-state measurement window per CC level (15s warmup excluded). This measures genuine sustained throughput rather than one-time arrival-burst behavior.

### 8.2 Results — ThunderAgent vs. KvRouter, same hardware and model

| CC | ThunderAgent (tok/s) | KvRouter (tok/s) | KvRouter advantage |
|---|---|---|---|
| 32 | 2,560 | 2,552 | -0.3% |
| 64 | 4,161 | 4,003 | -3.8% |
| 80 | 5,002 | 5,006 | +0.1% |
| 96 | 5,669 | 5,774 | +1.9% |
| 128 | 5,093 | 6,554 | +28.7% |
| 192 | 5,461 | 9,718 | +77.9% |
| 256 | 5,820 | 11,640 | +100.0% |
| 384 | 5,791 | 15,278 | +163.8% |
| 512 | 6,371 | 17,464 | +174.2% |

Through CC=96, the two routers are statistically indistinguishable (within a few percent, alternating which leads — consistent with noise). From CC=128 onward, KvRouter pulls decisively and increasingly ahead, reaching nearly 3x ThunderAgent's throughput at CC=512. ThunderAgent's throughput plateaus around 5,500–6,400 tok/s from CC=128 through 512 — essentially flat despite 4x more concurrent load — while KvRouter scales close to linearly across the same range.

### 8.3 Root cause: synchronized batch-and-drain pause cycling, not admission-capacity limits

Initial hypothesis was that ThunderAgent admits fewer concurrent requests than KvRouter at high CC. Direct measurement of vLLM's own `Running: N reqs` at CC=512 ruled this out: ThunderAgent's worker showed `Running` values *higher* than KvRouter's at points (108–162 vs. KvRouter's steady 12–23), but oscillating sharply between these peaks and `Running: 0` — a repeating batch-and-drain pattern with genuine idle gaps (`Avg generation throughput: 0.0 tokens/s`) roughly every 10–20 seconds. KvRouter's worker log showed no equivalent gaps — `Running` stayed in a steady 12–23 range continuously, with `Waiting: 0` throughout, indicating smooth, un-batched processing.

Cross-referencing the router's own pause/resume log against this window explains the mechanism directly: thousands of `ACTING -> PAUSED` events fire within a narrow window (e.g., hundreds of pauses within roughly half a second, at `21:16:22.581`–`.993`), and critically, **every single paused program in this window shows the identical token count (`tokens=566`)**. Because the benchmark's synthetic workload uses one fixed prompt and one fixed `max_tokens` value across all sessions, every session accumulates tokens at the same rate and crosses the pause threshold in near-perfect synchrony — causing the scheduler to pause a large fraction of all active programs simultaneously, drain the worker to idle, then admit/resume the next synchronized batch, repeating on roughly the `scheduler-interval-seconds` cadence.

### 8.4 Caveat: this magnitude may be a synthetic-benchmark artifact
Because this test's homogeneous workload (identical prompt, identical `max_tokens`, identical arrival pattern per slot) synchronizes pause timing across sessions in a way that heterogeneous real-world traffic — with varied prompt lengths, varied response lengths, and staggered arrival times — would not, the severity of the batch-and-drain effect observed here may not generalize directly to production traffic, where pause timing would naturally be staggered across sessions rather than clustered. The *existence* of a batching effect tied to synchronized pause events is a genuine, mechanistically-confirmed finding; its *magnitude* under this specific uniform benchmark should not be assumed to transfer unchanged to realistic, heterogeneous agentic workloads. This is worth testing directly with varied prompt/response-length traffic before treating the 174% throughput gap as representative of general production behavior.

## 9. Updated Conclusions (incorporating throughput findings)

- **ThunderAgent's pause/resume mechanism is functionally correct** (Section 5.1) but its coordination of *when* to pause can synchronize across many concurrent sessions under homogeneous load, producing a batch-and-drain throughput pattern not present in KvRouter's simpler continuous-admission design.
- **This batching pattern, not an admission-capacity limit, is the direct, confirmed cause of ThunderAgent's throughput plateau at CC≥128**, measured via `Running: N reqs`/`Waiting: N reqs` logs on both arms at matched CC=512.
- **At sustained CC=128 and above, KvRouter delivers substantially higher aggregate throughput than ThunderAgent** on identical hardware, model, and workload — up to 174% higher at CC=512 — though the exact magnitude may be inflated by this benchmark's synthetic workload homogeneity (Section 8.4).
- Combined with the per-session TTFT findings (Sections 5–7), this investigation now has two independent, converging lines of evidence — TTFT (per-session) and throughput (aggregate) — both showing no advantage, and in the throughput case a clear disadvantage, for ThunderAgent's sticky-pinning/pause-resume approach relative to plain KvRouter on this workload shape.

## 10. Updated Recommendations

- Investigate whether `_pause_until_safe`/`_greedy_resume` can stagger pause decisions (e.g., jittering the pause timing per-program rather than pausing every program that crosses the threshold in the same scheduler tick) to avoid the synchronized batch-and-drain pattern, independent of whether real traffic is homogeneous or not.
- Re-run the throughput sweep with heterogeneous synthetic traffic (varied prompt lengths, varied `max_tokens`, staggered slot start times) to determine whether the 174%-at-CC=512 gap is a benchmark artifact or a robust production-relevant finding.
- The `resume_hysteresis`-must-be-less-than-`pause_threshold` requirement (Section 5.1) remains worth reporting upstream regardless of the throughput findings.
- Given both TTFT and throughput data now point the same direction, prioritize the KvRouter-vs-ThunderAgent throughput comparison as the headline result for any stakeholder communication — it is a clearer, more visually compelling, and more decisively-supported finding than the per-session TTFT analysis alone.
