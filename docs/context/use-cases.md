# Background & use cases

<p class="research-lead">This project begins with a practical mismatch: a wearable records dense motion signals, while people make decisions from sparse events. A useful system must bridge those two views without hiding where one activity record begins, ends, or fails.</p>

!!! info "Intended scope"

    The repository is a research and teaching prototype for **segment-level
    activity recognition**. Its strongest fit is reproducible long-session
    evaluation and end-to-end deployment experiments. Any new device,
    population, sport protocol, coaching workflow, or clinical setting needs
    its own validation.

## An illustrative session

Imagine a participant wearing a six-axis IMU for one mixed workout. The device
does not receive a “start badminton” command. It only receives acceleration,
angular velocity, and timestamps while the person prepares, exercises, pauses,
changes activity, and walks away.

<div class="session-story" aria-label="Illustrative mixed workout timeline">
  <article class="story-moment">
    <time>08:55</time>
    <div><strong>Sensor connected</strong><span>Background movement, equipment setup, and walking all enter the stream.</span></div>
  </article>
  <article class="story-moment">
    <time>09:02</time>
    <div><strong>Badminton begins</strong><span>Repetitive wrist motion becomes locally recognizable, but brief pauses can interrupt confidence.</span></div>
  </article>
  <article class="story-moment">
    <time>09:17</time>
    <div><strong>Transition and rest</strong><span>The system must close one record without mistaking nearby motion for another event.</span></div>
  </article>
  <article class="story-moment">
    <time>09:25</time>
    <div><strong>Jump rope begins</strong><span>Short and long windows provide different views of the same repeated action.</span></div>
  </article>
  <article class="story-moment">
    <time>09:34</time>
    <div><strong>Session reviewed</strong><span>The desired result is a short record list with labels, starts, ends, durations, and visible evidence.</span></div>
  </article>
</div>

<p class="story-caption">This timeline is fictional and contains no participant data. It illustrates why continuous sensing is a segmentation problem rather than a collection of independent clip classifications.</p>

The output contract is deliberately small:

~~~text
user_id, category, start, end
~~~

Yet producing those four fields reliably requires the full chain: physical-unit
signal handling, multi-scale window evidence, temporal consistency, boundary
refinement, false-positive control, and segment-level evaluation.

## Why ordinary window accuracy is not enough

A window classifier answers “what does this short slice resemble?” A session
record answers a different question: “how many activities happened, what were
they, and where were their boundaries?” Converting the first answer into the
second introduces failure modes that window accuracy does not measure:

- one true activity can fragment into several short records;
- two nearby events can merge into one;
- motion-like background can become a false activity;
- a correct class can receive an unusable start or end time; and
- adjacent same-class actions can be difficult to count separately.

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="Open the full-resolution window-to-record gap figure">
    <img src="../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="Posterior trajectories, naive fragmented activity records, and the stabilized records produced by the Temporal Record Layer" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 1 makes the central gap visible: plausible local evidence can still yield the wrong record list.</figcaption>
</figure>

## Four concrete use scenarios

<div class="scenario-grid detailed">
  <article class="scenario-card">
    <span class="scenario-tag established">Primary research use</span>
    <h3>Long-session HAR evaluation</h3>
    <p><strong>Situation.</strong> A researcher has continuous wearable recordings and needs to compare complete segmentation systems.</p>
    <p><strong>How this helps.</strong> The repository provides user-level splits, multi-scale inference, one-to-one same-class IoU matching, segment F1, matched IoU, and false-positive analysis.</p>
    <p><strong>Decision supported.</strong> Whether a method creates better activity records—not merely better window labels.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">Controlled prototype</span>
    <h3>Automatic workout diary</h3>
    <p><strong>Situation.</strong> A controlled session contains badminton, rope skipping, dumbbell fly, running, or table tennis.</p>
    <p><strong>How this helps.</strong> The output can become a candidate activity log with event count, timing, and duration for later review.</p>
    <p><strong>Validation needed.</strong> New users, devices, sensor positions, and protocols must be tested before the log is treated as reliable.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">Implemented system path</span>
    <h3>Edge and mobile deployment research</h3>
    <p><strong>Situation.</strong> An engineer wants to test whether a research pipeline survives the move from saved files to a physical sensor and phone.</p>
    <p><strong>How this helps.</strong> The Android module covers BLE acquisition, signal views, CSV recording, offline recognition, and selected ONNX models.</p>
    <p><strong>Decision supported.</strong> Whether the data contract, model assets, and temporal logic remain coherent end to end.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">Candidate workflow</span>
    <h3>Human-assisted annotation or QA</h3>
    <p><strong>Situation.</strong> A reviewer needs to inspect many hours of continuous motion and find likely foreground intervals.</p>
    <p><strong>How this helps.</strong> Candidate records and probability plots can direct attention to boundaries, splits, merges, and false alarms.</p>
    <p><strong>Human role.</strong> A reviewer must confirm or correct every consequential label; this use was not the paper’s primary evaluated endpoint.</p>
  </article>
</div>

## Physical path: from a wrist to a record

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="Open the full-resolution physical deployment chain figure">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="Physical deployment chain from a wearable IMU through BLE and Android inference to activity records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Repository figure: WT9011DCL-BT50 wearable IMU → BLE acquisition → Android signal processing → on-device multi-scale inference → activity recognition.</figcaption>
</figure>

The browser demo provides a second entry point. It runs the tracked 3 s, 5 s,
and 8 s models on a synthetic or uploaded compatible session, then shows the
six signals, class probabilities, decoded timeline, final records, and a CSV
download. It is the fastest way to understand the output before reading the
implementation.

## What the current evidence supports

| Question | Current answer |
| --- | --- |
| Does the repository implement a complete sensing-to-record path? | Yes: Python, public Gradio, and Android paths are documented and tested. |
| Are records evaluated as segments rather than independent windows? | Yes: same-class one-to-one matching at IoU > 0.5. |
| Is there a fixed external test? | Yes: 37 long recordings with 114 labeled foreground segments. |
| Does the final reported system cover arbitrary activities? | No: the evaluated foreground vocabulary contains five activities. |
| Is cross-device or cross-population generalization established? | No. |
| Is this a clinical, coaching, or safety product? | No. |

!!! warning "Do not silently broaden the claim"

    The existing results measure **activity-record quality** under the studied
    protocol. They do not establish clinical outcomes, coaching correctness,
    injury prevention, safety monitoring, or production reliability. A useful
    prototype can motivate those studies, but it cannot replace them.

## Follow the route that matches your role

<div class="route-grid compact">
  <a class="route-card" href="../../deployment/hugging-face/">
    <span>Visitor or reviewer</span>
    <h3>See one complete run</h3>
    <p>Start with the live demo, then inspect the fixed paper evidence and its limitations.</p>
  </a>
  <a class="route-card" href="../../guide/pipeline/">
    <span>HAR researcher</span>
    <h3>Trace and reproduce the method</h3>
    <p>Read the architecture, data contract, training, inference, and segment evaluator in order.</p>
  </a>
  <a class="route-card" href="../../deployment/android/">
    <span>Mobile or edge engineer</span>
    <h3>Follow the physical deployment path</h3>
    <p>Review the sensor assumptions, ONNX assets, Android build, and BLE runtime chain.</p>
  </a>
</div>

<div class="cta-panel">
  <div>
    <h3>Now that the setting is clear, see how the records are built.</h3>
    <p>The architecture page moves from channel order to multi-scale evidence, LBSA, TRL, and final output.</p>
  </div>
  <a class="md-button md-button--primary" href="../../guide/pipeline/">Continue to the architecture</a>
</div>
