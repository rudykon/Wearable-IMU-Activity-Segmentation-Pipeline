# Scenarios

<p class="research-lead">A wearable records motion. This project turns it into an activity log.</p>

!!! info "Scope"

    This is a research prototype for **segment-level activity recognition**.
    New devices, users, activities, and high-stakes uses need new validation.

## Example

A six-axis IMU receives no activity commands. It records acceleration, angular
velocity, and time through exercise, pauses, and transitions.

<div class="session-story" aria-label="Illustrative mixed workout timeline">
  <article class="story-moment">
    <time>08:55</time>
    <div><strong>Connected</strong><span>Setup and walking enter the stream.</span></div>
  </article>
  <article class="story-moment">
    <time>09:02</time>
    <div><strong>Badminton</strong><span>Repeated wrist motion becomes recognizable.</span></div>
  </article>
  <article class="story-moment">
    <time>09:17</time>
    <div><strong>Rest</strong><span>The first record must end cleanly.</span></div>
  </article>
  <article class="story-moment">
    <time>09:25</time>
    <div><strong>Jump rope</strong><span>Short and long windows see the motion differently.</span></div>
  </article>
  <article class="story-moment">
    <time>09:34</time>
    <div><strong>Review</strong><span>The output is a short activity list.</span></div>
  </article>
</div>

<p class="story-caption">Fictional times; no participant data.</p>

Output:

~~~text
user_id, category, start, end
~~~

## Window vs record

A good window classifier can still produce a bad activity log:

- one true activity can fragment into several short records;
- two nearby events can merge into one;
- motion-like background can become a false activity;
- a correct class can receive an unusable start or end time; and
- adjacent same-class actions can be difficult to count separately.

<figure class="paper-figure">
  <a class="pipeline-image-link" href="../../assets/manuscript-figures/fig01_window_to_record_gap.png" target="_blank" rel="noopener" aria-label="Open the full-resolution window-to-record gap figure">
    <img src="../../assets/manuscript-figures/fig01_window_to_record_gap.png" alt="Posterior trajectories, naive fragmented activity records, and the stabilized records produced by the Temporal Record Layer" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">Paper Fig. 1. Good local predictions can still produce bad records.</figcaption>
</figure>

## Uses

<div class="scenario-grid detailed">
  <article class="scenario-card">
    <span class="scenario-tag established">Research</span>
    <h3>Long-session evaluation</h3>
    <p>Compare full records with fixed splits, segment F1, overlap, and false alarms.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag scoped">Prototype</span>
    <h3>Workout log</h3>
    <p>Create candidate records for five supported activities. New settings need validation.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag established">Mobile</span>
    <h3>Edge deployment</h3>
    <p>Test BLE capture, ONNX inference, and timelines on Android.</p>
  </article>
  <article class="scenario-card">
    <span class="scenario-tag exploratory">Review</span>
    <h3>Annotation aid</h3>
    <p>Find likely events and boundary errors. A reviewer confirms every label.</p>
  </article>
</div>

## Sensor to phone

<figure class="pipeline-frame">
  <a class="pipeline-image-link" href="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" target="_blank" rel="noopener" aria-label="Open the full-resolution physical deployment chain figure">
    <img src="https://raw.githubusercontent.com/rudykon/Wearable-IMU-Activity-Segmentation-Pipeline/main/experiments/figures/fig03_physical_deployment_chain.png" alt="Physical deployment chain from a wearable IMU through BLE and Android inference to activity records" loading="lazy" decoding="async">
  </a>
  <figcaption class="pipeline-caption">WT9011DCL-BT50 → BLE → Android → activity records.</figcaption>
</figure>

The [browser demo](../deployment/hugging-face.md) runs the 3 s, 5 s, and 8 s
models and returns plots, records, and CSV.

## Evidence

| Question | Current answer |
| --- | --- |
| End-to-end path? | Yes: Python, Gradio, and Android. |
| Segment-level evaluation? | Yes: same-class IoU > 0.5. |
| Fixed external test? | Yes: 37 recordings, 114 segments. |
| Any activity? | No: five evaluated activities. |
| New devices or users? | Not established. |
| Clinical, coaching, or safety use? | No. |

!!! warning "Limits"

    Results measure **activity-record quality** in the studied protocol. They
    do not establish clinical, coaching, safety, or production performance.

## Next

<div class="route-grid compact">
  <a class="route-card" href="../../deployment/hugging-face/">
    <span>Visitor</span>
    <h3>Demo</h3>
    <p>Run one complete example.</p>
  </a>
  <a class="route-card" href="../../guide/pipeline/">
    <span>Researcher</span>
    <h3>Pipeline</h3>
    <p>Trace and reproduce the method.</p>
  </a>
  <a class="route-card" href="../../deployment/android/">
    <span>Engineer</span>
    <h3>Android</h3>
    <p>Build the BLE and ONNX path.</p>
  </a>
</div>
