/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"context"
	"encoding/json"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui/history"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

// The planner's live retune (components/src/dynamo/planner/retune/live) replays each
// window of recorded traffic through DGDR v1beta2 Requests and keeps its decisions in a
// Lease. This file reads that contract: the Request labels, the state Lease annotations
// and the router adapter parameters of each Candidate. It never writes any of them.

var gvkLease = schema.GroupVersionKind{Group: "coordination.k8s.io", Version: "v1", Kind: "Lease"}

const (
	labelRetuneDGD     = "nvidia.com/planner-retune-dgd"
	labelRetuneDGDUID  = "nvidia.com/planner-retune-dgd-uid"
	labelRetuneWindow  = "nvidia.com/planner-retune-window"
	labelRetuneShard   = "nvidia.com/planner-retune-shard"
	labelRetuneAttempt = "nvidia.com/planner-retune-attempt"

	annotationRetuneState   = "nvidia.com/planner-retune-state"
	annotationRetuneHistory = "nvidia.com/planner-retune-history"
	retuneStateLeaseSuffix  = "-planner-retune-state"

	// shardProbe replays one grid point a second time as a determinism check; its point
	// is already in a grid shard. shardBaseline holds the deployed point when it is off
	// the search grid.
	shardProbe    = "probe"
	shardBaseline = "point-baseline"

	routerAdapter = "dynamo.router"
	plannerSweep  = "retune_sweep"

	// The planner's decision outcomes that the views tell apart. Only applied and
	// advisory decisions choose a point other than the deployed one.
	outcomeApplied    = "applied"
	outcomeAdvisory   = "advisory"
	outcomeRolledBack = "rolled_back"
)

// routerKnobs maps the router adapter's parameter names to the live router policy names
// the planner's state uses.
var routerKnobs = map[string]string{
	"overlap_score_credit": "overlap_score_credit",
	"prefill_load_scale":   "prefill_load_scale",
	"temperature":          "router_temperature",
}

// minimizedObjectives are the spec.objective.metric values where a lower score is better.
var minimizedObjectives = map[string]bool{"TTFT": true, "E2ELatency": true}

// retuneEntry is everything known about one DGD's live retune. state is nil when its
// Lease is gone or holds no snapshot; decisions outlive both.
type retuneEntry struct {
	namespace string
	dgd       string
	uid       string
	state     map[string]any
	decisions map[string]map[string]any // by decision_id
}

// retuneEntry returns the entry of a DGD UID, creating it. The caller holds ix.mu.
func (ix *Index) retuneEntry(namespace, dgd, uid string) *retuneEntry {
	entry, ok := ix.retunes[uid]
	if !ok {
		entry = &retuneEntry{namespace: namespace, dgd: dgd, uid: uid, decisions: map[string]map[string]any{}}
		ix.retunes[uid] = entry
	}
	return entry
}

// retuneRefOf reads the planner retune labels of a Request. It returns nil for a Request
// the planner's retune did not create.
func retuneRefOf(request map[string]any) *history.RetuneRef {
	labels := nestedMap(request, "metadata", "labels")
	uid := str(labels, labelRetuneDGDUID)
	window, err := strconv.ParseInt(str(labels, labelRetuneWindow), 10, 64)
	if uid == "" || err != nil {
		return nil
	}
	attempt, _ := strconv.Atoi(str(labels, labelRetuneAttempt))
	return &history.RetuneRef{
		DGD: str(labels, labelRetuneDGD), DGDUID: uid, WindowEndMs: window,
		Shard: str(labels, labelRetuneShard), Attempt: attempt,
	}
}

// --- writes ---

func (ix *Index) UpsertDGD(obj map[string]any) {
	key := types.NamespacedName{Namespace: str(obj, "metadata", "namespace"), Name: str(obj, "metadata", "name")}
	ix.mu.Lock()
	ix.dgds[key] = stripVolatile(obj)
	ix.mu.Unlock()
}

func (ix *Index) DeleteDGD(namespace, name string) {
	ix.mu.Lock()
	delete(ix.dgds, types.NamespacedName{Namespace: namespace, Name: name})
	ix.mu.Unlock()
}

// UpsertRetuneLease records the snapshot of a planner retune state Lease and stores the
// history entries it has not stored yet. Other Leases are ignored.
func (ix *Index) UpsertRetuneLease(ctx context.Context, obj map[string]any) error {
	name := str(obj, "metadata", "name")
	uid := str(obj, "metadata", "labels", labelRetuneDGDUID)
	if uid == "" || !strings.HasSuffix(name, retuneStateLeaseSuffix) {
		return nil
	}
	namespace := str(obj, "metadata", "namespace")
	dgd := strings.TrimSuffix(name, retuneStateLeaseSuffix)
	annotations := nestedMap(obj, "metadata", "annotations")

	// An unreadable annotation leaves the previous snapshot in place, as the planner does.
	var state map[string]any
	if raw := str(annotations, annotationRetuneState); raw != "" {
		if err := json.Unmarshal([]byte(raw), &state); err != nil {
			state = nil
		}
	}
	var entries []map[string]any
	if raw := str(annotations, annotationRetuneHistory); raw != "" {
		if err := json.Unmarshal([]byte(raw), &entries); err != nil {
			entries = nil
		}
	}

	ix.mu.Lock()
	entry := ix.retuneEntry(namespace, dgd, uid)
	entry.namespace, entry.dgd = namespace, dgd
	if state != nil {
		entry.state = state
	}
	var fresh []history.DecisionEnvelope
	for _, item := range entries {
		id := str(item, "decision_id")
		if id == "" || strings.ContainsAny(id, "/\\") {
			continue
		}
		if _, known := entry.decisions[id]; known {
			continue
		}
		entry.decisions[id] = item
		fresh = append(fresh, history.DecisionEnvelope{Namespace: namespace, DGD: dgd, DGDUID: uid, ID: id, Entry: item})
	}
	ix.mu.Unlock()

	if ix.store == nil {
		return nil
	}
	for _, env := range fresh {
		if err := ix.store.PutDecision(ctx, env); err != nil {
			return err
		}
	}
	return nil
}

// DeleteRetuneLease forgets the live snapshot of a DGD; its stored decisions stay.
func (ix *Index) DeleteRetuneLease(obj map[string]any) {
	uid := str(obj, "metadata", "labels", labelRetuneDGDUID)
	ix.mu.Lock()
	if entry, ok := ix.retunes[uid]; ok && strings.HasSuffix(str(obj, "metadata", "name"), retuneStateLeaseSuffix) {
		entry.state = nil
	}
	ix.mu.Unlock()
}

// --- views ---

// ComponentView is one spec.components entry of a DGD.
type ComponentView struct {
	Name     string `json:"name"`
	Type     string `json:"type,omitempty"`
	Replicas *int64 `json:"replicas,omitempty"`
}

// DecisionView is one planner retune decision. Baseline and Best are the planner's scores
// of the deployed point and of the best evaluated point of the window, in the units of
// the objective. HeadroomPct is how much better Best is than Baseline, in percent.
type DecisionView struct {
	ID            string             `json:"id"`
	WindowEndMs   int64              `json:"windowEndMs,omitempty"`
	Outcome       string             `json:"outcome"`
	Reason        string             `json:"reason,omitempty"`
	FailureReason string             `json:"failureReason,omitempty"`
	Baseline      *float64           `json:"baseline,omitempty"`
	Best          *float64           `json:"best,omitempty"`
	HeadroomPct   *float64           `json:"headroomPct,omitempty"`
	Chosen        map[string]float64 `json:"chosen,omitempty"`
	Planner       map[string]any     `json:"planner,omitempty"`
	AtMs          int64              `json:"atMs,omitempty"`
}

// RetuneSummary is the retune state of one DGD.
type RetuneSummary struct {
	Mode            string             `json:"mode,omitempty"`
	Objective       string             `json:"objective,omitempty"`
	Maximize        bool               `json:"maximize"`
	Current         map[string]float64 `json:"current,omitempty"`
	UpdatedMs       int64              `json:"updatedMs,omitempty"`
	LastWindowEndMs int64              `json:"lastWindowEndMs,omitempty"`
	Windows         int                `json:"windows"`
	Latest          *DecisionView      `json:"latest,omitempty"`
}

// DeploymentView is one DynamoGraphDeployment, live or known only from retune history.
type DeploymentView struct {
	Namespace  string          `json:"namespace"`
	Name       string          `json:"name"`
	UID        string          `json:"uid,omitempty"`
	Live       bool            `json:"live"`
	State      string          `json:"state,omitempty"`
	Created    string          `json:"created,omitempty"`
	Components []ComponentView `json:"components,omitempty"`
	Retune     *RetuneSummary  `json:"retune,omitempty"`
}

// WindowRunView is one shard Run of a retune window.
type WindowRunView struct {
	Name     string `json:"name"`
	Shard    string `json:"shard"`
	Attempt  int    `json:"attempt,omitempty"`
	Phase    string `json:"phase"`
	Retained bool   `json:"retained"`
}

// WindowView is one retune window: its decision, when the planner recorded one, and the
// Runs that replayed it, when the UI observed them.
type WindowView struct {
	WindowEndMs int64           `json:"windowEndMs"`
	Decision    *DecisionView   `json:"decision,omitempty"`
	Runs        []WindowRunView `json:"runs,omitempty"`
}

// DeploymentDetail adds the windows, newest first, and the decisions that belong to no
// window (promotions and rollbacks).
type DeploymentDetail struct {
	DeploymentView
	Windows []WindowView   `json:"windows"`
	Events  []DecisionView `json:"events,omitempty"`
}

// PointView is one evaluated point of a window. Deployed marks the router point that
// served the window's traffic, Chosen the one the decision applied or recommended. With a
// planner search several points share a router point, and each of them is marked.
type PointView struct {
	Candidate string             `json:"candidate"`
	Namespace string             `json:"namespace"`
	RunName   string             `json:"runName,omitempty"`
	Shard     string             `json:"shard,omitempty"`
	Router    map[string]float64 `json:"router,omitempty"`
	Planner   map[string]any     `json:"planner,omitempty"`
	Score     *float64           `json:"score,omitempty"`
	Metrics   map[string]float64 `json:"metrics,omitempty"`
	Deployed  bool               `json:"deployed"`
	Chosen    bool               `json:"chosen"`
}

// WindowDetail adds every evaluated point of the window.
type WindowDetail struct {
	WindowView
	Namespace  string             `json:"namespace"`
	Deployment string             `json:"deployment"`
	Objective  string             `json:"objective,omitempty"`
	Maximize   bool               `json:"maximize"`
	Deployed   map[string]float64 `json:"deployed,omitempty"`
	Points     []PointView        `json:"points"`
}

func floatMap(obj map[string]any, fields ...string) map[string]float64 {
	raw := nestedMap(obj, fields...)
	if raw == nil {
		return nil
	}
	out := make(map[string]float64, len(raw))
	for key, value := range raw {
		if f, ok := toFloat(value); ok {
			out[key] = f
		}
	}
	return out
}

func optionalNumber(obj map[string]any, fields ...string) *float64 {
	if f, ok := number(obj, fields...); ok && !math.IsNaN(f) && !math.IsInf(f, 0) {
		return &f
	}
	return nil
}

func decisionView(id string, entry map[string]any, maximize bool) DecisionView {
	v := DecisionView{
		ID:            id,
		WindowEndMs:   intField(entry, "window_end_ms"),
		Outcome:       str(entry, "outcome"),
		Reason:        str(entry, "reason"),
		FailureReason: str(entry, "failure_reason"),
		Baseline:      optionalNumber(entry, "baseline"),
		Best:          optionalNumber(entry, "best"),
		Chosen:        floatMap(entry, "flat"),
		Planner:       nestedMap(entry, "planner"),
		AtMs:          intField(entry, "decided_ms"),
	}
	if v.AtMs == 0 {
		v.AtMs = intField(entry, "at_ms")
	}
	if v.Baseline != nil && v.Best != nil && *v.Baseline != 0 {
		gain := *v.Best - *v.Baseline
		if !maximize {
			gain = -gain
		}
		pct := 100 * gain / math.Abs(*v.Baseline)
		v.HeadroomPct = &pct
	}
	return v
}

// isWindowDecision reports whether a history entry is the decision of a window, as
// opposed to a promotion or a rollback that only refers to one.
func isWindowDecision(id string) bool {
	return strings.HasPrefix(id, "retune-")
}

// sortedDecisions returns the entry's decisions oldest first.
func sortedDecisions(entry *retuneEntry, maximize bool) []DecisionView {
	out := make([]DecisionView, 0, len(entry.decisions))
	for id, item := range entry.decisions {
		out = append(out, decisionView(id, item, maximize))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].AtMs != out[j].AtMs {
			return out[i].AtMs < out[j].AtMs
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// deployedBefore returns the router point that was deployed while the window of
// decisions[at] ran, or nil when the history does not determine it. A decision's chosen
// point equals the deployed one unless its outcome is applied or advisory, so the search
// walks back to the closest decision that states what was deployed.
func deployedBefore(decisions []DecisionView, at int) map[string]float64 {
	if outcome := decisions[at].Outcome; outcome != outcomeApplied && outcome != outcomeAdvisory {
		return decisions[at].Chosen
	}
	for i := at - 1; i >= 0; i-- {
		switch decisions[i].Outcome {
		case outcomeAdvisory:
			continue
		case outcomeRolledBack:
			return nil
		default:
			return decisions[i].Chosen
		}
	}
	return nil
}

// sameRouterPoint reports whether every knob of point is in deployed with the same value.
func sameRouterPoint(point, deployed map[string]float64) bool {
	if len(point) == 0 || len(deployed) == 0 {
		return false
	}
	for knob, value := range point {
		other, ok := deployed[knob]
		if !ok || math.Abs(value-other) > 1e-9*math.Max(1, math.Max(math.Abs(value), math.Abs(other))) {
			return false
		}
	}
	return true
}

// retuneRuns returns the Runs of a DGD's retune windows. The caller holds ix.mu.
func (ix *Index) retuneRuns(dgdUID string) []*runEntry {
	var out []*runEntry
	for _, entry := range ix.runs {
		if entry.env.Retune != nil && entry.env.Retune.DGDUID == dgdUID {
			out = append(out, entry)
		}
	}
	return out
}

// retuneObjective returns the objective metric of the newest retune Run.
func retuneObjective(runs []*runEntry) string {
	var newest *runEntry
	for _, entry := range runs {
		if newest == nil || entry.env.Retune.WindowEndMs > newest.env.Retune.WindowEndMs {
			newest = entry
		}
	}
	if newest == nil {
		return ""
	}
	return str(newest.env.Object, "spec", "objective", "metric")
}

// windowsOf builds the windows of one DGD, newest first, and its windowless events.
func windowsOf(entry *retuneEntry, runs []*runEntry, maximize bool) ([]WindowView, []DecisionView) {
	windows := map[int64]*WindowView{}
	window := func(end int64) *WindowView {
		w, ok := windows[end]
		if !ok {
			w = &WindowView{WindowEndMs: end}
			windows[end] = w
		}
		return w
	}
	var events []DecisionView

	if entry != nil {
		for _, decision := range sortedDecisions(entry, maximize) {
			if !isWindowDecision(decision.ID) || decision.WindowEndMs == 0 {
				events = append(events, decision)
				continue
			}
			window(decision.WindowEndMs).Decision = &decision
		}
	}
	for _, run := range runs {
		ref := run.env.Retune
		phase := str(run.env.Object, "status", "phase")
		if phase == "" {
			phase = "Pending"
		}
		w := window(ref.WindowEndMs)
		w.Runs = append(w.Runs, WindowRunView{
			Name: run.env.Name, Shard: ref.Shard, Attempt: ref.Attempt, Phase: phase, Retained: !run.live,
		})
	}

	out := make([]WindowView, 0, len(windows))
	for _, w := range windows {
		sort.Slice(w.Runs, func(i, j int) bool {
			if w.Runs[i].Shard != w.Runs[j].Shard {
				return w.Runs[i].Shard < w.Runs[j].Shard
			}
			return w.Runs[i].Attempt < w.Runs[j].Attempt
		})
		out = append(out, *w)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].WindowEndMs > out[j].WindowEndMs })
	return out, events
}

func dgdState(obj map[string]any) string {
	if state := str(obj, "status", "state"); state != "" {
		return state
	}
	for _, c := range conditions(obj) {
		if c.Type == "Ready" {
			if c.Status == "True" {
				return "Ready"
			}
			return "NotReady"
		}
	}
	return ""
}

func dgdComponents(obj map[string]any) []ComponentView {
	var out []ComponentView
	for _, item := range nestedSlice(obj, "spec", "components") {
		component, ok := item.(map[string]any)
		if !ok {
			continue
		}
		view := ComponentView{Name: str(component, "name"), Type: str(component, "type")}
		if replicas, ok := number(component, "replicas"); ok {
			n := int64(replicas)
			view.Replicas = &n
		}
		out = append(out, view)
	}
	return out
}

// deploymentView builds the view of one DGD. obj is nil for a DGD the cluster no longer
// has; entry is nil for a DGD without retune. The caller holds ix.mu.
func (ix *Index) deploymentView(namespace, name string, obj map[string]any, entry *retuneEntry) (DeploymentView, []WindowView, []DecisionView) {
	view := DeploymentView{Namespace: namespace, Name: name, Live: obj != nil}
	if obj != nil {
		view.UID = str(obj, "metadata", "uid")
		view.State = dgdState(obj)
		view.Created = str(obj, "metadata", "creationTimestamp")
		view.Components = dgdComponents(obj)
	} else if entry != nil {
		view.UID = entry.uid
	}
	runs := ix.retuneRuns(view.UID)
	if entry == nil && len(runs) == 0 {
		return view, nil, nil
	}

	objective := retuneObjective(runs)
	maximize := !minimizedObjectives[objective]
	windows, events := windowsOf(entry, runs, maximize)
	summary := &RetuneSummary{Objective: objective, Maximize: maximize, Windows: len(windows)}
	if entry != nil && entry.state != nil {
		summary.Mode = str(entry.state, "mode")
		summary.Current = floatMap(entry.state, "flat")
		summary.UpdatedMs = intField(entry.state, "updated_ms")
		summary.LastWindowEndMs = intField(entry.state, "last_window_end_ms")
	}
	for _, w := range windows {
		if w.Decision != nil {
			summary.Latest = w.Decision
			break
		}
	}
	view.Retune = summary
	return view, windows, events
}

// retuneEntryOf finds the retune entry of a DGD: by UID for a live one, else by name.
// The caller holds ix.mu.
func (ix *Index) retuneEntryOf(namespace, name string, obj map[string]any) *retuneEntry {
	if obj != nil {
		return ix.retunes[str(obj, "metadata", "uid")]
	}
	var newest *retuneEntry
	for _, entry := range ix.retunes {
		if entry.namespace == namespace && entry.dgd == name && (newest == nil || entry.uid > newest.uid) {
			newest = entry
		}
	}
	return newest
}

// ListDeployments returns every live DGD and every DGD known only from retune history.
func (ix *Index) ListDeployments(namespace string) []DeploymentView {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	out := []DeploymentView{}
	liveUIDs := map[string]bool{}

	for key, obj := range ix.dgds {
		liveUIDs[str(obj, "metadata", "uid")] = true
		if namespace != "" && key.Namespace != namespace {
			continue
		}
		view, _, _ := ix.deploymentView(key.Namespace, key.Name, obj, ix.retuneEntryOf(key.Namespace, key.Name, obj))
		out = append(out, view)
	}
	for uid, entry := range ix.retunes {
		if liveUIDs[uid] || (namespace != "" && entry.namespace != namespace) {
			continue
		}
		if _, live := ix.dgds[types.NamespacedName{Namespace: entry.namespace, Name: entry.dgd}]; live {
			continue
		}
		view, _, _ := ix.deploymentView(entry.namespace, entry.dgd, nil, entry)
		out = append(out, view)
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Namespace != out[j].Namespace {
			return out[i].Namespace < out[j].Namespace
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// DeploymentDetail returns one DGD with its retune windows.
func (ix *Index) DeploymentDetail(namespace, name string) (DeploymentDetail, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	obj := ix.dgds[types.NamespacedName{Namespace: namespace, Name: name}]
	entry := ix.retuneEntryOf(namespace, name, obj)
	if obj == nil && entry == nil {
		return DeploymentDetail{}, false
	}
	view, windows, events := ix.deploymentView(namespace, name, obj, entry)
	if windows == nil {
		windows = []WindowView{}
	}
	return DeploymentDetail{DeploymentView: view, Windows: windows, Events: events}, true
}

// routerPoint reads a Candidate's router knobs under their live names.
func routerPoint(params map[string]any) map[string]float64 {
	adapter := nestedMap(params, "adapters", routerAdapter)
	if adapter == nil {
		return nil
	}
	out := map[string]float64{}
	for public, live := range routerKnobs {
		if f, ok := number(adapter, public); ok {
			out[live] = f
		}
	}
	return out
}

// plannerPoint reads the one planner point a Candidate was evaluated with, if any.
func plannerPoint(params map[string]any) map[string]any {
	points := nestedSlice(params, "adapters", "dynamo.planner", plannerSweep, "points")
	if len(points) != 1 {
		return nil
	}
	point, _ := points[0].(map[string]any)
	return point
}

// WindowDetail returns one window of a DGD with every point its latest shard attempts
// evaluated.
func (ix *Index) WindowDetail(namespace, name string, windowEndMs int64) (WindowDetail, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	obj := ix.dgds[types.NamespacedName{Namespace: namespace, Name: name}]
	entry := ix.retuneEntryOf(namespace, name, obj)
	view, windows, _ := ix.deploymentView(namespace, name, obj, entry)
	detail := WindowDetail{Namespace: namespace, Deployment: name, Points: []PointView{}}
	found := false
	for _, w := range windows {
		if w.WindowEndMs == windowEndMs {
			detail.WindowView, found = w, true
			break
		}
	}
	if !found {
		return WindowDetail{}, false
	}
	detail.Objective, detail.Maximize = view.Retune.Objective, view.Retune.Maximize

	// The deployed point comes from the decision history; the chosen one from this window.
	var chosen map[string]float64
	if entry != nil {
		decisions := sortedDecisions(entry, detail.Maximize)
		for i, decision := range decisions {
			if isWindowDecision(decision.ID) && decision.WindowEndMs == windowEndMs {
				detail.Deployed = deployedBefore(decisions, i)
				if decision.Outcome == outcomeApplied || decision.Outcome == outcomeAdvisory {
					chosen = decision.Chosen
				}
				break
			}
		}
	}

	// A shard that was retried contributes only its latest attempt, and the probe none.
	latest := map[string]*runEntry{}
	for _, run := range ix.retuneRuns(view.UID) {
		ref := run.env.Retune
		if ref.WindowEndMs != windowEndMs || ref.Shard == shardProbe {
			continue
		}
		if current, ok := latest[ref.Shard]; !ok || ref.Attempt > current.env.Retune.Attempt {
			latest[ref.Shard] = run
		}
	}
	for _, run := range latest {
		for _, candidate := range ix.candidatesOfRun(run.env.UID) {
			params := candidateParameters(candidate.env.Object)
			metrics := candidateMetrics(candidate.env.Object)
			point := PointView{
				Candidate: candidate.env.Name, Namespace: candidate.env.Namespace,
				RunName: run.env.Name, Shard: run.env.Retune.Shard,
				Router: routerPoint(params), Planner: plannerPoint(params),
				Score: optionalNumber(metrics, "score"), Metrics: map[string]float64{},
			}
			for key, value := range metrics {
				if f, ok := toFloat(value); ok && key != "score" {
					point.Metrics[key] = f
				}
			}
			point.Deployed = point.Shard == shardBaseline || sameRouterPoint(point.Router, detail.Deployed)
			point.Chosen = sameRouterPoint(point.Router, chosen)
			detail.Points = append(detail.Points, point)
		}
	}
	sort.Slice(detail.Points, func(i, j int) bool { return detail.Points[i].Candidate < detail.Points[j].Candidate })
	return detail, true
}
