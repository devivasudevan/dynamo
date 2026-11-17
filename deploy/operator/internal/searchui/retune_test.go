/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/searchui/history"
	"github.com/go-logr/logr"
)

const (
	retuneNS      = "prod"
	retuneDGD     = "qwen"
	retuneDGDUID  = "dgd-uid-1"
	windowOne     = int64(1_790_000_000_000)
	windowTwo     = int64(1_790_000_600_000)
	retuneRunsAPI = "/api/v1/deployments/prod/qwen"
)

func retuneDGDObject() map[string]any {
	return map[string]any{
		"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoGraphDeployment",
		"metadata": map[string]any{"name": retuneDGD, "namespace": retuneNS, "uid": retuneDGDUID},
		"spec": map[string]any{"components": []any{
			map[string]any{"name": "Frontend", "type": "frontend", "replicas": int64(1)},
			map[string]any{"name": "Worker", "type": "worker", "replicas": int64(2)},
		}},
		"status": map[string]any{"conditions": []any{map[string]any{"type": "Ready", "status": "True"}}},
	}
}

// retuneRequest and retuneRun build one shard of a window the way the planner and the
// operator do: the labels are on the Request, and the Run only names its owner.
func retuneRequest(window int64, shard string, attempt int) map[string]any {
	name := fmt.Sprintf("retune-%d-%s-%d", window, shard, attempt)
	return map[string]any{
		"apiVersion": "nvidia.com/v1beta2", "kind": "DynamoGraphDeploymentRequest",
		"metadata": map[string]any{"name": name, "namespace": retuneNS, "uid": "req-" + name, "labels": map[string]any{
			labelRetuneDGD: retuneDGD, labelRetuneDGDUID: retuneDGDUID, labelRetuneWindow: fmt.Sprint(window),
			labelRetuneShard: shard, labelRetuneAttempt: fmt.Sprint(attempt),
		}},
		"spec": map[string]any{"objective": map[string]any{"mode": "Optimize", "metric": "GoodputPerGPU"}},
	}
}

func retuneRun(window int64, shard string, attempt int) map[string]any {
	request := fmt.Sprintf("retune-%d-%s-%d", window, shard, attempt)
	return map[string]any{
		"apiVersion": "nvidia.com/v1beta2", "kind": "DynamoGraphDeploymentRun",
		"metadata": map[string]any{
			"name": request + "-1", "namespace": retuneNS, "uid": "run-" + request,
			"ownerReferences": []any{map[string]any{"kind": "DynamoGraphDeploymentRequest", "name": request}},
		},
		"spec":   map[string]any{"objective": map[string]any{"mode": "Optimize", "metric": "GoodputPerGPU"}},
		"status": map[string]any{"phase": "Succeeded"},
	}
}

func retuneCandidate(run map[string]any, id string, credit, score float64) map[string]any {
	runName, runUID := str(run, "metadata", "name"), str(run, "metadata", "uid")
	return map[string]any{
		"apiVersion": "nvidia.com/v1beta2", "kind": "DynamoGraphDeploymentCandidate",
		"metadata": map[string]any{
			"name": runName + "-" + id, "namespace": retuneNS, "uid": "cand-" + runName + id,
			"labels":          map[string]any{labelRunUID: runUID},
			"ownerReferences": []any{map[string]any{"kind": "DynamoGraphDeploymentRun", "name": runName, "uid": runUID}},
		},
		"status": map[string]any{
			"parameters": map[string]any{"backend": "vllm", "adapters": map[string]any{routerAdapter: map[string]any{
				"policy": "kv_router", "overlap_score_credit": credit, "prefill_load_scale": 2.0, "temperature": 0.0,
			}}},
			"metrics": map[string]any{"score": score, "mean_ttft_ms": 120.0, "goodput_output_throughput_tok_s": score},
		},
	}
}

func retuneLease(history []map[string]any, deployedCredit float64) map[string]any {
	state, _ := json.Marshal(map[string]any{
		"version": 1, "dgd_uid": retuneDGDUID, "mode": "active", "last_window_end_ms": windowTwo, "updated_ms": windowTwo + 90_000,
		"flat": map[string]any{"overlap_score_credit": deployedCredit, "prefill_load_scale": 2.0, "router_temperature": 0.0},
	})
	entries, _ := json.Marshal(history)
	return map[string]any{
		"apiVersion": "coordination.k8s.io/v1", "kind": "Lease",
		"metadata": map[string]any{
			"name": retuneDGD + retuneStateLeaseSuffix, "namespace": retuneNS,
			"labels":      map[string]any{labelRetuneDGD: retuneDGD, labelRetuneDGDUID: retuneDGDUID},
			"annotations": map[string]any{annotationRetuneState: string(state), annotationRetuneHistory: string(entries)},
		},
	}
}

func retuneDecision(window int64, outcome string, credit, baseline, best float64) map[string]any {
	return map[string]any{
		"decision_id": fmt.Sprintf("retune-w%d", window), "window_end_ms": window, "outcome": outcome,
		"baseline": baseline, "best": best, "decided_ms": window + 80_000,
		"flat": map[string]any{"overlap_score_credit": credit, "prefill_load_scale": 2.0, "router_temperature": 0.0},
	}
}

// retuneIndex holds one DGD with two windows. Window one kept credit 1.0. Window two
// applied credit 2.0; its grid shard was retried, and it has a probe.
func retuneIndex(t *testing.T, store history.Store) *Index {
	t.Helper()
	ctx := context.Background()
	ix := NewIndex(store)
	ix.UpsertDGD(retuneDGDObject())
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}

	shards := []struct {
		window  int64
		shard   string
		attempt int
		points  map[string][2]float64 // id -> credit, score
	}{
		{windowOne, "grid-0", 1, map[string][2]float64{"a": {1.0, 1000}, "b": {2.0, 1020}}},
		{windowTwo, "grid-0", 1, map[string][2]float64{"stale": {1.0, 1}}},
		{windowTwo, "grid-0", 2, map[string][2]float64{"a": {1.0, 1000}, "b": {2.0, 1200}}},
		{windowTwo, shardProbe, 1, map[string][2]float64{"p": {2.0, 1200}}},
	}
	for i, s := range shards {
		run := retuneRun(s.window, s.shard, s.attempt)
		// Half the Runs are observed before their Request, as separate informers allow.
		if i%2 == 0 {
			must(ix.UpsertRun(ctx, run))
			must(ix.UpsertRequest(ctx, retuneRequest(s.window, s.shard, s.attempt)))
		} else {
			must(ix.UpsertRequest(ctx, retuneRequest(s.window, s.shard, s.attempt)))
			must(ix.UpsertRun(ctx, run))
		}
		for id, point := range s.points {
			must(ix.UpsertCandidate(ctx, retuneCandidate(run, id, point[0], point[1])))
		}
	}
	must(ix.UpsertRetuneLease(ctx, retuneLease([]map[string]any{
		retuneDecision(windowOne, "kept", 1.0, 1000, 1020),
		retuneDecision(windowTwo, "applied", 2.0, 1000, 1200),
	}, 2.0)))
	return ix
}

func TestRetuneDeploymentAndWindowViews(t *testing.T) {
	ix := retuneIndex(t, nil)

	t.Log("The DGD lists with its retune state and the latest decision's headroom")
	deployments := ix.ListDeployments("")
	if len(deployments) != 1 {
		t.Fatalf("ListDeployments = %+v", deployments)
	}
	d := deployments[0]
	if !d.Live || d.State != "Ready" || len(d.Components) != 2 || d.Retune == nil {
		t.Fatalf("deployment = %+v", d)
	}
	r := d.Retune
	if r.Mode != "active" || r.Objective != "GoodputPerGPU" || !r.Maximize || r.Windows != 2 || r.Current["overlap_score_credit"] != 2.0 {
		t.Fatalf("retune summary = %+v", r)
	}
	if r.Latest == nil || r.Latest.Outcome != "applied" || r.Latest.HeadroomPct == nil || *r.Latest.HeadroomPct != 20 {
		t.Fatalf("latest decision = %+v", r.Latest)
	}

	t.Log("Windows are newest first and carry every shard Run")
	detail, ok := ix.DeploymentDetail(retuneNS, retuneDGD)
	if !ok || len(detail.Windows) != 2 || detail.Windows[0].WindowEndMs != windowTwo || len(detail.Windows[0].Runs) != 3 {
		t.Fatalf("detail windows = %+v", detail.Windows)
	}
}

func TestRetuneWindowPoints(t *testing.T) {
	ix := retuneIndex(t, nil)

	t.Log("A window plots the latest attempt of each shard without the probe")
	window, ok := ix.WindowDetail(retuneNS, retuneDGD, windowTwo)
	if !ok || len(window.Points) != 2 {
		t.Fatalf("window points = %+v", window.Points)
	}

	t.Log("The deployed point comes from the earlier kept decision, the chosen one from this window")
	for _, p := range window.Points {
		credit := p.Router["overlap_score_credit"]
		if p.Deployed != (credit == 1.0) || p.Chosen != (credit == 2.0) || p.Router["router_temperature"] != 0 {
			t.Fatalf("point = %+v", p)
		}
		if p.Score == nil || p.Metrics["mean_ttft_ms"] != 120 {
			t.Fatalf("point metrics = %+v", p)
		}
	}

	t.Log("Retune Requests are marked so the run list can fold them away")
	for _, request := range ix.ListRequests(retuneNS) {
		if request.Retune == nil || request.Retune.DGD != retuneDGD || request.Runs[0].Retune == nil {
			t.Fatalf("request = %+v", request)
		}
	}

	t.Log("An unknown window is not found")
	if _, ok := ix.WindowDetail(retuneNS, retuneDGD, 1); ok {
		t.Fatal("WindowDetail found a window that does not exist")
	}
}

func TestRetuneHistorySurvivesCleanup(t *testing.T) {
	ctx := context.Background()
	store, err := history.NewLocal(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	ix := retuneIndex(t, store)

	t.Log("The planner deletes the Requests and Runs of old windows, and later the DGD goes")
	for _, request := range ix.ListRequests(retuneNS) {
		ix.DeleteRequest(request.Namespace, request.Name)
		for _, run := range request.Runs {
			if err := ix.DeleteRun(ctx, run.UID); err != nil {
				t.Fatal(err)
			}
		}
	}

	t.Log("A fresh index still shows the DGD, both decisions and the window's points")
	reloaded := NewIndex(store)
	if err := reloaded.Load(ctx); err != nil {
		t.Fatal(err)
	}
	deployments := reloaded.ListDeployments(retuneNS)
	if len(deployments) != 1 || deployments[0].Live || deployments[0].Retune.Windows != 2 || deployments[0].Retune.Mode != "" {
		t.Fatalf("reloaded deployments = %+v", deployments)
	}
	window, ok := reloaded.WindowDetail(retuneNS, retuneDGD, windowTwo)
	if !ok || len(window.Points) != 2 || window.Decision == nil || window.Decision.Outcome != "applied" || window.Deployed["overlap_score_credit"] != 1.0 {
		t.Fatalf("reloaded window = %+v", window)
	}
}

func TestDeployedBefore(t *testing.T) {
	a, b := map[string]float64{"k": 1}, map[string]float64{"k": 2}
	cases := []struct {
		name      string
		decisions []DecisionView
		want      map[string]float64
	}{
		{"a kept window states its own deployed point", []DecisionView{{Outcome: "kept", Chosen: a}}, a},
		{"a failed window states its own deployed point", []DecisionView{{Outcome: "failed", Chosen: a}}, a},
		{"an applied window takes the previous applied point", []DecisionView{{Outcome: "applied", Chosen: a}, {Outcome: "applied", Chosen: b}}, a},
		{"advisory decisions deploy nothing and are skipped", []DecisionView{{Outcome: "kept", Chosen: a}, {Outcome: "advisory", Chosen: b}, {Outcome: "advisory", Chosen: b}}, a},
		{"a promotion deploys its point", []DecisionView{{Outcome: "promoted", Chosen: b}, {Outcome: "applied", Chosen: a}}, b},
		{"a rollback leaves the deployed point unknown", []DecisionView{{Outcome: "applied", Chosen: a}, {Outcome: "rolled_back"}, {Outcome: "applied", Chosen: b}}, nil},
		{"the first applied window has no earlier evidence", []DecisionView{{Outcome: "applied", Chosen: a}}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := deployedBefore(tc.decisions, len(tc.decisions)-1); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("deployedBefore = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDeploymentEndpoints(t *testing.T) {
	srv := httptest.NewServer(NewServer(retuneIndex(t, nil), NewLogManager(nil, nil, logr.Discard()), nil))
	defer srv.Close()
	cases := []struct {
		path string
		want int
	}{
		{"/api/v1/deployments", http.StatusOK},
		{"/api/v1/deployments?namespace=other", http.StatusOK},
		{retuneRunsAPI, http.StatusOK},
		{"/api/v1/deployments/prod/missing", http.StatusNotFound},
		{fmt.Sprintf("%s/windows/%d", retuneRunsAPI, windowTwo), http.StatusOK},
		{retuneRunsAPI + "/windows/1", http.StatusNotFound},
		{retuneRunsAPI + "/windows/latest", http.StatusBadRequest},
	}
	for _, tc := range cases {
		resp, err := http.Get(srv.URL + tc.path)
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
		if resp.StatusCode != tc.want {
			t.Errorf("GET %s = %d, want %d", tc.path, resp.StatusCode, tc.want)
		}
	}
}
