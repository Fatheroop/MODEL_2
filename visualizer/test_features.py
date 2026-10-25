#!/usr/bin/env python3
"""
===================================================================================================
COMPREHENSIVE HTTP & VISUALIZER FEATURE VERIFICATION SUITE
===================================================================================================
Tests all features requested by user over standard HTTP:
  1. What the model is SEEING (Retinal image, scene text, fixation coordinates, novelty)
  2. What the model is THINKING (Inner monologue stream, active concepts, workspace, drives)
  3. What the model is SPEAKING (Vocal output, vocabulary count, spoken history)
  4. What the model is LISTENING (Auditory input reception, intercom speech, listening history)
  5. Live HYPERPARAMETER updates & executive control actions (pause, resume, step, sleep, reward)
===================================================================================================
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
import sys

def run_feature_tests(base_url: str = "http://127.0.0.1:8080") -> bool:
    print("=" * 80)
    print(f"  🧪 RUNNING COMPREHENSIVE VISUALIZER FEATURE TESTS ON {base_url}")
    print("=" * 80)

    passed = 0
    total = 0

    def check(desc: str, condition: bool, extra: str = ""):
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {desc} {extra}")
        if not condition:
            print(f"         FAILED Assertion: {extra}")

    # Helper HTTP request functions
    def http_get(path: str) -> Dict[str, Any]:
        url = f"{base_url}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "AntigravityTest/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)

    def http_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)

    # 1. Server Connectivity
    try:
        t0 = time.time()
        init_data = http_get("/api/telemetry")
        check("HTTP Telemetry Endpoint Active (GET /api/telemetry)", True, f"({round((time.time()-t0)*1000, 1)}ms)")
    except Exception as e:
        check("HTTP Telemetry Endpoint Active (GET /api/telemetry)", False, f"Connection error: {e}")
        return False

    # 2. Test WHAT MODEL IS SEEING
    print("\n--- 1. Testing WHAT MODEL IS SEEING ---")
    seeing = init_data.get("seeing", {})
    check("Seeing: Retinal Image data URL is present", bool(seeing.get("image", "").startswith("data:image/png;base64,")), f"len={len(seeing.get('image', ''))}")
    check("Seeing: Visual text on screen is extracted", bool(seeing.get("scene_text")), f"text='{seeing.get('scene_text')[:40]}...'")
    check("Seeing: Fixation target coordinates and action are tracked", "col" in seeing.get("action", {}) and "name" in seeing.get("action", {}), f"action={seeing.get('action', {}).get('name')}")
    check("Seeing: Visual Novelty & Optical Motion are measured", isinstance(seeing.get("novelty"), (int, float)) and isinstance(seeing.get("change"), (int, float)), f"nov={seeing.get('novelty')}, chg={seeing.get('change')}")

    # 3. Test WHAT MODEL IS THINKING
    print("\n--- 2. Testing WHAT MODEL IS THINKING ---")
    thinking = init_data.get("thinking", {})
    check("Thinking: Inner thought monologue stream is generated", bool(thinking.get("current_thought")), f"thought='{thinking.get('current_thought')[:50]}...'")
    check("Thinking: Active conceptual tokens are populated", isinstance(thinking.get("active_concepts"), list), f"concepts={thinking.get('active_concepts')}")
    check("Thinking: Global Workspace competition items exist", isinstance(thinking.get("workspace_items"), list), f"items_count={len(thinking.get('workspace_items', []))}")
    check("Thinking: Homeostatic drives (curiosity, boredom, fatigue) are tracked", all(k in thinking for k in ("curiosity", "boredom", "fatigue", "confidence")), f"curiosity={thinking.get('curiosity')}, conf={thinking.get('confidence')}")

    # 4. Test WHAT MODEL IS SPEAKING
    print("\n--- 3. Testing WHAT MODEL IS SPEAKING ---")
    speaking = init_data.get("speaking", {})
    check("Speaking: Last vocal utterance is articulated", bool(speaking.get("last_spoken")), f"spoken='{speaking.get('last_spoken')[:50]}...'")
    check("Speaking: Model vocabulary counter is tracked", speaking.get("vocabulary_size", 0) > 0, f"vocab={speaking.get('vocabulary_size')} words")
    check("Speaking: Vocal history queue is maintained", isinstance(speaking.get("history"), list), f"history_len={len(speaking.get('history', []))}")

    # 5. Test WHAT MODEL IS LISTENING
    print("\n--- 4. Testing WHAT MODEL IS LISTENING ---")
    test_phrase = f"quantum photonic neural circuits test at {int(time.time())}"
    listen_resp = http_post("/api/listen", {"text": test_phrase})
    check("Listening: HTTP Speech Delivery (POST /api/listen)", listen_resp.get("status") == "success", f"resp={listen_resp}")
    
    # Wait one step cycle for auditory comprehension
    time.sleep(0.3)
    updated_telemetry = http_get("/api/telemetry")
    listening = updated_telemetry.get("listening", {})
    check("Listening: Organism auditory channel heard the input", listening.get("last_heard") == test_phrase, f"heard='{listening.get('last_heard')}'")
    check("Listening: Auditory history queue recorded the event", any(h.get("text") == test_phrase for h in listening.get("history", [])), f"history_entries={len(listening.get('history', []))}")

    # 6. Test LIVE HYPERPARAMETERS TUNING
    print("\n--- 5. Testing LIVE HYPERPARAMETERS TUNING ---")
    param_payload = {
        "k_sparsity": 32,
        "encoder_eta": 0.088,
        "policy_eta": 0.22,
        "policy_gamma": 0.88,
        "step_delay_ms": 80
    }
    hyper_resp = http_post("/api/hyperparameters", param_payload)
    check("Hyperparameters: HTTP Live Tuning (POST /api/hyperparameters)", hyper_resp.get("k_sparsity") == 32 and hyper_resp.get("encoder_eta") == 0.088, f"resp={hyper_resp}")

    time.sleep(0.15)
    # Verify parameters reflected in live telemetry
    telemetry_post_hyper = http_get("/api/telemetry")
    live_params = telemetry_post_hyper.get("hyperparameters", {})
    check("Hyperparameters: Live Organism reflected k_sparsity=32", live_params.get("k_sparsity") == 32, f"actual={live_params.get('k_sparsity')}")
    check("Hyperparameters: Live Organism reflected encoder_eta=0.088", abs(live_params.get("encoder_eta") - 0.088) < 1e-4, f"actual={live_params.get('encoder_eta')}")

    # 7. Test EXECUTIVE CONTROLS
    print("\n--- 6. Testing EXECUTIVE CONTROLS (Pause, Resume, Step, Sleep, Reward) ---")
    # Pause
    ctrl_pause = http_post("/api/control", {"action": "pause"})
    check("Control: Simulation Pause", ctrl_pause.get("paused") is True, f"resp={ctrl_pause}")
    
    # Verify paused
    step_frozen_1 = http_get("/api/telemetry").get("step")
    time.sleep(0.2)
    step_frozen_2 = http_get("/api/telemetry").get("step")
    check("Control: Step counter halts while paused", step_frozen_1 == step_frozen_2, f"step1={step_frozen_1}, step2={step_frozen_2}")

    # Step exactly one cycle
    ctrl_step = http_post("/api/control", {"action": "step"})
    time.sleep(0.2)
    step_advanced = http_get("/api/telemetry").get("step")
    check("Control: Single-step execution", step_advanced == step_frozen_2 + 1, f"step_advanced={step_advanced}")

    # Resume
    ctrl_resume = http_post("/api/control", {"action": "resume"})
    check("Control: Simulation Resume", ctrl_resume.get("paused") is False, f"resp={ctrl_resume}")

    # Reward feedback injection
    ctrl_reward = http_post("/api/control", {"action": "reward"})
    check("Control: Dopamine Reward Injection (+1.0)", ctrl_reward.get("feedback") == 1.0, f"resp={ctrl_reward}")

    # Sleep trigger
    ctrl_sleep = http_post("/api/control", {"action": "sleep"})
    check("Control: Sleep Consolidation Trigger", ctrl_sleep.get("slept") is True, f"resp={ctrl_sleep}")

    # Final Summary
    print("\n" + "=" * 80)
    print(f"  FEATURE VERIFICATION RESULT: {passed} / {total} CHECKS PASSED")
    print("=" * 80)
    return passed == total


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    success = run_feature_tests(f"http://127.0.0.1:{port}")
    sys.exit(0 if success else 1)
