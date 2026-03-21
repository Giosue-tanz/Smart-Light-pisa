# -*- coding: utf-8 -*-
"""Test rapido per verificare che Single-Agent e Multi-Agent funzionino."""
import sys
import os
import io
import time

# Fix encoding per Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("[TEST] SmartLight Pisa - Test rapido")
print("=" * 60)

results = []

# ===== TEST 1: SINGLE AGENT =====
print("\n[TEST 1] Single-Agent Environment")
print("-" * 40)
try:
    from smartlight_sumo_env import SumoIntersectionEnv
    
    t0 = time.time()
    env = SumoIntersectionEnv(use_gui=False, max_steps=50)
    obs, _ = env.reset()
    load_time = time.time() - t0
    
    print(f"  [OK] Ambiente creato in {load_time:.1f}s")
    print(f"  TLS: {env.tls_id[:55]}...")
    print(f"  Corsie monitorate: {len(env.lanes)}")
    print(f"  Fasi semaforo: {env.num_phases}")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs: [{', '.join(f'{x:.3f}' for x in obs)}]")
    
    # Esegui qualche step
    total_r = 0
    for i in range(30):
        obs, r, done, _, _ = env.step(env.action_space.sample())
        total_r += r
    
    print(f"  30 steps completati | Total reward: {total_r:.4f}")
    print(f"  Final obs: [{', '.join(f'{x:.3f}' for x in obs)}]")
    env.close()
    print("  >>> TEST 1 PASSED!")
    results.append(("Single-Agent", "PASSED"))
    
except Exception as e:
    print(f"  >>> TEST 1 FAILED: {e}")
    results.append(("Single-Agent", f"FAILED: {e}"))
    import traceback
    traceback.print_exc()

# ===== TEST 2: MULTI-AGENT =====
print("\n[TEST 2] Multi-Agent Environment")
print("-" * 40)
try:
    from smartlight_multi_agent_env import MultiAgentSumoEnv
    
    t0 = time.time()
    env = MultiAgentSumoEnv(
        use_gui=False,
        max_steps=30,
        top_n_tls=3,
        min_controlled_lanes=3,
        alpha=0.7,
        neighbor_radius=300,
    )
    obs_dict, _ = env.reset()
    load_time = time.time() - t0
    
    print(f"  [OK] Ambiente multi-agent creato in {load_time:.1f}s")
    print(f"  Agenti attivi: {env.n_agents}")
    
    for i, agent_id in enumerate(env.agent_ids):
        obs = obs_dict[agent_id]
        short_id = agent_id[:40] + "..." if len(agent_id) > 40 else agent_id
        print(f"  Agent {i}: {short_id}")
        print(f"    Corsie: {len(env.agent_lanes[agent_id])} | Fasi: {env.agent_num_phases[agent_id]}")
        print(f"    Vicini: {len(env.agent_neighbors.get(agent_id, []))}")
        print(f"    Obs: [{', '.join(f'{x:.3f}' for x in obs)}]")
    
    # Esegui qualche step
    total_rewards = {a: 0 for a in env.agent_ids}
    for step in range(20):
        actions = {a: env.action_space.sample() for a in env.agent_ids}
        obs_dict, rewards, done, _, _ = env.step(actions)
        for a in env.agent_ids:
            total_rewards[a] += rewards[a]
    
    print(f"\n  20 steps completati:")
    for i, agent_id in enumerate(env.agent_ids):
        print(f"    Agent {i}: Total reward = {total_rewards[agent_id]:.4f}")
    
    metrics = env.get_global_metrics()
    print(f"  Global: Wait={metrics['total_waiting_time']:.0f}s | Halt={metrics['total_halting']} | Speed={metrics['avg_speed']:.2f}m/s")
    
    env.close()
    print("  >>> TEST 2 PASSED!")
    results.append(("Multi-Agent", "PASSED"))
    
except Exception as e:
    print(f"  >>> TEST 2 FAILED: {e}")
    results.append(("Multi-Agent", f"FAILED: {e}"))
    import traceback
    traceback.print_exc()

# ===== TEST 3: ACTOR-CRITIC NETWORK =====
print("\n[TEST 3] Actor-Critic Network (PyTorch)")
print("-" * 40)
try:
    import torch
    from smartlight_train_multi_agent import ActorCritic
    
    model = ActorCritic(obs_dim=15, act_dim=2, hidden_dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Modello creato: {n_params:,} parametri")
    
    # Test forward pass
    dummy_obs = torch.randn(1, 15)
    probs, value = model(dummy_obs)
    print(f"  Action probs: {probs.detach().numpy()[0]}")
    print(f"  Value: {value.item():.4f}")
    
    # Test act
    action, log_prob, val = model.act(dummy_obs.numpy()[0])
    print(f"  Action: {action} | Log prob: {log_prob:.4f} | Value: {val:.4f}")
    print("  >>> TEST 3 PASSED!")
    results.append(("Actor-Critic", "PASSED"))
    
except Exception as e:
    print(f"  >>> TEST 3 FAILED: {e}")
    results.append(("Actor-Critic", f"FAILED: {e}"))
    import traceback
    traceback.print_exc()

# ===== RIEPILOGO =====
print("\n" + "=" * 60)
print("[RIEPILOGO]")
print("=" * 60)
all_passed = True
for name, status in results:
    marker = "[OK]" if "PASSED" in status else "[FAIL]"
    print(f"  {marker} {name}: {status}")
    if "FAILED" in status:
        all_passed = False

if all_passed:
    print("\n  TUTTI I TEST PASSATI!")
else:
    print("\n  ALCUNI TEST HANNO FALLITO!")
print("=" * 60)
