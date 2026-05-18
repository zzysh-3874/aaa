from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal


def _load_rsl_modules():
    repo_root = Path(__file__).resolve().parents[1]
    modules_dir = repo_root / "scripts/rsl_rl/modules"
    package_name = "_pie_test_rsl_modules"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(modules_dir)]
        sys.modules[package_name] = package
    return (
        importlib.import_module(f"{package_name}.ppo_with_extractor"),
        importlib.import_module(f"{package_name}.actor_critic_with_encoder"),
        importlib.import_module(f"{package_name}.feature_extractors.pie_estimator"),
    )


class _DummyPolicy(nn.Module):
    is_recurrent = False

    def __init__(self, obs_dim: int = 45, action_dim: int = 12, critic_obs_dim: int | None = None):
        super().__init__()
        if critic_obs_dim is None:
            critic_obs_dim = obs_dim
        self.actor = nn.Linear(obs_dim, action_dim)
        self.critic = nn.Linear(critic_obs_dim, 1)
        self.std = nn.Parameter(torch.ones(action_dim))
        self.distribution: Normal | None = None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, *_, **__):
        mean = self.actor(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution.sample()

    def act_inference(self, observations, *_, **__):
        return self.actor(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **_):
        return self.critic(critic_observations)

    def reset(self, dones=None):
        pass


def _make_obs_dict(batch_size: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    return {
        "policy": torch.randn(batch_size, 45),
        "depth_camera": torch.randn(batch_size, 2, 58, 87),
        "proprioception_history": torch.randn(batch_size, 10, 45),
        "estimator_targets": {
            "base_velocity": torch.randn(batch_size, 3),
            "foot_clearance": torch.randn(batch_size, 4),
            "height_scan": torch.randn(batch_size, 132),
            "next_proprioception": torch.randn(batch_size, 45),
        },
    }


def test_ppo_with_extractor_updates_pie_estimator_from_rollout_storage():
    ppo_module, _, pie_module = _load_rsl_modules()

    class _SpyPIEEstimator(pie_module.PIEEstimator):
        def __init__(self):
            super().__init__()
            self.hidden_none_calls = 0
            self.hidden_not_none_calls = 0

        def forward(self, depth, proprioception_history, hidden_state=None):
            if hidden_state is None:
                self.hidden_none_calls += 1
            else:
                self.hidden_not_none_calls += 1
            return super().forward(depth, proprioception_history, hidden_state=hidden_state)

    batch_size = 2
    num_steps = 3
    estimator = _SpyPIEEstimator()
    algorithm = ppo_module.PPOWithExtractor(
        policy=_DummyPolicy(),
        estimator=estimator,
        estimator_paras={
            "learning_rate": 1e-3,
            "train_with_estimated_states": False,
            "use_pie_estimator_rollout": True,
            "pie_train_gru_sequence": True,
            "pie_num_learning_epochs": 1,
            "pie_num_mini_batches": 1,
            "loss_weights": {"kl": 0.0},
        },
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )
    algorithm.init_pie_estimator_storage(batch_size, num_steps)

    for step in range(num_steps):
        cached_step = algorithm.cache_pie_estimator_step(_make_obs_dict(batch_size))
        next_obs_dict = _make_obs_dict(batch_size)
        dones = torch.tensor([False, step == num_steps - 1])
        algorithm.process_pie_estimator_env_step(cached_step, next_obs_dict, dones=dones)

    losses = algorithm.update_pie_estimator_from_storage()

    assert algorithm.uses_pie_estimator
    assert algorithm.pie_estimator_storage.step == 0
    assert estimator.hidden_none_calls > 0
    assert estimator.hidden_not_none_calls > 0
    expected_keys = {"loss", "loss_v", "loss_hf", "loss_height", "loss_next_proprio", "loss_kl"}
    assert set(losses.keys()) == expected_keys
    for value in losses.values():
        assert isinstance(value, float)
        assert torch.isfinite(torch.tensor(value))


def test_simple_actor_critic_accepts_flat_pie_policy_observation():
    _, actor_critic_module, _ = _load_rsl_modules()
    policy = actor_critic_module.SimpleActorCritic(num_critic_obs=45, num_actions=12)
    observations = torch.randn(8, 45)

    actions = policy.act(observations)
    values = policy.evaluate(observations)
    log_prob = policy.get_actions_log_prob(actions)

    assert tuple(actions.shape) == (8, 12)
    assert tuple(values.shape) == (8, 1)
    assert tuple(log_prob.shape) == (8,)
    assert tuple(policy.action_mean.shape) == (8, 12)
    assert tuple(policy.action_std.shape) == (8, 12)


def test_simple_actor_critic_supports_pie_actor_features_with_privileged_critic_obs():
    _, actor_critic_module, _ = _load_rsl_modules()
    policy = actor_critic_module.SimpleActorCritic(num_critic_obs=180, num_actions=12, num_actor_obs=116)
    actor_observations = torch.randn(8, 116)
    critic_observations = torch.randn(8, 180)

    actions = policy.act(actor_observations)
    values = policy.evaluate(critic_observations)
    log_prob = policy.get_actions_log_prob(actions)

    assert tuple(actions.shape) == (8, 12)
    assert tuple(values.shape) == (8, 1)
    assert tuple(log_prob.shape) == (8,)
    assert tuple(policy.action_mean.shape) == (8, 12)
    assert tuple(policy.action_std.shape) == (8, 12)


def test_ppo_with_extractor_builds_detached_pie_actor_observations():
    ppo_module, _, pie_module = _load_rsl_modules()
    batch_size = 2
    estimator = pie_module.PIEEstimator()
    algorithm = ppo_module.PPOWithExtractor(
        policy=_DummyPolicy(obs_dim=116, critic_obs_dim=45),
        estimator=estimator,
        estimator_paras={
            "learning_rate": 1e-3,
            "train_with_estimated_states": False,
            "use_pie_estimator_rollout": True,
            "use_pie_actor_features": True,
            "detach_pie_actor_features": True,
            "pie_actor_feature_keys": ("z_m", "z_mu", "v_hat", "h_f_hat"),
            "pie_num_learning_epochs": 1,
            "pie_num_mini_batches": 1,
            "loss_weights": {"kl": 0.0},
        },
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )

    policy_obs = torch.randn(batch_size, 45)
    actor_obs = algorithm.build_pie_actor_observations(policy_obs, _make_obs_dict(batch_size))

    assert tuple(actor_obs.shape) == (batch_size, 116)
    assert torch.allclose(actor_obs[:, :45], policy_obs)
    assert not actor_obs[:, 45:].requires_grad
    assert algorithm.pie_actor_rnn_hidden is not None
    assert tuple(algorithm.pie_actor_rnn_hidden.shape) == (1, batch_size, estimator.gru_hidden_dim)

    algorithm.reset_pie_actor_hidden(torch.tensor([True, False]))
    assert torch.allclose(algorithm.pie_actor_rnn_hidden[:, 0], torch.zeros_like(algorithm.pie_actor_rnn_hidden[:, 0]))
    assert torch.any(algorithm.pie_actor_rnn_hidden[:, 1] != 0)


def test_pie_actor_inference_wrapper_uses_obs_dict_and_resets_hidden():
    ppo_module, _, pie_module = _load_rsl_modules()
    batch_size = 2
    estimator = pie_module.PIEEstimator()
    algorithm = ppo_module.PPOWithExtractor(
        policy=_DummyPolicy(obs_dim=116, critic_obs_dim=45),
        estimator=estimator,
        estimator_paras={
            "learning_rate": 1e-3,
            "train_with_estimated_states": False,
            "use_pie_estimator_rollout": True,
            "use_pie_actor_features": True,
            "detach_pie_actor_features": True,
            "pie_actor_feature_keys": ("z_m", "z_mu", "v_hat", "h_f_hat"),
            "pie_num_learning_epochs": 1,
            "pie_num_mini_batches": 1,
            "loss_weights": {"kl": 0.0},
        },
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )
    wrapper = ppo_module.PIEActorInferenceWrapper(algorithm, normalizer=nn.Identity(), device="cpu")

    with torch.inference_mode():
        actions = wrapper(_make_obs_dict(batch_size), hist_encoding=True)

    assert tuple(actions.shape) == (batch_size, 12)
    assert algorithm.pie_actor_rnn_hidden is not None
    assert tuple(algorithm.pie_actor_rnn_hidden.shape) == (1, batch_size, estimator.gru_hidden_dim)

    wrapper.reset(torch.tensor([True, False]))
    assert torch.allclose(algorithm.pie_actor_rnn_hidden[:, 0], torch.zeros_like(algorithm.pie_actor_rnn_hidden[:, 0]))
    assert torch.any(algorithm.pie_actor_rnn_hidden[:, 1] != 0)


def test_pie_actor_estimator_joint_update_changes_estimator_from_actor_loss():
    torch.manual_seed(7)
    ppo_module, _, pie_module = _load_rsl_modules()
    num_envs = 2
    num_steps = 2
    estimator = pie_module.PIEEstimator()
    algorithm = ppo_module.PPOWithExtractor(
        policy=_DummyPolicy(obs_dim=116, critic_obs_dim=180),
        estimator=estimator,
        estimator_paras={
            "learning_rate": 1e-3,
            "train_with_estimated_states": False,
            "use_pie_estimator_rollout": True,
            "use_pie_actor_features": True,
            "detach_pie_actor_features": False,
            "pie_joint_actor_estimator": True,
            "pie_policy_obs_dim": 45,
            "pie_actor_feature_keys": ("z_m", "z_mu", "v_hat", "h_f_hat"),
            "pie_train_gru_sequence": True,
            "pie_num_learning_epochs": 1,
            "pie_num_mini_batches": 1,
            "loss_weights": {
                "v": 0.0,
                "h_f": 0.0,
                "height": 0.0,
                "next_proprio": 0.0,
                "kl": 0.0,
            },
        },
        num_learning_epochs=1,
        num_mini_batches=1,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        device="cpu",
    )
    algorithm.init_storage("rl", num_envs, num_steps, [116], [180], [12])
    algorithm.init_pie_estimator_storage(num_envs, num_steps)

    for _ in range(num_steps):
        obs_dict = _make_obs_dict(num_envs)
        cached_step = algorithm.cache_pie_estimator_step(obs_dict)
        stored_actor_obs = torch.cat((obs_dict["policy"], torch.zeros(num_envs, 71)), dim=-1)
        critic_obs = torch.randn(num_envs, 180)
        algorithm.act(stored_actor_obs, critic_obs)

        next_obs_dict = _make_obs_dict(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        algorithm.process_pie_estimator_env_step(cached_step, next_obs_dict, dones=dones)
        algorithm.process_env_step(torch.randn(num_envs), dones, {})

    algorithm.compute_returns(torch.randn(num_envs, 180))
    before = [parameter.detach().clone() for parameter in estimator.parameters()]
    losses = algorithm.update()

    assert losses["pie_actor_estimator_joint"] == 1.0
    assert any(
        not torch.allclose(parameter_before, parameter_after)
        for parameter_before, parameter_after in zip(before, estimator.parameters())
    )
