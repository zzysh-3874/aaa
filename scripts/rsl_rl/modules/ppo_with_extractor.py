
from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic_with_encoder import ActorCriticRMA
from .feature_extractors.pie_estimator_loss import PIEEstimatorLoss
from .feature_extractors.pie_estimator_rollout_storage import PIEEstimatorRolloutStorage, PIEEstimatorStepInput
from rsl_rl.algorithms import PPO


class PIEActorInferenceWrapper:
    """Inference-time policy that reconstructs PIE actor observations from env obs_dict."""

    def __init__(
        self,
        algorithm: "PPOWithExtractor",
        normalizer: nn.Module | None = None,
        device: torch.device | str | None = None,
    ):
        if not algorithm.use_pie_actor_features:
            raise ValueError("PIEActorInferenceWrapper requires use_pie_actor_features=True")
        self.algorithm = algorithm
        self.normalizer = normalizer if normalizer is not None else nn.Identity()
        self.device = device
        self.algorithm.reset_pie_actor_hidden()

    def __call__(
        self,
        obs_or_obs_dict: torch.Tensor | Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
        obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]] | None = None,
        dones: torch.Tensor | None = None,
        hist_encoding: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if dones is not None:
            self.reset(dones)
        policy_obs, obs_dict = self._resolve_inputs(obs_or_obs_dict, obs_dict)
        if self.device is not None:
            policy_obs = policy_obs.to(self.device)
        policy_obs = self.normalizer(policy_obs)
        actor_obs = self.algorithm.build_pie_actor_observations(policy_obs, obs_dict)
        return self.algorithm.policy.act_inference(actor_obs, hist_encoding=hist_encoding, **kwargs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.algorithm.reset_pie_actor_hidden(dones)

    @staticmethod
    def _resolve_inputs(
        obs_or_obs_dict: torch.Tensor | Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
        obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]]]:
        if obs_dict is None:
            if not isinstance(obs_or_obs_dict, Mapping):
                raise ValueError("obs_dict is required for PIE actor inference")
            obs_dict = obs_or_obs_dict
            policy_obs = obs_dict["policy"]
        else:
            if isinstance(obs_or_obs_dict, Mapping):
                policy_obs = obs_or_obs_dict["policy"]
            else:
                policy_obs = obs_or_obs_dict
        if not isinstance(policy_obs, torch.Tensor):
            raise TypeError("PIE policy observations must be a torch.Tensor")
        return policy_obs, obs_dict


class PPOWithExtractor(PPO):
    policy: ActorCriticRMA

    def __init__(
        self,
        policy,
        estimator,
        estimator_paras,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        priv_reg_coef_schedual = [0, 0, 0],
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            policy, 
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
            normalize_advantage_per_mini_batch,
            # RND parameters
            rnd_cfg,
            # Symmetry parameters
            symmetry_cfg,
            # Distributed training parameters
            multi_gpu_cfg,
            )

        self.estimator: nn.Module = estimator
        print(f"estimator MLP: {estimator}")

        self.uses_pie_estimator = estimator.__class__.__name__ == "PIEEstimator" or estimator_paras.pop(
            "use_pie_estimator_rollout", False
        )
        self.priv_states_dim = estimator_paras.get("num_priv_explicit", 0)
        self.num_prop = estimator_paras.get("num_prop", 0)
        self.num_scan = estimator_paras.get("num_scan", 0)
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras.get("train_with_estimated_states", False)
        self.pie_estimator_loss = PIEEstimatorLoss(weights=estimator_paras.get("loss_weights"))
        self.pie_estimator_storage: PIEEstimatorRolloutStorage | None = None
        self.use_pie_actor_features = bool(
            self.uses_pie_estimator and estimator_paras.get("use_pie_actor_features", False)
        )
        self.pie_actor_feature_keys = tuple(
            estimator_paras.get("pie_actor_feature_keys", ("z_m", "z_mu", "v_hat", "h_f_hat"))
        )
        self.detach_pie_actor_features = bool(estimator_paras.get("detach_pie_actor_features", True))
        self.pie_actor_feature_clip = estimator_paras.get("pie_actor_feature_clip")
        self.pie_joint_actor_estimator = bool(
            self.use_pie_actor_features and estimator_paras.get("pie_joint_actor_estimator", False)
        )
        self.pie_policy_obs_dim = int(estimator_paras.get("pie_policy_obs_dim", 45))
        self.pie_actor_estimator_grad_scale = float(estimator_paras.get("pie_actor_estimator_grad_scale", 1.0))
        self.pie_actor_rnn_hidden: torch.Tensor | None = None
        self.pie_train_gru_sequence = bool(estimator_paras.get("pie_train_gru_sequence", True))
        self.pie_estimator_num_learning_epochs = estimator_paras.get(
            "pie_num_learning_epochs", self.num_learning_epochs
        )
        self.pie_estimator_num_mini_batches = estimator_paras.get("pie_num_mini_batches", self.num_mini_batches)
        history_encoder = getattr(getattr(self.policy, "actor", None), "history_encoder", None)
        if history_encoder is not None:
            self.hist_encoder_optimizer = optim.Adam(history_encoder.parameters(), lr=learning_rate)
        else:
            self.hist_encoder_optimizer = None
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

    def build_pie_actor_observations(
        self,
        policy_obs: torch.Tensor,
        obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]] | None,
    ) -> torch.Tensor:
        """Append deterministic PIE estimator features to policy observations for the actor."""
        if not self.use_pie_actor_features:
            return policy_obs
        if obs_dict is None:
            raise ValueError("obs_dict is required when use_pie_actor_features=True")

        batch_size = policy_obs.shape[0]
        if self.pie_actor_rnn_hidden is None or self.pie_actor_rnn_hidden.shape[1] != batch_size:
            self.pie_actor_rnn_hidden = self.estimator.initial_hidden(batch_size, device=policy_obs.device)

        with torch.no_grad():
            predictions = self.estimator.forward_obs_dict(obs_dict, hidden_state=self.pie_actor_rnn_hidden)
            features = self._prepare_pie_actor_features(predictions)
            self.pie_actor_rnn_hidden = predictions["rnn_hidden"].detach()

        if self.detach_pie_actor_features:
            features = [feature.detach() for feature in features]
        return torch.cat((policy_obs, *features), dim=-1)

    def _prepare_pie_actor_features(self, predictions: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
        features = [predictions[key] for key in self.pie_actor_feature_keys]
        if self.pie_actor_feature_clip is None:
            return features
        clip = float(self.pie_actor_feature_clip)
        return [torch.clamp(feature, -clip, clip) for feature in features]

    def reset_pie_actor_hidden(self, dones: torch.Tensor | None = None) -> None:
        if self.pie_actor_rnn_hidden is None:
            return
        if dones is None:
            self.pie_actor_rnn_hidden = None
            return

        done_mask = dones.to(device=self.pie_actor_rnn_hidden.device).reshape(-1).bool()
        if done_mask.any():
            hidden = self.pie_actor_rnn_hidden.clone()
            hidden[:, done_mask, :] = 0.0
            self.pie_actor_rnn_hidden = hidden

    def act(self, obs, critic_obs, hist_encoding=False):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        if self.train_with_estimated_states and not self.uses_pie_estimator:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.policy.act(obs_est, hist_encoding).detach()
        else:
            try:
                self.transition.actions = self.policy.act(obs, hist_encoding).detach()
            except TypeError:
                self.transition.actions = self.policy.act(obs).detach()

        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs

        return self.transition.actions

    def init_pie_estimator_storage(self, num_envs: int, num_transitions_per_env: int):
        if not self.uses_pie_estimator:
            return
        self.pie_estimator_storage = PIEEstimatorRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            device=self.device,
        )

    def cache_pie_estimator_step(self, obs_dict) -> PIEEstimatorStepInput | None:
        if not self.uses_pie_estimator or self.pie_estimator_storage is None:
            return None
        return self.pie_estimator_storage.cache_step_input(obs_dict)

    def process_pie_estimator_env_step(
        self,
        step_input: PIEEstimatorStepInput | None,
        next_obs_dict,
        dones: torch.Tensor | None = None,
    ):
        if not self.uses_pie_estimator or self.pie_estimator_storage is None or step_input is None:
            return
        self.pie_estimator_storage.add_transition(step_input, next_obs_dict, dones=dones)

    def update_pie_estimator_from_storage(self) -> dict[str, float]:
        if not self.uses_pie_estimator or self.pie_estimator_storage is None or self.pie_estimator_storage.step == 0:
            return {}

        mean_losses: dict[str, float] = {}
        num_updates = 0
        if self.pie_train_gru_sequence:
            generator = self.pie_estimator_storage.sequence_mini_batch_generator(
                self.pie_estimator_num_mini_batches,
                self.pie_estimator_num_learning_epochs,
            )
            loss_fn = self._compute_pie_estimator_sequence_loss
        else:
            generator = self.pie_estimator_storage.mini_batch_generator(
                self.pie_estimator_num_mini_batches,
                self.pie_estimator_num_learning_epochs,
            )
            loss_fn = self._compute_pie_estimator_flat_loss

        for depth_batch, proprioception_history_batch, targets_batch, dones_batch in generator:
            losses = loss_fn(depth_batch, proprioception_history_batch, targets_batch, dones_batch)

            self.estimator_optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            for key, value in losses.items():
                mean_losses[key] = mean_losses.get(key, 0.0) + float(value.detach().cpu())
            num_updates += 1

        if num_updates == 0:
            return {}
        self.pie_estimator_storage.clear()
        self.reset_pie_actor_hidden()
        return {key: value / num_updates for key, value in mean_losses.items()}

    def _compute_pie_estimator_flat_loss(self, depth_batch, proprioception_history_batch, targets_batch, dones_batch):
        predictions = self.estimator(depth_batch, proprioception_history_batch)
        return self.pie_estimator_loss(predictions, targets_batch)

    def _compute_pie_estimator_sequence_loss(
        self,
        depth_sequence: torch.Tensor,
        proprioception_history_sequence: torch.Tensor,
        targets_sequence: Mapping[str, torch.Tensor],
        dones_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sequence_length = depth_sequence.shape[0]
        hidden_state = None
        loss_sums: dict[str, torch.Tensor] = {}

        for step_idx in range(sequence_length):
            predictions = self.estimator(
                depth_sequence[step_idx],
                proprioception_history_sequence[step_idx],
                hidden_state=hidden_state,
            )
            losses = self.pie_estimator_loss(
                predictions,
                {key: value[step_idx] for key, value in targets_sequence.items()},
            )
            for key, value in losses.items():
                loss_sums[key] = loss_sums.get(key, value.new_zeros(())) + value

            hidden_state = predictions["rnn_hidden"]
            done_mask = dones_sequence[step_idx].to(device=hidden_state.device).reshape(1, -1, 1).bool()
            if done_mask.any():
                hidden_state = hidden_state * (~done_mask).to(dtype=hidden_state.dtype)

        return {key: value / sequence_length for key, value in loss_sums.items()}

    def _pie_joint_actor_estimator_mini_batch_generator(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            tuple[None, None],
            None,
            torch.Tensor | None,
        ]
    ]:
        """Yield PPO minibatches with actor observations recomputed through the PIE estimator.

        The rollout stores actor observations produced under inference mode. For
        actor-estimator joint training, PPO must rebuild the estimator feature
        suffix with gradients enabled while preserving the estimator GRU time order.
        """
        if self.policy.is_recurrent:
            raise NotImplementedError("PIE actor-estimator joint training currently expects a feed-forward policy.")
        if self.pie_estimator_storage is None or self.pie_estimator_storage.step == 0:
            raise RuntimeError("PIE joint actor-estimator update requires non-empty PIE estimator storage.")
        if self.storage.step != self.pie_estimator_storage.step:
            raise RuntimeError(
                "PPO storage and PIE estimator storage are misaligned: "
                f"{self.storage.step} PPO steps vs {self.pie_estimator_storage.step} PIE steps."
            )
        if self.symmetry and self.symmetry.get("use_data_augmentation", False):
            raise NotImplementedError("PIE joint actor-estimator update does not support symmetry augmentation yet.")

        num_steps = self.pie_estimator_storage.step
        num_chunks = min(self.num_mini_batches, self.storage.num_envs)
        if num_chunks <= 0:
            raise ValueError(f"num_mini_batches must be positive, got {self.num_mini_batches}")

        depth, proprioception_history, _, dones = self.pie_estimator_storage.get()
        critic_observations = (
            self.storage.privileged_observations[:num_steps]
            if self.storage.privileged_observations is not None
            else self.storage.observations[:num_steps]
        )

        for _ in range(self.num_learning_epochs):
            env_indices = torch.randperm(self.storage.num_envs, device=self.device)
            for env_batch_idx in torch.chunk(env_indices, num_chunks):
                if env_batch_idx.numel() == 0:
                    continue

                stored_actor_obs = self.storage.observations[:num_steps, env_batch_idx]
                policy_obs_sequence = stored_actor_obs[..., : self.pie_policy_obs_dim]
                obs_batch = self._recompute_pie_actor_observation_sequence(
                    policy_obs_sequence,
                    depth[:, env_batch_idx],
                    proprioception_history[:, env_batch_idx],
                    dones[:, env_batch_idx],
                ).flatten(0, 1)

                rnd_state_batch = None
                if self.storage.rnd_state_shape is not None:
                    rnd_state_batch = self.storage.rnd_state[:num_steps, env_batch_idx].flatten(0, 1)

                yield (
                    obs_batch,
                    critic_observations[:, env_batch_idx].flatten(0, 1),
                    self.storage.actions[:num_steps, env_batch_idx].flatten(0, 1),
                    self.storage.values[:num_steps, env_batch_idx].flatten(0, 1),
                    self.storage.advantages[:num_steps, env_batch_idx].flatten(0, 1),
                    self.storage.returns[:num_steps, env_batch_idx].flatten(0, 1),
                    self.storage.actions_log_prob[:num_steps, env_batch_idx].flatten(0, 1),
                    self.storage.mu[:num_steps, env_batch_idx].flatten(0, 1),
                    self.storage.sigma[:num_steps, env_batch_idx].flatten(0, 1),
                    (None, None),
                    None,
                    rnd_state_batch,
                )

    def _recompute_pie_actor_observation_sequence(
        self,
        policy_obs_sequence: torch.Tensor,
        depth_sequence: torch.Tensor,
        proprioception_history_sequence: torch.Tensor,
        dones_sequence: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = policy_obs_sequence.shape[0]
        hidden_state = None
        actor_obs_steps = []

        for step_idx in range(sequence_length):
            predictions = self.estimator(
                depth_sequence[step_idx],
                proprioception_history_sequence[step_idx],
                hidden_state=hidden_state,
            )
            features = self._prepare_pie_actor_features(predictions)
            if self.pie_actor_estimator_grad_scale != 1.0:
                scale = self.pie_actor_estimator_grad_scale
                features = [feature.detach() + scale * (feature - feature.detach()) for feature in features]
            actor_obs_steps.append(torch.cat((policy_obs_sequence[step_idx], *features), dim=-1))

            hidden_state = predictions["rnn_hidden"]
            done_mask = dones_sequence[step_idx].to(device=hidden_state.device).reshape(1, -1, 1).bool()
            if done_mask.any():
                hidden_state = hidden_state * (~done_mask).to(dtype=hidden_state.dtype)

        return torch.stack(actor_obs_steps, dim=0)
    

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_priv_reg_loss = 0
        mean_entropy = 0
        mean_estimator_loss = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        num_updates = 0

        # generator for mini batches
        if self.pie_joint_actor_estimator:
            generator = self._pie_joint_actor_estimator_mini_batch_generator()
        elif self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.uses_pie_estimator:
                priv_reg_loss = obs_batch.new_tensor(0.0)
                priv_reg_coef = 0.0
            else:
                priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
                with torch.inference_mode():
                    hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

            # Estimator
            if self.uses_pie_estimator:
                estimator_loss = obs_batch.new_tensor(0.0)
            else:
                priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])  # obs in batch is with true priv_states
                estimator_loss = (priv_states_predicted - obs_batch[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim]).pow(2).mean()
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + \
                self.value_loss_coef * value_loss -\
                self.entropy_coef * entropy_batch.mean() + \
                priv_reg_coef * priv_reg_loss

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)


            self.optimizer.zero_grad()
            if self.pie_joint_actor_estimator:
                self.estimator_optimizer.zero_grad()
            loss.backward()

            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if self.pie_joint_actor_estimator:
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.pie_joint_actor_estimator:
                self.estimator_optimizer.step()

            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.mean().item()
            mean_estimator_loss += estimator_loss.item()

            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            num_updates += 1

        if num_updates == 0:
            raise RuntimeError("PPO update produced no minibatches.")
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_entropy /= num_updates
        if self.uses_pie_estimator:
            mean_estimator_loss = 0.0
            pie_estimator_losses = self.update_pie_estimator_from_storage()
            if pie_estimator_losses:
                mean_estimator_loss = pie_estimator_losses["loss"]
        else:
            pie_estimator_losses = {}
            mean_estimator_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()
        self.update_counter()
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "priv_reg": mean_priv_reg_loss,
            "entropy": mean_entropy,
            'estimator':mean_estimator_loss,
            'priv_reg_coef': priv_reg_coef
        }
        for key, value in pie_estimator_losses.items():
            loss_dict[f"pie_estimator/{key}"] = value
        if self.uses_pie_estimator:
            loss_dict["pie_actor_estimator_joint"] = float(self.pie_joint_actor_estimator)
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def reduce_parameters(self):
        if not self.pie_joint_actor_estimator:
            return super().reduce_parameters()

        parameters = list(self.policy.parameters()) + list(self.estimator.parameters())
        if self.rnd:
            parameters += list(self.rnd.parameters())

        grads = [param.grad.view(-1) for param in parameters if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in parameters:
            if param.grad is None:
                continue
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel

    def broadcast_parameters(self):
        if not self.pie_joint_actor_estimator:
            return super().broadcast_parameters()

        model_params = [self.policy.state_dict(), self.estimator.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.estimator.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def update_counter(self):
        self.counter += 1

    def update_dagger(self):
        if self.uses_pie_estimator or self.hist_encoder_optimizer is None:
            return 0.0
        mean_hist_latent_loss = 0
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            with torch.inference_mode():
                self.policy.act(obs_batch, 
                                hist_encoding=True, 
                                masks=masks_batch, 
                                hidden_states=hid_states_batch[0])

            # Adaptation module update
            with torch.inference_mode():
                priv_latent_batch = self.policy.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.policy.actor.infer_hist_latent(obs_batch)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss
