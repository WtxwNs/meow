"""CPU regressions for first-party MEow code, without simulator dependencies.

Run from the repository root: python -m unittest discover -s tests -v
Only NumPy and PyTorch are required. Environment, replay, plotting and logging
integrations are replaced; policy math and optimizers run with real PyTorch.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


@contextlib.contextmanager
def temporary_modules(dependencies):
    # Restore only substituted modules: discarding unrelated imports can break
    # PyTorch extensions that register native operators exactly once.
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in dependencies}
    sys.modules.update(dependencies)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def load_source(name, relative_path, dependencies):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    loaded = importlib.util.module_from_spec(spec)
    with temporary_modules({**dependencies, name: loaded}):
        spec.loader.exec_module(loaded)
    return loaded


class Box:
    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.dtype = np.float32

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class Environment:
    """One-step-free vector environment with no external simulator."""
    def __init__(self):
        self.num_envs = 1
        self.unwrapped = self
        self.single_action_space = Box((2,))
        self.single_observation_space = Box((2,))
        self.action_space = Box((1, 2))
        self.observation_space = Box((1, 2))

    def reset(self, **kwargs):
        return np.zeros((1, 2), dtype=np.float32), {}

    def step(self, actions):
        return (np.zeros((1, 2), dtype=np.float32), np.ones(1),
                np.array([False]), np.array([False]), {})

    def close(self):
        pass


class ToyCheckpointTests(unittest.TestCase):
    def test_plotting_cannot_replace_best_evaluated_checkpoint(self):
        utils = module("modules.utils", log=Mock(),
                       plot_traj_multigoal=Mock(return_value=([], [])),
                       plot_value=Mock())
        training = load_source("toy_training_regression", "toy/modules/train_loop.py",
                               {"modules": module("modules"), "modules.utils": utils})
        agent = Mock()
        agent.act.return_value = (torch.zeros((1, 2)), None)
        # A new model version is produced on every update.
        agent.policy.version = 0
        agent.update.side_effect = lambda: setattr(agent.policy, "version", agent.policy.version + 1) or {}
        saved = {}
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(seed=0, steps=3, warmup_steps=0,
                                         eval_every=1, plot_every=1, deterministic=True,
                                         save_path=directory)
            def save(policy, path):
                saved[Path(path).name] = policy.version
            with patch.object(training, "evaluate", side_effect=[10, 5, 20, 2]), \
                    patch.object(training.torch, "save", side_effect=save), \
                    contextlib.redirect_stdout(io.StringIO()):
                training.train_loop(agent, args, Mock(), Environment(), Environment(), Mock())
        self.assertEqual(saved["best.pt"], 3)
        self.assertEqual(saved["last.pt"], 4)
        self.assertEqual(utils.plot_value.call_count, 4)

    def test_plotting_without_evaluation_does_not_create_best_checkpoint(self):
        utils = module("modules.utils", log=Mock(),
                       plot_traj_multigoal=Mock(return_value=([], [])), plot_value=Mock())
        training = load_source("toy_no_eval_regression", "toy/modules/train_loop.py",
                               {"modules": module("modules"), "modules.utils": utils})
        agent = Mock()
        agent.act.return_value = (torch.zeros((1, 2)), None)
        agent.update.return_value = {}
        args = types.SimpleNamespace(seed=0, steps=1, warmup_steps=1,
                                     eval_every=2, plot_every=1, deterministic=True,
                                     save_path="unused")
        with patch.object(training.torch, "save") as save, \
                contextlib.redirect_stdout(io.StringIO()):
            training.train_loop(agent, args, Mock(), Environment(), Environment(), Mock())
        self.assertEqual([Path(call.args[1]).name for call in save.call_args_list], ["last.pt"])


class ToyPlotDeviceTests(unittest.TestCase):
    def test_value_plot_uses_the_policy_device(self):
        pyplot = module("matplotlib.pyplot", pcolormesh=Mock(), xlabel=Mock(),
                        ylabel=Mock(), savefig=Mock())
        utils = load_source("toy_plot_regression", "toy/modules/utils.py", {
            "matplotlib": module("matplotlib", pyplot=pyplot),
            "matplotlib.pyplot": pyplot,
            "gymnasium": module("gymnasium"),
        })
        observed_devices = []
        def get_v(obs):
            observed_devices.append(obs.device)
            return torch.zeros((len(obs), 1), device=obs.device)
        policy = types.SimpleNamespace(device=torch.device("cpu"), eval=Mock(), get_v=get_v)
        utils.plot_value(policy, "unused")
        self.assertEqual(observed_devices, [policy.device])
        pyplot.savefig.assert_called_once_with("unused.png", bbox_inches="tight")


class ReplayBuffer:
    def __init__(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        pass

    def sample(self, count):
        return types.SimpleNamespace(
            observations=torch.zeros((count, 2)), next_observations=torch.ones((count, 2)) * 0.1,
            actions=torch.zeros((count, 2)), rewards=torch.ones((count, 1)),
            dones=torch.zeros((count, 1)))


class CleanRLEntropyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gym = module("gymnasium", spaces=types.SimpleNamespace(Box=Box),
                         vector=types.SimpleNamespace(SyncVectorEnv=lambda _: Environment()),
                         make_vec=lambda *args, **kwargs: Environment(),
                         wrappers=types.SimpleNamespace(RescaleAction=lambda env, **kwargs: env))
        cls.sb3 = module("stable_baselines3", __version__="2.0.0")
        dependencies = {
            "gymnasium": cls.gym,
            "tyro": module("tyro"),
            "stable_baselines3": cls.sb3,
            "stable_baselines3.common": module("stable_baselines3.common"),
            "stable_baselines3.common.buffers": module("stable_baselines3.common.buffers", ReplayBuffer=ReplayBuffer),
            "torch.utils.tensorboard": module("torch.utils.tensorboard", SummaryWriter=lambda *args, **kwargs: Mock()),
        }
        with patch.object(sys, "path", [str(ROOT / "cleanrl"), *sys.path]):
            cls.training = load_source("cleanrl_meow_regression", "cleanrl/cleanrl/meow_continuous_action.py", dependencies)

    def run_training(self, autotune, total_timesteps):
        training = self.training
        policies = []
        class RecordingPolicy(training.FlowPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                policies.append(self)
        args = training.Args(cuda=False, track=False, capture_video=False, autotune=autotune,
                             total_timesteps=total_timesteps, learning_starts=0,
                             batch_size=2, buffer_size=8, policy_frequency=1, alpha=0.2)
        with temporary_modules({"stable_baselines3": self.sb3}), \
                patch.object(training, "FlowPolicy", RecordingPolicy):
            training.train(args)
        return policies

    def test_autotune_initial_temperature_reaches_both_policies(self):
        policy, target = self.run_training(autotune=True, total_timesteps=0)
        self.assertEqual(policy.alpha, 1.0)
        self.assertEqual(target.alpha, 1.0)

    def test_autotune_updates_temperature_in_both_policies(self):
        policy, target = self.run_training(autotune=True, total_timesteps=3)
        self.assertNotEqual(policy.alpha, 0.2)
        self.assertNotEqual(policy.alpha, 1.0)
        self.assertEqual(policy.alpha, target.alpha)
        self.assertTrue(np.isfinite(policy.alpha))

    def test_fixed_temperature_stays_unchanged(self):
        policy, target = self.run_training(autotune=False, total_timesteps=3)
        self.assertEqual(policy.alpha, 0.2)
        self.assertEqual(target.alpha, 0.2)

    def test_policy_q_and_v_use_current_temperature(self):
        policy = self.training.FlowPolicy(0.2, -0.3, -5, 2, 2, "cpu").eval()
        obs, act = torch.zeros((4, 2)), torch.zeros((4, 2))
        with torch.no_grad():
            q_before, v_before = policy.get_qv(obs, act)
            policy.alpha = 0.4
            q_after, v_after = policy.get_qv(obs, act)
            self.assertTrue(torch.allclose(q_after, 2 * q_before))
            self.assertTrue(torch.allclose(v_after, 2 * v_before))
            self.assertTrue(torch.allclose(policy.get_v(obs), v_after))


class SKRLInitializationTests(unittest.TestCase):
    def test_target_copies_online_weights_and_remains_frozen(self):
        gym = module("gym", Space=object)
        gymnasium = module("gymnasium", Space=object)
        model_source = load_source("skrl_base_model_regression", "skrl/skrl/models/torch/base.py", {
            "gym": gym, "gymnasium": gymnasium, "skrl": module("skrl", logger=Mock())})
        class Policy(model_source.Model):
            def __init__(self, value):
                super().__init__(2, 2, "cpu")
                self.weight = torch.nn.Parameter(torch.full((2, 2), value))
        class Agent:
            def __init__(self, models, cfg, **kwargs):
                self.models, self.cfg = models, cfg
                self.checkpoint_modules = {}
            @staticmethod
            def _empty_preprocessor(value, **kwargs):
                return value
        meow = load_source("skrl_meow_regression", "skrl/skrl/agents/torch/meow/meow.py", {
            "gym": gym, "gymnasium": gymnasium,
            "skrl.agents.torch": module("skrl.agents.torch", Agent=Agent),
            "skrl.memories.torch": module("skrl.memories.torch", Memory=object),
            "skrl.models.torch": module("skrl.models.torch", Model=model_source.Model),
        })
        policy, target = Policy(2.0), Policy(-3.0)
        meow.MEow(models={"policy": policy, "target_policy": target}, device="cpu")
        self.assertTrue(torch.equal(target.weight, policy.weight))
        self.assertFalse(target.weight.requires_grad)
        self.assertTrue(policy.weight.requires_grad)
        self.assertNotEqual(target.weight.data_ptr(), policy.weight.data_ptr())


if __name__ == "__main__":
    unittest.main()
