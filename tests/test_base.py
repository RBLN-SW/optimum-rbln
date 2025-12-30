import inspect
import os
import random
import shutil
import tempfile
import unittest
from enum import Enum
from typing import Iterable

import transformers
from diffusers import DiffusionPipeline
from transformers import AutoConfig, CLIPConfig

from optimum.rbln import __version__
from optimum.rbln.configuration_utils import ContextRblnConfig


def test_version_is_str():
    assert isinstance(__version__, str)


class TestLevel(Enum):
    ESSENTIAL = 1
    DEFAULT = 2
    FULL = 3
    DISABLED = 999
    UNKNOWN = -1


def require_hf_user_id(test_case):
    """
    Decorator marking a test that requires huggingface hub user id.
    """
    user_id = os.environ.get("HF_USER_ID", None)
    if not user_id:
        return unittest.skip("test requires hf token as `HF_USER_ID` environment variable")(test_case)
    else:
        return test_case


def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    use_auth_token = os.environ.get("HF_AUTH_TOKEN", None)
    if not use_auth_token:
        return unittest.skip("test requires hf token as `HF_AUTH_TOKEN` environment variable")(test_case)
    else:
        return test_case


class BaseHubTest:
    class TestHub(unittest.TestCase):
        @require_hf_token
        @require_hf_user_id
        def test_push_to_hub(self):
            """
            "HF_AUTH_TOKEN" should be set to execute this.
            """
            with tempfile.TemporaryDirectory() as tmpdirname:
                # create remote hash to check if file was updated.

                remote_hash = random.getrandbits(128)
                HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)
                HF_USER_ID = os.environ.get("HF_USER_ID", None)

                self.assertTrue(HF_AUTH_TOKEN)
                self.assertTrue(HF_USER_ID)
                TOKEN_KEY = "token"
                REPO_KEY = "repo_id"

                if self.is_diffuser():
                    self.model.text_encoder.config.from_local = remote_hash
                else:
                    self.model.config.from_local = remote_hash

                self.model.save_pretrained(
                    tmpdirname,
                    push_to_hub=True,
                    private=True,
                    **{
                        TOKEN_KEY: HF_AUTH_TOKEN,
                        REPO_KEY: f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                    },
                )

                # If our tests were moved to a public rather than a private repository,
                # this logic could be as simple as downloading the config file directly
                # and comparing it.
                if self.is_diffuser():
                    cfg = CLIPConfig.from_pretrained(
                        f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                        subfolder="text_encoder",
                        private=True,
                        **{TOKEN_KEY: HF_AUTH_TOKEN},
                    )
                else:
                    cfg = AutoConfig.from_pretrained(
                        f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                        private=True,
                        **{TOKEN_KEY: HF_AUTH_TOKEN},
                    )

                self.assertEqual(remote_hash, cfg.from_local)

        @require_hf_token
        @require_hf_user_id
        def test_pull_compiled_model_from_hub(self):
            HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)
            HF_USER_ID = os.environ.get("HF_USER_ID", None)

            _ = self.RBLN_CLASS.from_pretrained(
                f"{HF_USER_ID}/{self.get_hf_remote_dir()}",
                **self.HF_CONFIG_KWARGS,
                rbln_device=self.DEVICE,
                token=HF_AUTH_TOKEN,
            )


class BaseTest:
    """
    Base Class for other models.

    You should specify class attributes : RBLN_CLASS, HF_MODEL_ID
    """

    class TestModel(unittest.TestCase):
        RBLN_AUTO_CLASS = None
        RBLN_CLASS = None
        HF_MODEL_ID = None
        RBLN_CLASS_KWARGS = {}
        GENERATION_KWARGS = {}
        HF_CONFIG_KWARGS = {}
        EXPECTED_OUTPUT = None
        TEST_LEVEL = TestLevel.DEFAULT
        DEVICE = -1  # -1 indicates dummy device

        @classmethod
        def setUpClass(cls):
            env_coverage = os.environ.get("OPTIMUM_RBLN_TEST_LEVEL", "default")
            env_coverage = TestLevel[env_coverage.upper()]
            if env_coverage.value < cls.TEST_LEVEL.value:
                raise unittest.SkipTest(f"Skipped test : Test Coverage {env_coverage.name} < {cls.TEST_LEVEL.name}")

            REUSE_ARTIFACTS_PATH = os.environ.get("REUSE_ARTIFACTS_PATH", None)
            if REUSE_ARTIFACTS_PATH is None:
                if os.path.exists(cls.get_rbln_local_dir()):
                    shutil.rmtree(cls.get_rbln_local_dir())

                cls.model = cls.RBLN_CLASS.from_pretrained(
                    cls.HF_MODEL_ID,
                    model_save_dir=cls.get_rbln_local_dir(),
                    rbln_device=cls.DEVICE,
                    **cls.RBLN_CLASS_KWARGS,
                    **cls.HF_CONFIG_KWARGS,
                )
            else:
                if os.path.exists(REUSE_ARTIFACTS_PATH):
                    compiled_model_path = os.path.join(REUSE_ARTIFACTS_PATH, cls.get_rbln_local_dir())
                    if os.path.exists(compiled_model_path):
                        with ContextRblnConfig(device=-1):
                            cls.model = cls.RBLN_CLASS.from_pretrained(compiled_model_path)
                if not hasattr(cls, "model"):
                    raise unittest.SkipTest("Compiled model not found")

        @classmethod
        def get_rbln_local_dir(cls):
            return os.path.basename(cls.__module__.split(".")[-1]) + "_" + os.path.basename(cls.__name__) + "-artifact"

        @classmethod
        def get_hf_auto_class(cls):
            if cls.RBLN_AUTO_CLASS is not None:
                return getattr(transformers, cls.RBLN_AUTO_CLASS.__name__[4:])
            return None

        @classmethod
        def get_hf_remote_dir(cls):
            return "rbln-" + os.path.basename(cls.HF_MODEL_ID)

        def is_diffuser(self):
            # Note that This is only True when it is a pipeline, not model (i.e. AutoEncoderKL)
            return isinstance(self.model, DiffusionPipeline)

        @classmethod
        def tearDownClass(cls):
            if os.path.exists(cls.get_rbln_local_dir()):
                shutil.rmtree(cls.get_rbln_local_dir())

        # BC: Test save_artifacts and copy tree is successful
        def test_save_artifacts(self):
            SAVE_ARTIFACTS_PATH = os.environ.get("SAVE_ARTIFACTS_PATH", None)
            if SAVE_ARTIFACTS_PATH is None:
                return
            else:
                os.makedirs(SAVE_ARTIFACTS_PATH, exist_ok=True)
                saved_path = os.path.join(SAVE_ARTIFACTS_PATH, self.get_rbln_local_dir())
                shutil.copytree(self.get_rbln_local_dir(), saved_path, dirs_exist_ok=True)

                with ContextRblnConfig(create_runtimes=False):
                    _ = self.RBLN_CLASS.from_pretrained(
                        saved_path,
                        **self.HF_CONFIG_KWARGS,
                    )

        def test_model_save_dir(self):
            self.assertTrue(os.path.exists(self.get_rbln_local_dir()), "model_save_dir does not work.")

        def get_inputs(self):
            return self.GENERATION_KWARGS

        def postprocess(self, inputs, output):
            return output

        def test_generate(self):
            inputs = self.get_inputs()
            if self.is_diffuser():
                output = self.model(**inputs)[0]
            else:
                if self.model.can_generate():
                    output = self.model.generate(**inputs)
                else:
                    # encoder-only, resnet, etc..
                    output = self.model(**inputs)[0]

            output = self.postprocess(inputs, output)
            REUSE_ARTIFACTS_PATH = os.environ.get("REUSE_ARTIFACTS_PATH", None)
            if self.EXPECTED_OUTPUT and self.DEVICE is None and REUSE_ARTIFACTS_PATH is None:
                from simphile import jaccard_similarity

                if isinstance(self.EXPECTED_OUTPUT, str):
                    similarity = jaccard_similarity(output, self.EXPECTED_OUTPUT)
                    self.assertGreater(
                        similarity, 0.9, msg=f"self.EXPECTED_OUTPUT: {self.EXPECTED_OUTPUT}, output: {output}"
                    )
                else:
                    for o, e_o in zip(output, self.EXPECTED_OUTPUT):
                        similarity = jaccard_similarity(o, e_o)
                        self.assertGreater(
                            similarity, 0.9, msg=f"self.EXPECTED_OUTPUT: {self.EXPECTED_OUTPUT}, output: {output}"
                        )

        def _inner_test_save_load(self, tmpdir):
            with ContextRblnConfig(create_runtimes=False):
                with self.subTest():
                    self.model.save_pretrained(tmpdir)
                    config_path = os.path.join(tmpdir, self.RBLN_CLASS.config_name)
                    self.assertTrue(os.path.exists(config_path), "save_pretrained does not work.")

                with self.subTest():
                    # Test load
                    _ = self.RBLN_CLASS.from_pretrained(
                        tmpdir,
                        **self.HF_CONFIG_KWARGS,
                    )

                with self.subTest():
                    # Test saving from exported pipe
                    self.model.save_pretrained(tmpdir)
                    _ = self.RBLN_CLASS.from_pretrained(
                        tmpdir,
                        rbln_create_runtimes=False,
                        **self.HF_CONFIG_KWARGS,
                    )

        def test_save_load(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                self._inner_test_save_load(tmpdir)

        def test_model_save_dir_load(self):
            rbln_local_dir = self.get_rbln_local_dir()
            with ContextRblnConfig(create_runtimes=False):
                # Test model_save_dir
                _ = self.RBLN_CLASS.from_pretrained(
                    rbln_local_dir,
                    rbln_create_runtimes=False,
                    **self.HF_CONFIG_KWARGS,
                )

        def test_automap(self):
            if self.RBLN_AUTO_CLASS is None:
                self.skipTest("Skipping test because RBLN_AUTO_CLASS is None")

            if isinstance(self.RBLN_AUTO_CLASS, Iterable):
                for auto_class in self.RBLN_AUTO_CLASS:
                    assert self.RBLN_CLASS == auto_class.get_rbln_cls(
                        self.HF_MODEL_ID,
                        **self.RBLN_CLASS_KWARGS,
                        **self.HF_CONFIG_KWARGS,
                    )
            else:
                assert self.RBLN_CLASS == self.RBLN_AUTO_CLASS.get_rbln_cls(
                    self.HF_MODEL_ID,
                    **self.RBLN_CLASS_KWARGS,
                    **self.HF_CONFIG_KWARGS,
                )

        # check if this use a pipeline
        def test_infer_framework(self):
            class_hierarchy = inspect.getmro(self.model.__class__)

            is_valid_framework = (
                any(base_class.__name__ == "PreTrainedModel" for base_class in class_hierarchy)
                if any("transformers" in str(base_class) for base_class in class_hierarchy)
                else True
            )

            assert is_valid_framework, "Model does not inherit from PreTrainedModel."

        def test_get_rbln_config_class(self):
            assert self.RBLN_CLASS.get_rbln_config_class() is not None
            rbln_config_class_name = self.RBLN_CLASS.get_rbln_config_class().__name__
            assert self.RBLN_CLASS.__name__ == rbln_config_class_name[:-6]


class DisallowedTestBase:
    class DisallowedTest(unittest.TestCase):
        RBLN_CLASS = None
        HF_MODEL_ID = None
        RBLN_CLASS_KWARGS = {}
        GENERATION_KWARGS = {}
        HF_CONFIG_KWARGS = {}
        TEST_LEVEL = TestLevel.DEFAULT

        @classmethod
        def setUpClass(cls):
            env_coverage = os.environ.get("OPTIMUM_RBLN_TEST_LEVEL", "default")
            env_coverage = TestLevel[env_coverage.upper()]
            if env_coverage.value < cls.TEST_LEVEL.value:
                raise unittest.SkipTest(f"Skipped test : Test Coverage {env_coverage.name} < {cls.TEST_LEVEL.name}")

        def test_load(self):
            try:
                _ = self.RBLN_CLASS.from_pretrained(
                    self.HF_MODEL_ID,
                    model_save_dir=self.get_rbln_local_dir(),
                    **self.RBLN_CLASS_KWARGS,
                    **self.HF_CONFIG_KWARGS,
                )

                self.assertTrue(False, "This should be disallowed.")

            except ValueError:
                pass

            finally:
                if os.path.exists(self.get_rbln_local_dir()):
                    shutil.rmtree(self.get_rbln_local_dir())

        @classmethod
        def get_rbln_local_dir(cls):
            return os.path.basename(cls.HF_MODEL_ID) + "-local"
