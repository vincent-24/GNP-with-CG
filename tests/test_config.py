"""Tests for configuration module."""

import pytest


class TestConfig:
    """Tests for GNP configuration."""

    def test_config_imports(self):
        """Config module should import without errors."""
        from GNP import config
        assert config is not None

    def test_required_constants_exist(self):
        """Check that all required config constants exist."""
        from GNP import config

        required = [
            'SEED',
            'BATCH_SIZE',
            'EPOCHS',
            'LEARNING_RATE',
            'WEIGHT_DECAY',
            'TRAIN_VAL_SPLIT',
            'GRAD_CLIP_NORM',
            'NUM_LAYERS',
            'EMBED_DIM',
            'HIDDEN_DIM',
            'DROP_RATE',
            'NUM_LEVELS',
            'LINEAR_MGGNN_NUM_VCYCLES',
            'LINEAR_MGGNN_SMOOTHER_K',
            'LINEAR_MGGNN_COARSEST_K',
            'LINEAR_MGGNN_SHARE_SMOOTHERS',
            'MAX_ITERS',
            'TOLERANCE',
        ]

        for const in required:
            assert hasattr(config, const), f"Missing config constant: {const}"

    def test_train_val_split_valid(self):
        """TRAIN_VAL_SPLIT should be between 0 and 1."""
        from GNP import config

        assert 0 < config.TRAIN_VAL_SPLIT < 1, \
            f"TRAIN_VAL_SPLIT should be in (0, 1), got {config.TRAIN_VAL_SPLIT}"

    def test_solver_config_valid(self):
        """SOLVER_CONFIG should have expected solvers."""
        from GNP import config

        assert 'PCG' in config.SOLVER_CONFIG
        assert 'FCG' in config.SOLVER_CONFIG
        assert 'GMRES' in config.SOLVER_CONFIG

        # Each solver config should have required keys
        for name, cfg in config.SOLVER_CONFIG.items():
            assert 'solver_cls' in cfg, f"Missing solver_cls in {name}"
            assert 'default_net' in cfg, f"Missing default_net in {name}"


class TestFactory:
    """Tests for factory module."""

    def test_get_solver_class(self):
        """get_solver_class should return correct solver classes."""
        from GNP.factory import get_solver_class

        pcg = get_solver_class('PCG')
        assert pcg.__name__ == 'PCG'

        fcg = get_solver_class('FCG')
        assert fcg.__name__ == 'FCG'

        gmres = get_solver_class('GMRES')
        assert gmres.__name__ == 'GMRES'

    def test_get_solver_class_invalid(self):
        """get_solver_class should raise for unknown solvers."""
        from GNP.factory import get_solver_class

        with pytest.raises(ValueError,match="Unknown solver"):
            get_solver_class('InvalidSolver')

    def test_get_network_class(self):
        """get_network_class should return correct network classes."""
        from GNP.factory import get_network_class

        linear_mggnn = get_network_class('LinearMGGNN')
        assert linear_mggnn.__name__ == 'LinearMGGNN'

        resgcn = get_network_class('ResGCN')
        assert resgcn.__name__ == 'ResGCN'

    def test_get_network_class_invalid(self):
        """get_network_class should raise for unknown networks."""
        from GNP.factory import get_network_class

        with pytest.raises(ValueError, match="Unknown network"):
            get_network_class('InvalidNetwork')
