# test_api_imports.py

"""Test that all public API functions and classes are importable and visible."""

import linox


class TestPublicAPI:
    """Test that all items in __all__ are importable from linox."""

    def test_all_items_exist(self):
        """Test that all items declared in __all__ exist in the module."""
        for name in linox.__all__:
            assert hasattr(linox, name), f"{name} not found in linox module"

    def test_all_items_importable(self):
        """Test that all items can be imported from linox."""
        for name in linox.__all__:
            obj = getattr(linox, name)
            assert obj is not None, f"{name} is None"

    def test_classes_importable(self):
        """Test that all linear operator classes are importable."""
        from linox import (
            AddLinearOperator,
            ArrayKernel,
            BlockDiagonal,
            BlockMatrix,
            BlockMatrix2x2,
            Diagonal,
            EigenD,
            Identity,
            InverseLinearOperator,
            IsotropicAdditiveLinearOperator,
            IsotropicScalingPlusSymmetricLowRank,
            Kronecker,
            LinearOperator,
            LowRank,
            Matrix,
            Ones,
            Permutation,
            PositiveDiagonalPlusSymmetricLowRank,
            ProductLinearOperator,
            PseudoInverseLinearOperator,
            Scalar,
            ScaledLinearOperator,
            SymmetricLowRank,
            Toeplitz,
            TransposedLinearOperator,
            Zero,
        )

        # Verify they are classes
        assert isinstance(LinearOperator, type)
        assert isinstance(Matrix, type)
        assert isinstance(Identity, type)
        assert isinstance(Diagonal, type)
        assert isinstance(Scalar, type)
        assert isinstance(Zero, type)
        assert isinstance(Ones, type)
        assert isinstance(BlockMatrix, type)
        assert isinstance(BlockMatrix2x2, type)
        assert isinstance(BlockDiagonal, type)
        assert isinstance(LowRank, type)
        assert isinstance(SymmetricLowRank, type)
        assert isinstance(IsotropicScalingPlusSymmetricLowRank, type)
        assert isinstance(PositiveDiagonalPlusSymmetricLowRank, type)
        assert isinstance(Kronecker, type)
        assert isinstance(Permutation, type)
        assert isinstance(EigenD, type)
        assert isinstance(AddLinearOperator, type)
        assert isinstance(ProductLinearOperator, type)
        assert isinstance(ScaledLinearOperator, type)
        assert isinstance(TransposedLinearOperator, type)
        assert isinstance(InverseLinearOperator, type)
        assert isinstance(PseudoInverseLinearOperator, type)
        assert isinstance(ArrayKernel, type)
        assert isinstance(Toeplitz, type)
        assert isinstance(IsotropicAdditiveLinearOperator, type)

    def test_functions_importable(self):
        """Test that all linear operator functions are importable."""
        from linox import (
            congruence_transform,
            diagonal,
            is_square,
            kron,
            lcholesky,
            ldet,
            leigh,
            linverse,
            lpinverse,
            lpsolve,
            lqr,
            lsolve,
            lsqrt,
            slogdet,
            svd,
            symmetrize,
            transpose,
        )

        # Verify they are callable
        assert callable(linverse)
        assert callable(lsqrt)
        assert callable(transpose)
        assert callable(is_square)
        assert callable(congruence_transform)
        assert callable(diagonal)
        assert callable(symmetrize)
        assert callable(kron)
        assert callable(lcholesky)
        assert callable(ldet)
        assert callable(leigh)
        assert callable(lpinverse)
        assert callable(lpsolve)
        assert callable(lqr)
        assert callable(lsolve)
        assert callable(slogdet)
        assert callable(svd)

    def test_utility_functions_importable(self):
        """Test that utility functions are importable."""
        from linox import allclose, todense

        assert callable(allclose)
        assert callable(todense)

    def test_config_functions_importable(self):
        """Test that config functions are importable."""
        from linox import is_debug, set_debug

        assert callable(is_debug)
        assert callable(set_debug)


class TestConfig:
    """Test that config module works correctly."""

    def test_config_import(self):
        """Test that config module is importable."""
        from linox import config

        assert hasattr(config, "is_debug")
        assert hasattr(config, "set_debug")
        assert hasattr(config, "warn")

    def test_is_debug_returns_bool(self):
        """Test that is_debug returns a boolean."""
        from linox import is_debug

        result = is_debug()
        assert isinstance(result, bool)

    def test_set_debug_toggle(self):
        """Test that set_debug can toggle debug mode."""
        from linox import is_debug, set_debug

        original_state = is_debug()

        try:
            # Test setting to True
            set_debug(True)
            assert is_debug() is True

            # Test setting to False
            set_debug(False)
            assert is_debug() is False

            # Test setting back to True
            set_debug(True)
            assert is_debug() is True
        finally:
            # Restore original state
            set_debug(original_state)

    def test_config_direct_import(self):
        """Test that config functions work when imported directly."""
        from linox.config import is_debug, set_debug

        original_state = is_debug()

        try:
            set_debug(not original_state)
            assert is_debug() == (not original_state)
        finally:
            set_debug(original_state)

    def test_warn_function_exists(self):
        """Test that warn function exists in config."""
        from linox.config import warn

        assert callable(warn)
        # Test that it doesn't raise an error
        warn("Test message")


class TestModuleMetadata:
    """Test module metadata."""

    def test_version_exists(self):
        """Test that __version__ is defined."""
        assert hasattr(linox, "__version__")
        assert isinstance(linox.__version__, str)

    def test_all_exists(self):
        """Test that __all__ is defined."""
        assert hasattr(linox, "__all__")
        assert isinstance(linox.__all__, list)
        assert len(linox.__all__) > 0

    def test_docstring_exists(self):
        """Test that module has a docstring."""
        assert linox.__doc__ is not None
        assert len(linox.__doc__) > 0
