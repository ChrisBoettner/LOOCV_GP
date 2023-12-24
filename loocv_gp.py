import numba
import numpy as np
import george


@numba.jit(nopython=True)
def compiled_log_pseudo_likelihood(
    qn: np.ndarray,
    Kinv_diag: np.ndarray,
) -> float:
    """
    Calculate the log pseudo likelihood for Gaussian process regression.

    Parameters
    ----------
    qn : np.ndarray
        The vector `alpha` calculated in Gaussian process regression.
    Kinv_diag : np.ndarray
        The diagonal of the inverse of the covariance matrix `K`.

    Returns
    -------
    float
        The log pseudo likelihood value.

    Notes
    -----
    This method is based on the approach described in Sundararajan (2001).
    """
    k = len(qn)
    log_det = np.sum(np.log(Kinv_diag))
    squared_diff_sum = np.sum(qn**2 / Kinv_diag)

    log_predictive_prob = 0.5 * (-squared_diff_sum + log_det - k * np.log(2 * np.pi))
    return log_predictive_prob


class LOOCV_GP(george.GP):
    def _compute_K_inv(
        self, y: np.ndarray, cache: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the inverse of the covariance matrix K and its diagonal.

        Parameters
        ----------
        y : np.ndarray
            The target values.
        cache : bool
            Whether to use cached values.

        Returns
        -------
        tuple
            A tuple containing the inverse of K and its diagonal.
        """
        if not cache:
            K_inv = np.ascontiguousarray(self.solver.get_inverse())
            K_inv_diag = np.ascontiguousarray(np.diag(K_inv))

            return K_inv, K_inv_diag
        if self._alpha is None or not np.array_equiv(y, self._y):
            K_inv = np.ascontiguousarray(self.solver.get_inverse())
            K_inv_diag = np.ascontiguousarray(np.diag(K_inv))

            self._K_inv = K_inv
            self._K_inv_diag = K_inv_diag
        return self._K_inv, self._K_inv_diag

    def compute_predictive_estimates(
        self, y: np.ndarray, cache: bool = True
    ) -> np.ndarray:
        """
        Compute predictive mean estimates using leave-one-out cross-validation.

        Parameters
        ----------
        y : np.ndarray
            The target values.
        cache : bool, optional
            Whether to use cached values (default is True).

        Returns
        -------
        np.ndarray
            The predictive mean estimates.
        """
        K_inv, K_inv_diag = self._compute_K_inv(y, cache=cache)

        predictive_mu = y - self.apply_inverse(y) / K_inv_diag
        return predictive_mu

    def log_pseudo_likelihood(
        self, y: np.ndarray, quiet: bool = False, cache: bool = True
    ) -> float:
        """
        Compute the log pseudo likelihood for Gaussian process regression.

        Parameters
        ----------
        y : np.ndarray
            The target values.
        quiet : bool, optional
            If True, suppress exceptions and return negative infinity on
            failure (default is False).
        cache : bool, optional
            Whether to use cached values (default is True).

        Returns
        -------
        float
            The log pseudo likelihood value.

        Raises
        ------
        NotImplementedError
            If there are duplicates in X.
        ValueError
            If the mean function fails.
        """
        if not self.recompute(quiet=quiet):
            return -np.inf

        if len(np.unique(self._x)) != len(self._x):
            raise NotImplementedError(
                "There are duplicates in X. Leave-one-out Cross-Validation not "
                "implemented for this case."
            )

        try:
            mu = self._call_mean(self._x)
        except ValueError:
            if quiet:
                return -np.inf
            raise
        r = np.ascontiguousarray(self._check_dimensions(y) - mu, dtype=np.float64)

        _, K_inv_diag = self._compute_K_inv(r, cache=cache)
        alpha = self._compute_alpha(y, cache=cache)

        return compiled_log_pseudo_likelihood(alpha, K_inv_diag)

    def grad_log_pseudo_likelihood(
        self, y: np.ndarray, quiet: bool = False, cache: bool = True
    ) -> np.ndarray:
        """
        Compute the gradient of the log pseudo likelihood for Gaussian process
        regression.

        Parameters
        ----------
        y : np.ndarray
            The target values.
        quiet : bool, optional
            If True, suppress exceptions and return a zero vector on failure (default
            is False).
        cache : bool, optional
            Whether to use cached values (default is True).

        Returns
        -------
        np.ndarray
            The gradient of the log pseudo likelihood.
        """
        # Make sure that the model is computed and try to recompute it if it's
        # dirty.
        if not self.recompute(quiet=quiet):
            return np.zeros(len(self), dtype=np.float64)

        # Pre-compute some factors.
        try:
            alpha = self._compute_alpha(y, False)
        except ValueError:
            if quiet:
                return np.zeros(len(self), dtype=np.float64)
            raise

        K_inv, K_inv_diag, A = None, None, None
        if len(self.white_noise) or len(self.kernel):
            K_inv, K_inv_diag = self._compute_K_inv(y, cache=cache)
            A = np.einsum("i,j", alpha, alpha) - K_inv

        # Compute each component of the gradient.
        grad = np.empty(len(self))
        n = 0

        l_mean = len(self.mean)
        if l_mean:
            try:
                mu = self._call_mean_gradient(self._x)
            except ValueError:
                if quiet:
                    return np.zeros(len(self), dtype=np.float64)
                raise
            grad[n : n + l_mean] = np.dot(mu, alpha)
            n += l_mean

        l_noise = len(self.white_noise)
        if l_noise:
            assert A is not None
            wn = self._call_white_noise(self._x)
            wng = self._call_white_noise_gradient(self._x)
            grad[n : n + l_noise] = 0.5 * np.sum(
                (np.exp(wn) * np.diag(A))[None, :] * wng, axis=1
            )
            n += l_noise

        l_kernel = len(self.kernel)
        if l_kernel:
            assert K_inv is not None and K_inv_diag is not None
            Kg = self.kernel.get_gradient(self._x)
            Z = np.ascontiguousarray(np.einsum("ij,jkl", K_inv, Kg))
            Za = np.ascontiguousarray(np.einsum("ijk,j", Z, alpha))
            Zk_diag = np.ascontiguousarray(np.einsum("ijk,ji->ik", Z, K_inv))

            term_1 = (alpha / K_inv_diag) @ Za
            term_2 = -0.5 * np.einsum("ik, i", Zk_diag, 1 / K_inv_diag)
            term_3 = -0.5 * np.einsum("i, ik", (alpha / K_inv_diag) ** 2, Zk_diag)
            grad[n : n + l_kernel] = term_1 + term_2 + term_3
        return grad

    def nlpl(
        self, vector: np.ndarray, y: np.ndarray, quiet: bool = True, cache: bool = True
    ) -> float:
        """
        Negative log pseudo likelihood for a given parameter vector.

        Parameters
        ----------
        vector : np.ndarray
            The parameter vector.
        y : np.ndarray
            The target values.
        quiet : bool, optional
            If True, suppress exceptions (default is True).
        cache : bool, optional
            Whether to use cached values (default is True).

        Returns
        -------
        float
            The negative log pseudo likelihood value.
        """
        self.set_parameter_vector(vector)
        if not np.isfinite(self.log_prior()):
            return np.inf
        return -self.log_pseudo_likelihood(
            y,
            quiet=quiet,
            cache=cache,
        )

    def grad_nlpl(
        self, vector: np.ndarray, y: np.ndarray, quiet: bool = True, cache: bool = True
    ) -> np.ndarray:
        """
        Gradient of the negative log pseudo likelihood for a given parameter vector.

        Parameters
        ----------
        vector : np.ndarray
            The parameter vector.
        y : np.ndarray
            The target values.
        quiet : bool, optional
            If True, suppress exceptions (default is True).
        cache : bool, optional
            Whether to use cached values (default is True).

        Returns
        -------
        np.ndarray
            The gradient of the negative log pseudo likelihood.
        """
        self.set_parameter_vector(vector)
        if not np.isfinite(self.log_prior()):
            return np.zeros(len(vector))
        return -self.grad_log_pseudo_likelihood(
            y,
            quiet=quiet,
            cache=cache,
        )
