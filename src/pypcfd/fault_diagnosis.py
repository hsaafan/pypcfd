from typing import Literal, List, Tuple, Union, Optional, Dict

import numpy as np
import scipy.linalg as la
from scipy import stats

Indices = Union[str, List[str]]


def std_basis_vector(size: int, index: int,
                     shape: Literal["row", "col", "flat"] = "col"):
    """Create a vector of {size} values where all values are zero except at
    position {index} which is one. The shape can be specified as 'row', 'col',
    or 'flat' to generate vectors of shape (1, {size}), ({size}, 1), or
    ({size}, ) respectively. The default shape is 'col'."""
    e = np.zeros(size)
    e[index] = 1

    if shape.lower() == "col":
        e = np.reshape(e, (size, 1))
    elif shape.lower() == "row":
        e = np.reshape(e, (1, size))
    elif shape.lower() == "flat":
        pass
    else:
        raise ValueError(f"Cannot understand vector shape: '{shape}', use "
                         f"'row', 'col', or 'flat'")
    return(e)


class GenericDiagnosisMethod:

    def __init__(self) -> None:
        self.sample_size = 0

    def contribution(self, sample: np.ndarray, variable_index: int) -> float:
        """Return the error contribution of a variable in a sample"""
        raise NotImplementedError

    def expectation(self, variable_index: int) -> float:
        """Return the expected error contribution of a variable"""
        raise NotImplementedError

    def limits(self, variable_index: int,
               alpha: float) -> Tuple[float, float]:
        """Return the lower and upper limits of a variable at a given alpha"""
        e_contrib = self.expectation(variable_index)
        lower = stats.chi2.ppf(alpha, 1) * e_contrib
        upper = stats.chi2.ppf(1 - alpha, 1) * e_contrib
        return(lower, upper)

    def rel_contribution(self, sample: np.ndarray,
                         variable_index: int) -> float:
        """Return the relative error contribution of a variable in a sample"""
        c = self.contribution(sample, variable_index)
        E_c = self.expectation(variable_index)
        return(c / E_c)

    def all_contributions(self, sample: np.ndarray) -> np.ndarray:
        """Return the error contributions for all variables in a sample"""
        contribs = np.zeros(self.sample_size)
        for i in range(self.sample_size):
            contribs[i] = self.contribution(sample, i)
        return(contribs)

    def all_rel_contributions(self, sample: np.ndarray) -> np.ndarray:
        """Return the relative error contributions for all variables in a
        sample"""
        rel_contribs = np.zeros(self.sample_size)
        for i in range(self.sample_size):
            rel_contribs[i] = self.rel_contribution(sample, i)
        return(rel_contribs)

    def all_expectations(self) -> np.ndarray:
        """Return the expected error contribution for all variables"""
        e_contribs = np.zeros(self.sample_size)
        for i in range(self.sample_size):
            e_contribs[i] = self.expectation(i)
        return(e_contribs)

    def all_limits(self, alpha: float) -> np.ndarray:
        """Return the lower and upper limits for all variables at a given
        alpha"""
        lower_upper_limits = np.zeros((self.sample_size, 2))
        for i in range(self.sample_size):
            lower_upper_limits[i] = self.limits(i, alpha)
        return(lower_upper_limits)


class CDC(GenericDiagnosisMethod):
    def __init__(self, M: np.ndarray, S: Optional[np.ndarray]) -> None:
        """Complete Decomposition Contributions Diagnosis Method"""
        super().__init__()
        self.M = M
        self.S = S
        self.sqrt_M = np.real(la.fractional_matrix_power(M, 0.5))
        self.sample_size = M.shape[0]

    def contribution(self, sample: np.ndarray, variable_index: int) -> float:
        e_i = std_basis_vector(self.sample_size, variable_index, 'col')
        contrib = (e_i.T @ self.sqrt_M @ sample) ** 2
        return(contrib)

    def expectation(self, variable_index: int) -> float:
        if self.S is None:
            raise RuntimeError("S matrix must be set to use this function")
        e_i = std_basis_vector(self.sample_size, variable_index, 'col')
        e_contrib = e_i.T @ self.S @ self.M @ e_i
        return(e_contrib)


class PDC(GenericDiagnosisMethod):
    def __init__(self, M: np.ndarray, S: Optional[np.ndarray]) -> None:
        """Partial Decomposition Contributions Diagnosis Method"""
        super().__init__()
        self.M = M
        self.S = S
        self.sample_size = M.shape[0]

    def contribution(self, sample: np.ndarray, variable_index: int) -> float:
        e_i = std_basis_vector(sample.size, variable_index, 'col')
        contrib = sample.T @ self.M @ e_i @ e_i.T @ sample
        return(contrib)

    def expectation(self, variable_index: int) -> float:
        if self.S is None:
            raise RuntimeError("S matrix must be set to use this function")
        e_i = std_basis_vector(self.sample_size, variable_index, 'col')
        e_contrib = e_i.T @ self.S @ self.M @ e_i
        return(e_contrib)

    def limits(self, variable_index: int,
               alpha: float) -> Tuple[float, float]:
        e_contrib = self.expectation(variable_index)
        e_i = std_basis_vector(self.sample_size, variable_index, 'col')
        stdv_contrib = ((e_contrib) ** 2
                        + e_i.T @ self.S @ self.M
                        @ self.M @ e_i @ e_i.T @ self.S @ e_i) ** 0.5
        # Assumes n>=30 to use normal distribution rather than t distribution
        lower, upper = stats.norm.interval(alpha, e_contrib, stdv_contrib)
        return(lower, upper)


class DC(GenericDiagnosisMethod):
    def __init__(self, M: np.ndarray, S: Optional[np.ndarray]) -> None:
        """Diagonal Contributions Diagnosis Method"""
        super().__init__()
        self.M = M
        self.S = S
        self.sample_size = M.shape[0]

    def contribution(self, sample: np.ndarray, variable_index: int) -> float:
        e_i = std_basis_vector(self.sample_size, variable_index, 'col')
        contrib = sample.T @ e_i @ e_i.T @ self.M @ e_i @ e_i.T @ sample
        return(contrib)

    def expectation(self, variable_index: int) -> float:
        if self.S is None:
            raise RuntimeError("S matrix must be set to use this function")
        e_i = std_basis_vector(self.M.shape[1], variable_index, 'col')
        e_contrib = e_i.T @ self.S @ e_i @ e_i.T @ self.M @ e_i
        return(e_contrib)


class RBC(GenericDiagnosisMethod):
    def __init__(self, M: np.ndarray, S: Optional[np.ndarray]) -> None:
        """Reconstruction Based Contributions Diagnosis Method"""
        super().__init__()
        self.M = M
        self.S = S
        self.sample_size = M.shape[0]

    def contribution(self, sample: np.ndarray, variable_index: int) -> float:
        e_i = std_basis_vector(self.sample_size, variable_index, 'col')
        contrib = (e_i.T @ self.M @ sample) ** 2 / (e_i.T @ self.M @ e_i)
        return(contrib)

    def expectation(self, variable_index: int) -> float:
        if self.S is None:
            raise RuntimeError("S matrix must be set to use this function")
        e_i = std_basis_vector(self.sample_size, variable_index, 'col')
        e_contrib = (e_i.T @ self.M @ self.S @ self.M @ e_i
                     / (e_i.T @ self.M @ e_i))
        return(e_contrib)


class GenericFaultDiagnosisModel:
    def __init__(self, M: np.ndarray, S: Optional[np.ndarray]) -> None:
        """Generic Fault Diagnosis Model for any test statistic"""
        if S is not None:
            if not (M.shape[0] == M.shape[1] == S.shape[0] == S.shape[1]):
                raise ValueError("M and S need to be [n x n] matrices")
        else:
            if not (M.shape[0] == M.shape[1]):
                raise ValueError("M needs to be an [n x n] matrix")
        self.diagnosis_methods = {
            "CDC": CDC(M, S),
            "PDC": PDC(M, S),
            "DC": DC(M, S),
            "RBC": RBC(M, S)
        }
        self.sample_size = M.shape[0]
        indices = list(self.diagnosis_methods.keys())
        rel_indices = [f"r{i}" for i in indices]
        self.valid_indices = indices + rel_indices

    def validate_indices(self, indices: Indices) -> List[str]:
        """Validate list of requested indices"""
        if type(indices) == str:
            indices = [indices]

        for ind in indices:
            if ind not in self.valid_indices:
                raise ValueError(f"No contribution index {ind} exists")
        return(indices)

    def validate_sample(self, sample: np.ndarray) -> np.ndarray:
        """Validate passed sample"""
        if not isinstance(sample, np.ndarray):
            raise TypeError("Expected numpy array inputs for sample")
        if not (self.sample_size == sample.size):
            raise ValueError("M needs to be an [n x n] matrix and x needs to "
                             "be an [n x 1] vector")
        sample = np.reshape(sample, (-1, 1))  # Makes sure it's a column vector
        return(sample)

    def get_contributions(self, sample: np.ndarray,
                          indices: Indices = ['CDC']) -> Dict[str, np.ndarray]:
        """Get the fault contributions for the sample for each index passed"""
        indices = self.validate_indices(indices)
        sample = self.validate_sample(sample)

        index_values = dict()

        for ind in indices:
            if ind[0] == 'r':
                fd_method = self.diagnosis_methods[ind[1:]]
                index_values[ind] = fd_method.all_rel_contributions(sample)
            else:
                fd_method = self.diagnosis_methods[ind]
                index_values[ind] = fd_method.all_contributions(sample)
        return(index_values)

    def get_limits(self, alpha: float = 0.05,
                   indices: Indices = ['CDC']) -> Dict[str, np.ndarray]:
        """Get the lower and upper control limits for any non-relative
        contribution indices"""
        indices = self.validate_indices(indices)
        limits = dict()

        for ind in indices:
            if ind[0] == 'r':
                raise ValueError("Control limits are not defined for relative "
                                 "contribution indices")
            else:
                fd_method = self.diagnosis_methods[ind]
                limits[ind] = fd_method.all_limits(alpha)
        return(limits)


if __name__ == "__main__":
    import random
    print("Module ran as script: Running example fault diagnosis with PCA")

    def example_process_model(num_samples):
        A = [
            [-0.3441, 0.4815, 0.6637],
            [-0.2313, -0.5936, 0.3545],
            [-0.5060, 0.2495, 0.0739],
            [-0.5552, -0.2405, -0.1123],
            [-0.3371, 0.3822, -0.6115],
            [-0.3877, -0.3868, -0.2045]
        ]
        A = np.asarray(A)
        num_vars = 6

        # Generate inputs t
        t1 = 2.0 * stats.uniform.rvs(size=num_samples)
        t2 = 1.6 * stats.uniform.rvs(size=num_samples)
        t3 = 1.2 * stats.uniform.rvs(size=num_samples)
        t = np.asarray([t1, t2, t3])

        # Generate noise
        noise = [None] * num_vars
        for i in range(num_vars):
            noise[i] = stats.norm.rvs(size=num_samples, scale=0.2)
        noise = np.asarray(noise)

        # Create samples
        X = A @ t + noise

        return(X)

    num_samples = 3000
    num_faults = 2000
    num_vars = 6

    X = example_process_model(num_samples)

    """ PCA Model """
    # Shift to 0 mean
    xmean = np.mean(X, 1).reshape((-1, 1))
    X = X - xmean

    # Scale to unit variance
    xstd = np.std(X, 1).reshape((-1, 1))
    X = X / xstd

    assert np.allclose(np.mean(X, 1), 0)
    assert np.allclose(np.std(X, 1), 1)

    S = np.cov(X)
    Lam, P = la.eig(S)
    Lam = np.real_if_close(Lam)
    order = np.argsort(-1 * Lam)
    Lam = Lam[order]
    P = P[:, order]

    # Plot cumulative variance of eigenvectors
    # cum_eig = np.cumsum(Lam) / np.sum(Lam)
    # plt.plot(cum_eig)
    # plt.show()
    principal_vectors = 3
    alpha = 0.01  # Confidence = (1 - alpha) x 100%

    P_resid = P[:, principal_vectors:]
    Lam_resid = Lam[principal_vectors:]
    P = P[:, :principal_vectors]
    Lam = Lam[:principal_vectors]
    D = P @ np.diag(Lam ** -1) @ P.T

    # Generate faults
    faults = np.zeros((num_vars, num_faults))

    for fault_sample in range(num_faults):
        fault_var = random.sample(range(num_vars), 1)[0]
        faults[fault_var, fault_sample] = 5.0 * stats.uniform.rvs()

    X_faulty = example_process_model(num_faults) + faults
    X_faulty = (X_faulty - xmean) / xstd

    T_sqr = [0] * num_faults
    for i in range(num_faults):
        T_sqr[i] = X_faulty[:, i].T @ D @ X_faulty[:, i]

    T_sqr_limit = [stats.chi2.ppf(1 - alpha, principal_vectors)] * num_faults

    detected_faults = []

    for i in range(num_faults):
        if T_sqr[i] > T_sqr_limit[i]:
            detected_faults.append(i)

    fault_detect_rate = len(detected_faults) / num_faults * 100
    print(f"T^2 Detected Faults: {fault_detect_rate:.2f} %")
    # plt.plot(T_sqr, label="\$T^2\$")
    # plt.plot(T_sqr_limit, label="Limit")
    # plt.legend()
    # plt.show()
    all_indices = ['CDC', 'rCDC', 'PDC', 'rPDC', 'DC', 'rDC', 'RBC', 'rRBC']
    FDModel = GenericFaultDiagnosisModel(D, S)
    cont_rates = dict()
    for ind in all_indices:
        # Tracks number of correct diagnoses and false diagnoses
        cont_rates[ind] = [0, 0, 0]

    for i in detected_faults:
        # Get index and limit for each fault sample
        cont = FDModel.get_contributions(X_faulty[:, i], all_indices)

        for ind in all_indices:
            highest_contrib = np.argmax(cont[ind])
            if highest_contrib == np.argmax(faults[:, i]):
                cont_rates[ind][0] += 1
            else:
                cont_rates[ind][1] += 1

    for ind in all_indices:
        diag_rate = cont_rates[ind][0] / len(detected_faults) * 100
        false_diag_rate = cont_rates[ind][1] / len(detected_faults) * 100
        # missed_rate = cont_rates[ind][2] / len(detected_faults) * 100
        print("--------------------------------")
        print(f"{ind} correct diagnosis: {diag_rate:.2f} %")
        print(f"{ind} false diagnosis: {false_diag_rate:.2f} %")
        # print(f"{ind} missed diagnosis: {missed_rate:.2f} %")
