from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, RationalQuadratic, Matern, WhiteKernel, ConstantKernel as C
from config import get_config

cfg = get_config()

# Threshold for classification/plot labeling based on predicted density
threshold = float(cfg["threshold"])

# Candidate kernels to evaluate via cross-validation
kernels = {
    'RBF': C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1)),
    'Rational Quadratic': C(1.0, (1e-4, 1e1)) * RationalQuadratic(),
    'Matern (ν=1.5)': C(1.0, (1e-4, 1e1)) * Matern(nu=1.5),
    'Matern (ν=0.5)': C(1.0, (1e-4, 1e1)) * Matern(nu=0.5),
    'Matern (ν=2.5)': C(1.0, (1e-4, 1e1)) * Matern(nu=2.5),
    'Exp-Sine-Squared': C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=1.0, periodicity=3.0),
    'Dot-Product': C(1.0, (1e-4, 1e1)) * DotProduct(sigma_0=1.0),
    'Dot-Product + RBF': C(1.0, (1e-4, 1e1)) * (DotProduct(sigma_0=1.0) + RBF(1, (1e-4, 1e1))),
    'Dot-Product * Rational Quadratic': C(1.0, (1e-4, 1e1)) * (DotProduct(sigma_0=1.0) * RationalQuadratic()),
    'Dot-Product + Matern (ν=1.5)': C(1.0, (1e-4, 1e1)) * (DotProduct(sigma_0=1.0) + Matern(nu=1.5)),
    'Dot-Product * Matern (ν=2.5)': C(1.0, (1e-4, 1e1)) * (DotProduct(sigma_0=1.0) * Matern(nu=2.5)),
    'RBF + WhiteKernel': C(1.0, (1e-4, 1e1)) * (RBF(1, (1e-4, 1e1)) + WhiteKernel(noise_level=1e-1)),
    'Rational Quadratic + WhiteKernel': RationalQuadratic() + WhiteKernel(noise_level=1),
    'RBF * Rational Quadratic': C(1.0, (1e-4, 1e1)) * (RBF(1, (1e-4, 1e1)) * RationalQuadratic()),
    'Matern (ν=1.5) * RBF': C(1.0, (1e-4, 1e1)) * (Matern(nu=1.5) * RBF(1, (1e-4, 1e1))),
    'Matern (ν=0.5) + RBF': C(1.0, (1e-4, 1e1)) * (Matern(nu=0.5) + RBF(1, (1e-4, 1e1))),
    'Matern (ν=2.5) * Rational Quadratic': C(1.0, (1e-4, 1e1)) * (Matern(nu=2.5) * RationalQuadratic()),
    'Additive (RBF + WhiteKernel)': C(1.0, (1e-4, 1e1)) * (RBF(1, (1e-4, 1e1)) + WhiteKernel()),
    'Additive (Rational Quadratic + WhiteKernel)': RationalQuadratic() + WhiteKernel(noise_level=1),
    'Multiplicative (RBF * Matern)': C(1.0, (1e-4, 1e1)) * (RBF(1, (1e-4, 1e1)) * Matern(nu=1.5)),
}