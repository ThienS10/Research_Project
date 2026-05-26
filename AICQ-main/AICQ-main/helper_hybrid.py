import numpy as np
from pyswarm import pso
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from config import get_config
from typing import Optional
CFG = get_config()
GS = float(CFG["soil"]["Gs"])

# SageMaker adapter
import os
import json
import boto3

class SageMakerPredictorAdapter:
    """Adapter that mimics scikit-learn's predict API, calling a SageMaker endpoint.

    Expected behavior:
    - predict(X) -> np.array of shape (n_samples,)
    - predict(X, return_std=True) -> (np.array, np.array) where std is approximated
    """
    def __init__(self, endpoint_name: str, region: Optional[str] = None):
        self.endpoint_name = endpoint_name
        self.client = boto3.client('sagemaker-runtime', region_name=region or os.getenv('AWS_REGION'))

    def predict(self, X, return_std: bool = False):
        # Ensure X is a 2D list
        if hasattr(X, 'tolist'):
            payload = X.tolist()
        else:
            payload = X
        body = json.dumps({"instances": payload})
        resp = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=body,
        )
        result = json.loads(resp['Body'].read().decode('utf-8'))
        # Expect {"predictions": [y1, y2, ...]} or raw list
        if isinstance(result, dict) and 'predictions' in result:
            preds = result['predictions']
        else:
            preds = result
        y = np.array(preds).reshape(-1)
        if return_std:
            # If endpoint does not return std, provide a small constant as placeholder
            std = np.full_like(y, 0.05, dtype=float)
            return y, std
        return y


def calculate_rmse(y_true, y_pred):
    """Compute root mean squared error (RMSE).

    Args:
        y_true: Array-like of ground-truth target values.
        y_pred: Array-like of predicted target values.

    Returns:
        float: The RMSE between predictions and ground truth.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def golden_section_search(f, a, b, tol=1e-5):
    """Minimize a univariate function on [a, b] via Golden Section search.

    Args:
        f: Callable taking a single float argument and returning a float objective.
        a: Lower bound of the search interval.
        b: Upper bound of the search interval.
        tol: Termination tolerance on the bracket size.

    Returns:
        float: Approximate minimizer location within [a, b].
    """
    invphi = (np.sqrt(5) - 1) / 2  # 1/phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1/phi^2
    
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
    
    return (b + a) / 2
 


# General helper functions (unchanged)
 

def calculate_suction(void_ratio, moisture, alpha):
    """Calculate suction using a mixed dry/wet SWCC formulation.

    Uses parameters from config: swcc.n, swcc.m, swcc.a, swcc.omega and soil.Gs.
    The mixing between dry and wet curves is controlled by alpha in [0, 1].

    Args:
        void_ratio: Soil void ratio e.
        moisture: Moisture content in percent (0-100).
        alpha: Mixing factor between dry and wet suction (1=dry, 0=wet).

    Returns:
        float: Suction value under given state.
    """
    n_swcc = float(CFG["swcc"]["n"]) 
    m_swcc = float(CFG["swcc"]["m"]) 
    a_swcc = float(CFG["swcc"]["a"]) 
    a_w_swcc = a_swcc / 2
    omega = float(CFG["swcc"]["omega"]) 
    sr = max(min(GS * (moisture/100) / void_ratio, 0.9999999999), 0.001)
    suc_d = a_swcc * (sr ** (-1 / m_swcc) - 1) ** (1 / n_swcc) / void_ratio ** omega
    suc_w = a_w_swcc * (sr ** (-1 / m_swcc) - 1) ** (1 / n_swcc) / void_ratio ** omega
    return suc_d * alpha + (1-alpha) * suc_w


def _finite_or_default(value, default):
    """Return a finite float, falling back when the equation becomes unstable."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _elastic_cfg(cfg=None):
    """Read elastic workflow constants from config with conservative fallbacks."""
    cfg = cfg or get_config()
    soil_cfg = cfg.get("soil", {})
    elastic_cfg = cfg.get("elastic_physics", {})
    return soil_cfg, elastic_cfg


def calculate_degree_of_saturation(void_ratio, moisture_percent, cfg=None):
    """Calculate Sw = w * Gs / e using water content as a decimal fraction."""
    soil_cfg, elastic_cfg = _elastic_cfg(cfg)
    gs = float(soil_cfg.get("Gs", GS))
    e_min = float(elastic_cfg.get("void_ratio_min", 0.25))
    e_max = float(elastic_cfg.get("void_ratio_max", 0.75))
    sw_min = float(elastic_cfg.get("saturation_min", 0.001))
    sw_max = float(elastic_cfg.get("saturation_max", 0.999))
    e = np.clip(_finite_or_default(void_ratio, e_min), e_min, e_max)
    water_content = max(_finite_or_default(moisture_percent, 0.0) / 100.0, 0.0)
    return float(np.clip(gs * water_content / e, sw_min, sw_max))


def calculate_matric_pressure(void_ratio, moisture_percent, cfg=None):
    """Calculate the suction-like pressure term used inside the Gmax equation."""
    soil_cfg, _ = _elastic_cfg(cfg)
    alpha = max(float(soil_cfg.get("alpha", 0.8)), 1e-9)
    m_const = max(float(soil_cfg.get("m", 0.5)), 1e-9)
    n_const = max(float(soil_cfg.get("n", 1.2)), 1e-9)
    sw = calculate_degree_of_saturation(void_ratio, moisture_percent, cfg)

    chi = sw ** (0.75 * sw ** -0.02)
    capillary = (1.0 / alpha) * ((sw ** (-1.0 / m_const) - 1.0) ** (1.0 / n_const))
    pressure = chi * capillary
    return max(_finite_or_default(pressure, 0.0), 0.0)


def calculate_effective_pressure(pnet, void_ratio, moisture_percent, cfg=None):
    """Calculate p' from global Pnet plus the moisture/saturation pressure term."""
    soil_cfg, elastic_cfg = _elastic_cfg(cfg)
    pa = float(soil_cfg.get("Pa", 101.325))
    pressure_ratio_min = float(elastic_cfg.get("pressure_ratio_min", 0.001))
    pnet_value = max(_finite_or_default(pnet, elastic_cfg.get("pnet_initial", 50.0)), 0.0)
    min_pressure = pa * pressure_ratio_min
    return max(pnet_value + calculate_matric_pressure(void_ratio, moisture_percent, cfg), min_pressure)


def calculate_gmax(void_ratio, effective_pressure, cfg=None):
    """Calculate scaled Gmax for use with ELWD-compatible modulus equations."""
    _, elastic_cfg = _elastic_cfg(cfg)
    gmax_raw = calculate_gmax_raw(void_ratio, effective_pressure, cfg)
    return float(gmax_raw * float(elastic_cfg.get("gmax_to_modulus_factor", 1.0)))


def calculate_gmax_raw(void_ratio, effective_pressure, cfg=None):
    """Calculate unscaled Gmax from the workflow equation."""
    soil_cfg, elastic_cfg = _elastic_cfg(cfg)
    pa = float(soil_cfg.get("Pa", 101.325))
    e_min = float(elastic_cfg.get("void_ratio_min", 0.25))
    e_max = float(elastic_cfg.get("void_ratio_max", 0.75))
    pressure_ratio_min = float(elastic_cfg.get("pressure_ratio_min", 0.001))
    pressure_ratio_max = float(elastic_cfg.get("pressure_ratio_max", 1000.0))

    e = np.clip(_finite_or_default(void_ratio, e_min), e_min, e_max)
    pressure = max(_finite_or_default(effective_pressure, pa * pressure_ratio_min), pa * pressure_ratio_min)
    pressure_ratio = np.clip(pressure / pa, pressure_ratio_min, pressure_ratio_max)
    gmax = (
        float(elastic_cfg.get("gmax_coefficient", 0.066))
        * e ** float(elastic_cfg.get("gmax_void_ratio_exponent", -4.67))
        * pressure_ratio ** float(elastic_cfg.get("gmax_stress_exponent", 0.39))
    )
    return float(np.clip(
        _finite_or_default(gmax, elastic_cfg.get("gmax_min", 1e-6)),
        float(elastic_cfg.get("gmax_min", 1e-6)),
        float(elastic_cfg.get("gmax_max", 1e6)),
    ))


def calculate_bulk_modulus(void_ratio, effective_pressure, cfg=None):
    """Calculate scaled bulk modulus for use with ELWD-compatible equations."""
    _, elastic_cfg = _elastic_cfg(cfg)
    bulk_raw = calculate_bulk_modulus_raw(void_ratio, effective_pressure, cfg)
    unit_factor = float(elastic_cfg.get("pa_to_modulus_factor", 0.001))
    calibration_factor = float(elastic_cfg.get("bulk_to_modulus_factor", 1.0))
    return float(bulk_raw * unit_factor * calibration_factor)


def calculate_bulk_modulus_raw(void_ratio, effective_pressure, cfg=None):
    """Calculate raw K = K0 * Pa * ((1+e)/e) * (p'/Pa)^(2/3)."""
    soil_cfg, elastic_cfg = _elastic_cfg(cfg)
    pa = float(soil_cfg.get("Pa", 101.325))
    e_min = float(elastic_cfg.get("void_ratio_min", 0.25))
    e_max = float(elastic_cfg.get("void_ratio_max", 0.75))
    pressure_ratio_min = float(elastic_cfg.get("pressure_ratio_min", 0.001))
    pressure_ratio_max = float(elastic_cfg.get("pressure_ratio_max", 1000.0))

    e = np.clip(_finite_or_default(void_ratio, e_min), e_min, e_max)
    pressure_ratio = np.clip(_finite_or_default(effective_pressure, pa) / pa, pressure_ratio_min, pressure_ratio_max)
    bulk = float(elastic_cfg.get("bulk_k0", 500.0)) * pa * ((1.0 + e) / e) * pressure_ratio ** (2.0 / 3.0)
    return float(np.clip(
        _finite_or_default(bulk, elastic_cfg.get("bulk_min", 1e-6)),
        float(elastic_cfg.get("bulk_min", 1e-6)),
        float(elastic_cfg.get("bulk_max", 1e6)),
    ))


def calculate_elastic_modulus(gmax, bulk_modulus):
    """Combine G and K into E using E = 9KG / (3K + G)."""
    g = max(_finite_or_default(gmax, 0.0), 0.0)
    k = max(_finite_or_default(bulk_modulus, 0.0), 0.0)
    denominator = 3.0 * k + g
    if denominator <= 0:
        return 0.0
    return float((9.0 * k * g) / denominator)


def transform_physics_feature(value, cfg=None):
    """Transform sensitive physics features before sending them to the GPR."""
    _, elastic_cfg = _elastic_cfg(cfg)
    transform = str(elastic_cfg.get("feature_transform", "log1p")).lower()
    value = max(_finite_or_default(value, 0.0), 0.0)
    if transform == "none":
        return value
    if transform == "sqrt":
        return float(np.sqrt(value))
    return float(np.log1p(value))


def calculate_elastic_state(void_ratio, moisture_percent, pnet, cfg=None):
    """Calculate Sw, Gmax, K, and E for one candidate void ratio."""
    effective_pressure = calculate_effective_pressure(pnet, void_ratio, moisture_percent, cfg)
    gmax_raw = calculate_gmax_raw(void_ratio, effective_pressure, cfg)
    gmax = calculate_gmax(void_ratio, effective_pressure, cfg)
    bulk_raw = calculate_bulk_modulus_raw(void_ratio, effective_pressure, cfg)
    bulk = calculate_bulk_modulus(void_ratio, effective_pressure, cfg)
    elastic_modulus = calculate_elastic_modulus(gmax, bulk)
    return {
        "void_ratio": float(void_ratio),
        "saturation": calculate_degree_of_saturation(void_ratio, moisture_percent, cfg),
        "matric_pressure": calculate_matric_pressure(void_ratio, moisture_percent, cfg),
        "effective_pressure": effective_pressure,
        "gmax_raw": gmax_raw,
        "gmax": gmax,
        "bulk_modulus_raw": bulk_raw,
        "bulk_modulus": bulk,
        "elastic_modulus": elastic_modulus,
        "gmax_feature": transform_physics_feature(gmax, cfg),
        "bulk_feature": transform_physics_feature(bulk, cfg),
    }


def estimate_void_ratio_from_elwd(elwd, moisture_percent, pnet, cfg=None):
    """Find the void ratio whose physics E is closest to measured ELWD."""
    _, elastic_cfg = _elastic_cfg(cfg)
    e_min = float(elastic_cfg.get("void_ratio_min", 0.25))
    e_max = float(elastic_cfg.get("void_ratio_max", 0.75))
    target_elwd = max(_finite_or_default(elwd, 0.0), 1e-9)

    def objective(e_value):
        state = calculate_elastic_state(e_value, moisture_percent, pnet, cfg)
        return ((state["elastic_modulus"] - target_elwd) / target_elwd) ** 2

    grid = np.linspace(e_min, e_max, 101)
    errors = np.array([objective(e_value) for e_value in grid])
    best_idx = int(np.argmin(errors))
    lo = grid[max(best_idx - 1, 0)]
    hi = grid[min(best_idx + 1, len(grid) - 1)]
    if hi > lo:
        best_e = golden_section_search(objective, float(lo), float(hi), 1e-5)
    else:
        best_e = float(grid[best_idx])
    return calculate_elastic_state(best_e, moisture_percent, pnet, cfg)


def calculate_elastic_state_array(elwd_array, moisture_percent_array, pnet, cfg=None):
    """Estimate elastic physics states for every row in a dataset."""
    states = [
        estimate_void_ratio_from_elwd(elwd, moisture, pnet, cfg)
        for elwd, moisture in zip(elwd_array, moisture_percent_array)
    ]
    return {key: np.array([state[key] for state in states], dtype=float) for key in states[0]}


def _mean_stress_cfg(cfg=None):
    """Read mean-effective-stress configuration."""
    cfg = cfg or get_config()
    return cfg.get("mean_stress", {})


def calculate_lwd_contact_pressure(cfg=None):
    """Calculate LWD vertical contact stress q = F / A in kPa."""
    mean_cfg = _mean_stress_cfg(cfg)
    peak_load_kn = float(mean_cfg.get("lwd_peak_load_kn", 6.5))
    plate_radius_m = max(float(mean_cfg.get("lwd_plate_radius_m", 0.15)), 1e-9)
    plate_area_m2 = np.pi * plate_radius_m ** 2
    return float(peak_load_kn / plate_area_m2)


def estimate_unit_weight_kn_m3(ndg_density_array=None, used_mask=None, cfg=None):
    """Estimate unit weight from density in t/m3, returning kN/m3."""
    mean_cfg = _mean_stress_cfg(cfg)
    source = str(mean_cfg.get("unit_weight_density_source", "training_ndg_mean")).lower()
    density_t_m3 = float(mean_cfg.get("density_for_unit_weight_t_m3", 2.0))
    if source == "training_ndg_mean" and ndg_density_array is not None and used_mask is not None:
        used_mask = np.asarray(used_mask, dtype=bool)
        density_values = np.asarray(ndg_density_array, dtype=float)[used_mask]
        density_values = density_values[np.isfinite(density_values)]
        if density_values.size:
            density_t_m3 = float(np.mean(density_values))
    return density_t_m3 * 9.80665


def formulate_mean_effective_pnet(beta, unit_weight_kn_m3, cfg=None):
    """Formulate Pnet as mean overburden plus LWD-induced mean stress."""
    mean_cfg = _mean_stress_cfg(cfg)
    depth_m = float(mean_cfg.get("testbed_depth_m", 0.6))
    stress_offset_kpa = float(mean_cfg.get("mean_stress_offset_kpa", 0.0))
    earth_factor = float(mean_cfg.get("earth_pressure_mean_factor", 1.0))
    lwd_divisor = max(float(mean_cfg.get("lwd_mean_stress_divisor", 3.0)), 1e-9)
    overburden_mean_kpa = earth_factor * unit_weight_kn_m3 * depth_m
    lwd_contact_kpa = calculate_lwd_contact_pressure(cfg)
    lwd_mean_kpa = lwd_contact_kpa / lwd_divisor
    pnet = stress_offset_kpa + overburden_mean_kpa + float(beta) * lwd_mean_kpa
    return {
        "pnet": float(max(pnet, 0.0)),
        "beta": float(beta),
        "stress_offset_kpa": float(stress_offset_kpa),
        "unit_weight_kn_m3": float(unit_weight_kn_m3),
        "overburden_mean_kpa": float(overburden_mean_kpa),
        "lwd_contact_kpa": float(lwd_contact_kpa),
        "lwd_mean_kpa": float(lwd_mean_kpa),
    }


def optimize_global_mean_stress_beta(elwd_array, moisture_percent_array, ndg_density_array, used_mask, cfg=None):
    """Calibrate beta in Pnet = gamma*z + beta*q_lwd/3 using training rows."""
    cfg = cfg or get_config()
    mean_cfg = _mean_stress_cfg(cfg)
    elastic_cfg = cfg.get("elastic_physics", {})
    if not bool(mean_cfg.get("enabled", True)):
        pnet = float(elastic_cfg.get("pnet_initial", 50.0))
        return {
            "pnet": pnet,
            "beta": np.nan,
            "unit_weight_kn_m3": np.nan,
            "overburden_mean_kpa": np.nan,
            "lwd_contact_kpa": np.nan,
            "lwd_mean_kpa": np.nan,
        }

    used_mask = np.asarray(used_mask, dtype=bool)
    elwd_train = np.asarray(elwd_array, dtype=float)[used_mask]
    moisture_train = np.asarray(moisture_percent_array, dtype=float)[used_mask]
    density_train = np.asarray(ndg_density_array, dtype=float)[used_mask]
    unit_weight = estimate_unit_weight_kn_m3(ndg_density_array, used_mask, cfg)

    def objective(beta_value):
        stress = formulate_mean_effective_pnet(beta_value, unit_weight, cfg)
        states = calculate_elastic_state_array(elwd_train, moisture_train, stress["pnet"], cfg)
        physics_density = np.array([
            void_ratio_to_density(e_value, moisture)
            for e_value, moisture in zip(states["void_ratio"], moisture_train)
        ])
        return calculate_rmse(density_train, physics_density)

    beta_min = float(mean_cfg.get("beta_min", 0.0))
    beta_max = float(mean_cfg.get("beta_max", 1.0))
    grid = np.linspace(beta_min, beta_max, 25)
    errors = np.array([objective(beta_value) for beta_value in grid])
    best_idx = int(np.argmin(errors))
    lo = grid[max(best_idx - 1, 0)]
    hi = grid[min(best_idx + 1, len(grid) - 1)]
    if hi > lo:
        beta = float(golden_section_search(objective, float(lo), float(hi), float(mean_cfg.get("beta_tol", 0.0001))))
    else:
        beta = float(grid[best_idx])
    return formulate_mean_effective_pnet(beta, unit_weight, cfg)


def optimize_global_pnet(elwd_array, moisture_percent_array, ndg_density_array, used_mask, cfg=None):
    """Backward-compatible wrapper returning mean-stress-derived Pnet."""
    mean_stress = optimize_global_mean_stress_beta(
        elwd_array,
        moisture_percent_array,
        ndg_density_array,
        used_mask,
        cfg,
    )
    return mean_stress["pnet"]


def optimize_free_global_pnet(elwd_array, moisture_percent_array, ndg_density_array, used_mask, cfg=None):
    """Calibrate one free global Pnet by minimizing training-row physics-density RMSE."""
    _, elastic_cfg = _elastic_cfg(cfg)
    if not bool(elastic_cfg.get("enabled", True)):
        return float(elastic_cfg.get("pnet_initial", 50.0))

    used_mask = np.asarray(used_mask, dtype=bool)
    elwd_train = np.asarray(elwd_array, dtype=float)[used_mask]
    moisture_train = np.asarray(moisture_percent_array, dtype=float)[used_mask]
    density_train = np.asarray(ndg_density_array, dtype=float)[used_mask]

    def objective(pnet_value):
        states = calculate_elastic_state_array(elwd_train, moisture_train, pnet_value, cfg)
        physics_density = np.array([
            void_ratio_to_density(e_value, moisture)
            for e_value, moisture in zip(states["void_ratio"], moisture_train)
        ])
        return calculate_rmse(density_train, physics_density)

    pnet_min = float(elastic_cfg.get("pnet_min", 1.0))
    pnet_max = float(elastic_cfg.get("pnet_max", 500.0))
    grid = np.linspace(pnet_min, pnet_max, 25)
    errors = np.array([objective(pnet_value) for pnet_value in grid])
    best_idx = int(np.argmin(errors))
    lo = grid[max(best_idx - 1, 0)]
    hi = grid[min(best_idx + 1, len(grid) - 1)]
    if hi > lo:
        return float(golden_section_search(objective, float(lo), float(hi), float(elastic_cfg.get("pnet_tol", 0.01))))
    return float(grid[best_idx])


def void_ratio_to_density(void_ratio, moisture):
    """Convert void ratio and moisture to bulk density using Gs from config.

    Args:
        void_ratio: Soil void ratio e.
        moisture: Moisture content in percent (0-100).

    Returns:
        float: Bulk density.
    """
    dry_density = GS / (1 + void_ratio)
    return dry_density * (1.0 + moisture / 100)
 

def objective_function(void_ratio, lwd_modulus, moisture_numeric, alpha, model):
    """PSO objective for void ratio fit using surrogate model and suction physics.

    Args:
        void_ratio: Candidate void ratio (scalar or array-like from PSO).
        lwd_modulus: LWD modulus at the point.
        moisture_numeric: Moisture content (%) at the point.
        alpha: Mixing factor between dry and wet suction.
        model: Trained surrogate model mapping features to void ratio.

    Returns:
        float: Squared error between candidate and predicted void ratio.
    """
    # Accept vector input from PSO and extract scalar
    try:
        void_ratio_scalar = float(np.atleast_1d(void_ratio)[0])
    except Exception:
        void_ratio_scalar = void_ratio
    suction = calculate_suction(void_ratio_scalar, moisture_numeric, alpha)
    X_new = np.array([[moisture_numeric / 100, lwd_modulus, suction]])
    predicted_void_ratio = model.predict(X_new)[0]
    return (void_ratio_scalar - predicted_void_ratio)**2

def predict_void_ratio(lwd_modulus, moisture_numeric, alpha, initial_void_ratio_guess, model):
    """Estimate void ratio via PSO by minimizing objective_function.

    Args:
        lwd_modulus: LWD modulus value.
        moisture_numeric: Moisture content (%) value.
        alpha: Mixing factor between dry and wet suction.
        initial_void_ratio_guess: Historical parameter (kept for API compatibility).
        model: Trained surrogate model mapping features to void ratio.

    Returns:
        tuple[float, float]: (optimal_void_ratio, std_of_model_prediction).
    """
    cfg = get_config()
    lb = [cfg["pso"]["lb_void_ratio"]]
    ub = [cfg["pso"]["ub_void_ratio"]]
    xopt, fopt = pso(
        objective_function,
        lb,
        ub,
        args=(lwd_modulus, moisture_numeric, alpha, model),
        swarmsize=cfg["pso"]["swarmsize"],
        omega=cfg["pso"]["omega"],
        phip=cfg["pso"]["phip"],
        phig=cfg["pso"]["phig"],
        maxiter=cfg["pso"]["maxiter"],
        debug=cfg["pso"]["debug"],
    )
    suction = calculate_suction(xopt[0], moisture_numeric, alpha)
    X_new = np.array([[moisture_numeric / 100, lwd_modulus, suction]])
    predicted_void_ratio, std_predicted_void_ratio = model.predict(X_new, return_std=True)
    return xopt[0], std_predicted_void_ratio
 

def physical_model(lwd_modulus, moisture_input, initial_moisture_values, initial_void_ratio_guess, wd, wo, wl, alpha,gpr_model):
    """Run the physics + surrogate model to predict void ratio and density.

    Note: The wd/wo/wl parameters are ignored; per-row moisture values are
    provided via initial_moisture_values dict keyed by moisture label.

    Args:
        lwd_modulus: LWD modulus value.
        moisture_input: Moisture label ('low'|'medium'|'high').
        initial_moisture_values: Dict mapping label to numeric moisture (%).
        initial_void_ratio_guess: Kept for API compatibility; unused by PSO backend.
        wd: Deprecated.
        wo: Deprecated.
        wl: Deprecated.
        alpha: Mixing factor between dry and wet suction.
        gpr_model: Surrogate model for void ratio prediction.

    Returns:
        tuple[float, float, float, float]: (void_ratio, std_void_ratio, density, moisture_numeric).
    """
    # 'wd/wo/wl' kept for signature compatibility; values come from per-row initial_moisture_values
    moisture_numeric = initial_moisture_values[moisture_input]
    predicted_void_ratio, std_predicted_void_ratio = predict_void_ratio(lwd_modulus, moisture_numeric, alpha, initial_void_ratio_guess, gpr_model)
    predicted_density = void_ratio_to_density(predicted_void_ratio, moisture_numeric)
    return predicted_void_ratio, std_predicted_void_ratio, predicted_density, moisture_numeric
# Load GPR model
def load_gpr_model(filepath):
    """Load a trained GPR model or return a SageMaker endpoint adapter if configured."""
    endpoint = os.getenv('SAGEMAKER_ENDPOINT_NAME')
    if endpoint:
        return SageMakerPredictorAdapter(endpoint, os.getenv('AWS_REGION'))
    return joblib.load(filepath)

# Function to load data
 
 

def train_and_evaluate_gpr_kernels(kernels, X_a, residuals, physical_density, NDG_density, lane_size, option, cv_splits=3):
    """
    Train and evaluate Gaussian Process Regressor models with different kernels using cross-validation.

    Parameters:
    - kernels: A dictionary of kernel names and corresponding kernel objects.
    - X_a: The input matrix for the GPR model (e.g., LWD, x_coord, y_coord, moisture, suction).
    - residuals: The residuals from the physical model (observed - predicted).
    - physical_density: The predicted physical density values.
    - NDG_density: The observed NDG density values.
    - lane_size: The number of data points per lane.
    - option: The option number (1 or 2) for distinguishing between different experiments.
    - cv_splits: Number of cross-validation splits (default is 5).

    Returns:
    - best_gpr: The GPR model with the highest R² score.
    - best_kernel_name: The name of the best-performing kernel.
    - best_r2: The highest cross-validated R² score achieved.
    - best_rmse: The lowest cross-validated RMSE achieved.
    """
    # Convert lists to NumPy arrays if they aren't already
    X_a = np.array(X_a)
    residuals = np.array(residuals)
    physical_density = np.array(physical_density)
    NDG_density = np.array(NDG_density)

    best_r2 = -np.inf
    best_rmse = np.inf
    best_kernel_name = None
    best_gpr = None

    # Define the cross-validation scheme
    cfg = get_config()
    kf = KFold(n_splits=int(cfg["cv"]["splits"]), shuffle=True, random_state=42)

    # Open the file for logging R² and RMSE scores for this option
    with open(f"performance_metrics_option_{option}.txt", "w", encoding="utf-8") as file:
        for kernel_name, kernel in kernels.items():
            r2_scores = []
            rmse_scores = []

            for train_index, test_index in kf.split(X_a):
                X_train, X_test = X_a[train_index], X_a[test_index]
                residuals_train, residuals_test = residuals[train_index], residuals[test_index]
                physical_density_train, physical_density_test = physical_density[train_index], physical_density[test_index]
                NDG_density_test = NDG_density[test_index]

                # Debugging: Print shapes and types
                print(f"Kernel: {kernel_name}")
                print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
                print(f"residuals_train shape: {residuals_train.shape}, residuals_test shape: {residuals_test.shape}")
                print(f"physical_density_train shape: {physical_density_train.shape}, physical_density_test shape: {physical_density_test.shape}")
                print(f"NDG_density_test shape: {NDG_density_test.shape}")

                # Train GPR model
                gpr_option1 = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=int(cfg["gpr"]["n_restarts_optimizer"]),
                    alpha=float(cfg["gpr"]["alpha"]),
                )
                gpr_option1.fit(X_train, residuals_train)
                
                # Predict on the test data
                y_pred = gpr_option1.predict(X_test) + physical_density_test

                # Debugging: Print predictions and actual values
                print(f"y_pred shape: {y_pred.shape}, NDG_density_test shape: {NDG_density_test.shape}")
                print(f"y_pred: {y_pred}")
                print(f"NDG_density_test: {NDG_density_test}")

                # Calculate R² score
                r2 = r2_score(NDG_density_test, y_pred)
                r2_scores.append(r2)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(NDG_density_test, y_pred))
                rmse_scores.append(rmse)

            # Average the cross-validation scores
            avg_r2 = np.mean(r2_scores)
            avg_rmse = np.mean(rmse_scores)

            # Print and log the results
            print(f'{kernel_name} Kernel Avg R²: {avg_r2:.4f}, Avg RMSE: {avg_rmse:.4f}')
            file.write(f'{kernel_name} Kernel Avg R²: {avg_r2:.4f}, Avg RMSE: {avg_rmse:.4f}\n')
            
            # Check if this is the best model
            if avg_r2 > best_r2 or (avg_r2 == best_r2 and avg_rmse < best_rmse):
                best_r2 = avg_r2
                best_rmse = avg_rmse
                best_kernel_name = kernel_name
                best_gpr = gpr_option1
        
        file.write(f'\nBest Kernel: {best_kernel_name} with Avg R²: {best_r2:.4f}, Avg RMSE: {best_rmse:.4f}\n')

    print(f'\nBest Kernel: {best_kernel_name} with Avg R²: {best_r2:.4f}, Avg RMSE: {best_rmse:.4f}')
    
    return best_gpr, best_kernel_name, best_r2, best_rmse
 

def calculate_ndg_suction(NDG_void, moisture_numeric_array, alpha_list, moisture_input_list):
    """
    Calculate NDG suction for each data point.

    Parameters:
    - NDG_void: Array of NDG void ratios.
    - moisture_numeric_array: Array of moisture numeric values.
    - alpha_list: List of alpha values for each data point.
    - moisture_input_list: List of moisture inputs corresponding to each data point.

    Returns:
    - NDG_suction_array: Array of calculated NDG suction values.
    """
    NDG_suction_list = []

    for i in range(len(NDG_void)):
        # Use the alpha value provided in the alpha_list
        alpha = alpha_list[i]
        
        # Calculate suction using the NDG_void, moisture_numeric_array, and alpha
        NDG_suction = calculate_suction(NDG_void[i], moisture_numeric_array[i], alpha)
        
        # Append the calculated suction to the list
        NDG_suction_list.append(NDG_suction)

    # Convert the list to a numpy array
    NDG_suction_array = np.array(NDG_suction_list)
    
    return NDG_suction_array


def write_optimal_alpha_to_file(optimal_alpha, option):
    """Persist the optimized alpha value for a given option to text file.

    Args:
        optimal_alpha: The optimized alpha scalar.
        option: Option name/identifier used in file naming.
    """
    with open(f"optimal_alpha_option_{option}.txt", "w", encoding="utf-8") as file:
        file.write(f"Optimal alpha for Option {option}: {optimal_alpha:.6f}\n")
