import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import get_config
from data_interface import ensure_json_from_csv, load_dataset_json
from helper_hybrid import (
    calculate_elastic_state_array,
    calculate_rmse,
    load_gpr_model,
    optimize_global_mean_stress_beta,
    train_and_evaluate_gpr_kernels,
    void_ratio_to_density,
)
from hybrid_initial_settings import kernels


def read_coords(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read relative coordinates from dataset frame."""
    return df['rel_X (m)'].to_numpy(), df['rel_Y (m)'].to_numpy()


def prepare_dataset(json_path: str, default_moisture_label: str):
    """Load and transform dataset for model training and evaluation."""
    payload = load_dataset_json(json_path)
    df = pd.DataFrame(payload['data'])
    ndg_density = df['NDG-density'].to_numpy()
    lwd_av = df['LWD_av'].to_numpy()
    used_mask = df['usedfortraining'].astype(bool).to_numpy()
    x_coord, y_coord = read_coords(df)

    omc = df['OMC'].astype(float).to_numpy()
    moisture_input_list = df.get(
        'moisture_label',
        pd.Series([default_moisture_label] * len(df)),
    ).astype(str).str.lower().tolist()

    omc_offset = float(get_config()['omc']['offset'])
    moisture_numeric_list = []
    for i, label in enumerate(moisture_input_list):
        if label == 'low':
            moisture_numeric_list.append(omc[i] - omc_offset)
        elif label == 'high':
            moisture_numeric_list.append(omc[i] + omc_offset)
        else:
            moisture_numeric_list.append(omc[i])
    moisture_numeric_array = np.array(moisture_numeric_list)

    cfg = get_config()
    gs = float(cfg['soil']['Gs'])
    ndg_void = gs / (ndg_density / (1 + moisture_numeric_array / 100.0)) - 1.0
    return ndg_density, lwd_av, moisture_input_list, moisture_numeric_array, ndg_void, x_coord, y_coord, used_mask, omc


def build_elastic_feature_matrix(lwd_av, x_coord, y_coord, moisture_numeric_array, physics_state):
    """Create ML features from measured inputs plus physics-derived states."""
    return np.column_stack((
        lwd_av,
        x_coord,
        y_coord,
        moisture_numeric_array,
        physics_state['void_ratio'],
        physics_state['saturation'],
        physics_state['gmax_feature'],
        physics_state['bulk_feature'],
        physics_state['elastic_modulus'],
        lwd_av - physics_state['elastic_modulus'],
    ))


def run_option(option_name: str, csv_path: str, json_path: str, cfg, gpr_model):
    """Run the physics-based residual ML pipeline for one data option."""
    json_ready = ensure_json_from_csv(csv_path, json_path)
    ndg_density, lwd_av, moisture_input_list, moisture_numeric_array, ndg_void, x_coord, y_coord, used_mask, omc = prepare_dataset(
        json_ready, cfg['moisture']['default_label']
    )

    mean_stress = optimize_global_mean_stress_beta(lwd_av, moisture_numeric_array, ndg_density, used_mask, cfg)
    global_pnet = mean_stress['pnet']
    physics_state = calculate_elastic_state_array(lwd_av, moisture_numeric_array, global_pnet, cfg)
    physical_density = np.array([
        void_ratio_to_density(e_value, moisture)
        for e_value, moisture in zip(physics_state['void_ratio'], moisture_numeric_array)
    ])
    residuals = ndg_density - physical_density
    elastic_features = build_elastic_feature_matrix(
        lwd_av, x_coord, y_coord, moisture_numeric_array, physics_state
    )

    xa_train = elastic_features[used_mask]
    residual_train = residuals[used_mask]
    ndg_train = ndg_density[used_mask]
    physics_train = physical_density[used_mask]

    gpr_model_opt, best_kernel_name, best_r2, best_rmse = train_and_evaluate_gpr_kernels(
        kernels=kernels,
        X_a=xa_train,
        residuals=residual_train,
        physical_density=physics_train,
        NDG_density=ndg_train,
        lane_size=residual_train.shape[0],
        option=option_name,
    )

    gpr_model_opt.fit(xa_train, residual_train)

    hold_mask = ~used_mask
    pred_residual = gpr_model_opt.predict(elastic_features[hold_mask])
    pred_density = physical_density[hold_mask] + pred_residual
    actual_density = ndg_density[hold_mask]

    rmse = calculate_rmse(actual_density, pred_density)
    r2 = r2_score(actual_density, pred_density)

    pd.DataFrame({
        'Actual_NDG_Density': actual_density,
        'Predicted_Density': pred_density,
        'Physical_Density': physical_density[hold_mask],
        'Predicted_Residual': pred_residual,
        'Estimated_Void_Ratio': physics_state['void_ratio'][hold_mask],
        'Estimated_Saturation': physics_state['saturation'][hold_mask],
        'Global_Pnet': np.full(np.sum(hold_mask), global_pnet),
        'Global_Beta': np.full(np.sum(hold_mask), mean_stress['beta']),
        'Mean_Stress_Offset_kPa': np.full(np.sum(hold_mask), mean_stress['stress_offset_kpa']),
        'Unit_Weight_kN_m3': np.full(np.sum(hold_mask), mean_stress['unit_weight_kn_m3']),
        'Overburden_Mean_kPa': np.full(np.sum(hold_mask), mean_stress['overburden_mean_kpa']),
        'LWD_Contact_kPa': np.full(np.sum(hold_mask), mean_stress['lwd_contact_kpa']),
        'LWD_Mean_kPa': np.full(np.sum(hold_mask), mean_stress['lwd_mean_kpa']),
        'Effective_Pressure': physics_state['effective_pressure'][hold_mask],
        'Gmax_Raw': physics_state['gmax_raw'][hold_mask],
        'Gmax_Scaled': physics_state['gmax'][hold_mask],
        'Gmax_Feature': physics_state['gmax_feature'][hold_mask],
        'Bulk_Modulus_Raw': physics_state['bulk_modulus_raw'][hold_mask],
        'Bulk_Modulus': physics_state['bulk_modulus'][hold_mask],
        'Bulk_Feature': physics_state['bulk_feature'][hold_mask],
        'Elastic_Modulus_Physics': physics_state['elastic_modulus'][hold_mask],
        'ELWD_minus_Ephysics': (lwd_av - physics_state['elastic_modulus'])[hold_mask],
    }).to_csv(f'prediction_vs_ndg_{option_name}.csv', index=False)

    _, rounded_counts = np.unique(np.round(pred_density, 6), return_counts=True)
    repeated_prediction_count = int(np.sum(rounded_counts[rounded_counts > 1] - 1))
    gmax_unique_count = int(len(np.unique(np.round(physics_state['gmax_feature'][hold_mask], 6))))

    with open(f'final_results_{option_name}.txt', 'w', encoding='utf-8') as f:
        f.write(f'Final R2 on holdout with physics-based residual ML: {r2:.4f}\n')
        f.write(f'Final RMSE on holdout with physics-based residual ML: {rmse:.4f}\n')
        f.write(f'Best residual kernel: {best_kernel_name}\n')
        f.write(f'Global Pnet: {global_pnet:.6f}\n')
        f.write(f'Global beta: {mean_stress["beta"]:.6f}\n')
        f.write(f'Mean stress offset: {mean_stress["stress_offset_kpa"]:.6f} kPa\n')
        f.write(f'Unit weight: {mean_stress["unit_weight_kn_m3"]:.6f} kN/m3\n')
        f.write(f'Overburden mean stress: {mean_stress["overburden_mean_kpa"]:.6f} kPa\n')
        f.write(f'LWD contact stress: {mean_stress["lwd_contact_kpa"]:.6f} kPa\n')
        f.write(f'LWD mean stress contribution at beta=1: {mean_stress["lwd_mean_kpa"]:.6f} kPa\n')
        f.write(f'Gmax scale factor: {float(cfg.get("elastic_physics", {}).get("gmax_to_modulus_factor", 1.0)):.6g}\n')
        f.write(f'Pa to modulus factor: {float(cfg.get("elastic_physics", {}).get("pa_to_modulus_factor", 0.001)):.6g}\n')
        f.write(f'Bulk scale factor: {float(cfg.get("elastic_physics", {}).get("bulk_to_modulus_factor", 1.0)):.6g}\n')
        f.write(f'Elastic physics enabled: {bool(cfg.get("elastic_physics", {}).get("enabled", True))}\n')
        f.write(f'Unique Gmax feature values on holdout: {gmax_unique_count}\n')
        f.write(f'Repeated predicted density count at 6 decimals: {repeated_prediction_count}\n')


def main():
    """Entry point: configure logging, load model, run both options."""
    cfg = get_config()
    log_path = cfg.get('logging', {}).get('file', 'run.log')
    if not os.path.isabs(log_path):
        log_path = os.path.join(os.path.dirname(__file__), log_path)
    mode = 'a' if cfg.get('logging', {}).get('append', True) else 'w'
    log_fh = open(log_path, mode)
    sys.stdout = log_fh
    sys.stderr = log_fh
    np.random.seed(int(cfg['seed']))
    gpr_model = load_gpr_model(cfg['model']['path'])

    run_option('option1', cfg['data']['csv_option1'], cfg['data']['json_option1'], cfg, gpr_model)
    run_option('option2', cfg['data']['csv_option2'], cfg['data']['json_option2'], cfg, gpr_model)
    log_fh.flush()
    log_fh.close()


if __name__ == '__main__':
    main()
