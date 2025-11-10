import os
from lib.test.evaluation.environment import EnvSettings
from lib.train.admin.environment import env_settings as train_env_settings

def local_env_settings():
    """Setup paths for Kaggle environment."""
    train_env = train_env_settings()

    # Base workspace
    workspace_dir = getattr(train_env, 'workspace_dir', '/kaggle/working') or os.getcwd()

    # Set evaluation outputs inside Kaggle /working folder
    results_path = os.path.join(workspace_dir, 'test', 'tracking_results')
    segmentation_path = os.path.join(workspace_dir, 'test', 'segmentation_results')
    network_path = os.path.join(workspace_dir, 'test', 'networks')
    result_plot_path = os.path.join(workspace_dir, 'test', 'result_plots')

    # Ensure directories exist
    for p in [results_path, segmentation_path, network_path, result_plot_path]:
        os.makedirs(p, exist_ok=True)

    s = EnvSettings()
    s.results_path = results_path
    s.segmentation_path = segmentation_path
    s.network_path = network_path
    s.result_plot_path = result_plot_path

    # Mirror training dataset paths (adjust to Kaggle /kaggle/working or /kaggle/temp)
    s.lasot_path = getattr(train_env, 'lasot_dir', '/kaggle/temp/LaSOT_partial')
    s.got10k_path = getattr(train_env, 'got10k_dir', '/kaggle/temp/got10k')
    s.trackingnet_path = getattr(train_env, 'trackingnet_dir', '/kaggle/temp/trackingnet')
    s.youtubevos_dir = getattr(train_env, 'youtubevos_dir', '/kaggle/temp/youtubevos')
    s.davis_dir = getattr(train_env, 'davis_dir', '/kaggle/temp/davis')

    # Optional empty datasets (if not used)
    s.otb_path = ''
    s.nfs_path = ''
    s.uav_path = ''
    s.tpl_path = ''
    s.vot_path = ''

    # Empty results for packed datasets
    s.got_packed_results_path = ''
    s.got_reports_path = ''
    s.tn_packed_results_path = ''

    return s
