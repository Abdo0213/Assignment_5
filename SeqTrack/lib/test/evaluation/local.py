import os
from lib.test.evaluation.environment import EnvSettings
from lib.train.admin.environment import env_settings as train_env_settings


def local_env_settings():
    """
    Test/evaluation paths are copied from the training local settings.
    Edit lib/train/admin/local.py (EnvironmentSettings) to change roots,
    then this file will reflect the same dataset/workspace locations.
    """
    train_env = train_env_settings()

    workspace_dir = getattr(train_env, 'workspace_dir', '') or os.getcwd()
    results_path = os.path.join(workspace_dir, 'test', 'tracking_results')
    segmentation_path = os.path.join(workspace_dir, 'test', 'segmentation_results')
    network_path = os.path.join(workspace_dir, 'test', 'networks')
    result_plot_path = os.path.join(workspace_dir, 'test', 'result_plots')

    s = EnvSettings()
    s.results_path = results_path
    s.segmentation_path = segmentation_path
    s.network_path = network_path
    s.result_plot_path = result_plot_path

    # Datasets: mirror training local paths
    s.otb_path = ''
    s.nfs_path = ''
    s.uav_path = ''
    s.tpl_path = ''
    s.vot_path = ''
    s.got10k_path = getattr(train_env, 'got10k_dir', '')
    s.lasot_path = getattr(train_env, 'lasot_dir', '')
    s.trackingnet_path = getattr(train_env, 'trackingnet_dir', '')
    s.davis_dir = getattr(train_env, 'davis_dir', '')
    s.youtubevos_dir = getattr(train_env, 'youtubevos_dir', '')
    s.got_packed_results_path = ''
    s.got_reports_path = ''
    s.tn_packed_results_path = ''

    # Ensure output dirs exist
    for p in [s.results_path, s.segmentation_path, s.network_path, s.result_plot_path]:
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass

    return s
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k_lmdb'
    settings.got10k_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot_extension_subset'
    settings.lasot_lmdb_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot_lmdb'
    settings.lasot_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\lasot'
    settings.network_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/networks'    # Where tracking networks are stored.
    settings.nfs_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\nfs'
    settings.otb_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\OTB2015'
    settings.prj_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack'
    settings.result_plot_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/result_plots'
    settings.results_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack'
    settings.segmentation_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\test/segmentation_results'
    settings.tc128_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\trackingnet'
    settings.uav_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\UAV123'
    settings.vot_path = 'D:\College\Level 4\Semester 1\Image Processing\Assignments_Solve\VideoX\SeqTrack\data\VOT2019'
    settings.youtubevos_dir = ''

    return settings

