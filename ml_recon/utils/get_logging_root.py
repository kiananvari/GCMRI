# Utility to get logging path from config dict

def get_logging_root(config):
    """
    Returns the root logging directory as 'logging_dir/project_run_name'.
    config: dict loaded from YAML or Namespace with attributes.
    """
    if isinstance(config, dict):
        project = config.get('training_params', {}).get('project', 'project')
        run_name = config.get('training_params', {}).get('run_name', 'run')
        logging_dir = config.get('logging_dir', './logs')
    else:
        # Namespace or argparse args
        project = getattr(config, 'project', 'project')
        run_name = getattr(config, 'run_name', 'run')
        logging_dir = getattr(config, 'logging_dir', './logs')
    return f"{logging_dir}/{project}_{run_name}"
