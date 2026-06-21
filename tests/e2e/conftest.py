import pytest
import subprocess
import sys
import os
import json
import shutil

@pytest.fixture
def override_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(project_root, "isf_optimization_config.json")
    backup_path = config_path + ".backup"
    
    backup_created = False
    if os.path.exists(config_path):
        shutil.copyfile(config_path, backup_path)
        backup_created = True
        
    def _override(new_config_dict):
        with open(config_path, "w") as f:
            json.dump(new_config_dict, f, indent=2)
            
    yield _override
    
    # Restore
    if backup_created:
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, config_path)
            try:
                os.remove(backup_path)
            except Exception:
                pass
    else:
        if os.path.exists(config_path):
            try:
                os.remove(config_path)
            except Exception:
                pass

@pytest.fixture
def subprocess_runner():
    def _run(script_relative_path, args=None, env=None):
        if args is None:
            args = []
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script_path = os.path.abspath(os.path.join(project_root, script_relative_path))
        
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
            
        # Ensure project root is in PYTHONPATH
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        if existing_pythonpath:
            run_env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}"
        else:
            run_env["PYTHONPATH"] = project_root

        cmd = [sys.executable, script_path] + args
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root,
            env=run_env
        )
        return result
        
    return _run
