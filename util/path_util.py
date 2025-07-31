from pathlib import Path

def get_root_dir():
    return Path(__file__).resolve().parent.parent

def get_config_dir():
    return Path(__file__).resolve().parent.parent / "config"

def get_user_define_dir():
    return Path(__file__).resolve().parent.parent / "user_define"

def get_result_dir():
    return Path(__file__).resolve().parent.parent / "result"