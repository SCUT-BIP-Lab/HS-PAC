import os

logger = None
need_loghead = True
def config_logging(log_file = "runtime.log", log_level = "INFO", from_info = True):
    global logger
    global need_loghead
    if not logger:
        import logging
        level = logging.INFO
        if log_level.lower() == "info":
            level = logging.INFO
        elif log_level.lower() == "debug":
            level = logging.DEBUG
        logger = logging.getLogger()
        logger.setLevel(level = log_level)
        log_dir = os.path.dirname(log_file)
        if(not os.path.isdir(log_dir)):
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        need_loghead = from_info
    return logger

def get_filename(whole_path):
    data = whole_path.strip().strip("/").split("/")
    return data[-1]

class logging(object):
    global logger
    @staticmethod
    def info(log_text):
        invoke_file = get_filename(inspect.stack()[1][1])
        invoke_line = inspect.stack()[1][2]
        log_head = "%s:%d " % (invoke_file, invoke_line) if need_loghead == True else ""
        config_logging().info(log_head + log_text)
    @staticmethod
    def debug(log_text):
        invoke_file = get_filename(inspect.stack()[1][1])
        invoke_line = inspect.stack()[1][2]
        log_head = "%s:%d " % (invoke_file, invoke_line) if need_loghead == True else ""
        config_logging().debug(log_head + log_text)
    @staticmethod
    def error(log_text):
        invoke_file = get_filename(inspect.stack()[1][1])
        invoke_line = inspect.stack()[1][2]
        log_head = "%s:%d " % (invoke_file, invoke_line) if need_loghead == True else ""
        config_logging().error(log_head + log_text)