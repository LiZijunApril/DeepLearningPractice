import logging.config
import logging.handlers
import json
import pathlib

# logger = logging.getLogger("my_app")

logging_config = {
    "version": 1, # 1是必须的，为了确保将来更改东西不会破坏旧代码
    "disable_existing_loggers": False,  # 禁用此配置中不存在的任何内容（disable anything that`s not explicity listed in this config）
    "formatters": {
        "simple": {
            "format":"%(levelname)s: %(message)s",
        },
        "detailed": {
            # "format": "[%(levelname)s - %(module)s - L%(lineno)d] %(asctime)s: %(message)s",
            # "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            "format": "[%(levelname)s - %(module)s] - %(asctime)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
            # "datefmt": "%Y-%m-%dT%H:%M:%S%z"
        }
    },
    "handlers": { # handler 处理器
        "stderr": { # 
            "class": "logging.StreamHandler", 
            "level": "WARNING",
            "formatter": "simple", # 把“simple”格式化器设置为此处理程序的格式化器(set our "simple" formatter as the formater for this handler)
            "stream": "ext://sys.stdout",    
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/my_app.log",
            "maxBytes": 1000000,
            "backupCount": 3
        },
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "logs/my_app.log",
            "maxBytes": 1000000,  # 每个日志文件的最大大小（字节）
            "backupCount": 3  # 备份日志文件的数量
        }
    },
    "loggers": {
        "my_logger": {
            "level": "DEBUG",
            "handlers":[
                "stderr",
                "file",
                # "file_handler"
            ]
        },
        "my_logger2": {
            "level": "DEBUG",
            "handlers":[
                "file_handler"
            ]
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers":[
            "stderr",
            # "file",
            # "file_handler"
        ]
    }
}

def setup_logging():
    config_file = pathlib.Path('config.json')
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)

def main():
    # logging.config.dictConfig(config=logging_config)
    # logging.config.dictConfig(logging_config)
    logger = logging.getLogger("my_logger2")
    # setup_logging()
        
    logger.debug("This is a debug message")  # 这条日志会被记录到文件中
    logger.info("This is an info message")    # 这条日志会被记录到文件中
    logger.warning("This is a warning message")  # 这条日志会被记录到文件中
    logger.error("This is an error message")  # 这条日志会被记录到文件中
    logger.critical("This is a critical message")  # 这条日志会被记录到文件中

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("exception message")

if __name__ == "__main__":
    main() 