import logging.config
import pathlib

# 获取脚本所在的目录
script_dir = pathlib.Path(__file__).resolve().parent

# 配置日志字典
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": script_dir / "logs/my_app.log",
            "maxBytes": 1000000,  # 每个日志文件的最大大小（字节）
            "backupCount": 3  # 备份日志文件的数量
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["file_handler"]
    }
}

# 配置日志系统
logging.config.dictConfig(logging_config)

# 创建一个日志记录器
logger = logging.getLogger("my_app")

# 输出日志消息
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
