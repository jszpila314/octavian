import logging

def setup_logger(log_file: str) -> logging.Logger:
  logger = logging.getLogger('OCTAVIAN')
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
  sh = logging.StreamHandler()
  sh.setLevel(logging.ERROR)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  fh = logging.FileHandler(log_file)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
  
  return logger