import time
import logging
from functools import wraps
from typing import Callable


def timing(method: Callable) -> Callable:

    @wraps(method)
    def timed(*args, **kw):

        ts = time.time()
        result = method(*args, **kw)
        log = logging.getLogger(__name__)

        te = time.time()
        log.info("Run time: %r  %2.2f s" % (method.__name__, (te - ts)))
        return result

    return timed
