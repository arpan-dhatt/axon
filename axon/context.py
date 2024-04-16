from contextvars import ContextVar

backend = ContextVar('axon.context.backend', default=None)
