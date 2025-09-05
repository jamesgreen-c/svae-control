from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def members(cls) -> list:
        return [e.value for e in cls]


class Dataset(BaseEnum):
    PENDULUM = "pendulum"
    LDS = "lds"
    CARTPOLE = "cartpole-swingup"
    WALKER = "walker-run"
    GPS = "gps"


class NNType(BaseEnum):
    NoOp = "NoOp"
    MLP = "MLP"
    CNN = "CNN"


class NNDistType(BaseEnum):
    GAUSSIAN = "GaussianNN"
    DIAG_GAUSSIAN = "DiagGaussianNN"
    DIAG_GAUSSIAN_FIXED_VAR = "DiagGaussianFixedVarNN"


class BetaSchedule(BaseEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
