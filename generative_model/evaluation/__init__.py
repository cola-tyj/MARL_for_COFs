"""模型无关的 COF 分子生成评测接口。

The heavier report validator depends on :mod:`jsonschema`, while generation
environments only need geometry helpers.  Public names are therefore imported
lazily so importing a focused evaluation submodule does not pull optional
reporting dependencies into ``env_etflow``.
"""

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "REPORT_SCHEMA_PATH",
    "EvaluationConfig",
    "EvaluationInputError",
    "EvaluationReference",
    "EvaluationSample",
    "FailureReason",
    "build_reference",
    "evaluate",
    "validate_report",
    "write_report",
]


def __getattr__(name: str):
    if name in {"EvaluationReference", "build_reference"}:
        from .chemistry import EvaluationReference, build_reference

        return {
            "EvaluationReference": EvaluationReference,
            "build_reference": build_reference,
        }[name]
    if name in {"EvaluationConfig", "evaluate", "validate_report", "write_report"}:
        from .evaluator import EvaluationConfig, evaluate, validate_report, write_report

        return {
            "EvaluationConfig": EvaluationConfig,
            "evaluate": evaluate,
            "validate_report": validate_report,
            "write_report": write_report,
        }[name]
    if name in {
        "EVALUATION_SCHEMA_VERSION",
        "REPORT_SCHEMA_PATH",
        "EvaluationInputError",
        "EvaluationSample",
        "FailureReason",
    }:
        from .schema import (
            EVALUATION_SCHEMA_VERSION,
            REPORT_SCHEMA_PATH,
            EvaluationInputError,
            EvaluationSample,
            FailureReason,
        )

        return {
            "EVALUATION_SCHEMA_VERSION": EVALUATION_SCHEMA_VERSION,
            "REPORT_SCHEMA_PATH": REPORT_SCHEMA_PATH,
            "EvaluationInputError": EvaluationInputError,
            "EvaluationSample": EvaluationSample,
            "FailureReason": FailureReason,
        }[name]
    raise AttributeError(name)
