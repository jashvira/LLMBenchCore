import hashlib
import inspect
from typing import Optional, Union, Callable, Tuple

# Placebo data provider - must be set by consuming benchmark
_placebo_data_provider: Optional[Callable[[str, int, int], Tuple[Optional[Union[dict, str]],
                                                                 str]]] = None

_models = []


def set_placebo_data_provider(
    models: list, provider: Callable[[str, int, int], Tuple[Optional[Union[dict, str]],
                                                            str]]) -> None:
  """
  Set the placebo data provider function.
  
  This should be called by the consuming benchmark to provide pre-defined
  responses for placebo baselines.
  
  Args:
      provider: A function that takes (model_name, question_num, subpass) and returns
                the expected response, or None if no response is available, plus
                an optional reasoning string.
  """
  global _placebo_data_provider
  global _models
  _models = models
  _placebo_data_provider = provider


class PlaceboEngine:
  """
  Placebo AI Engine class for testing with pre-defined responses.
  
  This engine returns pre-defined placebo responses for benchmark tests.
  It doesn't make any API calls and allows per-model response sets.
  
  Consuming benchmarks must call set_placebo_data_provider() to register their
  response data before using this engine.
  """

  def __init__(self, model_name: str):
    self.model_name = model_name
    self.configAndSettingsHash = hashlib.sha256(f"Placebo:{model_name}".encode("utf-8")).hexdigest()

  @staticmethod
  def _call_provider(provider: Callable[..., Tuple[Optional[Union[dict, str]], str]],
                     model_name: str, questionNum: int, subPass: int):
    signature = inspect.signature(provider)
    params = list(signature.parameters.values())
    supports_varargs = any(param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
                           for param in params)
    if supports_varargs or len(params) >= 3:
      return provider(model_name, questionNum, subPass)
    if len(params) == 2:
      return provider(questionNum, subPass)
    raise ValueError("Placebo data provider must accept (model_name, question_num, subpass).")

  def AIHook(self, prompt: str, structure: Optional[dict], questionNum: int, subPass: int):
    """Dispatch to the appropriate question module for placebo responses."""
    if _placebo_data_provider is not None:
      result, reasoning = self._call_provider(
        _placebo_data_provider,
        self.model_name,
        questionNum,
        subPass,
      )
      if result is not None:
        return result, reasoning

      # No response found for this question/subpass
      return "", reasoning

    # No response found for this question/subpass
    return "", ""


def get_placebo_model_configs() -> list:
  configs = []
  for raw_name in _models:
    name = raw_name.strip()
    if not name:
      continue
    configs.append({
      "name": name,
      "engine": "placebo",
      "env_key": None,
      "placebo_id": name,
    })
  return configs
