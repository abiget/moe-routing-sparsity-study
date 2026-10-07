from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM


class MoEAdapter(ABC):
    @abstractmethod
    def update_model_top_k(self, model, new_top_k) -> AutoModelForCausalLM:
        """
        Update the top_k value for each layer in the model.
        """
        pass
