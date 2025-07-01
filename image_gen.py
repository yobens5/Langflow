import base64
from io import BytesIO

from pydantic.v1 import SecretStr
from langflow.base.models.model import LCModelComponent
from langflow.base.models.openai_constants import OPENAI_MODEL_NAMES
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import BoolInput, DictInput, DropdownInput, IntInput, SecretStrInput, SliderInput, StrInput

OPENAI_MODEL_NAMES = ["gpt-image-1"]

class OpenAIModelComponent(LCModelComponent):
    display_name = "OpenAI Image"
    description = "Generates an image using OpenAI GPT-image-1 and returns base64 if needed."
    icon = "OpenAI"
    name = "OpenAIImageModel"

    inputs = [
        *LCModelComponent._base_inputs,
        
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=OPENAI_MODEL_NAMES,
            combobox=True,
        ),
        StrInput(
            name="openai_api_base",
            display_name="OpenAI API Base",
            advanced=True,
            info="The base URL of the OpenAI API. Defaults to https://api.openai.com/v1.",
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="The OpenAI API Key to use for the OpenAI model.",
            advanced=False,
            value="OPENAI_API_KEY",
            required=True,
        ),
        BoolInput(
            name="return_base64_only",
            display_name="Return Base64 Only",
            info="If True, only return the base64 image string.",
            advanced=False,
            value=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        from openai import OpenAI

        api_key = SecretStr(self.api_key).get_secret_value()
        base_url = self.openai_api_base or "https://api.openai.com/v1"
        model = self.model_name
        return_base64_only = self.return_base64_only

        client = OpenAI(api_key=api_key, base_url=base_url)

        def invoke(input_dict=None):
            prompt = str(input_dict[0]) if isinstance(input_dict, list) and len(input_dict) > 0 else ""
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size="1024x1024",
                quality="high",
                n=1
            )
            b64_data = response.data[0].b64_json

            if return_base64_only:
                return b64_data

            image_bytes = base64.b64decode(b64_data)
            buffer = BytesIO(image_bytes)
            buffer.name = "generated_image.png"
            return buffer

        class Runnable:
            def invoke(self, input_dict=None):
                return invoke(input_dict)

            def with_config(self, config):
                return self

        return Runnable()

    def _get_exception_message(self, e: Exception):
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            message = e.body.get("message")
            if message:
                return message
        return None
