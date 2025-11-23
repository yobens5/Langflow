
   
# gamma_generate_component.py
import time
from typing import Dict, Any

import requests
from langflow.custom import Component
from langflow.io import StrInput, DropdownInput, MessageTextInput, Output
from langflow.schema import Message

GAMMA_BASE_URL = "https://public-api.gamma.app/v1.0/generations"


class GammaGenerateComponent(Component):
    """
    Gamma Deck Generator

    - Takes text and optional params
    - Calls Gamma's /generations API
    - Polls until status == 'completed'
    - Returns Gamma URL (+ PDF/PPTX URLs if available) as a Message
    """

    display_name = "Gamma Deck Generator"
    description = "Generate a Gamma deck from text using Gamma's Generate API."
    icon = "presentation"
    name = "GammaGenerateComponent"

    # ------------------------
    # Inputs (none marked required)
    # ------------------------
    inputs = [
        StrInput(
            name="api_key",
            display_name="Gamma API Key",
            value="",
        ),
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
        ),
        DropdownInput(
            name="text_mode",
            display_name="Text Mode",
            options=["generate", "condense", "preserve"],
            value="generate",
        ),
        DropdownInput(
            name="format",
            display_name="Format",
            options=["presentation", "document", "webpage", "social"],
            value="presentation",
        ),
        StrInput(
            name="theme_id",
            display_name="Theme ID",
            value="",
        ),
        StrInput(
            name="additional_instructions",
            display_name="Additional Instructions",
            value="",
        ),
        DropdownInput(
            name="export_as",
            display_name="Export As",
            options=["", "pdf", "pptx"],
            value="",
        ),
    ]

    # ------------------------
    # Output
    # ------------------------
    outputs = [
        Output(
            name="gamma_deck",
            display_name="Gamma Deck",
            method="build",  # Langflow will call self.build()
        ),
    ]

    # ------------------------
    # Main logic
    # ------------------------
    def build(self) -> Message:
        # Values come directly from self.<name> (per docs)
        api_key = (self.api_key or "").strip()
        input_text = (self.input_text or "").strip()  # MessageTextInput → plain text

        if api_key == "":
            return Message(text="ERROR: Missing Gamma API key.")

        if input_text == "":
            return Message(text="ERROR: No input text provided.")

        text_mode = self.text_mode or "generate"
        fmt = self.format or "presentation"
        theme_id = (self.theme_id or "").strip() or None
        extra = (self.additional_instructions or "").strip() or None
        export_as = (self.export_as or "").strip() or None
        if export_as == "":
            export_as = None

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        }

        body: Dict[str, Any] = {
            "inputText": input_text,
            "textMode": text_mode,
            "format": fmt,
        }
        if theme_id:
            body["themeId"] = theme_id
        if extra:
            body["additionalInstructions"] = extra
        if export_as:
            body["exportAs"] = export_as

        # Call Gamma POST /generations
        try:
            resp = requests.post(GAMMA_BASE_URL, headers=headers, json=body, timeout=30)
        except Exception as e:
            return Message(text=f"ERROR: Error calling Gamma API: {e}")

        if not (200 <= resp.status_code < 300):
            return Message(text=f"ERROR: Gamma error: {resp.status_code} {resp.text}")

        try:
            data = resp.json()
        except Exception as e:
            return Message(
                text=f"ERROR: Could not decode Gamma response as JSON: {e}. Raw response: {resp.text}"
            )

        generation_id = data.get("generationId")
        if not generation_id:
            return Message(text=f"ERROR: Gamma did not return generationId: {data}")

        # Poll generation status
        status_url = f"{GAMMA_BASE_URL}/{generation_id}"
        gamma_url = None
        pdf_url = None
        pptx_url = None

        for _ in range(24):  # ~1 minute with 5 sec sleep
            time.sleep(5)
            try:
                r = requests.get(
                    status_url,
                    headers={"X-API-KEY": api_key, "accept": "application/json"},
                    timeout=120,
                )
            except Exception as e:
                return Message(text=f"ERROR: Error polling Gamma: {e}")

            if not (200 <= r.status_code < 300):
                return Message(text=f"ERROR: Poll error: {r.status_code} {r.text}")

            try:
                status_data = r.json()
            except Exception as e:
                return Message(
                    text=f"ERROR: Could not decode poll response as JSON: {e}. Raw response: {r.text}"
                )

            status = status_data.get("status")

            if status == "pending":
                continue

            if status == "completed":
                gamma_url = status_data.get("gammaUrl")
                pdf_url = status_data.get("pdfUrl")
                pptx_url = status_data.get("pptxUrl")
                break

            return Message(text=f"ERROR: Gamma generation error: {status_data}")

        if not gamma_url:
            return Message(text="ERROR: Timeout. Gamma generation did not finish in time.")

        msg = f"Gamma deck: {gamma_url}"
        if pdf_url:
            msg += f"\nPDF: {pdf_url}"
        if pptx_url:
            msg += f"\nPPTX: {pptx_url}"

        return Message(text=msg)
