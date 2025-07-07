from langflow.custom import Component
from langflow.io import FileInput, Output
from langflow.schema import Data
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


class PDFPageExtractorComponent(Component):
    display_name = "PDF Page Extractor"
    description = "Extracts text from each page of a PDF and assigns page number metadata."
    icon = "file-text"
    name = "PDFPageExtractor"

    inputs = [
        FileInput(
            name="pdf_file",
            display_name="PDF File",
            info="Upload a PDF file to extract its pages as Data objects.",
            required=True,
            file_types=["pdf"],
        )
    ]

    outputs = [
        Output(display_name="Pages", name="pages", method="extract_pages")
    ]

    def extract_pages(self) -> list[Data]:
        if not self.pdf_file:
            raise ValueError("No PDF file provided")

        try:
            file_path = self.pdf_file if isinstance(self.pdf_file, str) else getattr(self.pdf_file, "path", None)
            if not file_path:
                raise ValueError("Invalid file input. Cannot determine file path.")

            page_data = []
            for i, page_layout in enumerate(extract_pages(file_path)):
                texts = [element.get_text() for element in page_layout if isinstance(element, LTTextContainer)]
                page_text = "".join(texts).strip()
                if page_text:
                    page_data.append(Data(text=page_text, data={"page": i + 1}))
                else:
                    self.log(f"Page {i + 1} has no extractable text.")
            return page_data
        except Exception as e:
            self.status = f"Failed to extract PDF pages: {e}"
            return []
