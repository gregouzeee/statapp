from typing import Union

from crawl import fetch_html_and_extract_text, ErrorModel
from summarize import summarize_text_with_uncertainty


async def fetch_and_summarize(url: str) -> Union[dict, ErrorModel]:
    result = await fetch_html_and_extract_text(url)

    if isinstance(result, ErrorModel):
        return result

    try:
        summary_result = summarize_text_with_uncertainty(result["source_text"])
    except Exception as e:
        return ErrorModel(error=f"Erreur lors du résumé pour {url}: {e}")

    return {
        "url": result["url"],
        "title": result["title"],
        "html": result["html"],
        "source_text": result["source_text"],
        "summary": summary_result["summary"],
        "uncertainty": summary_result["uncertainty"],
        "model": summary_result["model"],
    }