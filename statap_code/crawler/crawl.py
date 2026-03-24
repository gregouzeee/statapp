from typing import Optional, Union
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
import re
import logging
from pydantic import BaseModel

class ErrorModel(BaseModel):
    error: str

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)   # supprime [1], [2], etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_title(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return None


def extract_wikipedia_text(soup: BeautifulSoup) -> str:
    """
    Extrait le texte principal d'un article Wikipédia.
    On essaie plusieurs noeuds typiques de Wikipédia.
    """
    main_candidates = [
        soup.find("main"),
        soup.find(id="bodyContent"),
        soup.find(id="mw-content-text"),
        soup.find(class_="mw-parser-output"),
    ]

    content_node = next((node for node in main_candidates if node is not None), None)
    if content_node is None:
        return ""

    # Supprimer les éléments parasites
    for bad in content_node.find_all([
        "script", "style", "table", "sup", "figure", "nav",
        "footer", "aside", "noscript"
    ]):
        bad.decompose()

    # Supprimer certains blocs spécifiques à Wikipédia
    wikipedia_bad_classes = [
        "reference",
        "reflist",
        "navbox",
        "infobox",
        "thumb",
        "mw-editsection",
        "metadata",
        "ambox",
        "hatnote",
        "toc"
    ]

    for cls in wikipedia_bad_classes:
        for tag in content_node.find_all(class_=cls):
            tag.decompose()

    paragraphs = []
    for p in content_node.find_all("p"):
        txt = p.get_text(" ", strip=True)
        txt = clean_text(txt)
        if len(txt) >= 40:
            paragraphs.append(txt)

    return "\n".join(paragraphs)


async def fetch_html_and_extract_text(url: str) -> Union[dict, ErrorModel]:
    logger.info(f"Début du traitement pour l'URL : {url}")

    # Étape 1 : récupération du HTML
    try:
        logger.info("Téléchargement du HTML...")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            html = result.html

        if not html or not isinstance(html, str) or not html.strip():
            logger.warning("HTML vide ou non récupéré.")
            return ErrorModel(error=f"HTML vide ou non récupéré pour {url}")

        logger.info("HTML récupéré avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement HTML : {e}", exc_info=True)
        return ErrorModel(error=f"Erreur lors du téléchargement de la page pour {url}: {e}")

    # Étape 2 : parsing HTML
    try:
        logger.info("Parsing HTML avec BeautifulSoup...")
        soup = BeautifulSoup(html, "html.parser")
        logger.info("Parsing terminé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors du parsing HTML : {e}", exc_info=True)
        return ErrorModel(error=f"Erreur lors du parsing HTML pour {url}: {e}")

    # Étape 3 : extraction du texte principal
    try:
        logger.info("Extraction du texte principal...")
        title = get_title(soup)
        source_text = extract_wikipedia_text(soup)

        if not source_text or not source_text.strip():
            logger.warning("Aucun texte principal extrait.")
            return ErrorModel(error=f"Aucun texte principal extrait pour {url}")

        logger.info("Texte principal extrait avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du texte : {e}", exc_info=True)
        return ErrorModel(error=f"Erreur lors de l'extraction du texte principal pour {url}: {e}")

    return {
        "url": url,
        "title": title,
        "html": html,
        "source_text": source_text,
    }