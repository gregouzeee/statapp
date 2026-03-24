import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote


LIST_URL = "https://fr.wikipedia.org/wiki/Wikipédia:Liste_d%27articles_que_toutes_les_encyclopédies_devraient_avoir"
BASE = "https://fr.wikipedia.org"


def is_valid_article_href(href: str) -> bool:
    if not href:
        return False

    if not href.startswith("/wiki/"):
        return False

    bad_prefixes = [
        "/wiki/Wikipédia:",
        "/wiki/Aide:",
        "/wiki/Spécial:",
        "/wiki/Special:",
        "/wiki/Catégorie:",
        "/wiki/Category:",
        "/wiki/Fichier:",
        "/wiki/File:",
        "/wiki/Portail:",
        "/wiki/Modèle:",
        "/wiki/Template:",
        "/wiki/Discussion:",
        "/wiki/Projet:",
    ]

    if any(href.startswith(prefix) for prefix in bad_prefixes):
        return False

    if "#" in href:
        return False

    return True


def get_essential_wikipedia_urls() -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; WikiArticleCollector/1.0)"
    }

    response = requests.get(LIST_URL, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    urls = []
    seen = set()

    # On cible le contenu principal de la page
    content = soup.find(id="mw-content-text")
    if content is None:
        raise RuntimeError("Impossible de trouver le contenu principal de la page.")

    for a in content.find_all("a", href=True):
        href = a["href"]

        if not is_valid_article_href(href):
            continue

        full_url = urljoin(BASE, href)

        if full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)

    return urls


if __name__ == "__main__":
    urls = get_essential_wikipedia_urls()
    print(f"Nombre d'URLs récupérées : {len(urls)}")
    print()

    for url in urls[:50]:
        print(url)