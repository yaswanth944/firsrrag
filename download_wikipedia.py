
import requests
from pathlib import Path

# Wikipedia article titles
articles = {
    "Google": "Google",
    "Microsoft": "Microsoft",
    "Tesla": "Tesla, Inc.",
    "Nvidia": "Nvidia",
    "SpaceX": "SpaceX",
}

# Folder where text files will be saved
output_dir = Path("docs")
output_dir.mkdir(exist_ok=True)

# Wikipedia API endpoint
url = "https://en.wikipedia.org/w/api.php"

headers = {
    "User-Agent": "MyRAGProject/1.0 (educational project)"
}

for filename, title in articles.items():
    try:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "titles": title,
            "redirects": 1,
        }

        response = requests.get(
            url, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()

        pages = response.json()["query"]["pages"]

        page = next(iter(pages.values()))
        text = page.get("extract", "")

        if not text:
            print(f"Could not retrieve: {title}")
            continue

        # Save the article as a UTF-8 text file
        file_path = output_dir / f"{filename.lower()}.txt"
        file_path.write_text(
            f"{page['title']}\n"
            f"Source: https://en.wikipedia.org/wiki/"
            f"{requests.utils.quote(title.replace(' ', '_'))}\n\n"
            f"{text}",
            encoding="utf-8",
        )

        print(f"Downloaded: {file_path} ({len(text)} characters)")

    except (requests.RequestException, KeyError, ValueError) as error:
        print(f"Failed to download {title}: {error}")

print("\nDownload process completed.")
