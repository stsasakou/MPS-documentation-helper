import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# The URL to scrape
url = "https://coolya.github.io/maintainable-generators/"

# The directory to store files in
output_dir = "./mps-other/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all links
links = soup.find_all("a", href=True)

# Print all original hrefs to verify what's being found
print("Original hrefs found on the page:")
print([link["href"] for link in links])

for link in links:
    href = link["href"]

    # Make a full URL if necessary
    full_url = urljoin(url, href)
    print(f"Full URL: {full_url}")  # Debug: print the full URL

    # Check if ".html" is in the URL or if it's a directory link
    if ".html" in href or href.endswith("/"):
        print(f"Attempting to download {full_url}")
        file_response = requests.get(full_url)

        # Check if the response was successful and if the content is HTML
        if file_response.status_code == 200 and "text/html" in file_response.headers["Content-Type"]:
            # Determine file name
            file_name = os.path.basename(href.strip("/")) + ".html"
            file_path = os.path.join(output_dir, file_name)

            # Write the HTML content to file
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(file_response.text)
            print(f"Saved: {file_path}")
        else:
            print(f"Skipping {full_url} (status: {file_response.status_code}, content type: {file_response.headers.get('Content-Type')})")
