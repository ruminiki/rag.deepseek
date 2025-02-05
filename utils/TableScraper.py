import requests
from bs4 import BeautifulSoup
import pandas as pd

class TableScraper:
    def __init__(self, url):
        self.url = url
        self.dataframe = pd.DataFrame()

    def fetch_content(self):
        """Fetch the HTML content from the URL."""
        response = requests.get(self.url)
        response.raise_for_status()  # Check for request errors
        return response.content

    def parse_tables(self, html_content):
        """Parse all tables from the HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')

        # Loop through each table found
        for table in tables:
            # Extract headers
            headers = [header.get_text(strip=True) for header in table.find_all('th')]

            # Extract rows
            rows = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = row.find_all('td')
                cols = [col.get_text(strip=True) for col in cols]
                rows.append(cols)

            # Create a temporary DataFrame for the current table
            temp_df = pd.DataFrame(rows, columns=headers)

            # Append the current table's data to the main DataFrame
            self.dataframe = pd.concat([self.dataframe, temp_df], ignore_index=True)

    def save_to_csv(self, filename):
        """Save the combined DataFrame to a CSV file."""
        self.dataframe.to_csv(filename, index=False)

    def scrape(self):
        """Main method to perform scraping."""
        html_content = self.fetch_content()
        self.parse_tables(html_content)
        return self.dataframe

# Main function to execute the scraping
def main():
    url = 'https://servicos.unila.edu.br/catalogo/tecnologia/'
    scraper = TableScraper(url)
    
    # Perform scraping
    df = scraper.scrape()
    
    # Display the combined scraped data from all tables
    print("Combined Table:")
    print(df)

    # Save to a CSV file
    scraper.save_to_csv('ti_services.csv')

if __name__ == "__main__":
    main()
