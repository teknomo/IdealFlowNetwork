# plugins/crawler_plugin.py
import time
from plugin import PluginInterface
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from urllib.parse import urljoin, urlparse
import validators
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom


class CrawlerPlugin(PluginInterface):
    """
    A simple web crawler that collects links within the same domain up to a specified maximum size.
        Args:
            start_url (str): The URL to start crawling from.
            max_size (int): The maximum number of unique links to collect.
        Returns:
            set: A set of visited links.
    @author: Kardi Teknomo
    Basic Algorithm:
        start from url, visit the starting page then get all links
        visit the next page in the list of links
        continue until maxSize is reached or the stack is empty
        the output are the set of unique links   
    """
    def get_actions(self):
        return {
            'browse': self.browse,
            'crawl': self.crawl,
            'generate_site_map': self.create_sitemap        
        }

    def browse(self, url, **kwargs):
        r = requests.get(url)
        print(r.status_code)
        print(r.headers['content-type'])
        print(r.encoding)
        data=r.text
        soup = BeautifulSoup(data, 'html.parser')
        print(soup.prettify())
        print(soup.title)
        print(soup.title.name)
        print(soup.title.string)
        print(soup.title.parent.name)
        print(soup.a)
        for no,link in enumerate(soup.find_all('a')):
            print(no,link.get('href'),'\n')

    

    def crawl(self, start_url, max_size = 1000, delay = 0, **kwargs):
        """
        return set of visited links start from the given url
        The crawl will stop when the number of elements in visited set 
        is equal or larger than maxSize
        """
        original_domain = urlparse(start_url).netloc
        stack = {start_url}
        visited = set()

        while stack and len(visited) < max_size:
            current_url = stack.pop()
            print("Visiting:", current_url)
            visited.add(current_url)
            time.sleep(delay)
            for link in self._getPage(current_url):
                valid_link = self._get_valid_links(link, current_url)
                if valid_link and valid_link not in visited and urlparse(valid_link).netloc == original_domain:
                    stack.add(valid_link)
        return visited  # return set of visited urls


    def create_sitemap(self, start_url, filename='sitemap.xml', **kwargs):
        urls = self.crawl(start_url, max_size = 100000)
        self._generate_xml_sitemap(urls, filename)


    def _getPage(self,url, **kwargs):
        """
        go to page of the URL
        get all links from a page
        return set of links from the page
        """
        links=set()     # set of internal state
        response = requests.get(url)
        if response.status_code==200:
            page = BeautifulSoup(response.content, 'html.parser')
            for link in page.find_all('a'):
                links.add(link.get('href'))
        return links
        
    def _get_valid_links(self, url, current_url, **kwargs):
        """
        Validates a URL and returns its absolute form if it's relative.
        If the URL is invalid, returns None.

        Args:
            url (str): The URL to validate and possibly convert.
            current_url (str): The current URL from which `url` was extracted, used to resolve relative URLs.

        Returns:
            str or None: The absolute URL if valid, or None if the URL is invalid.
        """
        # Convert relative URL to absolute URL
        absolute_url = urljoin(current_url, url)

        # Validate the absolute URL
        if validators.url(absolute_url):
            return absolute_url
        else:
            return None
    
    
        
    def _generate_xml_sitemap(self, urls, filename='sitemap.xml'):
        """
        Generates an XML sitemap from a set of URLs and saves it to a file.

        Args:
            urls (set): A set of URLs collected by the crawler.
            filename (str): The filename for the saved XML sitemap.
        """
        urlset = Element('urlset', xmlns='http://www.sitemaps.org/schemas/sitemap/0.9')
        for url in urls:
            url_element = SubElement(urlset, 'url')
            loc = SubElement(url_element, 'loc')
            loc.text = url

        # Generate a pretty-printed XML string
        rough_string = tostring(urlset, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

        # Save the XML sitemap to a file
        with open(filename, 'w') as file:
            file.write(pretty_xml_as_string)
        print(f'Sitemap saved to {filename}')
