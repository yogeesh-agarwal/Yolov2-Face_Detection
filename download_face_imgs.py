import os
from icrawler.builtin import BingImageCrawler

def download_images(query , num_images , download_dir):
    bing_crawler = BingImageCrawler(downloader_threads=4 , storage={'root_dir': download_dir})
    bing_crawler.crawl(keyword=query , filters=None , offset=0 , max_num=num_images)

if __name__ == '__main__':
    download_dir = "./data/random_faces/"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    queries = ["human faces"]
    for query in queries:
        download_images(query , 500 , download_dir)
