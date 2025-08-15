import aiohttp
import tempfile
import os
async def download_file(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError("Failed to fetch document")
            content = await response.read()
            _, extension = os.path.splitext(url.split('?')[0])
            if not extension:
                extension = ".pdf"
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
            temp.write(content)
            temp.close()
            return temp.name
async def fetch_website_text(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"Failed to fetch website content from {url}")
            return await response.text()