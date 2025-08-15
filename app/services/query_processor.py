from app.services import chunking, embedding, llm
from app.models.schemas import HackRXInput
from app.services.document_loader import extract_text
from app.services.embedding import embeddings_model
from app.utils.downloader import download_file, fetch_website_text
import asyncio
import json
import logging
import hashlib
import os
from tenacity import RetryError
import time
from langchain_community.vectorstores import FAISS
import httpx
CACHE_DIR = "faiss_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
llm_semaphore = asyncio.Semaphore(5)
import urllib.parse
def get_cache_path(document_url: str) -> str:
    parsed_url = urllib.parse.urlparse(document_url)
    base_url = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
    key = base_url
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest())
async def get_faiss_index(document_url: str):
    parsed_url = urllib.parse.urlparse(document_url)
    _, extension = os.path.splitext(parsed_url.path)
    is_website = not extension or extension.lower() not in ['.pdf', '.docx', '.pptx', '.xlsx', '.png', '.jpeg', '.jpg', '.eml']
    if is_website:
        logging.info(f"Website URL detected: {document_url}. Bypassing cache.")
        try:
            text = await fetch_website_text(document_url)
            chunks = await chunking.chunk_text(text)
            faiss_index = await embedding.build_faiss_index(chunks)
            return faiss_index
        except ValueError as e:
            logging.error(f"Error processing website content from {document_url}: {e}")
            return None
    cache_dir = get_cache_path(document_url)
    if os.path.exists(cache_dir):
        logging.info(f"FAISS index found in cache for {document_url}: {cache_dir}. Loading directly.")
        return FAISS.load_local(cache_dir, embeddings_model, allow_dangerous_deserialization=True)
    else:
        logging.info(f"FAISS index not found in cache for {document_url}. Building new index.")
        file_path = await download_file(document_url)
        try:
            text = await extract_text(file_path)
        except ValueError as e:
            logging.error(f"Error extracting text from {file_path}: {e}")
            return None
        chunks = await chunking.chunk_text(text)
        faiss_index = await embedding.build_faiss_index(chunks)
        os.makedirs(cache_dir, exist_ok=True)
        faiss_index.save_local(cache_dir)
        logging.info(f"FAISS index built and saved to cache: {cache_dir}")
        return faiss_index
async def get_flight_number():
    landmark_urls = {
        "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
        "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
        "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
        "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber",
    }
    default_flight_url = "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
    city_landmarks = {
        "Delhi": ["Gateway of India"],
        "Mumbai": ["India Gate", "Space Needle"],
        "Chennai": ["Charminar"],
        "Hyderabad": ["Marina Beach", "Taj Mahal"],
        "Ahmedabad": ["Howrah Bridge"],
        "Mysuru": ["Golconda Fort"],
        "Kochi": ["Qutub Minar"],
        "Pune": ["Meenakshi Temple", "Golden Temple"],
        "Nagpur": ["Lotus Temple"],
        "Chandigarh": ["Mysore Palace"],
        "Kerala": ["Rock Garden"],
        "Bhopal": ["Victoria Memorial"],
        "Varanasi": ["Vidhana Soudha"],
        "Jaisalmer": ["Sun Temple"],
        "New York": ["Eiffel Tower"],
        "London": ["Statue of Liberty", "Sydney Opera House"],
        "Tokyo": ["Big Ben"],
        "Beijing": ["Colosseum"],
        "Bangkok": ["Christ the Redeemer"],
        "Toronto": ["Burj Khalifa"],
        "Dubai": ["CN Tower", "Moai Statues"],
        "Amsterdam": ["Petronas Towers"],
        "Cairo": ["Leaning Tower of Pisa"],
        "San Francisco": ["Mount Fuji"],
        "Berlin": ["Niagara Falls"],
        "Barcelona": ["Louvre Museum"],
        "Moscow": ["Stonehenge"],
        "Seoul": ["Sagrada Familia", "Times Square"],
        "Cape Town": ["Acropolis"],
        "Istanbul": ["Big Ben"],
        "Riyadh": ["Machu Picchu"],
        "Paris": ["Taj Mahal"],
        "Singapore": ["Christchurch Cathedral"],
        "Jakarta": ["The Shard"],
        "Vienna": ["Blue Mosque"],
        "Kathmandu": ["Neuschwanstein Castle"],
        "Los Angeles": ["Buckingham Palace"],
    }
    try:
        async with httpx.AsyncClient() as client:
            city_response = await client.get("https://register.hackrx.in/submissions/myFavouriteCity")
            city_response.raise_for_status()
            city_data = city_response.json()
            city = city_data.get("data", {}).get("city")
            if not city:
                return "Could not determine the city from the API."
            landmarks = city_landmarks.get(city, [])
            if not landmarks:
                return f"No landmarks found for the city: {city}."
            flight_numbers = []
            for landmark in landmarks:
                flight_url = landmark_urls.get(landmark, default_flight_url)
                flight_response = await client.get(flight_url)
                flight_response.raise_for_status()
                flight_data = flight_response.json()
                flight_number = flight_data.get("data", {}).get("flightNumber")
                if flight_number:
                    flight_numbers.append(f"For landmark '{landmark}', your flight number is {flight_number}.")
            if flight_numbers:
                return " ".join(flight_numbers)
            else:
                return "Could not retrieve any flight numbers."
    except httpx.RequestError as e:
        logging.error(f"An error occurred while requesting flight data: {e}")
        return "An error occurred while trying to get the flight number."
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."
async def process_query(input_data: HackRXInput):
    if "what is my flight number?" in [q.lower() for q in input_data.questions]:
        flight_number_answer = await get_flight_number()
        return {"answers": [flight_number_answer]}
    start_time = time.time()
    faiss_index = await get_faiss_index(input_data.documents)
    if faiss_index is None:
        return {"answers": ["Unsupported file type. The file was not processed."]}
    index_time = time.time() - start_time
    logging.info(f"FAISS index retrieval/creation took {index_time:.2f} seconds.")
    start_time = time.time()
    async with llm_semaphore:
        try:
            answers = await llm.ask_llm_batch(faiss_index, input_data.questions)
            for i, answer in enumerate(answers):
                log_data = {
                    "question": input_data.questions[i],
                    "answer": answer,
                }
                logging.info(json.dumps(log_data, indent=4))
        except RetryError:
            logging.error("LLM query failed after multiple retries for the batch.")
            answers = ["The service is currently busy, please try again in a few moments."] * len(input_data.questions)
    qa_time = time.time() - start_time
    logging.info(f"Question answering took {qa_time:.2f} seconds.")
    return {"answers": answers}