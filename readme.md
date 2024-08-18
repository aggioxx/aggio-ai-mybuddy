# OpsBuddy Chatbot

OpsBuddy is a chatbot developed as part of a conceptual project for Aggiocorp, a fictional company specializing in IT operations. The chatbot leverages FastAPI, Weaviate, and OpenAI to offer conversational retrieval capabilities.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/aggioxx/opsbuddy-chatbot.git
    cd opsbuddy-chatbot
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Set the required environment variables:
    ```sh
    export WEAVIATE_URL=your_weaviate_url
    export OPENAI_API_KEY=your_openai_api_key
    ```

2. Run the FastAPI application:
    ```sh
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

3. Open `index.html` in your browser to interact with the chatbot.

## Endpoints

- **GET /**: Serves the `index.html` file.
- **POST /query**: Accepts a JSON payload with a `query` field and returns the chatbot's response.

## Environment Variables

- `WEAVIATE_URL`: The URL of the Weaviate instance.
- `OPENAI_API_KEY`: The API key for OpenAI.

## Docker

1. Build the Docker image:
    ```sh
    docker build -t opsbuddy-chatbot .
    ```

2. Run the Docker container:
    ```sh
    docker run -p 8000:8000 -e WEAVIATE_URL=your_weaviate_url -e OPENAI_API_KEY=your_openai_api_key opsbuddy-chatbot
    ```

3. Open `index.html` in your browser to interact with the chatbot.