import os
import openai
import pandas as pd
import weaviate
from langchain_openai import OpenAI  # Update the import
from weaviate.connect import ConnectionParams, ProtocolParams

# Initialize the OpenAI model
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


# Function to read and filter CSV
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def filter_data(df, criterion):
    return df[df['Questions'].str.contains(criterion, case=False)]


def insert_data_weaviate(data, client, class_name):
    for item in data:
        # Ensure the keys match those used in the data preparation step
        prepared_data = {
            "question": item["Questions"],  # Adjusted to use 'question'
            "answer": item["Answers"]       # Adjusted to use 'answer'
        }
        client.data_object.create(prepared_data, class_name)


def get_question():
    return input("Please enter your question: ")


def enrich_question(question):
    response = llm.generate(f"Enrich the following question for better understanding: {question}", temperature=0.5,
                            max_tokens=60)
    return response.strip()


def search_weaviate(client, question, class_name):
    result = client.query.get(class_name, ["Questions", "Answers"]).with_near_text({"concepts": [question]}).do()
    return result


def generate_response(context, question, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"{context}\n\nQuestion: {question}\nAnswer:",
        temperature=0.7,
        max_tokens=150,
        top_p=1.0
    )
    return response


def main():
    # Configuration
    file_path = "Mental_Health_FAQ.csv"
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_class_name = "MentalHealthFAQ"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Step 1: Read and filter CSV data
    df = read_csv(file_path)
    filtered_df = filter_data(df, "depression")
    data_for_weaviate = [
        {"Questions": row["Questions"], "Answers": row["Answers"]}
        for index, row in filtered_df.iterrows()
    ]
    print("Data for Weaviate:", data_for_weaviate)
    # Step 2: Insert data into Weaviate

    client = weaviate.Client(weaviate_url)
    try:
        client.schema.delete_class(weaviate_class_name)
        print("Schema deleted successfully.")
    except:
        print("Schema does not exist, skipping deletion.")
        pass
    finally:
        # Corrected class schema definition
        class_schema = {
            "class": weaviate_class_name,
            "description": "A class to store FAQs related to mental health",
            "properties": [
                {
                    "name": "question",
                    "dataType": ["text"],
                    "description": "The FAQ question",
                    "moduleConfig": {
                        "multi2vec-clip": {
                            "vectorizePropertyName": "question"
                        }
                    }
                },
                {
                    "name": "answer",
                    "dataType": ["text"],
                    "description": "The FAQ answer",
                    "moduleConfig": {
                        "multi2vec-clip": {
                            "vectorizePropertyName": "answer"
                        }
                    }
                }
            ],
            "vectorIndexType": "hnsw",
            "vectorizer": "multi2vec-clip"
        }
        client.schema.create_class(class_schema)
        print("Schema created successfully.")

    # Adjusted data preparation for insertion
    data_for_weaviate = [
        {"question": row["Questions"], "answer": row["Answers"]}
        for index, row in filtered_df.iterrows()
    ]

    insert_data_weaviate(data_for_weaviate, client, weaviate_class_name)

    # Step 3: Get user question
    question = get_question()

    # Step 4: Enrich the question
    enriched_question = enrich_question(question)

    # Step 5: Query Weaviate for context
    weaviate_results = search_weaviate(client, enriched_question, weaviate_class_name)
    context = " ".join([item['properties']['Answers'] for item in weaviate_results['data']['Get'][weaviate_class_name]])

    # Step 6: Generate response using OpenAI
    response = generate_response(context, question, openai_api_key)
    print("Response:", response.choices[0].text.strip())


if __name__ == "__main__":
    main()
