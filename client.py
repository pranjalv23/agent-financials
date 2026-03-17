import requests
import json
import uuid

API_URL = "http://127.0.0.1:8000/ask"

def main():
    print("========================================")
    print("Welcome to the Financial AI Agent CLI")
    print("Type 'exit' or 'quit' to end the session.")
    print("========================================\n")

    # Start with no session ID
    current_session_id = None

    while True:
        try:
            # 1. Get user input
            user_input = input("\n\033[94mYou:\033[0m ")
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("Ending session. Goodbye!")
                break
                
            if not user_input.strip():
                continue

            print("\033[90mAgent is thinking...\033[0m", end="\r")

            # 2. Build the payload
            payload = {
                "query": user_input
            }
            
            # Auto-attach the session ID if we already have one from a previous turn
            if current_session_id:
                payload["session_id"] = current_session_id

            # 3. Send the request
            response = requests.post(API_URL, json=payload)
            
            # 4. Handle the response
            if response.status_code == 200:
                data = response.json()
                
                # Save the session ID automatically for the next loop!
                current_session_id = data.get("session_id")
                
                # Print the response
                print(f"\033[92mAgent:\033[0m {data.get('response')}")
                print(f"\n\033[90m[Session ID: {current_session_id}]\033[0m")
            else:
                print(f"\033[91mError: {response.status_code}\033[0m")
                print(response.text)

        except KeyboardInterrupt:
            print("\nEnding session. Goodbye!")
            break
        except requests.exceptions.ConnectionError:
            print("\n\033[91mError: Could not connect to the API. Is Uvicorn running?\033[0m")
            break

if __name__ == "__main__":
    main()
