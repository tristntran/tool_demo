# LangChain Tool-Calling Agent Demo

This project demonstrates a LangChain tool-calling agent with two defined tools: a Knapsack Problem Solver and a Fibonacci Number Finder. The agent uses OpenAI's language model to decide which tool to use based on the input query.

## Prerequisites

- Docker installed on your system
- An OpenAI API key

## Project Structure

```
langchain-demo/
│
├── .devcontainer/
│   └── devcontainer.json
├── app.py
├── Dockerfile
├── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

1. Clone this repository or create a new directory and add the files as described in the project structure.

2. Create a `.env` file in the project root and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_api_key_here
   REPLICATE_API_TOKEN=your_api_token_here
   ```

   Replace `your_api_key_here` with your actual OpenAI API key.

3. Ensure that your main Python script (containing the LangChain demo code) is named `app.py`. If it's not, either rename your script or update the `CMD` line in the Dockerfile to match your script's name.

## Running the Demo

### Option 1: Using Docker

1. Build the Docker image:
   ```
   docker build -t langchain-demo .
   ```

2. Run the container:
   ```
   docker run -it --env-file .env langchain-demo
   ```

This will execute your `app.py` script inside the container.

### Option 2: Using Visual Studio Code with Dev Containers

1. Install the "Remote - Containers" extension in VS Code.

2. Open the project folder in VS Code.

3. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container".

4. Once the container is built and running, open a terminal in VS Code and run:
   ```
   python app.py
   ```

## Modifying the Demo

- To modify the LangChain demo code, edit the `app.py` file.
- If you need to add new dependencies, add them to the `requirements.txt` file and rebuild the Docker image.

## Troubleshooting

- If you encounter any issues related to missing dependencies, make sure all required packages are listed in `requirements.txt`.
- If you're having problems with the OpenAI API key, check that it's correctly set in the `.env` file and that the file is in the project root.
- For Docker-related issues, ensure Docker is installed and running on your system.

## Additional Notes

- The demo uses OpenAI's language model. Make sure you have sufficient credits in your OpenAI account.
- The Dockerfile exposes port 80, which can be used if you decide to add a web interface to your application in the future.

## Contributing

Feel free to fork this project and submit pull requests with any enhancements.

## License

This project is open-source and available under the MIT License.