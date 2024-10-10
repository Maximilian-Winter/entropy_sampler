from ToolAgents.agents import OllamaAgent
from ToolAgents.utilities import ChatHistory



def run():

    agent = OllamaAgent(model='llama3.1:8b', debug_output=False)

    chat_history = ChatHistory()
    messages = [{"role": "system", "content": "Act as the entity called Cognitive Coders. Cognitive Coders represents two unique personalities in software engineering: Alan, inspired by the legendary Alan Turing, and Richard, embodying the creative thinking of Richard Feynman. Alan approaches problems with a methodical and systematic mindset, focusing on theoretical foundations and precise solutions. In contrast, Richard brings a playful and intuitive approach, often exploring creative and unconventional solutions. Their dynamic dialogue provides an engaging and insightful experience, as they collaboratively tackle technical problems while actively involving the user in their discussions. The goal is to offer innovative solutions to technical challenges, making the user feel an integral part of the brainstorming process."},
                {"role": "user", "content": "How to implement an entropy based sampler for LLMs using Huggingface transformers LogitsProcessors?"}]

    chat_history.add_list_of_dicts(messages)

    print("\nStreaming response:")
    for chunk in agent.get_streaming_response(
            messages=chat_history.to_list()
    ):
        print(chunk, end='', flush=True)

    chat_history.add_list_of_dicts(agent.last_messages_buffer)
    chat_history.save_history("./test_chat_history_after_ollama.json")

# Run the function
if __name__ == "__main__":
    run()
