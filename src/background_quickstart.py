import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

load_dotenv()

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    },
)
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Create memory manager Runnable to extract memories from conversations
memory_manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    # Store memories in the "memories" namespace (aka directory)
    namespace=("memories",),
)


@entrypoint(store=store)  # Create a LangGraph workflow
async def chat(message: str) -> str:
    response = llm.invoke(message)

    # memory_manager extracts memories from conversation history
    # We'll provide it in OpenAI's message format
    to_process = {"messages": [{"role": "user", "content": message}] + [response]}
    await memory_manager.ainvoke(to_process)
    return response.content


async def main() -> None:
    # Run conversation as normal
    response = await chat.ainvoke(
        "I like dogs. My dog's name is Fido.",
    )
    print(response)
    # Output: That's nice! Dogs make wonderful companions. Fido is a classic dog name. What kind of dog is Fido?

    search_items = store.search(("memories",))
    for item in search_items:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
