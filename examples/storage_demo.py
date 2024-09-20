import sys

sys.path.append('..')
from agentica import Assistant, AzureOpenAIChat
from agentica.storage.sqlite_storage import SqlAssistantStorage

if __name__ == '__main__':
    llm = AzureOpenAIChat()
    print(llm)
    assistant = Assistant(
        llm=llm,
        add_chat_history_to_messages=True,
        storage=SqlAssistantStorage(table_name="assistant_runs", db_file="outputs/assistant_runs.db"),
        output_dir="outputs"
    )
    r = assistant.run("How many people live in Canada?")
    print("".join(r))
    r = assistant.run("What is their national anthem called?")
    print("".join(r))


    # Function to print all stored runs
    def print_all_runs(storage):
        runs = storage.get_all_runs()
        print(f"Total runs: {len(runs)}")
        for run in runs:
            print(run)


    # Print all stored runs
    print_all_runs(assistant.storage)
