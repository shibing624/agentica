import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM
from actionflow.sqlite_storage import SqliteStorage

if __name__ == '__main__':
    llm = AzureOpenAILLM()
    print(llm)
    assistant = Assistant(
        llm=llm,
        add_chat_history_to_messages=True,
        storage=SqliteStorage(table_name="assistant_runs", db_file="outputs/assistant_runs.db"),
        output_dir="outputs"
    )
    assistant.print_response("How many people live in Canada?")
    assistant.print_response("What is their national anthem called?")


    # Function to print all stored runs
    def print_all_runs(storage):
        runs = storage.get_all_runs()
        print(f"Total runs: {len(runs)}")
        for run in runs:
            print(run)


    # Print all stored runs
    print_all_runs(assistant.storage)
