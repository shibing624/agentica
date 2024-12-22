import sys

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat
from agentica import SqlAgentStorage

if __name__ == '__main__':
    llm = AzureOpenAIChat()
    print(llm)
    m = Agent(
        llm=llm,
        add_history_to_messages=True,
        storage=SqlAgentStorage(table_name="assistant_runs", db_file="outputs/assistant_runs.db"),
        output_dir="outputs"
    )
    r = m.run("How many people live in Canada?")
    print(r)
    r = m.run("What is their national anthem called?")
    print(r)


    # Function to print all stored runs
    def print_all_runs(storage):
        runs = storage.get_all_sessions()
        print(f"Total runs: {len(runs)}")
        for run in runs:
            print(run)


    # Print all stored runs
    print_all_runs(m.storage)
