from .base_table_task import BaseTableTask
from .base_table_task import BaseTableTask
import os
import json
from copy import deepcopy
import pandas as pd

class TableSummary(BaseTableTask):
    def get_task_descriptions(self, test_example):
        descriptions = [
            "Please look at the table below and provide a title for the table.",
            "Kindly refer to the table below and suggest a suitable title for it.",
            "I'd appreciate it if you could glance at the table and offer a title for it.",
            "Could you spare a moment to look at the table and give it an appropriate title?",
            "Your task is to review the table and come up with a title for it.",
            "Given the table below, could you provide a title that accurately represents its contents?",
            "Here's a table for your consideration; please suggest a title that fits its contents.",
            "I request that you provide a summary of the input table's content.",
            "Kindly give a concise overview of what the input table represents.",
            "Take a moment to summarize the key points of the input table.",
            "Your task is to give a summary of the input table's main information.",
            "Could you spare a moment to summarize the input table's key findings?",
            "I'd appreciate it if you could provide a summary of the input table after examining it.",
            "Given the input table, can you provide a summary that captures its main data?",
            "Here's an input table for your consideration; please offer a summary of its key aspects.",
            "After reviewing the input table, could you provide a brief summary of its main points?",
            "I'd like your input on this table – can you summarize its contents for me?",
            "Please take a look at the input table and provide a concise summary of its data.",
            "Your help is needed in summarizing the input table and its main information.",
            "Summarize the input table and its key details for easy understanding.",
            "Your task is to analyze the input table and provide a summary of its main aspects.",
            "Could you please glance at the input table and offer a summary that captures its essence?",
            "Please provide a summary for the input table after reviewing its contents.",
            "Your input is valued – kindly summarize the input table's data.",
            "Having looked at the input table, can you give a summary that reflects its main points?",
            "Here's an input table that needs summarizing; can you do that for me?",
            "After considering the input table, please provide a summary that best represents it.",
            "I request that you review the table below and give a brief summary of its contents.",
            "Kindly examine the table and provide a concise overview of what it represents.",
            "Take a moment to look at the table and summarize its key points.",
            "Your task is to glance at the table and provide a summary of its contents.",
            "Could you spare a moment to review the table and give a summary of its main information?",
            "I'd appreciate it if you could summarize the table's content after looking at it.",
            "Given the table below, can you provide a summary that captures its main data?",
            "Here's a table for your consideration; please offer a summary of its key findings.",
            "After examining the table, could you provide a brief summary of its main points?",
            "Please take a look at the table and provide a concise summary of its data.",
            "Your help is needed in summarizing the table below and its main information.",
            "Summarize the table and its key details for easy understanding.",
            "Your task is to analyze the table and provide a summary of its main aspects.",
            "Could you please glance at the table and offer a summary that captures its essence?",
            "Please provide a summary for the table after reviewing its contents.",
            "Having looked at the table, can you give a summary that reflects its main points?",
            "Here's a table that needs summarizing; can you do that for me?",
            "After considering the table, please provide a summary that best represents it.",
        ]
        return descriptions

    def get_input(self, test_example):
        return self.serialize_df(test_example["input_table"])

    def get_output(self, test_example):
        return self.answer_to_json("summary", test_example["label"])

    def get_output_template(self, test_example):
        return self.answer_to_json("summary", "<summary of table>")

    def load_datasets(self, task_data_dir, n_jobs=1, max_size=None, random_state=1):
        if task_data_dir.rstrip("/").endswith("train"):
            mode = "train"
        elif task_data_dir.rstrip("/").endswith("test"):
            mode = "test"
        else:
            raise Exception(
                "cannot identify mode for table summary from task data dir"
            )

        datasets = []
        if os.path.exists(task_data_dir):
            for dataset in sorted(os.listdir(task_data_dir)):
                table_path = os.path.join(task_data_dir, dataset, "input_table.csv")
                info_path = os.path.join(task_data_dir, dataset, "info.json")

                table = self.load_df(table_path)
                info = self.load_json(info_path)

                test_example = {
                    "input_table": table,
                    "label": info.get("label", ""),
                    "metadata": {
                        "dataset": dataset
                    }
                }

                datasets.append(test_example)

        if max_size is not None:
            datasets = self._random_sample(
                datasets, max_size, random_state=random_state
            )

        return datasets

    def load_df(self, file_path):
        return pd.read_csv(file_path)

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def _random_sample(self, data, size, random_state=1):
        import random
        random.seed(random_state)
        return random.sample(data, size)