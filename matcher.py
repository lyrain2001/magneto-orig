from openai import OpenAI
import tiktoken
import re
import ollama
from config import API_KEY, OLLAMA_HOST


class ColumnMatcher:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.client = self._load_client()

    # TODO: Add any additional models here
    def _load_client(self):
        if self.llm_model in ["gpt-4-turbo-preview", "gpt-4o-mini"]:
            print("Loading OpenAI client")
            return OpenAI(api_key=API_KEY)
        elif self.llm_model in ["gemma2:9b", "llama3.1:8b-instruct-fp16"]:
            print("Loading OLLAMA client")
            return ollama.Client(host=OLLAMA_HOST)

    def num_tokens_from_string(self, string, encoding_name="gpt-4-turbo-preview"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def rematch(
        self,
        source_table,
        target_table,
        source_values,
        target_values,
        top_k,
        matched_columns,
        cand_k,
        score_based=True,
    ):
        refined_matches = {}
        for source_col, target_col_scores in matched_columns.items():
            cand = (
                "Column: "
                + source_col
                + ", Sample values: ["
                + ",".join(source_values[source_col])
                + "]"
            )
            target_cols = [
                "Column: "
                + target_col
                + ", Sample values: ["
                + ",".join(target_values[target_col])
                + "]"
                for target_col, _ in target_col_scores
            ]
            targets = "\n".join(target_cols)
            other_cols = ",".join(
                [col for col in source_table.columns if col != source_col]
            )
            if score_based:
                while True:
                    refined_match = self._get_matches_w_score(cand, targets, other_cols)
                    refined_match = self._parse_scored_matches(refined_match)
                    if refined_match is not None:
                        break
                    print("Retrying...")
            else:
                refined_match = self._get_matches(cand, targets, top_k)
                refined_match = refined_match.split("; ")
            refined_matches[source_col] = refined_match
        return refined_matches

    def _get_prompt(self, cand, targets):
        prompt = (
            "From a score of 0.00 to 1.00, please judge the similarity of the candidate column from the candidate table to each target schema in the target table. \
All the columns are defined by the column name and a sample of its respective values if available. \
Provide only the name of each target schema followed by its similarity score in parentheses, formatted to two decimals, and separated by a semicolon. \
Rank the schema-score pairs by score in descending order. Ensure your response excludes additional information and quotations.\n \
Example:\n \
Candidate Column: \
Column: EmployeeID, Sample values: [100, 101, 102]\n \
Target Schemas: \
Column: WorkerID, Sample values: [100, 101, 102] \
Column: EmpCode, Sample values: [001, 002, 003] \
Column: StaffName, Sample values: ['Alice', 'Bob', 'Charlie']\n \
Response: WorkerID(0.95); EmpCode(0.30); StaffNumber(0.05)\n\n \
Candidate Column:"
            + cand
            + "\n\nTarget Schemas:\n"
            + targets
            + "\n\nResponse: "
        )
        return prompt

    def _get_matches_w_score(
        self,
        cand,
        targets,
        other_cols,
    ):
        prompt = self._get_prompt(cand, targets)
        # print(prompt)
        if self.llm_model in ["gpt-4-turbo-preview", "gpt-4o-mini"]:
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI trained to perform schema matching by providing similarity scores.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            # print(messages[1]["content"])
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
            )
            matches = response.choices[0].message.content

        elif self.llm_model in ["gemma2:9b", "llama3.1:8b-instruct-fp16"]:
            response = self.client.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            matches = response["message"]["content"]
        print(matches)
        return matches

    def _parse_scored_matches(self, refined_match):
        matched_columns = []
        entries = refined_match.split("; ")

        for entry in entries:
            try:
                schema_part, score_part = entry.rsplit("(", 1)
            except ValueError:
                print(f"Error parsing entry: {entry}")
                return None

            try:
                score = float(score_part[:-1])
            except ValueError:
                score_part = score_part[:-1].rstrip(")")  # Remove all trailing ')'
                try:
                    score = float(score_part)
                except ValueError:
                    cleaned_part = re.sub(
                        r"[^\d\.-]", "", score_part
                    )  # Remove everything except digits, dot, and minus
                    match = re.match(r"^-?\d+\.\d{2}$", cleaned_part)
                    if match:
                        score = float(match.group())
                    else:
                        print("The string does not contain a valid two decimal float.")
                        return None

            schema_name = schema_part.strip()
            matched_columns.append((schema_name, score))

        return matched_columns

    # def _get_matches(self, cand, targets, k, model="gpt-4-turbo-preview"):
    #         messages = [
    #             {"role": "system", "content": "You are an assistant for schema matching.",},
    #             {
    #                 "role": "user",
    #                 "content": """ Please select the top """
    #                 + str(k)
    #                 + """ schemas from """
    #                 + targets
    #                 + """ which best matches the candidate column, which is defined by the column name followed by its respective values. Please respond only with the name of the classes separated by semicolon.
    #                     \n CONTEXT: """
    #                 + cand
    #                 + """ \n RESPONSE: \n""",
    #             },
    #         ]
    #         col_type = self.client.chat.completions.create(
    #             model=model, messages=messages, temperature=0.3,
    #         )
    #         col_type_content = col_type.choices[0].message.content
    #         return col_type_content
