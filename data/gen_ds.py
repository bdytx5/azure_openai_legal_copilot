

from datasets import load_dataset
import json

# List of subsets to load
subsets = [
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_inclusion_of_verbally_conveyed_information",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_permissible_acquirement_of_similar_information",
    "contract_nli_permissible_copy",
    "contract_nli_permissible_development_of_similar_information",
    "contract_nli_permissible_post-agreement_possession",
    "contract_nli_return_of_confidential_information",
    "contract_nli_sharing_with_employees",
    "contract_nli_sharing_with_third-parties",
    "contract_nli_survival_of_obligations",
    "contract_qa"
]

# Dictionary to hold subsets as keys and their prompts as values
subsets_prompts = {
    "contract_nli_confidentiality_of_agreement": "Identify if the clause provides that the Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
    "contract_nli_explicit_identification": "Identify if the clause provides that all Confidential Information shall be expressly identified by the Disclosing Party.",
    "contract_nli_inclusion_of_verbally_conveyed_information": "Identify if the clause provides that Confidential Information may include verbally conveyed information.",
    "contract_nli_limited_use": "Identify if the clause provides that the Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
    "contract_nli_no_licensing": "Identify if the clause provides that the Agreement shall not grant Receiving Party any right to Confidential Information.",
    "contract_nli_notice_on_compelled_disclosure": "Identify if the clause provides that the Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
    "contract_nli_permissible_acquirement_of_similar_information": "Identify if the clause provides that the Receiving Party may acquire information similar to Confidential Information from a third party.",
    "contract_nli_permissible_copy": "Identify if the clause provides that the Receiving Party may create a copy of some Confidential Information in some circumstances.",
    "contract_nli_permissible_development_of_similar_information": "Identify if the clause provides that the Receiving Party may independently develop information similar to Confidential Information.",
    "contract_nli_permissible_post-agreement_possession": "Identify if the clause provides that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
    "contract_nli_return_of_confidential_information": "Identify if the clause provides that the Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
    "contract_nli_sharing_with_employees": "Identify if the clause provides that the Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    "contract_nli_sharing_with_third-parties": "Identify if the clause provides that the Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
    "contract_nli_survival_of_obligations": "Identify if the clause provides that some obligations of Agreement may survive termination of Agreement.",
    "contract_qa": "Answer questions about whether contractual clauses discuss particular issues."
}


# Path to save the JSONL files
output_file_train = './legalbench_subsets_test.jsonl'
output_file_test = './legalbench_subsets_train.jsonl'

def write_to_file(subset, output_file, split):
    with open(output_file, 'a') as file:
        dataset = load_dataset("nguha/legalbench", subset, split=split)

        for example in dataset:
            # Constructing the dialogue format with the prompt and example data
            prompt = subsets_prompts.get(subset, "")
            dialogue = {
                "messages": [
                    {"role": "system", "content": "Answer yes/no "+ prompt},
                    {"role": "user", "content": example['text']},
                    {"role": "assistant", "content": example['answer']}
                ]
            }
            file.write(json.dumps(dialogue) + '\n')

# Write train and test datasets to separate files
for subset in subsets:
    # Write the test data to the train file (since they are mixed up on Hugging Face)
    write_to_file(subset, output_file_train, "test")

    # Write the train data to the test file
    write_to_file(subset, output_file_test, "train")