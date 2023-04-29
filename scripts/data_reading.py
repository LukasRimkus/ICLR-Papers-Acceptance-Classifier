import os
import json
import numpy as np
import pandas as pd


TEXT_SELECTION_MODES = {"INTRODUCTION_WITH_ABSTRACT": 0, 
                        "INTRODUCTION_WITHOUT_ABSTRACT": 1, 
                        "MIDDLE": 2, 
                        "TAIL": 3,
                        "ABSTRACT_WITH_TAIL": 4}


def get_papers_paths(dataset_path: str) -> list:
    """
    This method obtains the paths of papers inside the ICLR conference folders in the dataset.
    There are two important .json files for each paper, namely {PAPER}_content.json (contains 
    the parsed paper contents) and {PAPER}_paper.json (contains the paper metadata like the acceptance/rejection
    decision).
    """
    paper_and_content_paths = []  # stores (paper_path, content_path) tuples
    conference_dirs = [conference_dir for conference_dir in os.listdir(dataset_path) if 'ICLR' in conference_dir]

    for conference in conference_dirs:
        coference_dir = os.path.join(dataset_path, conference)
        conference_content_dir = os.path.join(coference_dir, f"{conference}_content")
        conference_paper_dir = os.path.join(coference_dir, f"{conference}_paper")
        
        for i, paper_filename in enumerate(os.listdir(conference_paper_dir)):
            paper_path = os.path.join(conference_paper_dir, paper_filename)
            content_filename = paper_filename.replace("paper", "content")
            content_path = os.path.join(conference_content_dir, content_filename)

            if os.path.exists(content_path):
                paper_and_content_paths.append((paper_path, content_path)) 

    return paper_and_content_paths


def extract_documents_text(paper_and_content_paths: list, mode: int=0, add_paper_metadata: bool=True, MAX_TOKENS_NUMBER: int=512) -> tuple:
    """
    This method extracts the text from each paper json files which are later used for training and evaluating the models.
    """
    number_of_papers = len(paper_and_content_paths)
    documents = np.full(number_of_papers, "", dtype=object)
    labels = np.full(number_of_papers, 0, dtype=int)
    
    for i, (paper_path, content_path) in enumerate(paper_and_content_paths):

        with open(paper_path) as paper_file, open(content_path) as content_file:
            paper_json = json.load(paper_file)
            content_json = json.load(content_file)

            documents[i] = parse_paper_from_json(content_json, add_paper_metadata, mode, MAX_TOKENS_NUMBER)
            
            # Check if the paper was accepted or rejected, where True (or 1) is for acceptance
            # whereas for rejection - False (or 0)
            label = 'Accept' in paper_json['decision']
            labels[i] = int(label)
            
    return np.array(documents), np.array(labels)


def parse_paper_from_json(content_json: dict, add_paper_metadata: bool, mode: int, MAX_TOKENS_NUMBER: int) -> str:
    """
    This method reads the json files for a given paper, applies the selected splitting strategy and 
    add optional Metadata features to the parsed text which can be helpful in deciding if the paper was accepted 
    or rejected. Adding these properties is treated as a hyperparameter. 

    Optionally added Metadata features:
    1. Title of the paper
    2. The existance of an Appendix
    3. The number of Sections
    4. The number of references
    """
    all_paper_text = str()

    abstract = content_json['metadata']['abstractText'] if 'abstractText' in content_json['metadata'] and \
                                                        content_json['metadata']['abstractText'] else ""

    reached_conclusion = False
    use_of_appendix = False

    for section in content_json['metadata']['sections']:
        if section["heading"]:
            # Check if it is appendix or not as appendix text is not used for models   
            if 'appendix' in section["heading"].lower() or section["heading"].split()[0].lower() == "a":
                use_of_appendix = True
                break

            # After detecting the conclusion section, allow the code to search for any more remaining sections
            # If there are appendices, then let the model know that and break the loop
            if reached_conclusion:
                break

            all_paper_text += f' SECTION {section["heading"]}: {section["text"]}'

            if 'conclusion' in section["heading"].lower():
                reached_conclusion = True
        else:
            # if a heading does not exist, then just append the text 
            all_paper_text += " " + section['text']

    # Split the text by the selected mode
    selected_text = apply_chosen_text_splitting_strategy(mode, abstract, all_paper_text, MAX_TOKENS_NUMBER)

    if add_paper_metadata:
        # Add some metadata which may be useful for the model to learn
        selected_text = selected_text if not use_of_appendix else f"APPENDIX. {selected_text}" 
        selected_text = f"TITLE: {content_json['metadata']['title']}. REFERENCES: {len(content_json['metadata']['references'])}. NUMBER OF SECTIONS: {len(content_json['metadata']['sections'])}. {selected_text}" 

    return selected_text


def parse_paper_from_pdf_document(content_json: dict, add_paper_metadata: bool, mode: int, MAX_TOKENS_NUMBER: int) -> str:
    """
    This method reads the provided paper for the demo demonstration, but this needs to be parsed a bit differently. 
    Also, this method applies the selected splitting strategy and optionally adds metadata features.

    Optionally added Metadata features:
    1. Title of the paper
    2. The existance of an Appendix
    3. The number of Sections
    4. The number of references
    """
    all_paper_text = str()

    abstract = content_json['abstractText'] if 'abstractText' in content_json and \
                                                        content_json['abstractText'] else ""

    reached_conclusion = False
    use_of_appendix = False

    for section in content_json['sections']:
        if "heading" in section and section["heading"]:
            # Check if it is appendix or not as appendix text is not used for models   
            if 'appendix' in section["heading"].lower() or section["heading"].split()[0].lower() == "a":
                use_of_appendix = True
                break

            # After detecting the conclusion section, allow the code to search for any more remaining sections
            # If there are appendices, then let the model know that and break the loop
            if reached_conclusion:
                break

            all_paper_text += f' SECTION {section["heading"]}: {section["text"]}'

            if 'conclusion' in section["heading"].lower():
                reached_conclusion = True
        else:
            # if a heading does not exist, then just append the text 
            all_paper_text += " " + section['text']

    # Split the text by the selected mode
    selected_text = apply_chosen_text_splitting_strategy(mode, abstract, all_paper_text, MAX_TOKENS_NUMBER)

    if add_paper_metadata:
        # Add some metadata which may be useful for the model to learn
        selected_text = selected_text if not use_of_appendix else f"APPENDIX. {selected_text}" 
        selected_text = selected_text if 'sections' not in content_json else f"NUMBER OF SECTIONS: {len(content_json['sections'])}. {selected_text}" 
        selected_text = selected_text if 'references' not in content_json else f"REFERENCES: {len(content_json['references'])}. {selected_text}" 
        selected_text = selected_text if 'title' not in content_json else f"TITLE: {content_json['title']}. {selected_text}" 

    return selected_text


def apply_chosen_text_splitting_strategy(mode: int, abstract: str, all_paper_text: str, MAX_TOKENS_NUMBER: int) -> str:
    """
    This method applies the chosen text splitting strategy for papers. There are 5 of them:
    1. INTRODUCTION_WITH_ABSTRACT - start from abstract till MAX_TOKENS_NUMBER
    2. INTRODUCTION_WITHOUT_ABSTRACT - skip the abstract and start from the introduction section
    3. MIDDLE - take the middle of the paper
    4. TAIL - take MAX_TOKENS_NUMBER from the end of the paper including conclusion
    5. ABSTRACT_WITH_TAIL - take the abstract and the tail (conclusion)
    """
    selected_text = str()

    if TEXT_SELECTION_MODES["INTRODUCTION_WITH_ABSTRACT"] == mode:
        tokens = (abstract + all_paper_text).split()
        selected_text = " ".join(tokens[:MAX_TOKENS_NUMBER])

    elif TEXT_SELECTION_MODES["INTRODUCTION_WITHOUT_ABSTRACT"] == mode:
        tokens = all_paper_text.split()
        selected_text = " ".join(tokens[:MAX_TOKENS_NUMBER])

    elif TEXT_SELECTION_MODES["MIDDLE"] == mode:
        tokens = (abstract + all_paper_text).split()
        selected_text = " ".join(tokens[len(tokens)//2 - MAX_TOKENS_NUMBER//2: len(tokens)//2 + MAX_TOKENS_NUMBER//2])

    elif TEXT_SELECTION_MODES["TAIL"] == mode:
        tokens = (abstract + all_paper_text).split()
        selected_text = " ".join(tokens[len(tokens) - MAX_TOKENS_NUMBER:])

    elif TEXT_SELECTION_MODES["ABSTRACT_WITH_TAIL"] == mode:
        abstract_tokens = abstract.split()
        abstract_tokens = abstract_tokens if len(abstract_tokens) < MAX_TOKENS_NUMBER*0.4 else abstract_tokens[:int(MAX_TOKENS_NUMBER*0.4)]
        text_tokens = all_paper_text.split()
        tail_text = " ".join(text_tokens[len(text_tokens) - MAX_TOKENS_NUMBER + len(abstract_tokens):])
        selected_text = f'START: {" ".join(abstract_tokens)} \n TAIL: {tail_text}'

    else:
        raise Exception(f"The mode {mode} is not supported!")
    
    return selected_text


def subsample_documents(data_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """
    This method takes all data and subsamples it by discarding rejected papers which make
    up the majority of all the dataset. Seeds ensure that this can be reproducible. 
    """
    rejected_papers = data_df[data_df.Decision == 0].copy()
    accepted_papers = data_df[data_df.Decision == 1].copy()

    if len(accepted_papers) < len(rejected_papers):
        rejected_papers = rejected_papers.sample(len(accepted_papers), random_state=random_state)
    
    # Accommodate the case when there are more accepted papers if the dataset was changed
    elif len(accepted_papers) > len(rejected_papers):
        accepted_papers = accepted_papers.sample(len(rejected_papers), random_state=random_state)

    subsampled_data_df = pd.concat([accepted_papers, rejected_papers])
    return subsampled_data_df
