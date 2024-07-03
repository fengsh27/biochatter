import os
from biochatter.prompts import BioCypherPromptEngine
import pytest

## THIS IS LARGELY BENCHMARK MATERIAL, TO BE MOCKED FOR UNIT TESTING

schema_dict = {
    'drug': {
        'represented_as': 'node', 
        'preferred_id': 'ddinter', 
        'input_label': 'drug', 
        'present_in_knowledge_graph': True, 
        'is_relationship': False
    }, 
   'drug interaction': {
        'is_a': 'association', 
        'represented_as': 'edge', 
        'input_label': 'drug_drug_interaction', 
        'properties': {
            'level': 'str', 
            'class': 'str'
        }, 
        'preferred_id': 'id', 
        'present_in_knowledge_graph': True, 
        'is_relationship': True
    }, 
    'is_schema_info': True
}

@pytest.fixture
def prompt_engine_1():
    return BioCypherPromptEngine(
        schema_config_or_info_path="./test/genomicKB_schema_config.yaml"
    )

def test_biocypher_prompts_1(prompt_engine_1):
    assert len(list(prompt_engine_1.entities.keys())) > 0
    assert len(list(prompt_engine_1.relationships.keys())) > 0
    print("test completed")

def test_biocypher_prompts_2(prompt_engine_1):
    res = prompt_engine_1.generate_query("show me the relationship between 'gene' and 'cell line or tissue'")
    assert(res is not None)

