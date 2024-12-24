from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from eve.base import VersionableBaseModel, generate_edit_model, apply_edit


"""
base tests
- generate_edit_model: produces correct edit model, with annotations
- apply_edit: correct edit, preserves annotations
- versionable model: versions correctly, reconstructs version

mongo model tests
- mongo model: saves, updates, loads
- versionable mongo model: saves, updates, loads, versions correctly, reconstructs version
- misc/all: targets correct env

tool tests
- load basemodel from yaml
- save basemodel to yaml (?)
- generates correct llm schema

document tests
- setup document schema from yaml
- edit document
- save, version, load document


"""


class InnerModel(BaseModel):
    """
    This is an inner model which is contained in a TestModel
    """
    
    string_field: Optional[str] = Field(None, description="Another optional string field in inner model")
    number_field: Optional[int] = Field(None, description="Another optional number field in inner model")


class TestModel(BaseModel):
    """
    This is a pydantic base model
    """

    string_field: str = Field(..., description="A string field")
    string_list_field: Optional[List[str]] = Field(None, description="An optional string list field")
    dict_field: Optional[Dict[str, Any]] = Field(None, description="An optional dictionary field")
    base_model_field: Optional[InnerModel] = Field(None, description="An optional base model field")


def test_generated_edit_model():
    """
    Test if TestModelEdit and TestModelEditExpected have the same fields
    """

    TestModelEdit = generate_edit_model(TestModel)

    class TestModelEditExpected(BaseModel):
        edit_string_field: Optional[str] = Field(None, description="Edit TestModel string_field (A string field)")
        add_string_list_field: Optional[Dict[str, Union[int, str]]] = Field(None, description="Add TestModel string_list_field (An optional string list field)")
        edit_string_list_field: Optional[Dict[str, Union[int, str]]] = Field(None, description="Edit TestModel string_list_field (An optional string list field)")
        remove_string_list_field: Optional[int] = Field(None, description="Remove TestModel string_list_field (An optional string list field)")
        add_dict_field: Optional[Dict[str, Any]] = Field(None, description="Add TestModel dict_field (An optional dictionary field)")
        edit_dict_field: Optional[Dict[str, Any]] = Field(None, description="Edit TestModel dict_field (An optional dictionary field)")
        remove_dict_field: Optional[str] = Field(None, description="Remove TestModel dict_field (An optional dictionary field)")
        edit_base_model_field: Optional[InnerModel] = Field(None, description="Edit TestModel base_model_field (An optional base model field)")


    edit_fields = set(TestModelEdit.model_fields.keys())
    expected_fields = set(TestModelEditExpected.model_fields.keys())
    
    assert edit_fields == expected_fields, f"Fields mismatch. TestModelEdit: {edit_fields}, TestModelEditExpected: {expected_fields}"
    
    for field_name in expected_fields:
        edit_field = TestModelEdit.model_fields[field_name]
        expected_field = TestModelEditExpected.model_fields[field_name]

        assert edit_field.annotation == expected_field.annotation, \
            f"Field type mismatch for {field_name}.\n\tTestModelEdit: {edit_field.annotation}\n\tTestModelEditExpected: {expected_field.annotation}"
        assert edit_field.description == expected_field.description, \
            f"Field description mismatch for {field_name}.\n\tTestModelEdit: {edit_field.description}\n\tTestModelEditExpected: {expected_field.description}"
        


def test_apply_edit():
    """
    Test if apply_edit edits a model correctly
    """

    t1 = TestModel(
        string_field="test", 
        string_list_field=["test1", "test2"], 
        dict_field={"test3": "test4"},
        base_model_field=InnerModel(string_field="test5", number_field=5)
    )

    TestModelEdit = generate_edit_model(TestModel)

    t1_edit = TestModelEdit(
        edit_string_field="test6",
        add_string_list_field={"index": 1, "value": "test8"},
        edit_dict_field={"test3": "test11"},
        add_dict_field={"test12": "test13"},
        edit_base_model_field={"string_field": "test14"}
    )

    print("lets apply edit")
    print(type(t1))
    print(type(t1_edit))
    t2 = apply_edit(t1, t1_edit)
    print(type(t2))
    t2_expected = TestModel(
        string_field="test6",
        string_list_field=["test1", "test8", "test2"],
        dict_field={"test3": "test11", "test12": "test13"},
        base_model_field=InnerModel(
            string_field="test14", 
            number_field=5
        )
    )

    assert t2 == t2_expected


def test_versionable_base_model():
    """
    Test if VersionableBaseModel works correctly
    """

    t = VersionableBaseModel(
        TestModel(
            string_field="test0"
        )
    )

    assert t.current == TestModel(string_field="test0")

    TestModelEdit = t.get_edit_model()

    t.apply_edit(
        TestModelEdit(
            add_string_list_field={"index": 0, "value": "test1"},
            add_dict_field={"test2": "test3"},
        )
    )

    assert t.current == TestModel(
        string_field="test0",
        string_list_field=["test1"],
        dict_field={"test2": "test3"},
    )

    t.apply_edit(
        TestModelEdit(
            edit_string_field="test4",
            add_string_list_field={"index": 1, "value": "test4"},
            add_dict_field={"test4": "test5"},
            edit_base_model_field={"string_field": "test6"}
        )
    )

    assert t.current == TestModel(
        string_field="test4",
        string_list_field=["test1", "test4"],
        dict_field={"test2": "test3", "test4": "test5"},
        base_model_field=InnerModel(string_field="test6")
    )

    t.apply_edit(
        TestModelEdit(
            edit_string_list_field={"index": 1, "value": "test9"},
            remove_dict_field="test2",
            edit_base_model_field={"string_field": "test7", "number_field": 2}
        )
    )

    assert t.current == TestModel(
        string_field="test4",
        string_list_field=["test1", "test9"],
        dict_field={"test4": "test5"},
        base_model_field=InnerModel(string_field="test7", number_field=2)
    )

    # test reconstructions
    assert t.reconstruct_version(2) == TestModel(
        string_field="test4",
        string_list_field=["test1", "test4"],
        dict_field={"test2": "test3", "test4": "test5"},
        base_model_field=InnerModel(string_field="test6")
    )

    assert t.reconstruct_version(0) == TestModel(
        string_field="test0",
    )



if __name__ == "__main__":
    # test_generated_edit_model()
    test_apply_edit()
    # test_versionable_base_model()