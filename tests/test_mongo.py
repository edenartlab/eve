"""
document tests
- setup document schema from yaml
- edit document
- save, version, load document


Todo:
VersionableMongoModel.load(t1.id, collection_name="stories")
-> schema = recreate_base_model(document['schema'])
* this works but strong typing is not working. 

it gets {'base_model_field': FieldInfo(annotation=Union[Any, NoneType], required=False, default=None),
 'dict_field': FieldInfo(annotation=Union[Any, NoneType], required=False, default=None),
 'string_field': FieldInfo(annotation=str, required=False, default=None),
 'string_list_field': FieldInfo(annotation=Union[Any, NoneType], required=False, default=None)}

 but dict_field  should be Dict[str, Any]
 string_list_field should be List[str]
 base_model_field should be Union[InnerModel, NoneType]

"""

from pydantic import Field
from typing import Dict, Any
from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

from eve.mongo import Document, Collection, VersionableDocument



def test_mongo_document():
    """
    Test save, load, and update
    """

    @Collection("tests")
    class MongoModelTest(Document):
        num: int = Field(ge=1, le=10, default=1)
        args: Dict[str, Any]
        user: ObjectId

    t = MongoModelTest(
        num=2,
        args={"foo": "bar"}, 
        user=ObjectId("666666663333366666666666")
    )

    t.save()

    t2 = MongoModelTest.from_mongo(t.id)

    assert t2 == MongoModelTest(
        num=2, 
        args={"foo": "bar"}, 
        user=ObjectId("666666663333366666666666"), 
        id=t.id, 
        createdAt=t2.createdAt, 
        updatedAt=t2.updatedAt
    )

    # t2.update(invalid_arg="this is ignored", num=7, args={"foo": "hello world"})
    t2.update(num=7, args={"foo": "hello world"})

    t3 = MongoModelTest.from_mongo(t2.id)

    assert t.id == t2.id == t3.id

    assert t3 == MongoModelTest(
        num=7, 
        args={"foo": "hello world"}, 
        user=ObjectId("666666663333366666666666"), 
        id=t2.id, 
        createdAt=t3.createdAt, 
        updatedAt=t3.updatedAt
    )


def test_versionable_document():


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


    t1 = VersionableDocument(
        instance = TestModel(
            string_field="hello world 11", 
            string_list_field=["test1", "test2"], 
            dict_field={"test3": "test4"},
            base_model_field=InnerModel(string_field="test5", number_field=7)
        ),
        collection_name="tests",
    )

    t1.save()

    assert t1.current == TestModel(
        string_field="hello world 11", 
        string_list_field=["test1", "test2"], 
        dict_field={"test3": "test4"},
        base_model_field=InnerModel(
            string_field="test5", 
            number_field=7
        )
    )

    TestModelEdit = t1.get_edit_model()

    t1.apply_edit(
        TestModelEdit(
            add_string_list_field={"index": 0, "value": "test3"},
            add_dict_field={"test2": "test3"},
        )
    )

    assert t1.current == TestModel(
        string_field="hello world 11", 
        string_list_field=["test3", "test1", "test2"], 
        dict_field={"test3": "test4", "test2": "test3"},
        base_model_field=InnerModel(
            string_field="test5", 
            number_field=7
        )
    )

    t1.save()

    t1.apply_edit(
        TestModelEdit(
            edit_string_field="test4999",
            add_string_list_field={"index": 1, "value": "test4"},
            add_dict_field={"test4": "test56"},
            edit_base_model_field={"string_field": "test6", "number_field": 3}
        )
    )

    t1.save()

    assert t1.current == TestModel(
        string_field="test4999",
        string_list_field=["test3", "test4", "test1", "test2"],
        dict_field={"test3": "test4", "test2": "test3", "test4": "test56"},
        base_model_field=InnerModel(
            string_field="test6", 
            number_field=3
        )
    )

    t1.save()

    # Todo: this is still not working correctly
    t2 = VersionableDocument.load(t1.id, collection_name="tests")

    assert t2.current == t1.current

    # assert t2.current.model_dump() == t2_expected.model_dump()

