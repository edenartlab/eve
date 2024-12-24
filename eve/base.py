import copy
from pydantic import BaseModel, Field, create_model
from typing import Any, Optional, Type, List, Dict, Union, get_origin, get_args, Literal, Tuple

from . import eden_utils


class VersionableBaseModel(BaseModel):
    """
    A versioned wrapper for Pydantic BaseModels that tracks changes over time.

    Attributes:
        schema: The Pydantic model class
        initial: Initial state of the model
        current: Current state of the model
        edits: List of applied edits
    """
    schema: Type[BaseModel]
    initial: BaseModel
    current: BaseModel
    edits: List[BaseModel] = Field(default_factory=list)

    def __init__(self, instance: BaseModel=None, **kwargs):
        if instance is not None:
            data = {
                "schema": type(instance),
                "initial": instance,
                "current": instance
            }
            super().__init__(**data)
        else:
            super().__init__(**kwargs)

    @classmethod
    def model_validate(cls, obj: Any):
        obj['schema'] = recreate_base_model(obj['schema'])
        return super().model_validate(obj)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data['schema'] = {
            'name': data['schema'].__name__,
            'schema': data['schema'].model_json_schema()
        }
        data['current'] = self.current.model_dump()
        data['initial'] = self.initial.model_dump()
        data['edits'] = [edit.model_dump() for edit in self.edits]
        return data
    
    def get_edit_model(self) -> Type[BaseModel]:
        return generate_edit_model(self.schema)

    def apply_edit(self, edit: BaseModel):
        self.current = apply_edit(self.current, edit)
        self.edits.append(edit)    

    def reconstruct_version(self, version: int) -> BaseModel:
        if version < 0 or version > len(self.edits):
            raise ValueError("Invalid version number")
        instance = copy.deepcopy(self.initial)
        for edit in self.edits[:version]:
            instance = apply_edit(instance, edit)
        return instance
    

def generate_edit_model(
    model: Type[BaseModel]
) -> Type[BaseModel]:
    """
    Generate an edit model for a given Pydantic model.

    Args:
        model (Type[BaseModel]): The source Pydantic model to generate an edit model from

    Returns:
        Type[BaseModel]: A new Pydantic model class that represents possible edits to the source model
    """

    edit_fields: Dict[str, Any] = {}
    model_description = model.__doc__ or ""
    edit_model_description = f"Edit a {model.__name__} ({model_description.strip()})"

    for name, field in model.__annotations__.items():
        origin = get_origin(field)
        args = get_args(field)

        # Optional[actual_type] detected
        if origin is Union and type(None) in args:
            actual_type = next(arg for arg in args if arg is not type(None))
            origin = get_origin(actual_type)
            args = get_args(actual_type)

        field_info = model.model_fields[name]
        field_description = field_info.description or ""

        if origin in (list, List):
            item_type = args[0]
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                nested_edit_model = generate_edit_model(item_type)
                edit_fields[f'add_{name}'] = (Optional[Dict[str, Union[int, item_type]]], None, f"Add {model.__name__} {name} ({field_description})")
                edit_fields[f'edit_{name}'] = (Optional[Dict[str, Union[int, nested_edit_model]]], None, f"Edit {model.__name__} {name} ({field_description})")
                edit_fields[f'remove_{name}'] = (Optional[int], None, f"Remove {model.__name__} {name} ({field_description})")
            else:
                edit_fields[f'add_{name}'] = (Optional[Dict[str, Union[int, item_type]]], None, f"Add {model.__name__} {name} ({field_description})")
                edit_fields[f'edit_{name}'] = (Optional[Dict[str, Union[int, item_type]]], None, f"Edit {model.__name__} {name} ({field_description})")
                edit_fields[f'remove_{name}'] = (Optional[int], None, f"Remove {model.__name__} {name} ({field_description})")
        elif origin in (dict, Dict):
            key_type, value_type = args
            if isinstance(value_type, type) and issubclass(value_type, BaseModel):
                nested_edit_model = generate_edit_model(value_type)
                edit_fields[f'add_{name}'] = (Optional[Dict[key_type, value_type]], None, f"Add {model.__name__} {name} ({field_description})")
                edit_fields[f'edit_{name}'] = (Optional[Dict[key_type, nested_edit_model]], None, f"Edit {model.__name__} {name} ({field_description})")
                edit_fields[f'remove_{name}'] = (Optional[key_type], None, f"Remove {model.__name__} {name} ({field_description})")
            else:
                edit_fields[f'add_{name}'] = (Optional[Dict[key_type, value_type]], None, f"Add {model.__name__} {name} ({field_description})")
                edit_fields[f'edit_{name}'] = (Optional[Dict[key_type, value_type]], None, f"Edit {model.__name__} {name} ({field_description})")
                edit_fields[f'remove_{name}'] = (Optional[key_type], None, f"Remove {model.__name__} {name} ({field_description})")
        elif isinstance(field, type) and issubclass(field, BaseModel):
            nested_edit_model = generate_edit_model(field)
            edit_fields[f'edit_{name}'] = (Optional[nested_edit_model], None, f"Edit {model.__name__} {name} ({field_description})")
        else:
            edit_fields[f'edit_{name}'] = (Optional[field], None, f"Edit {model.__name__} {name} ({field_description})")
    
    edit_model = create_model(
        f'{model.__name__}Edit',
        **{key: (value[0], Field(default=value[1], description=value[2])) for key, value in edit_fields.items()},
        __base__=BaseModel
    )

    edit_model.__doc__ = edit_model_description

    return edit_model


def apply_edit(
    instance: BaseModel, 
    edit: BaseModel
) -> BaseModel:
    """
    Apply modifications specified in an edit model to a BaseModel instance.

    This function handles three types of edits:
    - add_*: Add new items to lists or dictionaries
    - edit_*: Modify existing values, including nested models
    - remove_*: Remove items from lists or dictionaries

    Args:
        instance (BaseModel): The original model instance to be modified
        edit (BaseModel): An edit model containing the changes to apply

    Returns:
        BaseModel: A new instance with the edits applied, leaving the original unchanged

    Example:
        original = MyModel(field=[1, 2, 3])
        edit = MyModelEdit(add_field={'index': 1, 'value': 4})
        result = apply_edit(original, edit)  # result.field = [1, 4, 2, 3]
    """

    instance_copy = copy.deepcopy(instance)
    updates = {}
    for field_name, value in edit:
        if value is not None:
            
            if field_name.startswith('add_'):
                original_field = field_name.replace('add_', '')
                if isinstance(value, dict) and 'index' in value:
                    index = value['index']
                    new_value = value['value']
                    if getattr(instance_copy, original_field) is None:
                        setattr(instance_copy, original_field, [])
                    current_list = getattr(instance_copy, original_field)
                    current_list.insert(index, new_value)
                    setattr(instance_copy, original_field, current_list)
                elif isinstance(value, dict):
                    if getattr(instance_copy, original_field) is None:
                        setattr(instance_copy, original_field, {})
                    for key, new_value in value.items():
                        key = int(key) if isinstance(getattr(instance_copy, original_field), list) else key
                        getattr(instance_copy, original_field)[key] = new_value
            
            elif field_name.startswith('edit_'):
                original_field = field_name.replace('edit_', '')
                if isinstance(value, dict) and 'index' in value and 'value' in value:
                    index = value['index']
                    new_value = value['value']
                    if isinstance(new_value, BaseModel):
                        current_value = getattr(instance_copy, original_field)[index]
                        nested_updated = apply_edit(current_value, new_value)
                        getattr(instance_copy, original_field)[index] = nested_updated
                    else:
                        getattr(instance_copy, original_field)[index] = new_value
                elif isinstance(value, dict):
                    for key, new_value in value.items():
                        key = int(key) if isinstance(getattr(instance_copy, original_field), list) else key
                        if isinstance(new_value, BaseModel):
                            current_value = getattr(instance_copy, original_field)[key]
                            nested_updated = apply_edit(current_value, new_value)
                            getattr(instance_copy, original_field)[key] = nested_updated
                        else:
                            getattr(instance_copy, original_field)[key] = new_value
                elif isinstance(value, BaseModel):
                    nested_instance = getattr(instance_copy, original_field)
                    nested_updated = apply_edit(nested_instance, value)
                    setattr(instance_copy, original_field, nested_updated)
                else:
                    updates[original_field] = value
            
            elif field_name.startswith('remove_'):
                original_field = field_name.replace('remove_', '')
                if getattr(instance_copy, original_field) is None:
                    continue
                if isinstance(value, int):
                    getattr(instance_copy, original_field).pop(value)
                else:
                    getattr(instance_copy, original_field).pop(value, None)
            else:
                original_field = field_name.replace('edit_', '')
                updates[original_field] = value

    if instance_copy is None:
        instance_copy = type(edit)()
    elif isinstance(instance_copy, dict):
        instance_copy = type(edit).model_validate(instance_copy)

    return instance_copy.model_copy(update=updates)


def recreate_base_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Build a BaseModel from a type model data object.
    """

    model_name = schema['name']
    model_schema = schema['schema']
    base_model = create_model(model_name, **{
        field: (get_python_type(info), ... if info.get('required', False) else None)
        # for field, info in model_schema['parameters'].items()
        for field, info in model_schema['properties'].items()
    })

    return base_model


def get_python_type(field_info):
    """
    Retrieve the Python type from a field info object.
    """

    type_map = {
        'string': str,
        'integer': int,
        'float': float,
        'boolean': bool,
        'array': List,
        'object': Dict,
        'image': str,
        'video': str,
        'audio': str,
        'lora': str,
        'zip': str
    }
    field_type = field_info.get('type')
    if field_type == 'array' and 'items' in field_info:
        item_type = get_python_type(field_info['items'])
        output_type = List[item_type]
    elif field_type == 'object':
        output_type = Dict[str, Any]
    else:
        output_type = type_map.get(field_type, Any)
    return output_type


def parse_props(field: str, props: dict) -> Tuple[Type, dict, dict]:
    field_kwargs = {}
    json_schema_extra = {}
    
    if 'description' in props:
        field_kwargs['description'] = props['description']
        if 'tip' in props:
            field_kwargs['description'] = eden_utils.concat_sentences(field_kwargs['description'], props['tip'])
    if 'example' in props:
        json_schema_extra['example'] = props['example']
    
    if 'default' in props:
        field_kwargs['default'] = props['default']

    # Handle array
    if props['type'] == 'array':
        if 'min_length' in props:
            field_kwargs['min_length'] = props['min_length']
        if 'max_length' in props:
            field_kwargs['max_length'] = props['max_length']

    # Handle min and max for numeric types
    if props['type'] in ['integer', 'float']:
        if 'minimum' in props:
            field_kwargs['ge'] = props['minimum']
        if 'maximum' in props:
            field_kwargs['le'] = props['maximum']
    
    # Handle choices
    if props['type'] in ['integer', 'float', 'string'] and 'choices' in props:
        field_kwargs['choices'] = props['choices']
        return Literal[tuple(props['choices'])], field_kwargs, json_schema_extra
    
    # Handle file types
    if props['type'] in ['image', 'video', 'audio', 'lora', 'zip']:
        json_schema_extra['file_type'] = props['type']
        
    # Handle different types
    if props['type'] == 'object':
        fields, model_config = parse_schema(props)
        type_annotation = create_model(
            field, 
            __config__=model_config,
            **fields
        )
    elif props['type'] == 'array':
        json_schema_extra['is_array'] = True
        if props['items']['type'] == 'object':
            fields, model_config = parse_schema(props['items'])
            item_type = create_model(
                f"{field}Item", 
                __config__=model_config,
                **fields
            )
        else:
            item_type = get_python_type(props['items'])
            if props['items']['type'] in ['image', 'video', 'audio', 'lora', 'zip']:
                json_schema_extra['file_type'] = props['items']['type']
        type_annotation = List[item_type]
    else:
        type_annotation = get_python_type(props)
        
    return type_annotation, field_kwargs, json_schema_extra


def parse_schema(schema: dict) -> Tuple[Dict[str, Tuple[Type, Any]], dict]:
    fields = {}
    
    for field, props in schema.get('parameters', {}).items():
        # anyOf makes a Union of its types
        if props.get('anyOf'):
            types = []
            field_kwargs = {}
            json_schema_extra = {}
            
            for prop in props['anyOf']:
                type_annotation, prop_kwargs, prop_extra = parse_props(field, prop)
                types.append(type_annotation)
                field_kwargs.update(prop_kwargs)
                if 'file_type' in json_schema_extra:
                    json_schema_extra['file_type'] += f"|{prop_extra['file_type']}"
                    prop_extra.pop('file_type')
                elif 'file_type' in prop_extra:
                    json_schema_extra['file_type'] = prop_extra['file_type']
                json_schema_extra.update(prop_extra)
                
            type_annotation = Union[tuple(types)]
        else:
            type_annotation, field_kwargs, json_schema_extra = parse_props(field, props)

        if 'examples' in props:
            json_schema_extra['examples'] = props['examples']

        if props.get('required'):
            field_kwargs['default'] = ...
        else:
            field_kwargs['default'] = props.get("default", None)

        if field_kwargs.get('default') == "random":
            field_kwargs['default'] = None

        fields[field] = (type_annotation, Field(**field_kwargs, json_schema_extra=json_schema_extra))

    model_config = {}
    if schema.get('examples'):
        model_config['json_schema_extra'] = {'examples': schema['examples']}

    return fields, model_config







##### Old, deprecated
from pydantic import ConfigDict
from pydantic.json_schema import SkipJsonSchema
from bson import ObjectId
from datetime import datetime
from typing import Annotated

class VersionableMongoModel(VersionableBaseModel):
    id: Annotated[ObjectId, Field(default_factory=ObjectId, alias="_id")]
    collection_name: SkipJsonSchema[str] = Field(..., exclude=True)
    db: SkipJsonSchema[str] = Field(..., exclude=True)
    # createdAt: datetime = Field(default_factory=lambda: datetime.utcnow().replace(microsecond=0))
    # updatedAt: Optional[datetime] = None #Field(default_factory=lambda: datetime.utcnow().replace(microsecond=0))

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, **data):
        if 'instance' in data:
            instance = data.pop('instance')
            collection_name = data.pop('collection_name')
            db = data.pop('db')
            super().__init__(
                schema=type(instance),
                initial=instance,
                current=instance,
                collection_name=collection_name,
                db=db,
                **data
            )
        else:
            super().__init__(**data)

    @classmethod
    def load(cls, document_id: str, collection_name: str, db: str):
        collection = get_collection(collection_name, db)
        document = collection.find_one({"_id": ObjectId(document_id)})
        if document is None:
            raise MongoDocumentNotFound(collection_name, db, document_id)
        
        schema = recreate_base_model(document['schema'])
        initial = schema(**document['initial'])
        current = schema(**document['current'])
        
        edits = [
            generate_edit_model(schema)(**edit) 
            for edit in document['edits']
        ]
        
        versionable_data = {
            "id": document['_id'],
            "collection_name": collection_name, 
            "db": db,
            # "createdAt": document['createdAt'],
            # "updatedAt": document['updatedAt'],
            "schema": schema,
            "initial": initial,
            "current": current,
            "edits": edits
        }
        
        return cls(**versionable_data)

    def save(self, upsert_filter=None):
        data = self.model_dump(by_alias=True, exclude_none=True)
        collection = get_collection(self.collection_name, self.db)

        document_id = data.get('_id')
        if upsert_filter:
            document_id_ = collection.find_one(upsert_filter, {"_id": 1})
            if document_id_:
                document_id = document_id_["_id"]

        if document_id:
            # data['updatedAt'] = datetime.utcnow().replace(microsecond=0)
            collection.update_one({'_id': document_id}, {'$set': data}, upsert=True)
        else:
            collection.insert_one(data)
