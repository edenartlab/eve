import copy
from enum import Enum
from pydantic import BaseModel, Field, create_model
from typing import Any, Optional, Type, List, Dict, Union, get_origin, get_args

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

    # @classmethod
    # def load_from(cls, **kwargs):
    #     return cls(**kwargs)

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


def get_python_type(field_info):
    """
    Retrieve the Python type from a field info object.
    """

    type_map = {
        'str': str,
        'string': str,
        'int': int,
        'integer': int,
        'float': float,
        'bool': bool,
        'boolean': bool,
        'array': List,
        'object': Dict,
        'image': str,
        'video': str,
        'audio': str,
        'lora': str,
        'zip': str
    }
    # print("\n\n\n========\nFIELD INFO")
    # print(field_info)
    optional = 'anyOf' in field_info and {"type": "null"} in field_info['anyOf']

    # if 'anyOf' in field_info:
    #     types = field_info['anyOf']
    #     print("THE TYPES")
    #     print(types)
    #     null_type = next((t for t in types if t.get('type') == 'null'), None)
    #     types = [t for t in types if t != null_type]
    #     assert len(types) == 1 and null_type is not None
    #     field_info = types[0]
    #     is_optional = True


    field_type = field_info.get('type')
    if field_type == 'array' and 'items' in field_info:
        item_type = get_python_type(field_info['items'])
        output_type = List[item_type]
    elif field_type == 'object':
        output_type = Dict[str, Any]
    else:
        output_type = type_map.get(field_type, Any)
    # if optional:
    #     output_type = Optional[output_type]
    # print("OUTPUT TYPE")
    # print(output_type)
    # print("\n\n\n\n")
    return output_type



# def get_type(type_str: str):
#     type_mapping = {
#         'str': str,
#         'int': int,
#         'float': float,
#         'bool': bool,
#         'array': List,
#         'object': Dict[str, Any]
#     }
#     return type_mapping.get(type_str, Any)




def recreate_base_model(schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Build a BaseModel from a type model data object.
    """

    model_name = schema['name']
    model_schema = schema['schema']
    # print("lets create the model.........")
    # print(model_schema['properties'].keys())
    base_model = create_model(model_name, **{
        field: (get_python_type(info), ... if info.get('required', False) else None)
        # for field, info in model_schema['parameters'].items()
        for field, info in model_schema['properties'].items()
    })


    # print("BASE MODEL")
    # from pprint import pprint
    # pprint(base_model.model_fields)
    # print("\n\n\n\n")


    return base_model







def create_enum(name: str, choices: List[str]):
    return Enum(name, {str(choice): choice for choice in choices})

def parse_schema(schema: dict):
    fields = {}
    required_fields = schema.get('required', [])
    for field, props in schema.get('parameters', {}).items():
        field_kwargs = {}
        json_schema_extra = {}  # New dict for extra parameters
        
        if 'description' in props:
            field_kwargs['description'] = props['description']
            if 'tip' in props:
                field_kwargs['description'] = eden_utils.concat_sentences(field_kwargs['description'], props['tip'])
        if 'example' in props:
            json_schema_extra['example'] = props['example']  # Moved to json_schema_extra
        if 'default' in props:
            field_kwargs['default'] = props['default']
        
        # Store additional parameters
        # additional_props = {}
        
        # Handle min and max for int and float
        if props['type'] in ['int', 'float']:
            if 'minimum' in props:
                field_kwargs['ge'] = props['minimum']
            if 'maximum' in props:
                field_kwargs['le'] = props['maximum']
        
        # Handle enum for strings
        if props['type'] in ['int', 'float', 'str'] and 'choices' in props:
            enum_type = create_enum(f"{field.capitalize()}Enum", props['choices'])
            fields[field] = (enum_type, Field(**field_kwargs, json_schema_extra=json_schema_extra))
            continue
        
        # Add special handling for file types
        if props['type'] in ['image', 'video', 'audio', 'lora', 'zip']:
            json_schema_extra['file_type'] = props['type']  # Moved to json_schema_extra
        if props['type'] == 'array' and 'items' in props:
            if props['items']['type'] in ['image', 'video', 'audio', 'lora', 'zip']:
                json_schema_extra['file_type'] = props['items']['type']  # Moved to json_schema_extra

        if props['type'] == 'object':
            nested_model = create_model(field, **parse_schema(props))
            fields[field] = (nested_model, Field(**field_kwargs, json_schema_extra=json_schema_extra))
        elif props['type'] == 'array':
            item_type = get_python_type(props['items'])
            json_schema_extra['is_array'] = True  # Moved to json_schema_extra
            if props['items']['type'] == 'object':
                item_type = create_model(f"{field}Item", **parse_schema(props['items']))
            fields[field] = (List[item_type], Field(**field_kwargs, json_schema_extra=json_schema_extra))
        else:
            fields[field] = (get_python_type(props), Field(**field_kwargs, json_schema_extra=json_schema_extra))
    
        if 'alias' in props:
            json_schema_extra['alias'] = props['required']  # Moved to json_schema_extra

        # if 'hide_from_ui' in props:
        #     json_schema_extra['hide_from_ui'] = props['hide_from_ui']
        if 'hide_from_agent' in props:
            json_schema_extra['hide_from_agent'] = props['hide_from_agent']

        if not props.get('required') and not field in required_fields:
            fields[field] = (Optional[fields[field][0]], fields[field][1])
            fields[field][1].default = field_kwargs.get("default", None)

    return fields


