from eve.tool import Tool


def test_tool():
    tool = Tool.from_yaml('eve/tools/example_tool/api.yaml')

    args = tool.prepare_args({
        'type': 'thingy',
        'name': 'John', 
        'price': 1,
        'skills': ["cooking", "swimming"],
        'contacts': [
            {'type': 'email', 'value': 'widget@hotmail.com'},
            {'type': 'phone', 'value': '555-1234'},
        ]
    })

    age = args.get('age')  # this was randomly set
    
    assert isinstance(age, int)

    assert args == {
        'name': 'John', 
        'type': 'thingy', 
        'age': age,
        'price': 1, 
        'skills': ['cooking', 'swimming'], 
        'contacts': [
            {'type': 'email', 'value': 'widget@hotmail.com'}, 
            {'type': 'phone', 'value': '555-1234'}
        ]
    }
