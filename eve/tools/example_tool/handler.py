from ...eden_utils import download_file

async def handler(args: dict, db: str):   
    # print("args", args)
    
    # you can download files
    path = "https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/62946527441201f82e0e3d667fda480e176e9940a2e04f4e54c5230665dfc6f6.jpg"
    image_path = download_file(path, "myimg1.jpg")

    # you can call other tools
    #from ...tool import Tool
    #txt2img = Tool.load(key="txt2img")
    #result = await txt2img.run(args)

    result = {
        "output": image_path,
        "intermediate_outputs": {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }
    }
    return result