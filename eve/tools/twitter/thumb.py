import datetime
from bson import ObjectId
from eve.mongo import get_collection
from eve.s3 import *
from eve.utils import upload_media, url_exists

creations = get_collection("creations3")
models = get_collection("models3")

# for creation in creations.find({"createdAt": {"$gt": datetime.datetime(2025, 7, 20)}}).sort("createdAt", -1):
for creation in models.find({"createdAt": {"$gt": datetime.datetime(2025, 7, 1)}}).sort("createdAt", -1):
# for creation in models.find({"_id": {"$in": [ObjectId("6893b1cbf40c1fc4490ed04a"), ObjectId("6899f4466ebea246e1cfef04"), ObjectId("6899f184482e4ecbacea15ca")]}}):
    if True: # creation["tool"] in ["reel", "create", "create_image", "create_video", "txt2img", "media_editor", "flux_dev_lora", "flux_dev"]:
        filename = creation["thumbnail"]
        url = get_full_url(filename)
        if url.endswith(".png"):
            url_thumb = url.replace(".png", "_1024.webp")
        elif url.endswith(".jpg"):
            url_thumb = url.replace(".jpg", "_1024.webp")
        elif url.endswith(".jpeg"):
            url_thumb = url.replace(".jpeg", "_1024.webp")
        elif url.endswith(".mp4"):
            url_thumb = url.replace(".mp4", "_1024.webp")
        else:
            print("!!! unknown file type", url)
            continue
        
        if url_exists(url_thumb):
            print("---> skip")
            continue
        print("!!! gen thumbnmailk", creation["_id"], "uploading", url)
        upload_media(url, save_thumbnails=True)

    print("================================================")
