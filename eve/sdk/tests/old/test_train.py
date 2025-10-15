from eden import EdenClient
from loguru import logger

client = EdenClient(stage=True)

result = client.create("lora_trainer", {
    "name": "Verdelis",
    "lora_training_urls": [
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502851/user_uploads/kyuuefmnf56cnot8mqug.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502851/user_uploads/u215yeg4zrka2li0lcb7.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502851/user_uploads/pqu4j4n4ytptmlbcrb39.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502851/user_uploads/bvp3jz5onrswtxu7uf56.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502851/user_uploads/ncvw1ct6mx77vixi0stt.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502851/user_uploads/jljlboduzcneveg4kxpo.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502852/user_uploads/zmpzbyyycxazkr4blqhy.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502852/user_uploads/fvgtfar6ivjlnxx86kp4.jpg",
        "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1716502856/user_uploads/ovpw8eub0kwjshqiwfcy.jpg"
    ],
    "sd_model_version": "sdxl",
    "concept_mode": "face",
    "max_train_steps": 220
})

logger.info(result)
