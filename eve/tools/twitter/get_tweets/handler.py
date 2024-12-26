import tempfile
import subprocess
import json
from .... import eden_utils
# from ..common import X_client



async def handler(args: dict, db: str):
    pass
    # mentions = X_client.fetch_mentions()
    # mentions = {'output': ['{"data": [{"text": "Hey buddy  @SillySmile21038 @Akhil_lavadya @AjayVasava73118 @rsalusse @MelikianRa40150 @rajkumarnahar13 @elonfan1854 @Ernesto25954832 @US_Hot_Dog @JeremiahEr30464 https://t.co/RIgXToQVLM", "author_id": "1728653464601759744", "edit_history_tweet_ids": ["1814809535283916828"], "id": "1814809535283916828"}, {"text": "@cinebuzzbr @marinavgregory @rsalusse @paramountplusbr T\\u00e1 errado meu @ HAHAHAHAHH", "author_id": "1424172460153442308", "edit_history_tweet_ids": ["1533944964685635585"], "id": "1533944964685635585"}, {"text": "@rsalusse Nosso Amigo @ufc_matogrosso est\\u00e1 vendendo 4 ingressos para o UFC 142. Interessados add no msn diego_torrezini@live.com rf", "author_id": "456950209", "edit_history_tweet_ids": ["155611948959019009"], "id": "155611948959019009"}], "includes": {"users": [{"id": "1728653464601759744", "name": "idnaniotkirb", "username": "idnaniotki15704"}, {"id": "1424172460153442308", "name": "Bibi Lambe-Picas", "username": "servidorrrr"}, {"id": "456950209", "name": "Vivane Lins", "username": "VLins24625nes"}]}, "meta": {"result_count": 3, "newest_id": "1814809535283916828", "oldest_id": "155611948959019009"}}'], 'status': 'completed'}
    # mentions = {
    #     "output": {
    #         "data": [
    #         {
    #             "text": "Hey buddy  @SillySmile21038 @Akhil_lavadya @AjayVasava73118 @rsalusse @MelikianRa40150 @rajkumarnahar13 @elonfan1854 @Ernesto25954832 @US_Hot_Dog @JeremiahEr30464 https://t.co/RIgXToQVLM",
    #             "author_id": "1728653464601759744",
    #             "edit_history_tweet_ids": ["1814809535283916828"],
    #             "id": "1814809535283916828"
    #         },
    #         {
    #             "text": "@cinebuzzbr @marinavgregory @rsalusse @paramountplusbr Tá errado meu @ HAHAHAHAHH",
    #             "author_id": "1424172460153442308",
    #             "edit_history_tweet_ids": ["1533944964685635585"],
    #             "id": "1533944964685635585"
    #         },
    #         {
    #             "text": "@rsalusse Nosso Amigo @ufc_matogrosso está vendendo 4 ingressos para o UFC 142. Interessados add no msn diego_torrezini@live.com rf",
    #             "author_id": "456950209",
    #             "edit_history_tweet_ids": ["155611948959019009"],
    #             "id": "155611948959019009"
    #         }
    #         ],
    #         "includes": {
    #         "users": [
    #             {
    #             "id": "1728653464601759744",
    #             "name": "idnaniotkirb",
    #             "username": "idnaniotki15704"
    #             },
    #             {
    #             "id": "1424172460153442308",
    #             "name": "Bibi Lambe-Picas",
    #             "username": "servidorrrr"
    #             },
    #             {
    #             "id": "456950209",
    #             "name": "Vivane Lins",
    #             "username": "VLins24625nes"
    #             }
    #         ]
    #         },
    #         "meta": {
    #         "result_count": 3,
    #         "newest_id": "1814809535283916828",
    #         "oldest_id": "155611948959019009"
    #         }
    #     }
    # }
    # print(json.dumps(mentions, indent=2))

    # print("--------------------------------")
    # print(mentions)
    # print("--------------------------------")

    # return {
    #     "output": json.dumps(mentions)
    # }


