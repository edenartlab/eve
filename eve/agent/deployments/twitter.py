from typing import Optional
from pydantic import BaseModel
from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient, DeploymentSecrets, DeploymentConfig


class DeploymentSettingsTwitter(BaseModel):
    username: Optional[str] = None


class DeploymentSecretsTwitter(BaseModel):
    user_id: str
    bearer_token: str
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str


class TwitterClient(PlatformClient):
    TOOLS = {
        "tweet": {},
        "twitter_mentions": {},
        "twitter_search": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Twitter credentials and add Twitter tools to agent"""
        import tweepy

        # Validate Twitter credentials
        try:
            # Create Twitter client with OAuth 1.0a
            auth = tweepy.OAuth1UserHandler(
                consumer_key=secrets.twitter.consumer_key,
                consumer_secret=secrets.twitter.consumer_secret,
                access_token=secrets.twitter.access_token,
                access_token_secret=secrets.twitter.access_token_secret,
            )
            api = tweepy.API(auth)

            # Test the credentials by getting user info
            user = api.verify_credentials()
            print(f"Verified Twitter credentials for user: @{user.screen_name}")

            # Also test with v2 API using bearer token
            client = tweepy.Client(
                bearer_token=secrets.twitter.bearer_token,
                consumer_key=secrets.twitter.consumer_key,
                consumer_secret=secrets.twitter.consumer_secret,
                access_token=secrets.twitter.access_token,
                access_token_secret=secrets.twitter.access_token_secret,
            )

            # Test v2 API
            me = client.get_me()
            print(f"Verified Twitter v2 API for user: @{me.data.username}")

        except Exception as e:
            raise APIError(f"Invalid Twitter credentials: {str(e)}", status_code=400)

        # Add Twitter tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Twitter"""
        pass

    async def stop(self) -> None:
        """Stop Twitter client by removing tools"""
        self.remove_tools()
