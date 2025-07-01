from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig


class TwitterClient(PlatformClient):
    TOOLS = {
        "twitter_post": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Twitter credentials and add Twitter tools"""
        try:
            import tweepy

            # Initialize Twitter client
            client = tweepy.Client(
                bearer_token=secrets.twitter.bearer_token,
                consumer_key=secrets.twitter.consumer_key,
                consumer_secret=secrets.twitter.consumer_secret,
                access_token=secrets.twitter.access_token,
                access_token_secret=secrets.twitter.access_token_secret,
            )

            # Test credentials by getting user info
            user_info = client.get_me()
            print(f"Verified Twitter credentials for user: {user_info.data.username}")

            # Add Twitter tools to agent
            self.add_tools()

            return secrets, config
        except Exception as e:
            raise APIError(f"Invalid Twitter credentials: {str(e)}", status_code=400)

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Twitter"""
        pass

    async def stop(self) -> None:
        """Stop Twitter client"""
        self.remove_tools()
