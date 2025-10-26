## Local Gateway Setup

Use ngrok to get a public URL for your local API.

```
ngrok http --domain=eden.ngrok.dev --authtoken [AUTH_TOKEN] 8000
```

Configure Eve .env.STAGE

```
GATEWAY_ID=[your-id]
LOCAL_API_URL=https://edenartlab--discord-gateway-v2-stage-[your-id]-gateway-app-dev.modal.run
EDEN_API_URL=[your-ngrok-url]
```

Start processes:

```
cd eve
rye run eve api
```

```
cd eden
yarn dev:eden
```

Create deployments in the local Eden app, then mark them as local in the database. `deployment.local = True`

Finally, start the gateway. LOCAL_USER_ID must be set to your Eden staging user id and will filter all deployments but your own.

```
cd eve
export LOCAL_API_URL=[your-ngrok-url] && LOCAL_USER_ID=[eden-user-id]
rye run modal serve eve/agent/deployments/gateway_v2.py
```

This will create a fully local gateway setup where you can edit both Eve & Gateway code and test changes without pushing or disrupting the existing staging gateway.