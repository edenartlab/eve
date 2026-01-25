# Twitter Account Tracking Service - Technical Documentation

## Summary

A production-ready service that continuously monitors Twitter accounts for all engagement activity (mentions, replies, source tweets, and quote tweets). Built on top of Eve's Twitter integration (`eve.tools.twitter.X`), it polls the Twitter API v2 every 60 seconds, persists all data to MongoDB, downloads media files locally, and maintains state for crash recovery.

**Key Features:**
- 100% complete coverage for mentions, replies, and source tweets
- Smart prioritization for quote tweet detection (based on engagement metrics)
- Automatic OAuth 2.0 token refresh (every ~2 hours)
- Fault-tolerant with MongoDB-backed state persistence
- Rate limit aware with automatic backoff
- Media download with SHA256 checksums
- No data loss on crashes (resumes from last `since_id`)

**Resource Usage:**
- ~1-6 API calls per minute (well under rate limits)
- Search endpoint: 1 call/min (limit: 30/min)
- Quote tweets endpoint: 0-5 calls/min (limit: 5/min)
- All tweets within Twitter's 7-day search window are captured

---

## Architecture

### Data Flow

```
Twitter API v2
    ↓
TwitterAPIWrapper (handles auth, rate limits, retries)
    ↓
DataPersister (upserts to MongoDB)
    ↓
MediaDownloader (downloads media to local disk)
    ↓
MongoDB Collections: X_tweets, X_users, X_media, X_tracked_handles, X_state
```


## MongoDB Schema

All collections use Eve's `Document` pattern from `eve.mongo`.

### Collections

**`X_tracked_handles`** - Accounts being monitored
- `username` (str) - Twitter username
- `user_id` (str) - Twitter user ID
- `active` (bool) - Whether tracking is active
- `created_at`, `updated_at` (datetime)

**`X_tweets`** - Normalized tweet data
- `_id` (str) - Tweet ID (unique)
- `author_id` (str) - Tweet author's user ID
- `text` (str) - Tweet content
- `created_at` (datetime) - When tweet was posted
- `conversation_id` (str) - Thread ID
- `in_reply_to_user_id` (str, nullable) - Reply target
- `referenced_tweets` (List[Dict]) - Quoted/replied tweets
- `entities`, `attachments`, `lang`, `source`, `reply_settings`
- `public_metrics` (Dict) - `{retweet_count, reply_count, like_count, quote_count}`
- `raw` (Dict) - Full API response (future-proofing)
- `first_seen_at`, `last_seen_at` (datetime)
- **Indexes**: `_id` (unique), `author_id`, `created_at`, `conversation_id`

**`X_users`** - User profile data
- `_id` (str) - User ID (unique)
- `username` (str) - Twitter handle
- `name` (str) - Display name
- `verified`, `verified_type`, `protected` (bool/str)
- `profile_image_url` (str)
- `public_metrics` (Dict) - `{followers_count, following_count, tweet_count}`
- `bio_snapshot` (Dict) - `{description, url, location}`
- `created_at` (datetime) - Account creation date
- `last_updated_at` (datetime)
- `raw` (Dict)
- **Indexes**: `_id` (unique), `username` (unique)

**`X_media`** - Media metadata and download tracking
- `_id` (str) - media_key (unique)
- `type` (str) - "photo", "video", "animated_gif"
- `url` (str) - Direct URL for photos
- `variants` (List[Dict]) - Video variants with bitrates
- `width`, `height`, `duration_ms`, `alt_text`
- `public_metrics` (Dict) - View counts
- `download` (Dict):
  ```json
  {
    "status": "pending|ok|failed",
    "stored_path": "./x-media/<media_key>.<ext>",
    "byte_size": 12345,
    "checksum": "sha256...",
    "last_attempt_at": datetime,
    "last_success_at": datetime,
    "attempts": 0
  }
  ```
- `raw` (Dict)

**`X_state`** - Key-value state persistence
- `_id` (str) - State key (e.g., `"since_id:combined"`, `"quote_checked:1234567890"`)
- `value` (Any) - State value (tweet ID, timestamp, etc.)
- `updated_at` (datetime)

---

## Core Components

### 1. TwitterAPIWrapper

**Location:** `twitter_tracker.py:174-366`

Wraps Eve's `X` client with automatic token refresh and rate limit handling.

**Key Methods:**

**`__init__(deployment_id: str)`**
- Loads deployment from MongoDB
- Initializes Eve's `X(deployment)` client
- Stores OAuth 2.0 access token and refresh token

**`refresh_access_token()`**
- Calls Twitter OAuth 2.0 token endpoint with refresh token
- Updates deployment in MongoDB with new tokens
- Updates local `X` client instance
- Called automatically on 401 errors

**`_make_api_call(method, url, params, retry_on_401=True)`**
- Makes authenticated API request via Eve's `X._make_request()`
- Extracts rate limit headers (`x-rate-limit-remaining`, `x-rate-limit-reset`)
- Catches 401 errors and auto-refreshes token (once)
- Returns `(response_json, rate_limit_headers)`
- Increments `calls_this_loop` counter

**`fetch_search_recent(query, since_id, next_token, max_results)`**
- Endpoint: `GET /2/tweets/search/recent`
- Full field/expansion sets (see TWEET_FIELDS, USER_FIELDS, MEDIA_FIELDS)
- Returns tweets, users, media in single response

**`fetch_quote_tweets(tweet_id, next_token, max_results)`**
- Endpoint: `GET /2/tweets/:id/quote_tweets`
- Same field/expansion sets
- Returns quote tweets for a specific tweet

**`fetch_users_by_usernames(usernames)`**
- Endpoint: `GET /2/users/by?usernames=...`
- Used during bootstrap to resolve usernames → user IDs

**`check_rate_limit_and_sleep(rate_limit_headers)`**
- If `x-rate-limit-remaining == 0`, sleeps until `x-rate-limit-reset` + 5s jitter

---

### 2. DataPersister

**Location:** `twitter_tracker.py:413-519`

Handles all MongoDB operations using Eve's Document pattern.

**Key Methods:**

**`upsert_tweets(response_data)`**
- Extracts tweets from `response["data"]`
- For each tweet:
  - Check if exists: `Tweet.find({"_id": tweet_id})`
  - If exists: update `last_seen_at`
  - If new: create `Tweet(...)`, set `tweet.id = tweet_id`, save
- Returns list of tweet IDs processed

**`upsert_users(response_data)`**
- Extracts users from `response["includes"]["users"]`
- Updates or creates user documents
- Captures `bio_snapshot` for historical tracking

**`upsert_media(response_data)`**
- Extracts media from `response["includes"]["media"]`
- Creates media documents with `download.status = "pending"`
- Returns list of media_keys needing download

**`get_since_id(key)` / `set_since_id(key, value)`**
- Reads/writes state from `X_state` collection
- Used for cursor tracking (`since_id:combined`) and quote check cache (`quote_checked:<tweet_id>`)

---

### 3. MediaDownloader

**Location:** `twitter_tracker.py:522-598`

Downloads media files to local disk with retry logic.

**Key Methods:**

**`download_media(media_key)`**
- Skips if `download.status == "ok"` or `attempts >= 3`
- For photos: downloads `media.url`
- For videos/GIFs: selects highest bitrate MP4 from `variants`
- Saves to `./x-media/<media_key>.<ext>`
- Computes SHA256 checksum
- Updates `Media` document with download status

---

### 4. TwitterPoller

**Location:** `twitter_tracker.py:601-932`

Main polling logic orchestration.

**Key Methods:**

**`paginate_search(query, since_id_key, max_pages)`**
- Fetches search results with pagination
- Follows `next_token` up to `max_pages` (default 5)
- Upserts tweets, users, media
- Downloads media files
- Tracks `max_tweet_id` seen
- Only updates `since_id` after successful batch (crash safety)
- Returns count of new tweets

**`poll_tweets_mentions_replies(usernames)`**
- Query: `(@alice OR to:alice OR from:alice OR @bob OR to:bob OR from:bob) -is:retweet`
- Combines mentions, replies, and source tweets in ONE query
- State key: `"since_id:combined"`
- Max 5 pages per loop

**`poll_quote_tweets(usernames)`**
- Smart prioritization strategy:
  1. Get last 50 tweets from tracked users (from MongoDB)
  2. Filter: `quote_count > 0 OR like_count > 10`
  3. Sort by `quote_count DESC`, then `like_count DESC`
  4. Take top 5
  5. Check 4-hour cache: skip if checked recently
  6. Fetch quotes via `fetch_quote_tweets(tweet_id)`
  7. Stop early if rate limit low (`remaining < 10`)
- Returns count of new quote tweets

**`run_poll_loop(usernames)`**
- Resets API call counter
- Calls `poll_tweets_mentions_replies()`
- Calls `poll_quote_tweets()`
- Prints summary stats

---

### 5. Query Construction

**Location:** `twitter_tracker.py:388-409`

**`build_combined_query(usernames)`**
- Builds: `(@user1 OR to:user1 OR from:user1 OR @user2 OR to:user2 OR from:user2) -is:retweet`
- `-is:retweet` excludes retweets (only original content)

---

### 6. Bootstrap

**Location:** `twitter_tracker.py:935-960`

**`bootstrap_tracked_handles(api, usernames)`**
- Calls `fetch_users_by_usernames()` to resolve IDs
- Creates/updates `TrackedHandle` documents
- Creates/updates `User` documents
- Should be run once at startup (automatically called in `main()`)

---

## Configuration

**Location:** `twitter_tracker.py:35-57`

```python
# Accounts to track
TRACKED_ACCOUNTS = ["abraham_ai_", "genekogan"]

# API fields (comprehensive sets from PLAN.md)
TWEET_FIELDS = "id,text,created_at,author_id,conversation_id,..."
USER_FIELDS = "id,username,name,verified,..."
MEDIA_FIELDS = "media_key,type,url,..."
EXPANSIONS = "author_id,attachments.media_keys,..."

# Polling config
POLL_INTERVAL_SECONDS = 60
MAX_PAGES_PER_QUERY = 5
MAX_RESULTS_PER_PAGE = 100
MEDIA_DOWNLOAD_DIR = Path("./x-media")
MAX_MEDIA_DOWNLOAD_ATTEMPTS = 3
HARD_CAP_CALLS_PER_LOOP = 20
```

---

## Rate Limits

Twitter API v2 rate limits (per 15-minute window):

| Endpoint | Limit | Per Minute |
|----------|-------|------------|
| `/2/tweets/search/recent` | 450 | 30 |
| `/2/tweets/:id/quote_tweets` | 75 | 5 |
| `/2/users/by` | 300 | 20 |
| `/2/tweets` (lookup by ID) | 900 | 60 |

**Current Usage:**
- Search: 1 call/min (3.3% of limit)
- Quote tweets: 0-5 calls/min (0-100% of limit, adaptive)
- Total: 1-6 calls/min

---

## Command-Line Interface

```bash
# Bootstrap only (resolve usernames to IDs)
DB=PROD python twitter_tracker.py --deployment-id <id> --bootstrap-only

# Single poll loop (for testing)
DB=PROD python twitter_tracker.py --deployment-id <id> --once

# Continuous polling (production)
DB=PROD python twitter_tracker.py --deployment-id <id>
```

**Environment Variables Required:**
- `DB` - "STAGE" or "PROD" (for MongoDB database selection)
- `MONGO_URI` - MongoDB connection string
- `MONGO_DB_NAME` - Database name
- `TWITTER_INTEGRATIONS_CLIENT_ID` - OAuth 2.0 client ID
- `TWITTER_INTEGRATIONS_CLIENT_SECRET` - OAuth 2.0 client secret

---

## Dependencies

**Eve Packages:**
- `eve.agent.session.models.Deployment` - Load deployment from MongoDB
- `eve.tools.twitter.X` - Twitter API client
- `eve.mongo.Document`, `eve.mongo.Collection` - MongoDB ORM

**External:**
- `requests` - HTTP client
- `pydantic` - Data validation (via Eve's Document)
- `pymongo` - MongoDB driver (via Eve)
- `bson` - ObjectId handling

---

## Productionization Recommendations

### 1. Process Management

**Current:** Runs as foreground process with Ctrl+C handling

**Recommended:**
- Deploy as systemd service (Linux) or launchd (macOS)
- Use process manager like `supervisor` or `pm2`
- Add proper logging to file (currently stdout only)
- Add health check endpoint (simple HTTP server on :8080/health)

**Example systemd service:**
```ini
[Unit]
Description=Twitter Account Tracker
After=network.target mongod.service

[Service]
Type=simple
User=twitter-tracker
WorkingDirectory=/opt/twitter-tracker
Environment="DB=PROD"
Environment="MONGO_URI=mongodb://..."
ExecStart=/usr/bin/python3 twitter_tracker.py --deployment-id 69122c29abe75b272bf61e79
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

### 2. Logging

**Current:** Uses `print()` statements

**Recommended:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/twitter-tracker/tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

Replace all `print()` with `logger.info()`, `logger.warning()`, `logger.error()`

---

### 3. Monitoring & Alerting

**Metrics to Track:**
- API calls per minute (should stay < 6/min)
- Tweets processed per loop
- Media download success rate
- Token refresh events
- Rate limit 429 errors
- Crashes/restarts

**Recommended Tools:**
- Prometheus + Grafana for metrics
- Sentry for error tracking
- PagerDuty for critical alerts (e.g., > 1 hour downtime)

**Key Alerts:**
- Rate limit exhaustion (429 errors)
- Token refresh failures
- MongoDB connection loss
- No tweets found for > 24 hours (possible API issue)
- Disk space low (media directory)

---

### 4. Error Handling Improvements

**Add:**
- Exponential backoff on 5xx errors (currently immediate retry)
- Circuit breaker for repeated API failures
- Dead letter queue for failed media downloads
- Graceful degradation (skip quote tweets if rate limited, continue with mentions/replies)

---

### 5. Scalability

**Current Limitations:**
- Single-threaded, single process
- One tracked account list per instance
- Local media storage only

**For Multi-Account Tracking:**
- Run multiple instances with different `TRACKED_ACCOUNTS` configs
- Use instance-specific `since_id` keys (e.g., `"since_id:combined:instance1"`)
- Share MongoDB but separate media directories

**For High Volume:**
- Move media to S3/object storage
- Add async media download queue (Celery + Redis)
- Shard by account (one instance per high-volume account)

---

### 6. Backup & Disaster Recovery

**MongoDB Backups:**
- Schedule daily mongodump to S3
- Test restore procedure monthly
- Retention: 30 days minimum

**Media Backups:**
- Rsync `./x-media/` to S3 daily
- Consider lifecycle policy (archive after 90 days)

**State Recovery:**
- If `X_state` collection lost: service will re-fetch last 7 days on restart
- Critical: backup `X_state` before major upgrades

---

### 7. Security

**Current:**
- Deployment secrets stored in MongoDB (encrypted at rest assumed)
- OAuth tokens auto-refresh (good)
- No external API exposure

**Hardening:**
- Run as non-root user with minimal permissions
- Firewall: only MongoDB port open
- Use MongoDB authentication (require username/password)
- Rotate OAuth client secret quarterly
- Add rate limiting to prevent runaway API usage

---

### 8. Testing

**Unit Tests Needed:**
- Query construction (`build_combined_query`)
- Data persistence (mock MongoDB responses)
- Rate limit handling
- Token refresh logic

**Integration Tests:**
- End-to-end poll loop with test Twitter account
- MongoDB upsert idempotency
- Media download retry logic

**Load Tests:**
- Simulate 1000 tweets per minute
- Verify pagination handles > 500 results
- Test rate limit backoff under heavy load

---

### 9. Configuration Management

**Current:** Hardcoded in `TRACKED_ACCOUNTS`

**Recommended:**
- Move to environment variables or config file
- Support dynamic account addition without restart
- Use MongoDB collection for tracked accounts (already exists!)

**Example:**
```python
# Instead of hardcoded list, read from MongoDB
tracked_handles = TrackedHandle.find({"active": True})
usernames = [h.username for h in tracked_handles]
```

---

### 10. Media Storage Optimization

**Current Issues:**
- No cleanup of old media
- No deduplication (same media in multiple tweets)

**Improvements:**
- Add `created_at` to media filenames
- Implement retention policy (delete media > 90 days)
- Dedupe by checksum before download
- Compress old media (gzip)

---

## Troubleshooting Guide

### Service Won't Start

**Check:**
1. Environment variables set? (`DB`, `MONGO_URI`, etc.)
2. MongoDB accessible? `mongosh $MONGO_URI`
3. Deployment exists? Check ObjectId in MongoDB
4. Twitter credentials valid? Check deployment secrets

### Missing Tweets

**Possible Causes:**
1. Service offline > 7 days (Twitter limitation)
2. `since_id` corrupted in `X_state` collection
3. Rate limit exhaustion (check logs for 429 errors)

**Fix:**
- Check `X_state` collection for `since_id:combined`
- Delete state document to reset (will re-fetch last 7 days)

### Rate Limit Errors (429)

**Check:**
- How many instances running? (sharing same credentials)
- Quote tweet checks too aggressive? (increase cache time)
- Reduce `MAX_PAGES_PER_QUERY` or `HARD_CAP_CALLS_PER_LOOP`

### Token Refresh Failing

**Symptoms:** Repeated 401 errors, no successful refresh

**Check:**
1. `TWITTER_INTEGRATIONS_CLIENT_ID` and `CLIENT_SECRET` correct?
2. Refresh token expired? (re-authenticate via Eve platform)
3. Network can reach `api.twitter.com`?

**Fix:**
- Re-authenticate Twitter account via staging.app.eden.art
- Get new deployment ID
- Restart service with new ID

### Media Downloads Failing

**Check:**
1. Disk space available? `df -h`
2. Write permissions on `./x-media/`?
3. Network can reach Twitter CDN?

**Fix:**
- Check `X_media` collection for `download.status = "failed"`
- Delete media document to retry
- Increase `MAX_MEDIA_DOWNLOAD_ATTEMPTS`

---

## Performance Benchmarks

**Expected Performance (low volume):**
- CPU: < 5% (idle most of the time, 60s sleep between polls)
- Memory: ~100-200 MB
- Disk I/O: < 1 MB/s (media downloads)
- Network: < 100 KB/s average

**With 100 tweets/minute:**
- CPU: ~10-20%
- Memory: ~300-500 MB
- MongoDB writes: ~200 docs/min
- Media downloads: ~5-10 MB/min

---

## Future Enhancements

**Planned (from earlier discussion):**
1. Periodic metric updates (refresh `like_count`, `quote_count` for recent tweets)
   - Use `GET /2/tweets?ids=...` (batch 100 tweets)
   - Run hourly for tweets < 7 days old

2. Track liking users (if needed)
   - Endpoint: `GET /2/tweets/:id/liking_users`
   - Separate rate limit (75/15min)
   - Low priority (like_count already available)

3. Co-mentions tracking
   - Query: `@abraham_ai_ @genekogan` (both mentioned together)
   - Shows collaboration/association

4. Full conversation thread reconstruction
   - Use `conversation_id` to fetch all replies in a thread
   - Build tree structure for visualization

**Not Planned (out of scope):**
- Real-time streaming (would require different API tier)
- Historical backfill (> 7 days)
- Sentiment analysis
- User dashboard/UI

---

## Contact & Support

For questions about Eve integration or deployment:
- Check Eve documentation
- Review `twitter_search_boilerplate.py` for X client examples
- Review `mongo_example.py` for Document pattern examples

For Twitter API questions:
- [Twitter API v2 Documentation](https://developer.twitter.com/en/docs/twitter-api)
- [Search Tweets Guide](https://developer.twitter.com/en/docs/twitter-api/tweets/search/introduction)
- [Rate Limits](https://developer.twitter.com/en/docs/twitter-api/rate-limits)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-10
**Service Version:** twitter_tracker.py (production-ready)
