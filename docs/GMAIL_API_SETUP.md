# Gmail API Setup Guide

This guide walks through setting up Google Cloud credentials for Gmail API access.

## Prerequisites

- Google account with Gmail
- Access to Google Cloud Console
- rl-emails project installed locally

---

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)

2. Click **Select a project** → **New Project**

3. Enter project details:
   - **Project name**: `rl-emails` (or your preferred name)
   - **Organization**: Select if applicable
   - **Location**: Select if applicable

4. Click **Create**

5. Wait for project creation, then select it

---

## Step 2: Enable Gmail API

1. In Google Cloud Console, go to **APIs & Services** → **Library**

2. Search for "Gmail API"

3. Click **Gmail API** in results

4. Click **Enable**

5. Wait for API to be enabled

---

## Step 3: Configure OAuth Consent Screen

1. Go to **APIs & Services** → **OAuth consent screen**

2. Select **User Type**:
   - **Internal**: If using Google Workspace (recommended for testing)
   - **External**: For general users (requires verification for production)

3. Click **Create**

4. Fill in App Information:
   ```
   App name: rl-emails
   User support email: your-email@example.com
   Developer contact: your-email@example.com
   ```

5. Click **Save and Continue**

6. **Scopes**: Click **Add or Remove Scopes**
   - Add: `https://www.googleapis.com/auth/gmail.readonly`
   - Add: `https://www.googleapis.com/auth/gmail.labels` (optional)

7. Click **Save and Continue**

8. **Test Users** (External only):
   - Add your Gmail address for testing

9. Click **Save and Continue**

10. Review and click **Back to Dashboard**

---

## Step 4: Create OAuth 2.0 Credentials

1. Go to **APIs & Services** → **Credentials**

2. Click **Create Credentials** → **OAuth client ID**

3. Select **Application type**: `Web application`

4. Enter details:
   ```
   Name: rl-emails-client

   Authorized JavaScript origins:
   - http://localhost:8000
   - http://127.0.0.1:8000

   Authorized redirect URIs:
   - http://localhost:8000/auth/google/callback
   - http://127.0.0.1:8000/auth/google/callback
   ```

5. Click **Create**

6. **Download JSON** or copy:
   - **Client ID**: `123456789-xxx.apps.googleusercontent.com`
   - **Client Secret**: `GOCSPX-xxx`

7. Store these securely!

---

## Step 5: Configure rl-emails

### Option A: Environment Variables

Add to your `.env` file:

```bash
# Gmail API OAuth Credentials
GOOGLE_CLIENT_ID=123456789-xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxx
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

### Option B: Credentials File

1. Download the OAuth client JSON from Google Cloud Console

2. Save as `credentials.json` in project root (add to .gitignore!)

3. The app will auto-detect this file

---

## Step 6: Verify Setup

### Test OAuth Flow (when implemented)

```bash
# Start the auth flow
rl-emails auth connect --email your@gmail.com

# This will:
# 1. Open browser to Google consent screen
# 2. After approval, redirect to callback URL
# 3. Store tokens in database
```

### Check Token Status

```bash
rl-emails auth status --email your@gmail.com

# Expected output:
# Email: your@gmail.com
# Status: Connected
# Token expires: 2026-01-07 21:42:00
# Scopes: gmail.readonly
```

---

## OAuth Flow Diagram

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   User      │     │   rl-emails     │     │   Google OAuth   │
│   Browser   │     │   Backend       │     │   Server         │
└─────────────┘     └─────────────────┘     └──────────────────┘
       │                    │                        │
       │  1. Click Connect  │                        │
       │───────────────────▶│                        │
       │                    │                        │
       │  2. Redirect URL   │                        │
       │◀───────────────────│                        │
       │                    │                        │
       │  3. Visit Google OAuth consent              │
       │─────────────────────────────────────────────▶
       │                    │                        │
       │  4. User approves  │                        │
       │                    │                        │
       │  5. Redirect with auth code                 │
       │◀────────────────────────────────────────────│
       │                    │                        │
       │  6. Auth code to callback                   │
       │───────────────────▶│                        │
       │                    │                        │
       │                    │  7. Exchange code      │
       │                    │───────────────────────▶│
       │                    │                        │
       │                    │  8. Access + Refresh   │
       │                    │◀───────────────────────│
       │                    │                        │
       │  9. Success!       │                        │
       │◀───────────────────│                        │
```

---

## API Quotas and Limits

### Default Quotas

| Resource | Limit |
|----------|-------|
| Queries per day | 1,000,000,000 |
| Queries per 100 seconds per user | 25,000 |
| Messages.get per user per second | ~10-50 |
| Batch requests | 100 calls per batch |

### Rate Limiting Strategy

rl-emails implements conservative rate limiting:

```python
# Default settings
REQUESTS_PER_SECOND = 10       # Well under quota
BATCH_SIZE = 100               # Max per batch
RETRY_DELAYS = [1, 2, 4, 8]    # Exponential backoff
```

### Quota Monitoring

Monitor your usage in Google Cloud Console:
1. Go to **APIs & Services** → **Dashboard**
2. Select **Gmail API**
3. View **Metrics** tab

---

## Scopes Reference

| Scope | Access | Use Case |
|-------|--------|----------|
| `gmail.readonly` | Read emails, labels | Primary scope for rl-emails |
| `gmail.labels` | Manage labels | Optional: label management |
| `gmail.metadata` | Headers only, no body | Lighter weight alternative |

**Recommended**: Use `gmail.readonly` for full email analysis.

---

## Troubleshooting

### Error: redirect_uri_mismatch

**Cause**: Redirect URI doesn't match OAuth config

**Fix**:
1. Go to Google Cloud Console → Credentials
2. Edit OAuth 2.0 Client
3. Add exact URI from error message to Authorized redirect URIs

### Error: access_denied

**Cause**: User denied consent or not a test user

**Fix**:
1. For External apps: Add user to Test Users list
2. Verify scopes are correctly configured
3. Try incognito browser window

### Error: invalid_client

**Cause**: Wrong client ID or secret

**Fix**:
1. Re-download credentials from Google Cloud Console
2. Verify `.env` has correct values
3. Check for extra whitespace in values

### Error: Token refresh failed

**Cause**: Refresh token expired or revoked

**Fix**:
1. Re-authenticate: `rl-emails auth connect --email your@gmail.com`
2. In Google Account → Security → Third-party apps, revoke and re-grant

### Error: Rate limit exceeded

**Cause**: Too many API calls

**Fix**:
1. Wait for quota reset (usually per-second)
2. Reduce `BATCH_SIZE` in config
3. Implement exponential backoff (built-in)

---

## Security Best Practices

### Credential Storage

1. **Never commit credentials** to version control
2. Add to `.gitignore`:
   ```
   credentials.json
   .env
   *.pem
   ```

3. Use environment variables in production

### Token Storage

1. Tokens stored in PostgreSQL `oauth_tokens` table
2. Consider encrypting `access_token` and `refresh_token` columns
3. Implement token rotation on refresh

### Scope Minimization

1. Request only necessary scopes
2. `gmail.readonly` is sufficient for email analysis
3. Don't request `gmail.modify` unless needed

---

## Production Considerations

### OAuth Verification

For External user type:
1. Google requires app verification
2. Submit verification request in OAuth consent screen
3. Provide privacy policy and terms of service
4. May require security assessment

### Rate Limit Increases

If default quotas insufficient:
1. Go to Google Cloud Console → APIs & Services → Gmail API
2. Click **Quotas** tab
3. Request quota increase with justification

### Multi-Tenant Considerations

1. Each user has own OAuth tokens
2. Tokens stored per user_id in database
3. Rate limits are per-user, not per-app

---

## Environment Variables Reference

```bash
# Required for Gmail API
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your-secret

# OAuth callback (must match Google Cloud config)
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

# Optional: Override default rate limits
GMAIL_REQUESTS_PER_SECOND=10
GMAIL_BATCH_SIZE=100
GMAIL_MAX_RETRIES=5
```

---

## Next Steps

After completing this setup:

1. **Phase 1**: Implement multi-tenant foundation (Iterations 1-3)
2. **Phase 2, Iteration 4**: Implement OAuth flow in rl-emails
3. **Phase 2, Iteration 5**: Implement Gmail API client
4. **Phase 2, Iteration 6**: Implement initial sync with `--days` parameter

See [ARCHITECTURE_PLAN.md](./ARCHITECTURE_PLAN.md) for full implementation plan.
