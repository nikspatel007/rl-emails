# Email Pipeline Analysis Report

**Date**: January 9, 2026
**User**: me@nik-patel.com
**Period**: 30 Days (Dec 10, 2025 - Jan 9, 2026)
**Configuration**: 30 workers, full LLM classification

---

## Performance Matrix

### Timing Breakdown

| Stage | Duration | Throughput |
|-------|----------|------------|
| **Gmail Sync** | 199s | 9.6 emails/sec |
| Threads | 0.1s | 16,590 threads/sec |
| Action Labels | 0.1s | 19,120 emails/sec |
| ML Features | 0.5s | 3,554 emails/sec |
| **Embeddings** | 42.3s | 42 emails/sec (API bound) |
| Rule Classification | 0.2s | 8,885 emails/sec |
| User Profiles | 0.1s | 4,010 users/sec |
| Clustering | 1.4s | 1,269 emails/sec |
| Priority | 0.1s | 17,770 emails/sec |
| **LLM Classification** | 61.6s | 9.4 emails/sec (API bound) |
| **Pipeline Total** | **106.4s** | |
| **End-to-End** | **~306s (~5 min)** | |

### Worker Scaling Comparison

| Metric | 15 Workers (7d) | 30 Workers (30d) | Improvement |
|--------|-----------------|------------------|-------------|
| Emails | 451 | 1,912 | 4.2x data |
| Embeddings Time | 13.4s | 42.3s | ~3x (4x data) |
| Embeddings/sec | 32/sec | 42/sec | **31% faster** |
| LLM Time | 10.4s | 61.6s | ~6x (8x emails) |
| Total Pipeline | 24.6s | 106.4s | Scales linearly |

### API Usage

| API | Calls | Tokens | Cost Estimate |
|-----|-------|--------|---------------|
| OpenAI Embeddings | 1,777 | ~1.8M tokens | ~$0.02 |
| OpenAI GPT (LLM) | 582 | 319,235 tokens | ~$3.20 |
| **Total** | 2,359 calls | ~2.1M tokens | **~$3.22** |

---

## Email Overview

| Metric | Value |
|--------|-------|
| Total Emails | 1,912 |
| Received | 1,777 |
| Sent | 135 |
| Threads | 1,659 |
| Unique Contacts | 401 |
| Avg/Day Received | **~59 emails** |

### Weekly Pattern

| Day | Avg Emails | Load |
|-----|------------|------|
| Thursday | 86 | Peak |
| Wednesday | 79 | High |
| Tuesday | 73 | High |
| Friday | 72 | High |
| Monday | 59 | Medium |
| Saturday | 41 | Low |
| Sunday | 35 | Low |

---

## Priority Distribution

| Level | Count | % |
|-------|-------|---|
| High (0.6+) | 14 | 0.8% |
| Medium (0.4-0.6) | 756 | 43% |
| Low (0.2-0.4) | 998 | 56% |
| Minimal (<0.2) | 9 | 0.5% |

---

## Action Items Requiring Attention

### Immediate/Security Alerts

| Subject | From | Action |
|---------|------|--------|
| GuardianLife login detected | GuardianLife | Verify login |
| Apple iCloud sign-in iPhone 17 Pro Max | Apple | Verify device |
| Dropbox new sign-in | Dropbox | Verify login |
| Find My disabled on Nikunj's iPhone | Apple | Investigate |

### Today/This Week - Reply Needed

| Subject | From | Urgency |
|---------|------|---------|
| 529 Contribution ($20K) | Morgan Stanley | TODAY |
| Arc Support - lost part | Arc Support | TODAY |
| 5 Attributes - suggestions | William Villiers | TODAY |
| Third party lead Zapier | Kent Watson | THIS WEEK |

### Tasks/Decisions Pending

| Subject | From | Type |
|---------|------|------|
| TREHO SERVICE invoice overdue | TREHO SERVICE | TASK |
| Supabase invoice #QXYVKG-00008 | Supabase | TASK |
| Apple Developer age ratings | Apple Developer | TASK |
| GitHub invites (3) | prajan301, Bo Motlagh | DECISION |
| Tensorlake project invite | Tensorlake | DECISION |

### High Priority Business Emails

| Subject | From | Score |
|---------|------|-------|
| LTM requirements approval | Jason Grovert | 0.67 |
| ACH Setup CodeTicks | Ann Avery | 0.66 |
| Avenue Code C2C Onboarding | Olivia Page | 0.65 |
| TCB Tax Invoice 2347 | Tim Baze | 0.64 |
| Breadcrumbs Engagement | Bo Motlagh | 0.61 |

---

## Urgency Breakdown (LLM Classified)

| Urgency | Count | % |
|---------|-------|---|
| Immediate | 8 | 1.4% |
| Today | 42 | 7.2% |
| This Week | 68 | 11.7% |
| Whenever | 301 | 51.7% |
| None/FYI | 163 | 28.0% |

**20% of emails need action within the week** (118 emails)

---

## AI Automation Potential

| Category | Count | % |
|----------|-------|---|
| **Fully Automatable** | 565 | **97.1%** |
| Partially | 17 | 2.9% |

### Suggested AI Actions

| Action | Count |
|--------|-------|
| File to Folder | 539 |
| Draft Reply | 31 |
| Schedule Meeting | 3 |
| Summarize Attachment | 3 |
| Prepare Context | 3 |

---

## Top Email Sources (30 Days)

| Sender | Count | Category |
|--------|-------|----------|
| Chase Alerts | 74 | Banking |
| LendingTree | 47 | Finance Spam |
| Humble Bundle | 42 | Gaming |
| Sotheby's Motorsport | 41 | Shopping |
| Brilliant Earth | 37 | Shopping |
| Inc Authority | 33 | Business Formation |
| Robinhood | 32 | Investing |
| Ekster | 30 | Shopping |
| Medium Digest | 30 | News |
| Brooks Brothers | 28 | Shopping |

---

## Content Clusters

| Topic | Count | Source |
|-------|-------|--------|
| LendingTree Offers | 54 | Finance Spam |
| VC/Tech News (NFX) | 50 | Newsletter |
| Slack Messages | 49 | Work |
| Car Auctions | 42 | Sotheby's |
| FT Partners Reports | 40 | Finance |
| Chase Transactions | 39 | Banking |
| LinkedIn Messages | 38 | Social |
| Arc Browser | 37 | Product |
| Tech Articles | 34 | LinkedIn |
| ClickUp/Productivity | 33 | Work |

---

## People Clusters (Communication Patterns)

| Group | Emails | Primary Contact | Reply Rate |
|-------|--------|-----------------|------------|
| Slack/PhySec | 391 | Various | 50% |
| Real Estate | 305 | Michael Thornton | 0% |
| Reclaim AI | 226 | Notifications | 0% |
| LendingTree | 169 | Marketing | 0% |
| OpenAI | 166 | Billing | 60% |
| Inc Authority | 149 | Marketing | 0% |
| Coach Outlet | 124 | Marketing | 0% |
| Chase | 75 | Alerts | 0% |
| **TCB Tax (Tim Baze)** | 73 | Tim Baze | **45%** |
| Aditya Software | 23 | Contractor | **44%** |
| Tony Price | 13 | Business | **100%** |
| Morgan Stanley | 13 | Craig Yong | **100%** |

---

## Key Takeaways

1. **Signal-to-Noise Ratio**: Only ~30 emails out of 1,777 (1.7%) actually need human attention
2. **97% automatable** - vast majority can be filed, summarized, or auto-responded
3. **Financial matters pending**: 529 contribution, TREHO invoice, TCB Tax invoice
4. **Security alerts**: 4 immediate login notifications to verify
5. **Business priority**: Bo Motlagh, Tim Baze, Morgan Stanley, Avenue Code are high-value contacts
6. **Shopping/Marketing noise**: ~40% of email volume is retail marketing
7. **Peak activity**: Thursday (86 emails avg) vs Sunday (35 emails avg)
8. **Processing cost**: ~$3.22 for full 30-day analysis with LLM classification
