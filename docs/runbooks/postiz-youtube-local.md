# Local Postiz → YouTube publishing

This optional self-hosted integration adds a **send to postiz** button to each
completed OpenShorts clip. It streams that MP4 to a local Postiz instance, then
creates either a YouTube draft or an immediate YouTube post using the YouTube
account already connected in Postiz.

```text
OpenShorts clip → send to postiz → Postiz → connected YouTube channel
```

It does not use Upload-Post, n8n, a public callback URL, or a Google OAuth
secret in OpenShorts.

## Before starting

- Postiz must already be locally running and connected to YouTube.
- Upgrade Postiz to `v2.21.3` or newer.
- When Postiz uses `STORAGE_PROVIDER=local`, apply the companion local-media
  patch before using this integration. It lets Postiz read its own private
  `/uploads` files for both YouTube MP4 and thumbnail uploads without weakening
  SSRF protection.
- Create a **Postiz Public API key** in the same Postiz account that owns the
  YouTube connection.
- Keep both dashboard host ports on `127.0.0.1`.

## 1. Create the private Docker network

From PowerShell:

```powershell
docker network inspect openshorts-postiz *> $null
if ($LASTEXITCODE -ne 0) { docker network create openshorts-postiz }
```

## 2. Connect the Postiz Compose project

In the Postiz project's `docker-compose.yaml`, attach the `postiz` service to
the new network **without replacing its existing networks**.

Postiz itself needs to resolve `temporal:7233` for its background workflows.
In the current Postiz Compose layout, that requires its existing
`temporal-network` attachment. It also retains `postiz-network`. Add
`openshorts-postiz` as a third network; do not replace the list with
`default` and `openshorts-postiz`.

```yaml
services:
  postiz:
    networks:
      - postiz-network       # keep the existing entry
      - temporal-network     # keep: required to resolve temporal:7233
      - openshorts-postiz    # add this entry

networks:
  postiz-network:            # retain the existing definition unchanged
  temporal-network:          # retain the existing definition unchanged
  openshorts-postiz:
    external: true
    name: openshorts-postiz
```

The network names above are from the current Postiz Compose file. If your
local file already uses different network names, preserve its existing names
exactly and add only `openshorts-postiz`.

Before restarting, verify that the Postiz Compose project still includes the
Temporal service:

```powershell
docker compose config --services
```

It must list `postiz` and `temporal`. Current Postiz installations normally
also list `temporal-postgresql` and `temporal-elasticsearch`.

Restart the Postiz stack after correcting the Compose file:

```powershell
docker compose down
docker compose up -d
docker compose ps
docker compose logs --tail=100 temporal postiz
```

Do **not** add `-v`: these commands recreate containers but retain Postiz data
volumes. Do not publish Temporal port `7233`; it is used only within the
private Docker network.

Confirm that the Postiz container can resolve the internal Temporal hostname:

```powershell
docker compose exec postiz node -e "require('dns').lookup('temporal', (error, address) => { if (error) { console.error(error); process.exit(1); } console.log(address); })"
```

This must print a private Docker IP address. If `temporal` is absent from
`docker compose config --services`, restore or replace the incomplete Postiz
Compose file before continuing; adding the OpenShorts network alone cannot
create the required Temporal service.

## 3. Start OpenShorts with its Postiz override

This repository includes `docker-compose.postiz.local.yml`. It joins only the
OpenShorts backend to the private network; the frontend stays separate.

Keep any existing `docker-compose.override.yml` separate. It is the right
place for device-specific settings such as `gpus: all`; it must not become a
Postiz configuration file.

If you have the existing GPU override, start all three layers explicitly:

```powershell
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.postiz.local.yml up -d --build
```

If you do not have a device-specific override, use:

```powershell
docker compose -f docker-compose.yml -f docker-compose.postiz.local.yml up -d --build
```

The remaining commands show the GPU-override form. If your machine has no
`docker-compose.override.yml`, remove only
`-f docker-compose.override.yml` from each command.

Verify the internal hostname:

```powershell
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.postiz.local.yml exec backend python -c "import socket; print(socket.gethostbyname('postiz'))"
```

It must print a private Docker IP address. Do not add a Postiz port open to
your LAN to make this work.

## 4. Configure OpenShorts

Add the following to OpenShorts `.env`, without quotes around values:

```env
POSTIZ_API_BASE_URL=http://postiz:5000/api/public/v1
POSTIZ_API_KEY=paste-your-postiz-public-api-key

# Required if Postiz has more than one enabled YouTube connection. Leave blank
# only when exactly one exists.
POSTIZ_YOUTUBE_INTEGRATION_ID=

# draft: create a Postiz draft for review/scheduling (recommended).
# now: hand the clip to Postiz for immediate YouTube publishing.
POSTIZ_DEFAULT_POST_TYPE=draft

# private is the recommended first setting.
POSTIZ_YOUTUBE_PRIVACY_STATUS=private
POSTIZ_TIMEOUT_SECONDS=300
```

Allowed values:

- `POSTIZ_DEFAULT_POST_TYPE`: `draft` or `now`
- `POSTIZ_YOUTUBE_PRIVACY_STATUS`: `private`, `unlisted`, or `public`
- `POSTIZ_TIMEOUT_SECONDS`: 30–900

The base URL is intentionally fixed to the private Docker network. OpenShorts
rejects a different URL so the API key and MP4 cannot be redirected elsewhere.

Restart the backend after editing `.env`:

```powershell
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.postiz.local.yml up -d --force-recreate backend
```

## 5. Run the authenticated preflight

This verifies the network route, API key, API path, and selected enabled
YouTube integration. It does not print the key:

```powershell
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.postiz.local.yml exec backend python -c "from dotenv import load_dotenv; load_dotenv('/app/.env'); import postiz_client; print(postiz_client.preflight())"
```

If Postiz has multiple enabled YouTube integrations, set the returned desired
integration ID in `POSTIZ_YOUTUBE_INTEGRATION_ID` and recreate the backend.

## Use it

1. Generate clips in OpenShorts.
2. Optional thumbnail:
   - click **add thumbnail** and choose your own JPEG/PNG up to 2 MB; or
   - if configured, click **generate AI thumb**, review its Gemini-generated
     candidate(s), then select one.

   The AI option sends Gemini only one locally extracted JPG reference frame
   plus clip metadata; it never sends the short MP4. Choosing a manual image
   clears the selected AI candidate. You can click either option again to
   replace it, or **clear thumb** to send without a thumbnail.
3. Click **send to postiz**.
4. With `draft`, find the new draft in Postiz and review/schedule it.
5. With `now`, Postiz uploads it to YouTube using its connected account.

OpenShorts sends the current edited version of the MP4. A subtitle, hook,
reframe, or edit changes the file fingerprint, so the updated clip may be
sent once as its own version. An AI candidate is read directly from the
versioned OpenShorts output folder; a manual thumbnail travels from the
dashboard to OpenShorts. Both use the same private Docker network to Postiz.
The Postiz API key never reaches the dashboard or browser storage.

## Safety and recovery

- Without Postiz, start normally with `docker compose up -d --build`; Docker
  automatically includes the device-specific `docker-compose.override.yml`,
  so GPU support remains enabled.
- With Postiz, include all three files explicitly as shown above. This is
  necessary because supplying `-f` disables Compose's automatic discovery of
  `docker-compose.override.yml`.
- The Postiz API key never reaches the dashboard, browser storage, worker
  subprocesses, or API responses.
- A double-click produces one handoff.
- If Postiz confirms creation, another click for the same rendered file returns
  the saved result instead of creating a duplicate.
- The button is disabled while OpenShorts is editing, subtitling, adding a
  hook, dubbing, or bulk-processing clips, so an in-progress render is not
  handed off.
- If OpenShorts cannot know whether Postiz received the create-post request,
  it refuses a retry and tells you to inspect Postiz drafts/history. This
  avoids accidental duplicate YouTube uploads.
- A disabled **configure postiz** button means the local `POSTIZ_*`
  configuration is missing or invalid. Check `.env`, then recreate `backend`.
