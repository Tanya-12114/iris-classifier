# Jenkins Setup Guide

## 1. Start Jenkins (Docker)
From the project folder:
```bash
docker compose up -d --build
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```
Open http://localhost:8080, paste the password, choose **Install suggested plugins**, create an admin user.
Check that the **Git** and **Pipeline** plugins are installed (Manage Jenkins > Plugins). Add **GitHub** if missing.

## 2. Create the pipeline job
1. New Item > name `iris-classifier-pipeline` > **Pipeline** > OK.
2. Build Triggers: tick **GitHub hook trigger for GITScm polling** (the Jenkinsfile also polls every ~2 minutes as a fallback).
3. Pipeline > Definition: **Pipeline script from SCM** > SCM: **Git**.
4. Repository URL: `https://github.com/Tanya-12114/iris-classifier.git` (public repo, no credentials needed).
5. Branch: `*/main`. Script Path: `Jenkinsfile`. Save.

## 3. Webhook (optional, needs a public URL)
Jenkins on localhost cannot receive GitHub webhooks directly. Either rely on polling, or:
```bash
ngrok http 8080
```
GitHub repo > Settings > Webhooks > Add webhook:
- Payload URL: `https://<your-ngrok-id>.ngrok-free.app/github-webhook/`
- Content type: `application/json`, event: **Just the push event**.

## 4. Run it
Commit and push the new files, then click **Build Now** (or push again to test the trigger).
Expected stages: Checkout > Setup Environment > Test > Train Model (MLflow) > Archive Artifacts > Deploy.

## 5. Screenshots for the report
- Jenkins dashboard with the job
- Job configuration (Pipeline from SCM, repo URL, branch, trigger)
- Stage View with all stages green
- Console output ending in `Finished: SUCCESS`
- Test Result page (pytest results, 10 tests)
- Build artifacts (`model.pkl`, `junit.xml`)
- GitHub webhook page showing a successful delivery (if used)
- One failed build (e.g. break a test on purpose, then fix it) to show defect handling for section 7.7

## Troubleshooting
- `python3: not found` or venv errors: you are not running the custom image; rebuild with `docker compose up -d --build`.
- pip fails with a Python version error: change the first line of `jenkins/Dockerfile` to a newer Jenkins LTS image and rebuild.
- Deploy stage shows UNSTABLE: the Streamlit app may be asleep or the URL changed; open it once and rebuild.
