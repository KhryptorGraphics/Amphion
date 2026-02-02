# Port Configuration

## Port Assignments (14555-15000 Range)

All Amphion services use ports in the 14555-15000 range, except Apache which uses standard HTTPS port 443.

### Active Services

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| FastAPI Backend | 14555 | HTTP | REST API for TTS/VC models |
| Next.js Frontend | 14556 | HTTP | React web interface |
| MaskGCT Gradio | 14557 | HTTP | Gradio demo for MaskGCT TTS |
| Unified Web App | 14558 | HTTP | Unified Gradio interface |
| Apache | 443 | HTTPS | Reverse proxy (public) |

### Configuration Files

1. **FastAPI Backend**
   - Code: `models/web/api/main.py` (line ~435: `port=14555`)
   - Service: `/etc/systemd/system/amphion-api.service`

2. **Next.js Frontend**
   - Code: `models/web/react/package.json` (start script)
   - Deployed: `/var/www/aphion/frontend/package.json`
   - Service: `/etc/systemd/system/amphion-frontend.service`

3. **MaskGCT Gradio**
   - Code: `models/tts/maskgct/gradio_demo.py` (line 435: `server_port=14557`)
   - Service: `/etc/systemd/system/aphion-gradio.service`

4. **Unified Web App**
   - Code: `models/web/amphion_unified.py` (line 778: `server_port=14558`)
   - Service: `/etc/systemd/system/amphion-web.service`

5. **Apache Reverse Proxy**
   - Config: `/etc/apache2/sites-enabled/aphion.giggahost.com-le-ssl.conf`
   - Maps public paths to internal ports:
     - `/api/*` → `127.0.0.1:14555`
     - `/_next/*` and `/` → `127.0.0.1:14556`
     - `/gradio/*` → `127.0.0.1:14557`
     - `/unified/*` → `127.0.0.1:14558`

### Management Commands

```bash
# Restart all services
sudo systemctl restart amphion-api.service amphion-frontend.service aphion-gradio.service amphion-web.service

# Check service status
systemctl status amphion-api.service amphion-frontend.service aphion-gradio.service amphion-web.service

# Reload Apache after config changes
sudo systemctl reload apache2

# View service logs
journalctl -u amphion-api.service -f
journalctl -u amphion-frontend.service -f
journalctl -u aphion-gradio.service -f
journalctl -u amphion-web.service -f
```

### Network Access

All services bind to `127.0.0.1` (localhost only) and are only accessible through the Apache reverse proxy at:

**Public URL**: https://aphion.giggahost.com/

This configuration ensures:
- No direct external access to internal services
- All traffic goes through Apache SSL/TLS
- Centralized access control and logging
- Security headers applied by Apache
