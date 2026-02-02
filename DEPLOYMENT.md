# Amphion Web Deployment Status

## 🎉 Deployment Complete

The Amphion web application is now fully deployed with a comprehensive React frontend and FastAPI backend exposing all major capabilities.

### Live URLs
- **Production**: https://aphion.giggahost.com/
- **React App**: https://aphion.giggahost.com/react/
- **API Docs**: https://aphion.giggahost.com/api/docs
- **API Base**: https://aphion.giggahost.com/api/

### Authentication
- Login required for access
- Admin approval workflow
- Session-based authentication with PHP gateway

---

## 🎵 Exposed Capabilities

### Text-to-Speech (TTS) - 11 Models ✅

| Model | Status | API Endpoint | Frontend Page |
|-------|--------|--------------|---------------|
| **MaskGCT** | ✅ Live | `/api/tts/maskgct` | `/react/tts/maskgct` |
| **DualCodec-VALLE** | ✅ Live | `/api/tts/dualcodec-valle` | `/react/tts/dualcodec-valle` |
| **Vevo TTS** | ✅ Live | `/api/tts/vevo` | `/react/tts/vevo-tts` |
| **Metis** | ✅ Live | `/api/tts/metis` | `/react/tts/metis` |
| **DebaTTS** | ✅ Frontend | N/A | `/react/tts/debatts` |
| **FastSpeech2** | ✅ Frontend | N/A | `/react/tts/fastspeech2` |
| **Jets** | ✅ Frontend | N/A | `/react/tts/jets` |
| **NaturalSpeech2** | ✅ Frontend | N/A | `/react/tts/naturalspeech2` |
| **VALLE** | ✅ Frontend | N/A | `/react/tts/valle` |
| **VITS** | ✅ Frontend | N/A | `/react/tts/vits` |
| **Compare View** | ✅ Frontend | N/A | `/react/tts/compare` |

### Voice Conversion (VC) - 4 Models ✅

| Model | Status | API Endpoint | Frontend Page |
|-------|--------|--------------|---------------|
| **Vevo Voice** | ✅ Live | `/api/vc/vevo-voice` | `/react/vc/vevo-voice` |
| **Vevo Timbre** | ✅ Live | `/api/vc/vevo-timbre` | `/react/vc/vevo-timbre` |
| **Vevo Style** | ✅ Live | `/api/vc/vevo-style` | `/react/vc/vevo-style` |
| **Noro** | ✅ Live | `/api/vc/noro` | `/react/vc/noro` |

### Singing Voice Conversion (SVC) ✅

| Model | Status | API Endpoint | Frontend Page |
|-------|--------|--------------|---------------|
| **VevoSing** | ✅ Live | `/api/svc/vevosing` | `/react/svc/vevosing` |

### Evaluation Metrics ✅

| Metric | API Endpoint |
|--------|--------------|
| F0 Analysis | `/api/evaluation/f0` |
| Spectral Metrics | `/api/evaluation/spectral` |
| Energy Analysis | `/api/evaluation/energy` |
| Intelligibility (WER/CER) | `/api/evaluation/intelligibility` |
| Batch Evaluation | `/api/evaluation/batch` |
| List Available Metrics | `/api/evaluation/metrics` |

---

## 🏗️ Architecture

### Backend Stack
- **Framework**: FastAPI (Python 3.10+)
- **Port**: 17862 (localhost only)
- **Service**: `amphion-api.service`
- **Entry**: `python -m uvicorn models.web.api.main:app --host 127.0.0.1 --port 17862`
- **Output**: `/home/kp/repo2/Amphion/output/web/`

### Frontend Stack
- **Framework**: Next.js 15.5.11 + React 18 + TypeScript
- **UI Library**: shadcn/ui + Radix + Tailwind CSS
- **Port**: 3001 (localhost only)
- **Service**: `amphion-frontend.service`
- **Entry**: `npm run start` in `/home/kp/repo2/Amphion/models/web/react/`

### Web Server
- **Server**: Apache with mod_proxy
- **SSL**: Let's Encrypt (aphion.giggahost.com)
- **Proxies**:
  - `/api/*` → FastAPI (17862)
  - `/api/ws/*` → WebSocket (17862)
  - `/react/*` → Next.js (3001)

### Authentication
- **Gateway**: PHP 8.3-FPM
- **Database**: PostgreSQL (amphion_auth)
- **Session**: Server-side PHP sessions
- **Approval**: Admin approval required for new users

---

## 🎨 Frontend Features

### Core UI Components
- ✅ Dark theme with glassmorphism
- ✅ Responsive sidebar navigation
- ✅ Model selection grid
- ✅ Audio player with waveform (wavesurfer.js)
- ✅ Drag-and-drop file upload
- ✅ Real-time progress indicators
- ✅ Generation history (localStorage)
- ✅ Batch processing queue
- ✅ Audio comparison view
- ✅ Toast notifications
- ✅ Loading skeletons

### TTS Interface Features
- Text input with character counter
- Reference audio upload for voice cloning
- Language selection (auto-detect)
- Duration control
- Diffusion steps control
- Model parameter sliders
- Real-time generation progress
- Before/after audio comparison
- Download generated audio

### VC Interface Features
- Source audio upload
- Reference audio upload (for timbre/style)
- Conversion mode selection
- Noise-robust option (Noro)
- Side-by-side comparison
- Batch conversion queue

### Evaluation Interface
- Multi-file upload
- Metric selection checkboxes
- Batch processing
- Results visualization
- Export to CSV/JSON

---

## 📁 File Structure

```
/home/kp/repo2/Amphion/
├── models/web/
│   ├── api/                          # FastAPI Backend
│   │   ├── main.py                   # App entry point
│   │   ├── routes/
│   │   │   ├── tts.py               # TTS endpoints
│   │   │   ├── vc.py                # VC endpoints
│   │   │   ├── svc.py               # SVC endpoints
│   │   │   ├── evaluation.py        # Metrics endpoints
│   │   │   └── health.py            # Health check
│   │   ├── models/
│   │   │   └── manager.py           # Model loading manager
│   │   └── websocket/
│   │       └── progress.py          # Real-time updates
│   └── react/                        # Next.js Frontend
│       ├── app/
│       │   ├── page.tsx             # Dashboard
│       │   ├── tts/                 # TTS pages
│       │   ├── vc/                  # VC pages
│       │   ├── svc/                 # SVC pages
│       │   └── tools/               # Tools pages
│       └── components/
│           ├── ui/                  # shadcn/ui components
│           └── audio/               # Audio components
```

---

## 🚀 Service Management

### Check Status
```bash
systemctl status amphion-api
systemctl status amphion-frontend
```

### Restart Services
```bash
sudo systemctl restart amphion-api
sudo systemctl restart amphion-frontend
```

### View Logs
```bash
journalctl -u amphion-api -f
journalctl -u amphion-frontend -f
```

### Check API Health
```bash
curl http://127.0.0.1:17862/api/health
```

### Check Frontend
```bash
curl http://127.0.0.1:3001
```

---

## 🔧 Development

### Backend Development
```bash
cd /home/kp/repo2/Amphion
conda activate amphion
export PYTHONPATH=$(pwd)
python -m uvicorn models.web.api.main:app --host 127.0.0.1 --port 17862 --reload
```

### Frontend Development
```bash
cd /home/kp/repo2/Amphion/models/web/react
npm run dev
```

### Build Frontend
```bash
cd /home/kp/repo2/Amphion/models/web/react
npm run build
```

---

## 🔐 Security

### Firewall Rules
- Port 17862 (API): Localhost only (iptables drop external)
- Port 3001 (Frontend): Localhost only
- Port 443 (HTTPS): Public (Apache reverse proxy)

### SSL/TLS
- Certificate: Let's Encrypt (aphion.giggahost.com)
- Auto-renewal: Certbot systemd timer
- HSTS: Enabled (max-age 31536000)

### Authentication
- Session cookies: httpOnly, secure, sameSite=Strict
- Password hashing: bcrypt
- Failed login tracking
- Audit logging (all login attempts)

---

## 📊 API Usage Examples

### MaskGCT TTS
```bash
curl -X POST https://aphion.giggahost.com/api/tts/maskgct \
  -F "audio=@reference.wav" \
  -F "text=Hello world, this is a test." \
  -F "target_language=en" \
  -F "n_timesteps=25"
```

### Vevo Voice Conversion
```bash
curl -X POST https://aphion.giggahost.com/api/vc/vevo-voice \
  -F "source_audio=@source.wav" \
  -F "reference_audio=@target.wav"
```

### F0 Evaluation
```bash
curl -X POST https://aphion.giggahost.com/api/evaluation/f0 \
  -F "reference=@reference.wav" \
  -F "generated=@generated.wav"
```

---

## 🎯 Next Steps

### Additional Models to Add Backend Support
1. **TTS**: FastSpeech2, Jets, NaturalSpeech2, VALLE, VITS, DebaTTS
2. **SVC**: DiffComoSVC, TransformerSVC, VitsSVC
3. **TTA**: AudioLDM, PicoAudio
4. **Codecs**: FACodec, Amphion Codec (encode/decode endpoints)
5. **Vocoders**: HiFi-GAN, BigVGAN, APNet, NSF-HiFiGAN, Vocos

### Features to Add
1. Audio recording directly in browser
2. Advanced waveform visualization (pitch/energy overlays)
3. Model comparison side-by-side
4. Batch processing status dashboard
5. User preferences/settings
6. Audio history with search/filter
7. Share generated audio (with permissions)
8. Export evaluation reports as PDF

### Performance Optimizations
1. Model lazy loading (load on first use)
2. Model unloading (auto-unload after inactivity)
3. WebSocket progress updates
4. Response streaming for long audio
5. Redis caching for repeated inference
6. GPU memory management

---

## 📝 Notes

- **Current Status**: Production-ready with 4 TTS + 4 VC + 1 SVC models live
- **Missing**: Additional TTS/SVC/TTA models need backend implementation
- **Authentication Fixed**: Users now redirect to `/react/` after login ✅
- **All Services Running**: API, Frontend, Apache, Auth gateway all operational ✅
