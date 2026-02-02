# Amphion Project Testing Report

**Date:** 2026-02-02
**Tested By:** Claude Code (Automated Testing)
**Project:** Amphion Audio Generation Toolkit
**Environment:** Production (aphion.giggahost.com)

---

## Executive Summary

Comprehensive testing of the Amphion web application covering frontend, backend API, Gradio interfaces, security, and functionality. All critical functionality is working correctly. Several security enhancements have been identified and documented as improvement tasks.

**Overall Status:** ✅ **PASS** (with security recommendations)

---

## Testing Coverage

### 1. Frontend Testing (React/Next.js) ✅

#### TTS Interfaces
- **Main TTS Page** (`/tts`): ✅ Working
  - Model selector functional (4 models: MaskGCT, DualCodec-VALLE, Vevo TTS, Metis)
  - Generate/Batch/History tabs working
  - Browse All Models section added (links to 10 individual model pages)

- **Individual TTS Model Pages**: ✅ All 10 pages accessible
  - `/tts/maskgct` - MaskGCT interface ✅
  - `/tts/vits` - VITS interface ✅ (fixed 404 routing issue)
  - `/tts/fastspeech2` - FastSpeech 2 interface ✅
  - `/tts/jets` - JETS interface ✅
  - `/tts/naturalspeech2` - NaturalSpeech 2 interface ✅
  - `/tts/valle` - VALL-E interface ✅
  - `/tts/dualcodec-valle` - DualCodec-VALLE interface ✅
  - `/tts/vevo-tts` - Vevo TTS interface ✅
  - `/tts/metis` - Metis interface ✅
  - `/tts/debatts` - DebaTTS interface ✅

#### VC Interfaces
- **Main VC Page** (`/vc`): ✅ Working
  - Model selector functional
  - Proper navigation and layout

- **Individual VC Model Pages**: ✅ All 4 pages accessible
  - `/vc/noro` - Noro diffusion-based VC ✅
    - Source/Reference audio uploads
    - Diffusion parameters (Inference Steps: 200, Sigma: 1.20)
    - Convert button functional
  - `/vc/vevo-voice` - Full voice conversion ✅
    - Source/Reference audio uploads
    - Convert Voice button functional
  - `/vc/vevo-timbre` - Timbre-only conversion ✅
    - Source/Reference audio uploads
    - Convert Timbre button functional
  - `/vc/vevo-style` - Style/accent conversion ✅
    - Source/Style reference uploads
    - Convert Style button functional

#### Navigation & UX
- ✅ All navigation links working
- ✅ Back buttons functional
- ✅ Consistent UI/UX across pages
- ✅ Responsive design working
- ✅ Card hover effects functional
- ✅ File upload UI properly styled

---

### 2. Backend API Testing (FastAPI) ✅

#### Service Status
- **FastAPI Backend**: ✅ Active (Port 14555)
  - Service: `amphion-api.service`
  - PID: Running
  - Bind: 127.0.0.1 (localhost only) ✅ Secure

#### Endpoint Accessibility
- **Health Check**: ✅ `/api/health` - Working
- **OpenAPI Docs**: ✅ `/api/docs` - Accessible (publicly, see security notes)
- **ReDoc**: ✅ `/api/redoc` - Accessible

#### API Routes
- **TTS Routes** (`/api/tts/*`): ✅ Configured
  - MaskGCT, DualCodec-VALLE, Vevo TTS, Metis endpoints
- **VC Routes** (`/api/vc/*`): ✅ Configured
  - Noro, Vevo Voice/Timbre/Style endpoints
- **SVC Routes** (`/api/svc/*`): ✅ Configured
- **Evaluation Routes** (`/api/evaluation/*`): ✅ Configured

#### WebSocket Support
- **Progress Updates**: ✅ `/ws/progress/{task_id}` - Configured
- **Apache WebSocket Proxy**: ✅ Configured for `/api/ws/*`

---

### 3. Gradio Backend Integration ✅

#### MaskGCT Gradio Interface (`/gradio/`)
- **Service**: ✅ `aphion-gradio.service` active (PID 237944)
- **Port**: 14557 (proxied through Apache)
- **Title**: "MaskGCT TTS Demo"
- **Components**: ✅ All functional
  - Upload Prompt Wav (audio input)
  - Target Text (text input)
  - Target Duration slider
  - Number of Timesteps slider (15-100)
  - Generated Audio output
  - Clear/Submit buttons
- **API Endpoints**: ✅ `/inference`, `/Flag`, `/reset`
- **Version**: Gradio 6.4.0, SSE v3 protocol
- **Links**: arXiv paper, HuggingFace model & demo

#### Unified Gradio Interface (`/unified/`)
- **Service**: ✅ `amphion-web.service` active (PID 237918)
- **Port**: 14558 (proxied through Apache)
- **Title**: "Amphion TTS/VC Demo"
- **Tabs**: ✅ 4 model interfaces
  - MaskGCT TTS (zero-shot voice cloning)
  - DualCodec-VALLE (neural codec TTS)
  - Vevo TTS (style transfer)
  - Vevo VC (timbre/style control)
- **Welcome Message**: ✅ GitHub link, proper markdown rendering

---

### 4. Security Testing ⚠️

#### Security Strengths ✅
1. **Security Headers**: ✅ Properly configured
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: SAMEORIGIN
   - X-XSS-Protection: 1; mode=block
   - Referrer-Policy: strict-origin-when-cross-origin
   - Strict-Transport-Security: max-age=31536000 (HSTS)

2. **HTTPS/TLS**: ✅ Let's Encrypt certificates configured
3. **CORS**: ✅ Configured with specific allowed origins
4. **Service Isolation**: ✅ All services bind to localhost only
5. **Reverse Proxy**: ✅ All traffic through Apache
6. **File Cleanup**: ✅ Background tasks for temp file cleanup

#### Security Gaps ⚠️ (Improvement Tasks Created)

1. **❌ No Authentication/Authorization** (HIGH PRIORITY - Task AMP-cww)
   - All API endpoints publicly accessible
   - No API keys, tokens, or session management
   - Anyone can use compute-intensive models
   - **Recommendation**: Implement API key auth or OAuth2/JWT

2. **❌ No Rate Limiting** (HIGH PRIORITY - Task AMP-38e)
   - No protection against abuse or DoS
   - Could lead to resource exhaustion
   - **Recommendation**: Implement per-IP rate limiting (10 req/min)

3. **⚠️ API Documentation Publicly Accessible** (MEDIUM)
   - `/api/docs` and `/api/redoc` exposed publicly
   - Aids potential attackers in understanding system
   - **Recommendation**: Add basic auth or restrict access

4. **⚠️ No File Upload Size Limits** (MEDIUM - Task AMP-j0u)
   - No `LimitRequestBody` in Apache config
   - No validation in FastAPI routes
   - Could cause DoS with huge file uploads
   - **Recommendation**: Add 50MB limit

5. **⚠️ No Input Validation** (MEDIUM - Task AMP-j0u)
   - Text input has no length limits
   - Parameter values not validated (n_timesteps, temperature, etc.)
   - **Recommendation**: Add validation middleware

6. **⚠️ Limited File Type Validation** (LOW)
   - Relies only on MIME type checking
   - No content-based validation
   - **Recommendation**: Add magic number verification

#### Architecture Notes
- Application designed as public research demo
- Current security posture appropriate for research/demo tool
- **Not suitable for production deployment** without security enhancements

---

### 5. Infrastructure Testing ✅

#### System Services
- **Next.js Frontend**: ✅ `amphion-frontend.service` active
  - Port: 14556
  - Working directory: `/home/kp/repo2/Amphion/models/web/react`
  - Node.js: v24.11.1

- **FastAPI Backend**: ✅ `amphion-api.service` active
  - Port: 14555

- **MaskGCT Gradio**: ✅ `aphion-gradio.service` active
  - Port: 14557

- **Unified Web**: ✅ `amphion-web.service` active
  - Port: 14558

#### Apache Configuration
- **Reverse Proxy**: ✅ Properly configured for all services
- **WebSocket Support**: ✅ Configured for `/gradio/`, `/unified/`, `/api/ws/`
- **SSL/TLS**: ✅ Let's Encrypt certificates valid
- **Security Headers**: ✅ Applied to all responses
- **Directory Options**: ✅ `-Indexes` (directory listing disabled)

#### Port Configuration
All services use ports 14555-14558 (documented in `PORT_CONFIGURATION.md`):
- 14555: FastAPI Backend ✅
- 14556: Next.js Frontend ✅
- 14557: MaskGCT Gradio ✅
- 14558: Unified Web App ✅
- 443: Apache (HTTPS public) ✅

---

## Issues Found and Fixed

### Critical Issues (P0)
1. **✅ FIXED: /tts/vits 404 Error** (Issue AMP-0ie)
   - **Problem**: Individual TTS model pages inaccessible
   - **Root Cause**:
     - Next.js configured for static export instead of SSR
     - Systemd service pointed to wrong directory
     - No navigation links to individual model pages
   - **Fix**:
     - Removed `output: 'export'` from `next.config.js`
     - Updated systemd service WorkingDirectory
     - Added "Browse All Models" section with cards for all 10 TTS models
   - **Status**: Verified all routes return 200 OK

---

## Improvement Tasks Created

### Security Enhancements
1. **AMP-cww** (P1): Implement API authentication and authorization
2. **AMP-38e** (P1): Add rate limiting to prevent API abuse
3. **AMP-j0u** (P2): Add file upload size limits and input validation

---

## Test Metrics

- **Total Routes Tested**: 25+
- **Frontend Pages**: 16 (all working)
- **Backend Endpoints**: 5+ categories tested
- **Gradio Interfaces**: 2 (both working)
- **Issues Found**: 1 critical (fixed), 6 security improvements (documented)
- **Pass Rate**: 100% (all critical functionality working)

---

## Recommendations

### Immediate Actions (Before Production)
1. ✅ **COMPLETED**: Fix routing issues - All pages now accessible
2. ⚠️ **REQUIRED**: Implement authentication (Task AMP-cww)
3. ⚠️ **REQUIRED**: Add rate limiting (Task AMP-38e)
4. ⚠️ **RECOMMENDED**: Add file upload limits (Task AMP-j0u)

### Future Enhancements
1. Add comprehensive input validation
2. Implement API usage analytics
3. Add request logging and monitoring
4. Consider adding API versioning
5. Implement health check dashboard
6. Add automated testing suite

---

## Conclusion

The Amphion web application is **fully functional** with all critical features working correctly. The frontend, backend API, and Gradio interfaces are all properly integrated and accessible.

**Key Findings:**
- ✅ All user-facing functionality working
- ✅ No broken links or 404 errors (after fixes)
- ✅ All models accessible via UI
- ✅ Proper service configuration and isolation
- ⚠️ Security enhancements recommended before production use

**Security Status:** Appropriate for research demo/internal tool. Requires authentication, rate limiting, and input validation for production deployment.

**Next Steps:** Implement security enhancement tasks (AMP-cww, AMP-38e, AMP-j0u) before production release.

---

## Appendix: Testing Methodology

- **Approach**: Systematic testing following orchestration stack development methodology
- **Tools**: curl, systemd, Apache logs, browser testing
- **Coverage**: Frontend routes, API endpoints, service status, security headers
- **Documentation**: All findings tracked in beads task system
- **Verification**: Manual verification of all routes and interfaces

---

**Report Generated:** 2026-02-02
**Testing Epic:** AMP-s9e
**Status:** COMPLETE
