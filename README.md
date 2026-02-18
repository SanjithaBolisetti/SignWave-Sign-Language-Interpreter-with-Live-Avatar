# ResQSign : An Intelligent Sign Language System for Interaction and Emergency Assistance

SignWave is a React-based toolkit that translates spoken or typed text into Indian Sign Language (ISL) using animated 3D avatars. It also provides a sign-to-text flow powered by MediaPipe Hands and TensorFlow.js. The app is packaged as a Create React App project inside the `client` folder.

## Features
- Text-to-sign playback with animated avatar models (xbot/ybot glTF). 
- Pre-built ISL gestures for alphabets and common words. 
- Sign-to-text pipeline leveraging MediaPipe Hands + TensorFlow.js. 
- Browser-based UI with Bootstrap and React Router.

## Project Structure
- `client/` – React app (all source lives here)
  - `src/Animations/` – gesture clips for alphabets and words
  - `src/Models/` – 3D avatar models (glb/png)
  - `src/ML/` – model.json + weights (TensorFlow.js)
  - `src/Config/config.js` – API base URL
- `package.json` (root) – dependency mirror (no scripts)

## Prerequisites
- Node.js 16+ (tested with CRA 5 toolchain)
- npm 7+ (ships with Node)

## Setup & Run
1) Clone the repo
   ```bash
   git clone https://github.com/SanjithaBolisetti/SignWave-Sign-Language-Interpreter-with-Live-Avatar.git
   cd SignWave-Sign-Language-Interpreter-with-Live-Avatar
   ```

2) Install dependencies (app lives in `client`)
   ```bash
   npm install          # optional; root has no scripts but keeps lock in sync
   cd client
   npm install
   ```

3) Start the dev server
   ```bash
   npm start
   ```
   - Opens http://localhost:3000
   - Hot reload enabled

4) Build for production (optional)
   ```bash
   npm run build
   ```

## Configuration
- API base URL is defined in `client/src/Config/config.js` (currently `https://sign-kit-api.herokuapp.com/sign-kit`). Update if your backend differs.
- If you need environment variables (e.g., alt API endpoints), add a `.env` in `client/` (CRA uses `REACT_APP_*` vars) and restart the dev server.

## Notes
- `client/src/ML/weights.bin` must contain the TensorFlow.js weights. Ensure this file is present and non-empty; replace with your trained weights if required.
- `node_modules/` is not tracked—run installs as shown above.

## Scripts (from `client`)
- `npm start` – run dev server
- `npm run build` – production build
- `npm test` – CRA test runner
- `npm run eject` – eject CRA (irreversible)

## License
Not specified in the repository. Add one if you plan to distribute.
