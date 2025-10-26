import pythoncom
pythoncom.CoInitialize() 
from pywinauto import Desktop, Application
import threading
import time
import json
import asyncio
import sounddevice as sd
import numpy as np
import websockets
from collections import deque
from queue import Queue, Empty
import dotenv

dotenv.load_dotenv()

# Optional: pocket sphinx for wakeword
from pocketsphinx import LiveSpeech, get_model_path

OPENAI_REALTIME_WS = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

OPENAI_API_KEY =  dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

def window_worker():
    # this thread initializes COM fresh
    windows = Desktop(backend="uia").windows()
    print(windows)

threading.Thread(target=window_worker).start()

class VoiceAssistant(threading.Thread):
    def __init__(self, controller, wakeword="hi emofi", sample_rate=16000):
        super().__init__(daemon=True)
        self.controller = controller
        self.running = True
        self.wakeword = wakeword.lower()
        self.sample_rate = sample_rate
        self.audio_queue = Queue()
        self.in_session = False

        # pocketsphinx LiveSpeech object for wake word detection (very simple)
        model_path = get_model_path()
        self.speech = LiveSpeech(
            lm=False,
            keyphrase=self.wakeword,
            kws_threshold=1e-20,
            hmm=model_path + '/en-us',
            dict=model_path + '/cmudict-en-us.dict'
        )

    def run(self):
        # Start a sounddevice input stream in a background thread (fallback continuous capture)
        threading.Thread(target=self._start_mic_capture, daemon=True).start()
        # Start local wake-word loop (pocketsphinx)
        for phrase in self.speech:
            if not self.running:
                break
            # pocketsphinx triggers when wakeword is heard
            print("[Voice] Wake word detected")
            # start realtime session (async)
            asyncio.run(self._handle_session())
            # after session returns, continue listening

    def _start_mic_capture(self):
        def callback(indata, frames, time_info, status):
            if status:
                print("[mic status]", status)
            # convert float32 to 16-bit PCM
            pcm16 = (indata * 32767).astype(np.int16)
            self.audio_queue.put(pcm16.tobytes())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', callback=callback):
            while self.running:
                time.sleep(0.1)

    async def _handle_session(self):
        # Pause vision agent work while in conversation
        self.controller.agent_mode = True
        self.in_session = True

        # Build websocket connection per realtime docs; include Authorization header
        headers = [("Authorization", f"Bearer {OPENAI_API_KEY}")]
        async with websockets.connect(OPENAI_REALTIME_WS, extra_headers=headers, max_size=None) as ws:
            # Optionally send a "session.update" event to set voice params or context
            await ws.send(json.dumps({"type":"session.update", "data":{"voice":"alloy"}}))

            # create a consumer of audio queue that sends to WS as binary frames/events
            send_task = asyncio.create_task(self._send_audio_loop(ws))
            recv_task = asyncio.create_task(self._recv_loop(ws))

            # Wait until either finishes (recv may tell us session ended)
            done, pending = await asyncio.wait([send_task, recv_task], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

        # restore controller state
        self.controller.agent_mode = False
        self.in_session = False

    async def _send_audio_loop(self, ws):
        """
        Sends binary audio frames to the Realtime API. The exact message format
        (JSON wrapper event names, or raw binary chunks) depends on the Realtime
        API contract — consult the docs. Here we show a common pattern: a JSON event that wraps base64 audio.
        See docs: streaming / realtime channels. :contentReference[oaicite:4]{index=4}
        """
        import base64
        while self.in_session:
            try:
                pcm = self.audio_queue.get(timeout=1)
            except Empty:
                continue
            # wrap binary into base64 inside an event per some sample implementations:
            b64 = base64.b64encode(pcm).decode("ascii")
            event = {"type":"input_audio_buffer.append", "audio": b64}
            await ws.send(json.dumps(event))

            # occasionally send "input_audio_buffer.commit" so model processes audio
            # (timing and event names follow sample code in docs)
            await ws.send(json.dumps({"type":"input_audio_buffer.commit"}))

    async def _recv_loop(self, ws):
        """
        Handles messages from Realtime API: text transcripts, partial responses, and audio chunks for TTS.
        Many examples deliver TTS audio as base64 blobs; others stream audio via WebRTC.
        Check OpenAI docs for your chosen transport and parse accordingly. :contentReference[oaicite:5]{index=5}
        """
        import base64, wave, io
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            typ = data.get("type")
            # Example event types: "transcript.part", "response.delta", "response.audio_chunk", "response.complete"
            if typ == "transcript.part":
                text = data["text"]
                print("[ASR partial]", text)
                # optional hotword fallback if using server-side detection
                if "hi emofi" in text.lower():
                    print("[ASR] confirmed wakeword")
            elif typ == "response.delta":
                # text assistant contributes
                print("[Assistant delta]", data.get("content", ""))
            elif typ == "response.audio_chunk":
                # get base64 audio chunk, decode and play (blocking write to sounddevice or winsound)
                audio_b64 = data["data"]  # depends on message shape
                audio_bytes = base64.b64decode(audio_b64)
                # naive WAV playback (assuming WAV container); adjust to format returned by API
                # Here we write to a temp wav and play using sounddevice or winsound
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                    import sounddevice as sd
                    frames = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(frames, dtype=np.int16)
                    sd.play(arr, wf.getframerate())
                    sd.wait()
            elif typ == "response.complete":
                print("[Assistant] finished")
                break
            elif typ == "session.end":
                break

    # ----- Helper to inspect Windows apps -----
    def list_open_windows(self):
        desktop = Desktop(backend="uia")
        windows = []
        for w in desktop.windows():
            try:
                title = w.window_text()
                if title:
                    windows.append({"title": title, "handle": w})
            except Exception:
                continue
        return windows

    def activate_window_by_name(self, name_substr):
        windows = self.list_open_windows()
        for w in windows:
            if name_substr.lower() in w["title"].lower():
                try:
                    w["handle"].set_focus()
                    return True
                except Exception as e:
                    print("activate failed", e)
        return False
    
 
