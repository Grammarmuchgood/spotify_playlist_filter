import { useRef, useState } from 'react'

// The Web Speech API's SpeechRecognition has no official TypeScript
// definitions - unlike every other browser API used in this app so far,
// it's not part of the standard DOM spec (still marked experimental,
// and Firefox doesn't implement it at all). This is a minimal,
// hand-written shape covering only what's actually used below, not the
// full spec.
interface SpeechRecognitionResultLike {
  transcript: string
}
interface SpeechRecognitionEventLike extends Event {
  results: { [index: number]: { [index: number]: SpeechRecognitionResultLike } }
}
interface SpeechRecognitionLike extends EventTarget {
  lang: string
  interimResults: boolean
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: (() => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

declare global {
  interface Window {
    // Chrome/Edge/Safari all still ship this under the prefixed name;
    // unprefixed SpeechRecognition also exists in newer Chromium but
    // isn't universal yet - checking both is the standard pattern.
    SpeechRecognition?: new () => SpeechRecognitionLike
    webkitSpeechRecognition?: new () => SpeechRecognitionLike
  }
}

const SpeechRecognitionClass = window.SpeechRecognition ?? window.webkitSpeechRecognition

type Props = {
  onResult: (transcript: string) => void
}

function VoiceButton({ onResult }: Props) {
  const [listening, setListening] = useState(false)
  // A ref, not state - this holds the in-progress SpeechRecognition
  // instance so stopListening() can reach the same object startListening()
  // created, but changing which instance is "current" should never itself
  // cause a re-render (nothing on screen depends on the instance itself,
  // only on `listening`, which is already separate state).
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)

  if (!SpeechRecognitionClass) {
    return (
      <span className="flex-shrink-0 self-center text-xs text-neutral-500" title="Needs Chrome, Edge, or Safari">
        Voice input not supported in this browser
      </span>
    )
  }

  function startListening() {
    const recognition = new SpeechRecognitionClass!()
    recognition.lang = 'en-US'
    recognition.interimResults = false // only fire onresult once, with the final transcript
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      onResult(transcript)
    }
    recognition.onerror = () => setListening(false)
    recognition.onend = () => setListening(false)
    recognitionRef.current = recognition
    recognition.start() // triggers the browser's mic permission prompt on first use
    setListening(true)
  }

  function stopListening() {
    recognitionRef.current?.stop()
    setListening(false)
  }

  return (
    <button
      type="button"
      onClick={listening ? stopListening : startListening}
      title={listening ? 'Stop listening' : 'Search by voice'}
      className={`flex-shrink-0 rounded-full px-4 py-2 transition ${
        listening ? 'animate-pulse bg-red-500 text-white' : 'bg-white/10 text-white hover:bg-white/20'
      }`}
    >
      🎤
    </button>
  )
}

export default VoiceButton
