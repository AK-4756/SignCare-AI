import React, { useState, useRef, useEffect } from 'react';
import { Volume2, Settings, RotateCcw, Copy, Trash2, BookOpen, History, ArrowRight, Zap, Users } from 'lucide-react';

export default function SignCareAI() {
  const videoRef = useRef(null);
  const [currentWord, setCurrentWord] = useState('');
  const [currentSentence, setCurrentSentence] = useState('');
  const [suggestions, setSuggestions] = useState(['HELLO', 'HELP', 'HOPE']);
  const [confidence, setConfidence] = useState(0);
  const [isCamera, setIsCamera] = useState(false);
  const [history, setHistory] = useState(JSON.parse(localStorage.getItem('signcare_history') || '[]'));
  const [showHistory, setShowHistory] = useState(false);
  const [showGuide, setShowGuide] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [translation, setTranslation] = useState('');
  const [showTranslation, setShowTranslation] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const LANGUAGE_VOICES = {
    en: 'en-IN',
    hi: 'hi-IN',
    gu: 'gu-IN',
    mr: 'mr-IN',
    bn: 'bn-IN',
    ta: 'ta-IN',
    te: 'te-IN',
    kn: 'kn-IN',
    ml: 'ml-IN',
    pa: 'pa-IN',
    ur: 'ur-IN',
    or: 'or-IN',
    as: 'as-IN',
    sa: 'sa-IN',
    es: 'es-ES',
    fr: 'fr-FR'
  };

  const TRANSLATION_DICTIONARY = {
    "HELLO": { hi: "नमस्ते", gu: "નમસ્તે", mr: "नमस्कार", bn: "নমস্কার", ta: "வணக்கம்", te: "నమస్కారం", kn: "ನಮಸ್ಕಾರ", ml: "നമസ്കാരം", pa: "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", ur: "سلام", or: "ନମସ୍କାର", as: "নমস্কাৰ", sa: "नमस्ते", es: "Hola", fr: "Bonjour" },
    "HELP": { hi: "मदद", gu: "મદદ", mr: "मदत", bn: "সাহায্য", ta: "உதவி", te: "సహాయం", kn: "ಸಹಾಯ", ml: "സഹಾಯം", pa: "ਮਦਦ", ur: "مدد", or: "ସାହାଯ୍ୟ", as: "সহায়", sa: "सहायता", es: "Ayuda", fr: "Aide" },
    "THANK YOU": { hi: "धन्यवाद", gu: "આભાર", mr: "धन्यवाद", bn: "ধন্যবাদ", ta: "நன்றி", te: "ధన్యవాదాలు", kn: "ಧನ್ಯವಾದಗಳು", ml: "നന്ദി", pa: "ਧੰਨਵਾਦ", ur: "شکریہ", or: "ଧନ୍ୟବାଦ", as: "ধন্যবাদ", sa: "धन्यवादः", es: "Gracias", fr: "Merci" },
    "THANKS": { hi: "धन्यवाद", gu: "આભાર", mr: "धन्यवाद", bn: "ধন্যবাদ", ta: "நன்றி", te: "ధన్యవాదాలు", kn: "ಧನ್ಯವಾದಗಳು", ml: "നന്ദി", pa: "ਧੰਨਵਾਦ", ur: "شکریہ", or: "ଧନ୍ୟବାଦ", as: "ধন্যবাদ", sa: "धन्यवादः", es: "Gracias", fr: "Merci" },
    "PLEASE": { hi: "कृपया", gu: "કૃપા કરીને", mr: "कृपया", bn: "দয়া করে", ta: "தயவுசெய்து", te: "దయచేసి", kn: "ದಯವಿಟ್ಟು", ml: "ദയവായി", pa: "ਕਿਰਪਾ ਕਰਕੇ", ur: "براہ کرم", or: "ଦୟାକରି", as: "অনুগ্ৰহ কৰি", sa: "कृपया", es: "Por favor", fr: "S'il vous plaît" },
    "YES": { hi: "हाँ", gu: "હા", mr: "हो", bn: "হ্যাঁ", ta: "ஆம்", te: "అవును", kn: "ಹೌದು", ml: "അതെ", pa: "ਹਾਂ", ur: "ہاں", or: "ହଁ", as: "হয়", sa: "आम्", es: "Sí", fr: "Oui" },
    "NO": { hi: "नहीं", gu: "ના", mr: "नाही", bn: "না", ta: "இல்லை", te: "కాదు", kn: "ಇಲ್ಲ", ml: "ഇല്ല", pa: "ਨਹੀਂ", ur: "نہیں", or: "ନା", as: "নহয়", sa: "न", es: "No", fr: "Non" },
    "WATER": { hi: "पानी", gu: "પાણી", mr: "પાણી", bn: "জল", ta: "தண்ணீர்", te: "నీరు", kn: "ನೀರು", ml: "വെള്ളം", pa: "ਪਾਣੀ", ur: "پانی", or: "ପାଣି", as: "পানী", sa: "जलम्", es: "Agua", fr: "Eau" },
    "FOOD": { hi: "खाना", gu: "ખોરાક", mr: "અન્ન", bn: "খাবার", ta: "உணவு", te: "ఆహారం", kn: "ಆಹಾರ", ml: "ഭക്ഷണം", pa: "ਖਾਣਾ", ur: "کھانا", or: "ଖାଦ୍ୟ", as: "খাদ্য", sa: "ভোজনम्", es: "Comida", fr: "Nourriture" },
    "DOCTOR": { hi: "डॉक्टर", gu: "ડોક્ટર", mr: "डॉक्टर", bn: "ডাক্তার", ta: "மருத்துவர்", te: "వైద్యుడు", kn: "ವೈದ್ಯರು", ml: "ഡോക്ടർ", pa: "ਡਾਕਟਰ", ur: "ڈاکٹر", or: "ଡାକ୍ତର", as: "চিকিৎসক", sa: "चिकित्सकः", es: "Médico", fr: "Médecin" },
    "GOOD MORNING": { hi: "शुभ प्रभात", gu: "શુભ સવાર", mr: "शुभ सकाळ", bn: "সুপ্রভাত", ta: "காலை வணக்கம்", te: "శుభోదయం", kn: "ಶುಭೋದಯ", ml: "സുപ്രഭാതം", pa: "ਸ਼ੁਭ ਸਵੇਰ", ur: "صبح بخیر", or: "ଶୁଭ ସକାଳ", as: "সুপ্ৰভাত", sa: "सुप्रभातम्", es: "Buenos días", fr: "Bonjour" },
    "GOODBYE": { hi: "अलविदा", gu: "આવજો", mr: "निरोप", bn: "বিদায়", ta: "விடைபெறுகிறேன்", te: "వీడ్కోలు", kn: "ವಿದಾಯ", ml: "വിട", pa: "ਅਲਵਿਦਾ", ur: "الوداع", or: "ବିଦାୟ", as: "বিদায়", sa: "पुनर्मिलामः", es: "Adiós", fr: "Au revoir" },
    "LOVE": { hi: "प्यार", gu: "પ્રેમ", mr: "प्रेम", bn: "ভালোবাসা", ta: "அன்பு", te: "ಪ್ರೇಮ", kn: "ಪ್ರೀತಿ", ml: "സ്നേഹം", pa: "ਪਿਆਰ", ur: "محبت", or: "ଭଲପାଇବା", as: "মৰম", sa: "प्रेम", es: "Amor", fr: "Amour" },
    "FAMILY": { hi: "परिवार", gu: "પરિવાર", mr: "कुटुंब", bn: "পরিবার", ta: "குடும்பம்", te: "కుటుంబం", kn: "ಕುಟುಂಬ", ml: "കുടുംബം", pa: "ਪਰਿਵਾਰ", ur: "خاندان", or: "ପରିବାର", as: "পৰিয়াল", sa: "परिवारः", es: "Familia", fr: "Famille" },
    "FRIEND": { hi: "दोस्त", gu: "મિત્ર", mr: "मित्र", bn: "বন্ধু", ta: "நண்பன்", te: "స్నేహితుడు", kn: "ಸ್ನೇಹಿತ", ml: "സുഹൃത്ത്", pa: "ਦੋਸਤ", ur: "دوست", or: "ବନ୍ਧୁ", as: "বন্ধু", sa: "मित्रम्", es: "Amigo", fr: "Ami" },
    "FATHER": { hi: "पिता", gu: "પિતા", mr: "वडील", bn: "বাবা", ta: "அப்பா", te: "తండ్రి", kn: "ತಂದೆ", ml: "അച്ഛൻ", pa: "ਪਿਤਾ", ur: "والد", or: "ବାପା", as: "দেউতা", sa: "पिता", es: "Padre", fr: "Père" },
    "MOTHER": { hi: "माँ", gu: "માતા", mr: "आई", bn: "মা", ta: "அம்மா", te: "తల్లి", kn: "ತಾಯಿ", ml: "അമ്മ", pa: "ਮਾਂ", ur: "ماں", or: "ମା", as: "মা", sa: "माता", es: "Madre", fr: "Mère" },
    "EMERGENCY": { hi: "आपातकाल", gu: "કટોકટી", mr: "आणीबाणी", bn: "জরুরী", ta: "அவசரம்", te: "అత్యవసరం", kn: "ತುರ್ತು", ml: "അടിയന്തരാവസ്ഥ", pa: "ਸੰਕਟਕਾਲ", ur: "ہنگامی صورتحال", or: "ଜରୁରୀକାଳୀନ", as: "ଜৰୁৰীকালীন", sa: "आपत्कालः", es: "Emergencia", fr: "Urgence" },
    "HOSPITAL": { hi: "अस्पताल", gu: "હોસ્પિટલ", mr: "रुग्णालय", bn: "হাসপাতাল", ta: "மருத்துவமனை", te: "ఆసుపత్రి", kn: "ಆಸ್ಪತ್ರೆ", ml: "ആശുപത്രി", pa: "ਹਸਪਤਾਲ", ur: "ہسپتال", or: "ଡାକ୍ତରଖାନା", as: "চিকিৎসালয়", sa: "चिकित्सालयः", es: "Hospital", fr: "Hôpital" },
    "STOP": { hi: "रुको", gu: "રોકો", mr: "थांबा", bn: "থামুন", ta: "நிறுத்து", te: "ఆగు", kn: "ನಿಲ್ಲಿಸಿ", ml: "നിർത്തുക", pa: "ਰੁਕੋ", ur: "روکیں", or: "ଅଟକନ୍ତୁ", as: "ৰওক", sa: "તિષ્ઠ", es: "Alto", fr: "Arrêt" },
    "WELCOME": { hi: "स्वागत है", gu: "સ્વાગત છે", mr: "स्वागत आहे", bn: "স্বাগতম", ta: "வரவேற்கிறோம்", te: "స్వాగతం", kn: "ಸ್ವಾಗತ", ml: "സ്വാഗതം", pa: "ਸੁਆਗਤ ਹੈ", ur: "خوش آمدید", or: "ସ୍ୱାଗତ", as: "স্বাগতম", sa: "स्वागतम्", es: "Bienvenido", fr: "Bienvenue" }
  };

  // Common ASL dictionary for demonstrations
  const aslDictionary = {
    HELLO: 'Hello',
    HELP: 'Help',
    HOPE: 'Hope',
    THANK: 'Thank you',
    LOVE: 'Love',
    GOOD: 'Good',
    BAD: 'Bad',
    YES: 'Yes',
    NO: 'No',
    QUESTION: 'What?',
  };

  // Start camera
  useEffect(() => {
    if (isCamera) {
      navigator.mediaDevices
        .getUserMedia({ video: { facingMode: 'user' } })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        })
        .catch((err) => console.error('Camera error:', err));
    }
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isCamera]);

  // Simulate word detection from camera
  useEffect(() => {
    if (!isCamera) return;

    const interval = setInterval(() => {
      setConfidence(Math.floor(Math.random() * 40) + 60);
    }, 500);

    return () => clearInterval(interval);
  }, [isCamera]);

  // Update suggestions based on current word
  useEffect(() => {
    if (currentWord.length > 0) {
      const filtered = Object.keys(aslDictionary).filter((word) =>
        word.startsWith(currentWord.toUpperCase())
      );
      setSuggestions(filtered.length > 0 ? filtered : ['CONTINUE']);
    } else {
      setSuggestions(['HELLO', 'HELP', 'HOPE']);
    }
  }, [currentWord]);

  // Handle adding letter (simulated)
  const addLetter = (letter) => {
    setCurrentWord(currentWord + letter);
  };

  // Handle suggestion selection
  const selectSuggestion = (word) => {
    setCurrentWord(word);
  };

  // Add space - move word to sentence
  const addSpace = () => {
    if (currentWord) {
      setCurrentSentence(currentSentence ? `${currentSentence} ${currentWord}` : currentWord);
      setCurrentWord('');
      setSuggestions(['HELLO', 'HELP', 'HOPE']);
    }
  };

  // Clear current word
  const clearWord = () => {
    setCurrentWord('');
    setSuggestions(['HELLO', 'HELP', 'HOPE']);
  };

  // Clear sentence
  const clearSentence = () => {
    setCurrentSentence('');
  };

  // Text to speech
  const speak = async () => {
    if (!currentSentence) return;

    setIsSpeaking(true);

    try {
      const textToSpeak = translation || currentSentence;
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      const targetVoice = LANGUAGE_VOICES[selectedLanguage] || 'en-US';
      utterance.lang = targetVoice;
      utterance.rate = 1;
      utterance.pitch = 1;

      if (window.speechSynthesis) {
        const voices = window.speechSynthesis.getVoices();
        const voice = voices.find(v => v.lang === targetVoice || v.lang.startsWith(selectedLanguage));
        if (voice) {
          utterance.voice = voice;
        }
      }

      utterance.onend = () => setIsSpeaking(false);
      speechSynthesis.speak(utterance);
    } catch (err) {
      console.error('Speech error:', err);
      setIsSpeaking(false);
    }
  };

  // Translate
  const translateText = async () => {
    if (!currentSentence) return;
    const targetLang = selectedLanguage;
    if (targetLang === 'en') {
      setTranslation(currentSentence);
      setShowTranslation(true);
      return;
    }

    const upper = currentSentence.toUpperCase().trim();
    if (TRANSLATION_DICTIONARY[upper] && TRANSLATION_DICTIONARY[upper][targetLang]) {
      setTranslation(TRANSLATION_DICTIONARY[upper][targetLang]);
      setShowTranslation(true);
      return;
    }

    try {
      const res = await fetch(`https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=${targetLang}&dt=t&q=${encodeURIComponent(currentSentence)}`);
      if (res.ok) {
        const data = await res.json();
        if (data && data[0] && data[0][0]) {
          setTranslation(data[0].map(item => item[0]).join(''));
          setShowTranslation(true);
          return;
        }
      }
    } catch (err) {
      console.warn('Google translate error:', err);
    }

    setTranslation(currentSentence);
    setShowTranslation(true);
  };

  // Save to history
  const saveHistory = () => {
    if (currentSentence) {
      const newEntry = {
        id: Date.now(),
        asl: currentSentence,
        translation: translation || currentSentence,
        language: selectedLanguage,
        timestamp: new Date().toLocaleString(),
      };
      const updated = [newEntry, ...history];
      setHistory(updated);
      localStorage.setItem('signcare_history', JSON.stringify(updated));
      clearSentence();
      setShowTranslation(false);
    }
  };

  // Clear history
  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('signcare_history');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Navigation */}
      <nav className="border-b border-slate-700 bg-slate-900/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-lg flex items-center justify-center font-bold text-lg">
              🤟
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">SignCare AI</h1>
              <p className="text-xs text-slate-400">Real-time Sign Language Translator</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowGuide(true)}
              className="p-2 hover:bg-slate-700 rounded-lg transition"
              title="ASL Guide"
            >
              <BookOpen size={20} />
            </button>
            <button
              onClick={() => setShowHistory(true)}
              className="p-2 hover:bg-slate-700 rounded-lg transition relative"
              title="History"
            >
              <History size={20} />
              {history.length > 0 && (
                <span className="absolute top-1 right-1 w-2 h-2 bg-cyan-400 rounded-full"></span>
              )}
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Camera Section */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <Zap size={20} className="text-cyan-400" />
                  Hand Detection
                </h2>
                <button
                  onClick={() => setIsCamera(!isCamera)}
                  className={`px-4 py-2 rounded-lg font-medium transition ${
                    isCamera
                      ? 'bg-red-600 hover:bg-red-700'
                      : 'bg-cyan-600 hover:bg-cyan-700'
                  }`}
                >
                  {isCamera ? '⏹ Stop' : '▶ Start Camera'}
                </button>
              </div>

              {/* Video Feed */}
              <div className="relative bg-black rounded-xl overflow-hidden aspect-video mb-4">
                {isCamera ? (
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    muted
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center flex-col gap-4">
                    <div className="text-6xl">📷</div>
                    <p className="text-slate-400">Click "Start Camera" to begin</p>
                  </div>
                )}

                {/* Confidence Indicator */}
                {isCamera && (
                  <div className="absolute top-4 right-4 bg-black/60 backdrop-blur px-3 py-2 rounded-lg">
                    <div className="text-xs text-slate-300 mb-1">Confidence</div>
                    <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all"
                        style={{ width: `${confidence}%` }}
                      ></div>
                    </div>
                    <div className="text-xs text-cyan-400 mt-1">{confidence}%</div>
                  </div>
                )}
              </div>

              {/* Current Word */}
              <div className="bg-slate-700/50 rounded-xl p-4 mb-4">
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Current Word</div>
                <input
                  type="text"
                  value={currentWord}
                  onChange={(e) => setCurrentWord(e.target.value.toUpperCase())}
                  className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-2xl font-bold text-cyan-300 focus:outline-none focus:border-cyan-500"
                  placeholder="Type or sign a word..."
                />
              </div>

              {/* Word Suggestions */}
              <div className="mb-4">
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Suggestions</div>
                <div className="grid grid-cols-3 gap-2">
                  {suggestions.map((word, idx) => (
                    <button
                      key={idx}
                      onClick={() => selectSuggestion(word)}
                      className="bg-slate-700 hover:bg-slate-600 border border-slate-600 hover:border-cyan-500 rounded-lg py-2 px-3 transition font-medium text-sm"
                    >
                      {word}
                    </button>
                  ))}
                </div>
              </div>

              {/* Word Controls */}
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={addSpace}
                  className="bg-blue-600 hover:bg-blue-700 rounded-lg py-2 px-4 font-medium transition flex items-center justify-center gap-2"
                >
                  <ArrowRight size={18} />
                  Add Space
                </button>
                <button
                  onClick={clearWord}
                  className="bg-slate-700 hover:bg-slate-600 rounded-lg py-2 px-4 font-medium transition flex items-center justify-center gap-2"
                >
                  <Trash2 size={18} />
                  Clear Word
                </button>
              </div>
            </div>
          </div>

          {/* Sidebar - Sentence & Controls */}
          <div className="lg:col-span-1">
            {/* Current Sentence */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6 mb-4">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-3">
                Current Sentence
              </h3>
              <textarea
                value={currentSentence}
                onChange={(e) => setCurrentSentence(e.target.value.toUpperCase())}
                className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-3 py-3 text-base font-medium text-cyan-300 focus:outline-none focus:border-cyan-500 resize-none"
                rows="4"
                placeholder="Your sentence will appear here..."
              />
              <button
                onClick={clearSentence}
                className="w-full mt-3 bg-red-600/20 hover:bg-red-600/40 border border-red-600/50 rounded-lg py-2 text-sm font-medium transition"
              >
                Clear Sentence
              </button>
            </div>

            {/* Actions */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6 mb-4">
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-3">
                Actions
              </h3>

              {/* Language Selection */}
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm mb-3 focus:outline-none focus:border-cyan-500 text-slate-200"
              >
                <option value="en">English 🇺🇸 / 🇮🇳</option>
                <option value="hi">Hindi (हिन्दी) 🇮🇳</option>
                <option value="gu">Gujarati (ગુજરાતી) 🇮🇳</option>
                <option value="mr">Marathi (मराठी) 🇮🇳</option>
                <option value="bn">Bengali (বাংলা) 🇮🇳</option>
                <option value="ta">Tamil (தமிழ்) 🇮🇳</option>
                <option value="te">Telugu (తెలుగు) 🇮🇳</option>
                <option value="kn">Kannada (ಕನ್ನಡ) 🇮🇳</option>
                <option value="ml">Malayalam (മലയാളം) 🇮🇳</option>
                <option value="pa">Punjabi (ਪੰਜਾਬੀ) 🇮🇳</option>
                <option value="ur">Urdu (اردو) 🇮🇳</option>
                <option value="or">Odia (ଓଡ଼ିଆ) 🇮🇳</option>
                <option value="as">Assamese (অসমীয়া) 🇮🇳</option>
                <option value="sa">Sanskrit (संस्कृतम्) 🇮🇳</option>
                <option value="es">Spanish 🇪🇸</option>
                <option value="fr">French 🇫🇷</option>
              </select>

              <button
                onClick={translateText}
                disabled={!currentSentence}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg py-2 text-sm font-medium transition mb-2"
              >
                🌐 Translate
              </button>

              <button
                onClick={speak}
                disabled={!currentSentence || isSpeaking}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg py-2 text-sm font-medium transition flex items-center justify-center gap-2"
              >
                <Volume2 size={16} />
                {isSpeaking ? 'Speaking...' : 'Speak'}
              </button>

              {/* Translation Result */}
              {showTranslation && translation && (
                <div className="mt-3 bg-slate-700/50 rounded-lg p-3 border border-slate-600">
                  <div className="text-xs text-slate-400 mb-2">Translation</div>
                  <p className="text-sm font-medium text-cyan-300">{translation}</p>
                </div>
              )}

              <button
                onClick={saveHistory}
                disabled={!currentSentence}
                className="w-full mt-3 bg-cyan-600 hover:bg-cyan-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg py-2 text-sm font-medium transition"
              >
                💾 Save
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* History Modal */}
      {showHistory && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur flex items-center justify-center p-4 z-50">
          <div className="bg-slate-800 rounded-2xl max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="sticky top-0 bg-slate-900 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold">📖 History</h3>
              <button
                onClick={() => setShowHistory(false)}
                className="text-slate-400 hover:text-white text-2xl leading-none"
              >
                ✕
              </button>
            </div>

            {history.length === 0 ? (
              <div className="p-6 text-center text-slate-400">
                <p>No history yet. Save your first translation!</p>
              </div>
            ) : (
              <div className="divide-y divide-slate-700">
                {history.map((entry) => (
                  <div key={entry.id} className="p-4 hover:bg-slate-700/30 transition">
                    <div className="flex items-start justify-between mb-2">
                      <div className="text-xs text-slate-400">{entry.timestamp}</div>
                      <div className="text-xs bg-cyan-600/30 text-cyan-300 px-2 py-1 rounded">
                        {entry.language.toUpperCase()}
                      </div>
                    </div>
                    <p className="text-sm font-medium mb-2">{entry.asl}</p>
                    <p className="text-xs text-slate-400">{entry.translation}</p>
                  </div>
                ))}
              </div>
            )}

            {history.length > 0 && (
              <div className="border-t border-slate-700 p-4 bg-slate-900">
                <button
                  onClick={clearHistory}
                  className="w-full bg-red-600/20 hover:bg-red-600/40 border border-red-600/50 rounded-lg py-2 text-sm font-medium transition"
                >
                  Clear All History
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ASL Guide Modal */}
      {showGuide && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur flex items-center justify-center p-4 z-50">
          <div className="bg-slate-800 rounded-2xl max-w-2xl w-full max-h-96 overflow-y-auto">
            <div className="sticky top-0 bg-slate-900 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold">📚 ASL Alphabet Guide</h3>
              <button
                onClick={() => setShowGuide(false)}
                className="text-slate-400 hover:text-white text-2xl leading-none"
              >
                ✕
              </button>
            </div>

            <div className="p-6">
              <p className="text-slate-400 mb-4">
                Below is the American Sign Language (ASL) alphabet. Use this as a reference when learning to sign.
              </p>

              <div className="grid grid-cols-7 gap-3">
                {Object.keys(aslDictionary).slice(0, 26).map((letter, idx) => {
                  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
                  return (
                    <div key={idx} className="bg-slate-700/50 rounded-lg p-3 text-center">
                      <div className="text-2xl mb-2">🤟</div>
                      <div className="font-bold text-cyan-300">{letters[idx]}</div>
                    </div>
                  );
                })}
              </div>

              <p className="text-xs text-slate-400 mt-6 text-center">
                👉 Pro Tip: Position your hands clearly in front of the camera for best results
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-slate-700 bg-slate-900/80 mt-16 py-8">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            <div>
              <h4 className="font-semibold mb-3 flex items-center gap-2">
                <Zap size={18} className="text-cyan-400" />
                Real-Time Detection
              </h4>
              <p className="text-sm text-slate-400">
                Powered by AI to recognize your hand gestures instantly
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-3 flex items-center gap-2">
                <Users size={18} className="text-cyan-400" />
                For Everyone
              </h4>
              <p className="text-sm text-slate-400">
                Built for Deaf communities and ASL learners
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-3 flex items-center gap-2">
                <Volume2 size={18} className="text-cyan-400" />
                Multi-Language Support
              </h4>
              <p className="text-sm text-slate-400">
                Translate to English, Spanish, French, and more
              </p>
            </div>
          </div>

          <div className="border-t border-slate-700 pt-6 text-center text-sm text-slate-400">
            <p>
              SignCare AI © 2024 • Making communication accessible through AI-powered sign language translation
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
