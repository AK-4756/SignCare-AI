/**
 * SignCare AI — Multilingual Internationalization (i18n) System
 * Provides complete UI localization for English, Hindi (हिन्दी), and Gujarati (ગુજરાતી)
 */

const translations = {
    en: {
        // App Identity & Header
        appTitle: "SignCare AI",
        appSubtitle: "Real-time Sign Language Translator",
        languageLabel: "Language",
        selectLanguage: "Select UI Language",
        aslGuideTitle: "ASL Alphabet Guide",
        historyTitle: "Translation History",

        // Hand Detection & Camera Card
        handDetectionTitle: "Hand Detection",
        startCamera: "▶ Start Camera",
        stopCamera: "⏹ Stop Camera",
        videoPlaceholderText: 'Click "Start Camera" to begin',
        confidenceLabel: "Confidence",
        cameraStarting: "Camera Starting...",
        cameraActive: "✓ Camera Active",
        cameraHandDetected: "✓ Hand Detected • {letter}",
        noValidHand: "⚠ No valid hand detected",
        aiConnectionError: "⚠ AI connection error",
        cameraStopping: "Stopping camera...",
        motionTrainingTitle: "Moving-letter training (J/Z)",
        motionTrainingHelp: "One click records one sample: perform one complete sign while 24 valid hand frames are recorded (about 2.4 seconds at the current capture rate). Collect at least 30 different samples for each label. OTHER should include static signs and unrelated movement.",
        motionTrainingLabel: "Sample label",
        motionOtherLabel: "OTHER (static / other movement)",
        motionJCount: "J samples",
        motionZCount: "Z samples",
        motionOtherCount: "OTHER samples",
        motionCountProgress: "{count} / {minimum}",
        motionCountComplete: "{count} / {minimum} complete",
        motionCaptureStart: "Capture 24-frame sample",
        motionCaptureCancel: "Cancel capture",
        motionCameraReady: "Camera ready. Choose a label and capture a motion sample.",
        motionCameraRequired: "Start the camera before capturing a sample.",
        motionCameraStopped: "Camera stopped; motion capture cancelled.",
        motionCaptureProgress: "Recording camera frames {count}/{total}. Perform the complete sign.",
        motionSampleSaving: "Saving motion sample...",
        motionSampleProcessing: "Checking hand tracking and saving sample...",
        motionSampleSaved: "Saved {label} sample #{count}; hand detected in {valid}/{total} frames.",
        motionSampleSaveError: "Could not save this sample: {reason}",
        motionCaptureCancelled: "Motion capture cancelled.",
        motionCaptureTimeout: "Timed out waiting for camera frames. Check the camera and try again.",
        motionModelReady: "Motion model ready. Captured samples: {counts}.",
        motionModelNotReady: "Motion model not trained. Captured samples: {counts}. After collecting at least 30 per label, run: python train_motion_model.py",
        motionStatusError: "Could not check the motion model status.",

        // AI Detection Result Card
        detectedLetterLabel: "Detected Letter",
        waitingForCamera: "Waiting for camera...",
        handDetected: "✓ Hand detected",
        landmarksInitial: "21 landmarks will appear when a valid hand is detected",
        landmarksDetected: "21 / 21 landmarks detected",

        // Current Word Section
        currentWordLabel: "Current Word",
        currentWordPlaceholder: "Type or sign a word...",
        suggestionsLabel: "Word Suggestions",
        noSuggestions: "No matching words found",
        addSpace: "➜ Add Space",
        clearWord: "🗑 Clear Word",

        // Current Sentence Section
        currentSentenceLabel: "Current Sentence",
        currentSentencePlaceholder: "Your sentence will appear here...",
        clearSentence: "🗑 Clear Sentence",

        // Actions & Sentence Translation
        actionsTitle: "Actions",
        targetLanguageLabel: "Target Language for Sentence Translation",
        translateBtn: "🌐 Translate",
        speakBtn: "🔊 Speak",
        saveHistoryBtn: "💾 Save to History",
        translatingTo: "Translating to {lang}...",
        pleaseWait: "⏳ Please wait...",
        translationHeader: "Translation ({lang})",
        copyBtn: "📋 Copy",
        copied: "✓ Copied!",

        // History Modal
        historyModalTitle: "📖 Translation History",
        noHistory: "No history yet. Save your first translation!",
        clearHistoryBtn: "Clear All History",
        clearHistoryConfirm: "Are you sure? This cannot be undone.",
        closeBtn: "Close",

        // ASL Guide Modal
        guideModalTitle: "📚 ASL Alphabet Guide",
        guideModalDesc: "Below is the American Sign Language (ASL) alphabet. Use this as a reference when learning to sign.",
        guideModalTip: "👉 Pro Tip: Position your hands clearly in front of the camera for best results",
        aslPosterAlt: "American Sign Language alphabet showing hand signs from A to Z",

        // Footer Section
        footerFeature1Title: "⚡ Real-Time Detection",
        footerFeature1Desc: "Powered by AI to recognize your hand gestures instantly",
        footerFeature2Title: "👥 For Everyone",
        footerFeature2Desc: "Built for Deaf communities and ASL learners",
        footerFeature3Title: "🌐 Multi-Language Support",
        footerFeature3Desc: "Translate to English, Hindi, Gujarati, and 14+ languages",
        footerCopyright: "SignCare AI © 2024 • Making communication accessible through AI-powered sign language translation",

        // Dialogs, Alerts, Notifications
        cameraPermissionDenied: "❌ Camera permission denied. Please allow camera access and try again.",
        cameraNotFound: "❌ No camera found.",
        cameraError: "❌ Camera error: ",
        pleaseEnterSentence: "Please enter or sign a sentence first.",
        speechNotSupported: "Speech synthesis is not supported on this browser.",
        textCopied: "Text copied to clipboard!"
    },

    hi: {
        // App Identity & Header
        appTitle: "SignCare AI",
        appSubtitle: "रीयल-टाइम सांकेतिक भाषा अनुवादक",
        languageLabel: "भाषा",
        selectLanguage: "भाषा चुनें",
        aslGuideTitle: "एएसएल वर्णमाला मार्गदर्शिका",
        historyTitle: "अनुवाद इतिहास",

        // Hand Detection & Camera Card
        handDetectionTitle: "हस्त पहचान",
        startCamera: "▶ कैमरा शुरू करें",
        stopCamera: "⏹ कैमरा बंद करें",
        videoPlaceholderText: 'शुरू करने के लिए "कैमरा शुरू करें" पर क्लिक करें',
        confidenceLabel: "विश्वास स्तर",
        cameraStarting: "कैमरा शुरू हो रहा है...",
        cameraActive: "✓ कैमरा सक्रिय है",
        cameraHandDetected: "✓ हाथ पहचाना गया • {letter}",
        noValidHand: "⚠ कोई मान्य हाथ नहीं पहचाना गया",
        aiConnectionError: "⚠ AI कनेक्शन त्रुटि",
        cameraStopping: "कैमरा बंद हो रहा है...",
        motionTrainingTitle: "गतिशील अक्षर प्रशिक्षण (J/Z)",
        motionTrainingHelp: "एक क्लिक पर एक नमूना रिकॉर्ड होगा: 24 मान्य हाथ फ़्रेम रिकॉर्ड होते समय एक पूरा संकेत करें (वर्तमान गति पर लगभग 2.4 सेकंड)। हर लेबल के कम से कम 30 अलग नमूने लें। OTHER में स्थिर संकेत और अन्य गतिविधियाँ शामिल करें।",
        motionTrainingLabel: "नमूने का लेबल",
        motionOtherLabel: "OTHER (स्थिर / अन्य गतिविधि)",
        motionJCount: "J नमूने",
        motionZCount: "Z नमूने",
        motionOtherCount: "OTHER नमूने",
        motionCountProgress: "{count} / {minimum}",
        motionCountComplete: "{count} / {minimum} पूरे",
        motionCaptureStart: "24-फ़्रेम नमूना लें",
        motionCaptureCancel: "कैप्चर रद्द करें",
        motionCameraReady: "कैमरा तैयार है। लेबल चुनें और गति नमूना लें।",
        motionCameraRequired: "नमूना लेने से पहले कैमरा शुरू करें।",
        motionCameraStopped: "कैमरा बंद है; गति कैप्चर रद्द कर दिया गया।",
        motionCaptureProgress: "कैमरा फ़्रेम {count}/{total} रिकॉर्ड हो रहे हैं। पूरा संकेत करें।",
        motionSampleSaving: "गति नमूना सहेजा जा रहा है...",
        motionSampleProcessing: "हाथ की ट्रैकिंग जाँची जा रही है और नमूना सहेजा जा रहा है...",
        motionSampleSaved: "{label} नमूना #{count} सहेजा गया; {valid}/{total} फ़्रेम में हाथ मिला।",
        motionSampleSaveError: "नमूना सहेजा नहीं जा सका: {reason}",
        motionCaptureCancelled: "गति कैप्चर रद्द किया गया।",
        motionCaptureTimeout: "कैमरा फ़्रेम की प्रतीक्षा का समय समाप्त हुआ। कैमरा जाँचकर फिर प्रयास करें।",
        motionModelReady: "गति मॉडल तैयार है। रिकॉर्ड नमूने: {counts}।",
        motionModelNotReady: "गति मॉडल प्रशिक्षित नहीं है। रिकॉर्ड नमूने: {counts}। हर लेबल के कम से कम 30 नमूने लेने के बाद python train_motion_model.py चलाएँ।",
        motionStatusError: "गति मॉडल की स्थिति जाँची नहीं जा सकी।",

        // AI Detection Result Card
        detectedLetterLabel: "पहचाना गया अक्षर",
        waitingForCamera: "कैमरे की प्रतीक्षा है...",
        handDetected: "✓ हाथ पहचाना गया",
        landmarksInitial: "मान्य हाथ पहचाने जाने पर 21 लैंडमार्क दिखाई देंगे",
        landmarksDetected: "21 / 21 लैंडमार्क पहचाने गए",

        // Current Word Section
        currentWordLabel: "वर्तमान शब्द",
        currentWordPlaceholder: "शब्द टाइप करें या संकेत करें...",
        suggestionsLabel: "शब्द सुझाव",
        noSuggestions: "कोई मेल खाता शब्द नहीं मिला",
        addSpace: "➜ स्पेस जोड़ें",
        clearWord: "🗑 शब्द साफ़ करें",

        // Current Sentence Section
        currentSentenceLabel: "वर्तमान वाक्य",
        currentSentencePlaceholder: "आपका वाक्य यहाँ दिखाई देगा...",
        clearSentence: "🗑 वाक्य साफ़ करें",

        // Actions & Sentence Translation
        actionsTitle: "कार्य",
        targetLanguageLabel: "वाक्य अनुवाद के लिए लक्षित भाषा",
        translateBtn: "🌐 अनुवाद करें",
        speakBtn: "🔊 बोलें",
        saveHistoryBtn: "💾 इतिहास में सहेजें",
        translatingTo: "{lang} में अनुवाद हो रहा है...",
        pleaseWait: "⏳ कृपया प्रतीक्षा करें...",
        translationHeader: "अनुवाद ({lang})",
        copyBtn: "📋 कॉपी करें",
        copied: "✓ कॉपी हो गया!",

        // History Modal
        historyModalTitle: "📖 अनुवाद इतिहास",
        noHistory: "अभी तक कोई इतिहास नहीं है। अपना पहला अनुवाद सहेजें!",
        clearHistoryBtn: "सारा इतिहास साफ़ करें",
        clearHistoryConfirm: "क्या आप निश्चित हैं? इसे पूर्ववत नहीं किया जा सकता।",
        closeBtn: "बंद करें",

        // ASL Guide Modal
        guideModalTitle: "📚 एएसएल वर्णमाला मार्गदर्शिका",
        guideModalDesc: "नीचे अमेरिकन सांकेतिक भाषा (ASL) वर्णमाला दी गई है। संकेत करना सीखते समय इसे संदर्भ के रूप में उपयोग करें।",
        guideModalTip: "👉 उपयोगी सुझाव: सर्वोत्तम परिणामों के लिए अपने हाथों को कैमरे के सामने स्पष्ट रूप से रखें",
        aslPosterAlt: "अमेरिकन सांकेतिक भाषा वर्णमाला जिसमें A से Z तक के हाथ के संकेत दिखाए गए हैं",

        // Footer Section
        footerFeature1Title: "⚡ रीयल-टाइम पहचान",
        footerFeature1Desc: "आपके हाथ के इशारों को तुरंत पहचानने के लिए AI द्वारा संचालित",
        footerFeature2Title: "👥 सभी के लिए",
        footerFeature2Desc: "बधिर समुदाय और एएसएल शिक्षार्थियों के लिए निर्मित",
        footerFeature3Title: "🌐 बहु-भाषा समर्थन",
        footerFeature3Desc: "अंग्रेज़ी, हिन्दी, गुजराती और 14+ भाषाओं में अनुवाद करें",
        footerCopyright: "SignCare AI © 2024 • AI-संचालित सांकेतिक भाषा अनुवाद के माध्यम से संचार को सुलभ बनाना",

        // Dialogs, Alerts, Notifications
        cameraPermissionDenied: "❌ कैमरा अनुमति अस्वीकृत। कृपया कैमरे की अनुमति दें और पुनः प्रयास करें।",
        cameraNotFound: "❌ कोई कैमरा नहीं मिला।",
        cameraError: "❌ कैमरा त्रुटि: ",
        pleaseEnterSentence: "कृपया पहले एक वाक्य दर्ज करें या संकेत करें।",
        speechNotSupported: "इस ब्राउज़र पर वाक् संश्लेषण (Speech Synthesis) समर्थित नहीं है।",
        textCopied: "टेक्स्ट क्लिपबोर्ड पर कॉपी किया गया!"
    },

    gu: {
        // App Identity & Header
        appTitle: "SignCare AI",
        appSubtitle: "રીઅલ-ટાઇમ સાંકેતિક ભાષા અનુવાદક",
        languageLabel: "ભાષા",
        selectLanguage: "ભાષા પસંદ કરો",
        aslGuideTitle: "ASL મૂળાક્ષર માર્ગદર્શિકા",
        historyTitle: "અનુવાદ ઇતિહાસ",

        // Hand Detection & Camera Card
        handDetectionTitle: "હાથની ઓળખ",
        startCamera: "▶ કૅમેરો શરૂ કરો",
        stopCamera: "⏹ કૅમેરો બંધ કરો",
        videoPlaceholderText: 'શરૂ કરવા માટે "કૅમેરો શરૂ કરો" પર ક્લિક કરો',
        confidenceLabel: "વિશ્વાસ સ્તર",
        cameraStarting: "કૅમેરો શરૂ થઈ રહ્યો છે...",
        cameraActive: "✓ કૅમેરો સક્રિય છે",
        cameraHandDetected: "✓ હાથ ઓળખાયો • {letter}",
        noValidHand: "⚠ કોઈ માન્ય હાથ ઓળખાયો નથી",
        aiConnectionError: "⚠ AI કનેક્શન ભૂલ",
        cameraStopping: "કૅમેરો બંધ થઈ રહ્યો છે...",
        motionTrainingTitle: "ગતિશીલ અક્ષર તાલીમ (J/Z)",
        motionTrainingHelp: "એક ક્લિકથી એક નમૂનો રેકોર્ડ થશે: 24 માન્ય હાથની ફ્રેમ રેકોર્ડ થાય ત્યારે એક આખો સંકેત કરો (હાલની ગતિએ લગભગ 2.4 સેકન્ડ). દરેક લેબલના ઓછામાં ઓછા 30 અલગ નમૂનાઓ લો. OTHERમાં સ્થિર સંકેતો અને બીજી ગતિઓ સામેલ કરો.",
        motionTrainingLabel: "નમૂનાનું લેબલ",
        motionOtherLabel: "OTHER (સ્થિર / બીજી ગતિ)",
        motionJCount: "J નમૂનાઓ",
        motionZCount: "Z નમૂનાઓ",
        motionOtherCount: "OTHER નમૂનાઓ",
        motionCountProgress: "{count} / {minimum}",
        motionCountComplete: "{count} / {minimum} પૂર્ણ",
        motionCaptureStart: "24-ફ્રેમ નમૂનો લો",
        motionCaptureCancel: "કૅપ્ચર રદ કરો",
        motionCameraReady: "કૅમેરો તૈયાર છે. લેબલ પસંદ કરો અને ગતિનો નમૂનો લો.",
        motionCameraRequired: "નમૂનો લેતાં પહેલાં કૅમેરો શરૂ કરો.",
        motionCameraStopped: "કૅમેરો બંધ છે; ગતિ કૅપ્ચર રદ થયું.",
        motionCaptureProgress: "કૅમેરા ફ્રેમ {count}/{total} રેકોર્ડ થાય છે. આખો સંકેત કરો.",
        motionSampleSaving: "ગતિનો નમૂનો સાચવાઈ રહ્યો છે...",
        motionSampleProcessing: "હાથનું ટ્રૅકિંગ તપાસીને નમૂનો સાચવાઈ રહ્યો છે...",
        motionSampleSaved: "{label} નમૂનો #{count} સાચવાયો; {valid}/{total} ફ્રેમમાં હાથ મળ્યો.",
        motionSampleSaveError: "નમૂનો સાચવી શકાયો નથી: {reason}",
        motionCaptureCancelled: "ગતિ કૅપ્ચર રદ થયું.",
        motionCaptureTimeout: "કૅમેરા ફ્રેમની રાહ જોવાનો સમય પૂરો થયો. કૅમેરો તપાસીને ફરી પ્રયાસ કરો.",
        motionModelReady: "ગતિ મોડેલ તૈયાર છે. લીધેલા નમૂનાઓ: {counts}.",
        motionModelNotReady: "ગતિ મોડેલ તાલીમ પામ્યું નથી. લીધેલા નમૂનાઓ: {counts}. દરેક લેબલના ઓછામાં ઓછા 30 નમૂનાઓ પછી python train_motion_model.py ચલાવો.",
        motionStatusError: "ગતિ મોડેલની સ્થિતિ તપાસી શકાઈ નથી.",

        // AI Detection Result Card
        detectedLetterLabel: "ઓળખાયેલ અક્ષર",
        waitingForCamera: "કૅમેરાની રાહ જોઈ રહ્યા છીએ...",
        handDetected: "✓ હાથ ઓળખાયો",
        landmarksInitial: "માન્ય હાથ ઓળખાવા પર 21 લૅન્ડમાર્ક દેખાશે",
        landmarksDetected: "21 / 21 લૅન્ડમાર્ક ઓળખાયા",

        // Current Word Section
        currentWordLabel: "વર્તમાન શબ્દ",
        currentWordPlaceholder: "શબ્દ લખો અથવા સંકેત કરો...",
        suggestionsLabel: "શબ્દ સૂચનો",
        noSuggestions: "કોઈ મેળ ખાતો શબ્દ મળ્યો નથી",
        addSpace: "➜ સ્પેસ ઉમેરો",
        clearWord: "🗑 શબ્દ સાફ કરો",

        // Current Sentence Section
        currentSentenceLabel: "વર્તમાન વાક્ય",
        currentSentencePlaceholder: "તમારું વાક્ય અહીં દેખાશે...",
        clearSentence: "🗑 વાક્ય સાફ કરો",

        // Actions & Sentence Translation
        actionsTitle: "ક્રિયાઓ",
        targetLanguageLabel: "વાક્ય અનુવાદ માટે લક્ષિત ભાષા",
        translateBtn: "🌐 અનુવાદ કરો",
        speakBtn: "🔊 બોલો",
        saveHistoryBtn: "💾 ઇતિહાસમાં સાચવો",
        translatingTo: "{lang}માં અનુવાદ થઈ રહ્યો છે...",
        pleaseWait: "⏳ કૃપા કરીને રાહ જુઓ...",
        translationHeader: "અનુવાદ ({lang})",
        copyBtn: "📋 કૉપિ કરો",
        copied: "✓ કૉપિ થઈ ગયું!",

        // History Modal
        historyModalTitle: "📖 અનુવાદ ઇતિહાસ",
        noHistory: "હજી સુધી કોઈ ઇતિહાસ નથી. તમારો પહેલો અનુવાદ સાચવો!",
        clearHistoryBtn: "બધો ઇતિહાસ સાફ કરો",
        clearHistoryConfirm: "શું તમે ખરેખર ખાતરી કરો છો? આ ક્રિયા પૂર્વવત્ કરી શકાતી નથી.",
        closeBtn: "બંધ કરો",

        // ASL Guide Modal
        guideModalTitle: "📚 ASL મૂળાક્ષર માર્ગદર્શિકા",
        guideModalDesc: "નીચે અમેરિકન સાંકેતિક ભાષા (ASL) મૂળાક્ષરો આપેલા છે. સાઇન કરવાનું શીખતી વખતે આનો સંદર્ભ તરીકે ઉપયોગ કરો.",
        guideModalTip: "👉 ખાસ સલાહ: શ્રેષ્ઠ પરિણામો માટે તમારા હાથને કૅમેરાની સામે સ્પષ્ટપણે રાખો",
        aslPosterAlt: "અમેરિકન સાંકેતિક ભાષાના મૂળાક્ષરો A થી Z હાથના સંકેતો દર્શાવે છે",

        // Footer Section
        footerFeature1Title: "⚡ રીઅલ-ટાઇમ ઓળખ",
        footerFeature1Desc: "તમારા હાથના હાવભાવને તરત જ ઓળખવા માટે AI દ્વારા સંચાલિત",
        footerFeature2Title: "👥 દરેક માટે",
        footerFeature2Desc: "બહેરા સમુદાયો અને ASL શીખનારાઓ માટે બનાવેલ",
        footerFeature3Title: "🌐 બહુભાષી સપોર્ટ",
        footerFeature3Desc: "અંગ્રેજી, હિન્દી, ગુજરાતી અને 14+ ભાષાઓમાં અનુવાદ કરો",
        footerCopyright: "SignCare AI © 2024 • AI-સંચાલિત સાંકેતિક ભાષા અનુવાદ દ્વારા સંચારને સુલભ બનાવવું",

        // Dialogs, Alerts, Notifications
        cameraPermissionDenied: "❌ કૅમેરાની પરવાનગી નકારી દેવામાં આવી છે. કૃપા કરીને કૅમેરા ઍક્સેસ આપો અને ફરી પ્રયાસ કરો.",
        cameraNotFound: "❌ કોઈ કૅમેરો મળ્યો નથી.",
        cameraError: "❌ કૅમેરા ભૂલ: ",
        pleaseEnterSentence: "કૃપા કરીને પહેલાં વાક્ય લખો અથવા સંકેત કરો.",
        speechNotSupported: "આ બ્રાઉઝર પર સ્પીચ સિન્થેસિસ સમર્થિત નથી.",
        textCopied: "ટેક્સ્ટ ક્લિપબોર્ડ પર કૉપિ કરવામાં આવી!"
    }
};

// Supported UI Languages
const SUPPORTED_UI_LANGUAGES = ['en', 'hi', 'gu'];
const STORAGE_KEY = 'signcare_ui_language';

let currentUiLanguage = 'en';

/**
 * Get translated text by key with optional interpolation params
 * @param {string} key - Translation key
 * @param {Object} params - Template parameters (e.g. { letter: 'A' })
 * @returns {string} Translated string
 */
function t(key, params = {}) {
    const lang = currentUiLanguage || 'en';
    let text = (translations[lang] && translations[lang][key]) ||
               (translations['en'] && translations['en'][key]) ||
               key;

    if (params && typeof params === 'object') {
        for (const [k, v] of Object.entries(params)) {
            text = text.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
        }
    }
    return text;
}

/**
 * Retrieve saved UI language preference or default to English
 * @returns {string} 'en', 'hi', or 'gu'
 */
function getSavedLanguage() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved && SUPPORTED_UI_LANGUAGES.includes(saved)) {
            return saved;
        }
    } catch (e) {
        console.warn('Could not read UI language from localStorage:', e);
    }
    return 'en';
}

/**
 * Apply selected language across the entire application frontend
 * @param {string} langCode - 'en', 'hi', or 'gu'
 * @param {boolean} persist - Whether to persist to localStorage
 */
function setLanguage(langCode, persist = true) {
    if (!SUPPORTED_UI_LANGUAGES.includes(langCode)) {
        langCode = 'en';
    }

    currentUiLanguage = langCode;

    if (persist) {
        try {
            localStorage.setItem(STORAGE_KEY, langCode);
        } catch (e) {
            console.warn('Could not save UI language to localStorage:', e);
        }
    }

    // Update document language attribute
    document.documentElement.lang = langCode;

    // Update UI language dropdown if not already matched
    const uiSelector = document.getElementById('uiLanguageSelect');
    if (uiSelector && uiSelector.value !== langCode) {
        uiSelector.value = langCode;
    }

    // 1. Text elements with data-i18n
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');

        // Special handling for dynamic camera button
        if (el.id === 'cameraBtn') {
            const isCamActive = typeof cameraActive !== 'undefined' && cameraActive;
            el.textContent = isCamActive ? t('stopCamera') : t('startCamera');
            return;
        }

        if (el.id === 'motionCaptureBtn') {
            const isCapturing =
                typeof motionCaptureInProgress !== 'undefined' &&
                motionCaptureInProgress &&
                motionCaptureImages !== null;
            el.textContent = t(isCapturing ? 'motionCaptureCancel' : 'motionCaptureStart');
            return;
        }

        // Special handling for camera status text
        if (el.id === 'statusText') {
            const isCamActive = typeof cameraActive !== 'undefined' && cameraActive;
            if (!isCamActive) {
                el.textContent = t('cameraStarting');
            } else if (typeof stableLetter !== 'undefined' && stableLetter) {
                el.textContent = t('cameraHandDetected', { letter: stableLetter });
            } else {
                const handStatusEl = document.getElementById('handStatus');
                if (handStatusEl && handStatusEl.textContent.includes('✓')) {
                    el.textContent = t('cameraActive');
                } else {
                    el.textContent = t('noValidHand');
                }
            }
            return;
        }

        // Special handling for handStatus element
        if (el.id === 'handStatus') {
            const isCamActive = typeof cameraActive !== 'undefined' && cameraActive;
            const detectedLetterEl = document.getElementById('detectedLetter');
            if (!isCamActive) {
                el.textContent = t('waitingForCamera');
            } else if (detectedLetterEl && detectedLetterEl.textContent !== '—') {
                el.textContent = t('handDetected');
            } else {
                el.textContent = t('noValidHand');
            }
            return;
        }

        // Special handling for landmarkCount element
        if (el.id === 'landmarkCount') {
            const detectedLetterEl = document.getElementById('detectedLetter');
            if (detectedLetterEl && detectedLetterEl.textContent !== '—') {
                el.textContent = t('landmarksDetected');
            } else {
                el.textContent = t('landmarksInitial');
            }
            return;
        }

        const translation = t(key);
        if (translation) {
            el.textContent = translation;
        }
    });

    // 2. Placeholder attributes with data-i18n-placeholder
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        const translation = t(key);
        if (translation) {
            el.placeholder = translation;
        }
    });

    // 3. Tooltips and titles with data-i18n-title
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        const translation = t(key);
        if (translation) {
            el.title = translation;
        }
    });

    // 4. Accessible ARIA labels with data-i18n-aria
    document.querySelectorAll('[data-i18n-aria]').forEach(el => {
        const key = el.getAttribute('data-i18n-aria');
        const translation = t(key);
        if (translation) {
            el.setAttribute('aria-label', translation);
        }
    });

    // 5. Image Alt text with data-i18n-alt
    document.querySelectorAll('[data-i18n-alt]').forEach(el => {
        const key = el.getAttribute('data-i18n-alt');
        const translation = t(key);
        if (translation) {
            el.alt = translation;
        }
    });

    // 6. Update empty states if present
    const suggestionsGrid = document.getElementById('suggestionsGrid');
    if (suggestionsGrid) {
        const noSuggEl = suggestionsGrid.querySelector('.no-suggestions');
        if (noSuggEl) {
            noSuggEl.textContent = t('noSuggestions');
        }
    }

    const historyList = document.getElementById('historyList');
    if (historyList) {
        const emptyState = historyList.querySelector('.empty-state');
        if (emptyState) {
            emptyState.textContent = t('noHistory');
        }
    }

    // Dispatch an event in case other modules need to know about UI language change
    if (typeof CustomEvent !== 'undefined' && typeof window.dispatchEvent === 'function') {
        window.dispatchEvent(new CustomEvent('signcare:languageChanged', {
            detail: { language: langCode }
        }));
    }

    // Refresh word suggestions in the new language immediately if function is available
    if (typeof window.updateSuggestions === 'function') {
        try {
            window.updateSuggestions();
        } catch (e) {
            console.warn('Could not auto-refresh suggestions:', e);
        }
    } else if (typeof updateSuggestions === 'function') {
        try {
            updateSuggestions();
        } catch (e) {
            console.warn('Could not auto-refresh suggestions:', e);
        }
    }
}

// Export to window
window.translations = translations;
window.t = t;
window.setLanguage = setLanguage;
window.getSavedLanguage = getSavedLanguage;
window.SUPPORTED_UI_LANGUAGES = SUPPORTED_UI_LANGUAGES;
