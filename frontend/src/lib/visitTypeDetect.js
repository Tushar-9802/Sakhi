// Heuristic visit-type detection from a Hindi/Hinglish transcript.
// Port of app.py:642-677 — same ordering (delivery → ANC → PNC → child).
// Returns one of 'delivery' | 'anc_visit' | 'pnc_visit' | 'child_health'.
// Default = anc_visit (most common fallback).

const DELIVERY_KW = [
  'डिलीवरी हो गई', 'डिलीवरी हुई', 'delivery हुई',
  'डिलीवरी कब हुई', 'delivery कब',
  'जन्म हुआ', 'पैदा हुआ', 'प्रसव हुआ',
  'लड़का हुआ', 'लड़की हुई', 'लड़की हुआ',
  'घर पर ही हो गया', 'घर पर हुई', 'घर पर हुआ',
  'ऑपरेशन से हुई', 'caesarean', 'सिजेरियन',
  'जन्म का वजन', 'birth weight', 'birth_weight',
  'जन्म के समय', 'normal delivery', 'दाई ने',
]

const ANC_KW = [
  'गर्भ', 'प्रेग्नेंसी', 'pregnancy', 'anc', 'पेट में बच्चा',
  'गर्भवती', 'हफ्ते की', 'हफ्ते हो', 'महीने की',
  'lmp', 'edd', 'bp चेक', 'hb ', 'ifa', 'tt का टीका',
  'बच्चे की हलचल', 'fetal', 'डिलीवरी कहाँ', 'डिलीवरी के लिए',
  'जन्म के लिए तैयारी', 'birth preparedness',
]

const PNC_KW = [
  'नवजात', 'newborn', 'दूध पीना', 'दूध नहीं पीता', 'दूध पीता',
  'दूध पी रहा', 'दूध नहीं पी', 'दूध पिला',
  'नाभि', 'cord', 'नाल', 'स्तनपान',
  'breastfeed', 'imnci', 'hbnc', 'डिलीवरी के बाद',
  'डिलीवरी को', 'delivery को', 'pnc',
  'खून बहना', 'खून आ रहा', 'pad ', 'पैड ',
]

const CHILD_KW = [
  'बच्चे को', 'बच्चा कैसा', 'child', 'टीका', 'vaccine',
  'deworming', 'vitamin a', 'hbyc',
  'महीने का', 'महीने है', 'दस्त', 'diarrhea',
  'खाता है', 'खेलता है', 'आँखें धँसी',
  'सुस्त है', 'सुस्त हो', 'बहुत सुस्त',
]

function anyContains(haystack, needles) {
  for (const n of needles) {
    if (haystack.includes(n)) return true
  }
  return false
}

export function detectVisitType(transcript) {
  const t = (transcript || '').toLowerCase()
  if (anyContains(t, DELIVERY_KW)) return 'delivery'
  if (anyContains(t, ANC_KW)) return 'anc_visit'
  if (anyContains(t, PNC_KW)) return 'pnc_visit'
  if (anyContains(t, CHILD_KW)) return 'child_health'
  return 'anc_visit'
}
