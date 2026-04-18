# Hindi ASHA Role-Play Scripts — Week 1 Real-Voice Recording

**Purpose:** 4 scripts for real-voice ASHA visit recordings. One person (you) plays ASHA, helper plays patient/caregiver. Record on a real phone (not laptop mic). Noisy room, not a studio. Natural Hindi/Hinglish with interruptions, background noise, incomplete sentences.

**Output target:** `data/real_audio/<case>.wav` + `data/real_audio/<case>.expected.json` (for reproducibility).

**Recording tips:**
- Phone mic, 2–3 feet away — mimic real visit conditions
- Keep kitchen / fan / traffic sounds in the background
- Don't read word-for-word — glance at the script, then speak naturally
- 2–4 minutes per visit is realistic
- Don't restart on small mistakes — ASHA conversations aren't clean

---

## 1. ANC Normal — Routine Antenatal Check (no danger signs)

**Scenario:** ASHA Priya visits Sunita (28 years old, second pregnancy, 6 months / 24 weeks). Routine check. Everything normal.

**Expected extraction:** ANC form populated (gestation 24 weeks, BP normal, weight, IFA compliance, TT doses). Danger signs: **none**. Referral: **none**.

**Script outline:**

ASHA: नमस्ते सुनीता जी, कैसी हैं आप? आज छठा महीना चल रहा है ना?
Sunita: हाँ दीदी, सब ठीक है। बच्चा हिल रहा है अच्छे से।
ASHA: चलो BP देख लेते हैं पहले। (pause) एक सौ बीस बटा अस्सी, बिल्कुल ठीक है। वज़न कितना है अभी?
Sunita: पिछले हफ्ते तौला था — छप्पन किलो।
ASHA: अच्छा, दो किलो बढ़ा है, सही है। IFA की गोली रोज़ ले रही हो?
Sunita: हाँ रोज़ रात को खाने के बाद। कभी-कभी भूल जाती हूँ पर ज़्यादातर दिन लेती हूँ।
ASHA: कोशिश करो रोज़ लो, बच्चे के लिए ज़रूरी है। TT का दूसरा टीका लगवा लिया?
Sunita: हाँ पिछले महीने लगवाया था PHC में।
ASHA: बहुत बढ़िया। कोई तकलीफ़? सिरदर्द, चक्कर, पेट में दर्द — कुछ भी?
Sunita: नहीं दीदी, सब ठीक है। बस थोड़ी कमज़ोरी लगती है कभी-कभी।
ASHA: ये नॉर्मल है, खाना अच्छे से खाओ — दूध, दाल, हरी सब्ज़ी। पानी ज़्यादा पियो। अगले महीने फिर आऊँगी।

---

## 2. ANC Preeclampsia — Danger Case (must trigger referral)

**Scenario:** ASHA Priya visits Rekha (32 years old, first pregnancy, 32 weeks). Rekha complains of headache and blurred vision. BP reads **160/110**. This is a **preeclampsia danger sign** — must trigger urgent referral.

**Expected extraction:** ANC form with BP 160/110, gestation 32 weeks. Danger signs: **severe headache, blurred vision, elevated BP**. Referral: **urgent, within 24 hours, to CHC/district hospital**.

**Script outline:**

ASHA: नमस्ते रेखा जी। कैसी तबीयत है?
Rekha: दीदी, दो-तीन दिन से सिर बहुत दर्द कर रहा है। दवा से भी ठीक नहीं हो रहा।
ASHA: कहाँ दर्द होता है? पूरे सिर में या एक तरफ़?
Rekha: पूरे सिर में, माथे पे ज़्यादा। और कभी-कभी आँखों के सामने धुंधला हो जाता है।
ASHA: धुंधला? जैसे कि दिखाई कम देता है?
Rekha: हाँ दीदी, अभी-अभी भी थोड़ा ऐसा लगा। और पैर भी सूज रहे हैं।
ASHA: (concerned) रुको, BP चेक करती हूँ पहले। (pause) अरे... एक सौ साठ बटा एक सौ दस। ये बहुत हाई है रेखा।
Rekha: क्या हुआ दीदी?
ASHA: सुनो, ये ठीक नहीं है। तुम्हें और बच्चे को ख़तरा हो सकता है। अभी हमें तुरंत CHC जाना होगा, डॉक्टर को दिखाना होगा।
Rekha: अभी? पर घर पर कोई नहीं है।
ASHA: मैं साथ चलती हूँ। देर मत करो — ये preeclampsia का लक्षण है, बच्चे के लिए भी ख़तरा है। अभी चलते हैं।

---

## 3. PNC Day 7 — Normal Postnatal Check

**Scenario:** ASHA Priya visits Kavita (26 years old, delivered 7 days ago, normal vaginal delivery, baby girl 2.8 kg at birth). Routine PNC check. Everything normal.

**Expected extraction:** PNC form (day 7, mother vitals normal, baby feeding well, weight gain tracking, cord healed, no fever). Danger signs: **none**. Referral: **none**.

**Script outline:**

ASHA: कविता, कैसी हो? बच्ची कैसी है?
Kavita: दीदी सब ठीक है। दूध अच्छा पी रही है।
ASHA: कितनी बार फ़ीड करती हो दिन में?
Kavita: हर दो घंटे में — आठ-दस बार दिन में।
ASHA: बहुत अच्छा। तुम्हारा BP देख लूँ। (pause) एक सौ दस बटा सत्तर। बढ़िया। बुख़ार-वुख़ार तो नहीं है?
Kavita: नहीं दीदी।
ASHA: टाँके का दर्द?
Kavita: पहले था, अब कम है। थोड़ा खिंचता है बैठने में।
ASHA: ये नॉर्मल है। पानी से साफ़ रखो वहाँ। बच्ची का नाभि कैसी है? सूखी है?
Kavita: हाँ अब सूख गई है, दो दिन पहले गिर गई थी।
ASHA: अच्छा। वज़न कर लिया था बच्ची का?
Kavita: हाँ कल ANM दीदी आई थीं — तीन किलो हो गया है।
ASHA: सही है, दो सौ ग्राम बढ़ा है हफ्ते में — बहुत अच्छा। IFA और कैल्शियम ले रही हो अपनी?
Kavita: हाँ दोनों ले रही हूँ।
ASHA: बढ़िया। कोई दिक़्क़त लगे तो तुरंत बताओ।

---

## 4. Child Health — Diarrhea with Dehydration (danger case)

**Scenario:** ASHA Priya visits Sonam's home. Sonam's 14-month-old son Aarav has had diarrhea for 3 days, vomiting, and is very drowsy. Signs of moderate-to-severe dehydration — sunken eyes, dry mouth, reduced urine output, skin pinch slow return. Needs urgent referral.

**Expected extraction:** Child Health form (age 14 months, diarrhea 3 days, vomiting, reduced feeding). Danger signs: **dehydration, drowsiness/lethargy, persistent vomiting**. Referral: **urgent, same day, to nearest CHC with IV fluids**.

**Script outline:**

ASHA: सोनम, आरव कैसा है? कल तुमने बुलाया था फ़ोन पे।
Sonam: दीदी, तीन दिन से दस्त लग रहे हैं। पानी जैसे आते हैं। और दो बार से उल्टी भी कर रहा है।
ASHA: कितनी बार दस्त हो रहे हैं?
Sonam: गिनती नहीं है दीदी, आठ-दस बार दिन में। डायपर भीग जाता है हर बार।
ASHA: पानी पी रहा है? दूध?
Sonam: दूध नहीं ले रहा। पानी भी कम पी रहा है। थका रहता है बस।
ASHA: (looks at baby) आरव बेटा... (pause) सोनम ये बहुत सुस्त लग रहा है। आँखें भी धँसी हुई हैं।
Sonam: हाँ दीदी, कल रात से बहुत ढीला हो गया है।
ASHA: पेशाब कर रहा है?
Sonam: बहुत कम। सुबह से एक बार ही।
ASHA: (pinches skin gently) देखो, चमड़ी भी धीरे वापस जा रही है। इसको डीहाइड्रेशन हो रहा है — शरीर में पानी की कमी है। ORS दिया था?
Sonam: थोड़ा दिया था पर उल्टी कर देता है।
ASHA: सुनो, इसको अभी CHC ले जाना पड़ेगा — ड्रिप लगेगी। घर पे ये ठीक नहीं होगा। ये ख़तरे की स्थिति है। चलो तुरंत, मैं साथ आती हूँ।

---

## Recording Checklist (per case)

- [ ] 1. ANC Normal recorded
- [ ] 2. ANC Preeclampsia recorded
- [ ] 3. PNC Day 7 recorded
- [ ] 4. Child Health Diarrhea recorded

## Pipeline Validation (per case)

For each recording:
1. Upload via Voice Mode OR put in Field Mode queue + Sync
2. Check transcript captures key details (BP, symptoms, age, duration)
3. Check form fields populate correctly
4. Check danger signs fire only on cases 2 and 4
5. Save `data/real_audio/<case>.expected.json` from the extracted result (after manual review)

## When 4/4 pass

Update README Safety section: remove "all current test data is synthetic" caveat, replace with "validated on real-voice role-played ASHA conversations in noisy conditions, including two confirmed danger cases (preeclampsia, pediatric dehydration)."
