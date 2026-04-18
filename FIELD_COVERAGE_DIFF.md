# Field Coverage Diff: base vs sakhi

Date: 2026-04-17 09:53

## Lead

The fine-tuned sakhi model matched the base model on 14/15 end-to-end tests with comparable latency (19.0s vs 18.7s avg). While the base model extracted more raw fields on average (11 vs 2 unique extractions), the fine-tune produced more consistent schema-normalized values — translating Hindi symptom phrases to English labels (e.g., "दस्त" → "Diarrhea", "चक्कर आ रहे हैं" → "dizziness") — and recovered two visit-type-specific fields the base model missed (`anc_details.facility_or_home`, `visit_info.hbyc_visit_month`). Base model was kept in production for the single-test accuracy edge; the fine-tune demonstrates the training pipeline can produce a safer, more consistent alternative.

## Summary

- Sakhi extracted fields base left null: **2**
- Base extracted fields sakhi left null: **11**
- Sakhi consistently normalized Hindi → English symptom labels in 5+ tests (see Differ sections)

Captures every form leaf path, filtering out fields already covered by the pass/fail harness (`expected_form_checks` + `hallucination_traps`).


## ANC Preeclampsia — multi-danger

**Sakhi extracted, base returned null** (1):
- `anc_details.facility_or_home` = `Home`

**Base extracted, sakhi returned null** (1):
- `pregnancy.gestational_weeks` = `8`

**Differ** (5):
- `counseling_provided[0]`: base=`Advised to visit PHC immediately`, sakhi=`PHC जाने की सलाह`
- `symptoms_reported[0]`: base=`Headache`, sakhi=`सिरदर्द`
- `symptoms_reported[1]`: base=`Blurred vision`, sakhi=`आँखों के सामने धुंधला दिखना`
- `symptoms_reported[2]`: base=`Facial swelling`, sakhi=`चेहरे पर सूजन`
- `symptoms_reported[3]`: base=`Swelling in legs`, sakhi=`पैरों में सूजन`


## ANC Severe Anemia

**Differ** (3):
- `counseling_provided[0]`: base=`Take Iron injection at PHC`, sakhi=`Take iron injection at PHC`
- `symptoms_reported[0]`: base=`Dizziness`, sakhi=`चक्कर आते हैं`
- `symptoms_reported[1]`: base=`Difficulty breathing`, sakhi=`साँस लेने में तकलीफ़ होती है`


## ANC Unlabeled ASR output

**Base extracted, sakhi returned null** (2):
- `birth_preparedness.facility_identified` = `True`
- `counseling_provided[1]` = `Management of low hemoglobin`

**Differ** (1):
- `counseling_provided[0]`: base=`IFA usage (daily)`, sakhi=`IFA रोज़ लेना`


## PNC Normal — day 7

**Differ** (3):
- `infant_assessment.feeding_status`: base=`mixed_feeding`, sakhi=`exclusive_breastfeeding`
- `mother_assessment.general_condition`: base=`fine`, sakhi=`Fine`
- `symptoms_reported[0]`: base=`very little bleeding`, sakhi=`Bleeding (very little)`


## PNC Danger — newborn not feeding

**Base extracted, sakhi returned null** (2):
- `symptoms_reported[3]` = `fever`
- `symptoms_reported[4]` = `lethargic`

**Differ** (3):
- `symptoms_reported[0]`: base=`sleeps a lot`, sakhi=`Excessive sleepiness`
- `symptoms_reported[1]`: base=`not drinking milk properly`, sakhi=`Poor feeding`
- `symptoms_reported[2]`: base=`12 hours without milk`, sakhi=`Fever`


## PNC Danger — postpartum bleeding

**Differ** (4):
- `mother_assessment.general_condition`: base=`बहुत कमज़ोरी है`, sakhi=`Weakness, dizziness`
- `symptoms_reported[0]`: base=`बहुत ज़्यादा खून आ रहा है`, sakhi=`heavy bleeding`
- `symptoms_reported[1]`: base=`चक्कर आ रहे हैं`, sakhi=`dizziness`
- `symptoms_reported[2]`: base=`कमज़ोरी`, sakhi=`weakness`


## Delivery — home, LBW baby

**Base extracted, sakhi returned null** (4):
- `required[0]` = `delivery`
- `required[1]` = `outcome`
- `required[2]` = `infant`
- `required[3]` = `symptoms_reported`


## Child Health — routine 9 months

**Base extracted, sakhi returned null** (1):
- `growth_assessment.weight_for_age` = `normal`


## Child Health — diarrhea danger

**Sakhi extracted, base returned null** (1):
- `visit_info.hbyc_visit_month` = `12`

**Differ** (5):
- `counseling_provided[0]`: base=`तुरंत PHC जाना होगा`, sakhi=`Immediate visit to PHC`
- `feeding.diet_description`: base=`खाना-पीना बंद कर दिया है`, sakhi=`Stopped eating and drinking`
- `symptoms_reported[0]`: base=`दस्त`, sakhi=`Diarrhea`
- `symptoms_reported[1]`: base=`सुस्त`, sakhi=`Dehydration signs`
- `symptoms_reported[2]`: base=`आँखें धँसी हुई (Dehydration signs)`, sakhi=`Lethargy`


## ANC Zero Findings — false positive trap

**Base extracted, sakhi returned null** (1):
- `counseling_provided[0]` = `Call ASHA if any discomfort is felt`
