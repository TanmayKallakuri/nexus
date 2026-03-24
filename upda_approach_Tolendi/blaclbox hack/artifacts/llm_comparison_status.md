# LLM Comparison Status

- Sample JSON rows: 500
- Sample JSON respondents: 50
- Sample JSON questions: 10
- Sample JSON question ids: T5, T8, T10, T14, T17, T18, T22, T25, T26, T49

## predictions_233_respondents.csv

- Shape: 233 respondents x 11 question columns
- Overlapping respondents with sample JSON: 50
- Question columns: S1_Googled_Self, S2_AI_Outfit, S3_HotDog_Debate, S4_Cuisine, S5_Mass_Apology, S6_Share_Article, S7_Phone_Morning, S8_Work_Memes, S9_Judge_Coffee, S10_Zombie_Confidence, S11_Vacation

## respondent_predictions_233.csv

- Shape: 233 respondents x 11 question columns
- Overlapping respondents with sample JSON: 50
- Question columns: S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11

## predicted_answers-sonnet4.6-J.csv

- Shape: 233 respondents x 11 question columns
- Overlapping respondents with sample JSON: 50
- Question columns: S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11

## Blocker

- The sample JSON uses question ids T5, T8, T10, T14, T17, T18, T22, T25, T26, T49.
- The LLM CSVs use a different question set: S1..S11, with labels like Googled_Self, AI_Outfit, HotDog_Debate, Vacation.
- Because the questions do not align, scoring them against each other would be invalid and misleading.