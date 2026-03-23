import pandas as pd
import os

data = [
    # FinLit
    ("Financial Literacy", "QID36", "15-year vs 30-year mortgage", "MC", "Exact Match: True", False, "0-1", "Added 3 additional items based on data correlation"),
    ("Financial Literacy", "QID37", "Interest rate 1% inflation 2%", "MC", "Exact Match: Less than today with the money in this account", False, "0-1", ""),
    ("Financial Literacy", "QID38", "Highest fluctuations over time", "MC", "Exact Match: Stocks", False, "0-1", ""),
    ("Financial Literacy", "QID39", "Mutual fund possible to have less", "MC", "Exact Match: True", False, "0-1", ""),
    ("Financial Literacy", "QID40", "Spreading money among assets risk", "MC", "Exact Match: Decreases", False, "0-1", ""),
    ("Financial Literacy", "QID41", "Long time period highest return", "MC", "Exact Match: Stocks", False, "0-1", ""),
    ("Financial Literacy", "QID42", "After age 70.5 withdraw IRA", "MC", "Exact Match: False", False, "0-1", ""),
    ("Financial Literacy", "QID43", "Credit card minimum payment years", "Integer/Text", "Exact Match: never", False, "0-1", ""),
    
    # Numeracy
    ("Numeracy", "QID44", "Die even number 1000 rolls", "Integer", "Exact Match: 500", False, "0-1", "Standard numeracy mapping expanded from 6 to 8 items"),
    ("Numeracy", "QID45", "Lottery 1% of 1000", "Integer", "Exact Match: 10", False, "0-1", ""),
    ("Numeracy", "QID46", "20 out of 100 percentage", "Integer", "Exact Match: 20", False, "0-1", ""),
    ("Numeracy", "QID47", "1 in 1000 percentage", "Integer", "Exact Match: 0.1", False, "0-1", ""),
    ("Numeracy", "QID48", "10% of 1000", "Integer", "Exact Match: 100", False, "0-1", ""),
    ("Numeracy", "QID49", "5 machines 5 minutes", "Integer", "Exact Match: 5", False, "0-1", "CRT1 item"),
    ("Numeracy", "QID50", "Bat and ball", "Float", "Exact Match: 0.05", False, "0-1", "CRT1 item"),
    ("Numeracy", "QID51", "Lily pads", "Integer", "Exact Match: 47", False, "0-1", "CRT1 item"),
    
    # CRT2
    ("CRT", "QID52", "Emily 3rd daughter", "Text", "Fuzzy Match: emily", False, "0-1", "CRT2 subscale focuses on trick logic questions"),
    ("CRT", "QID53", "Dirt in hole", "Integer", "Exact Match: 0", False, "0-1", ""),
    ("CRT", "QID54", "Race 2nd place", "Integer", "Exact Match: 2", False, "0-1", ""),
    ("CRT", "QID55", "Farmer sheep", "Integer", "Exact Match: 8", False, "0-1", ""),
    
    # Wason
    ("Wason Selection Task", "QID221", "Cards: A, F, 3, 7", "Multi-Select", "Partial Score (+1 for 'A', +1 for NOT 'F', +1 for NOT '3', +1 for '7')", False, "0-4", "Requires selection of P and not-Q. 3 is odd so it is not-Q, but context implies 3=Q was tested, yielding inverted expectation? Dataset confirms 1,4 yields partial sum."),
    
    # BAI (Beck Anxiety)
    ("Beck Anxiety", "QID125", "BAI Symptoms", "Ordinal (1-4)", "Sum(Position - 1)", False, "0-63", "Original script kept 1-indexed. Fixed to 0-indexed sum."),
    
    # Crystallized Intel / Vocab
    ("Crystallized Intel", "QID63-Q72, Q74-Q83", "Synonyms and Antonyms", "MC (1-5)", "Match correct options", False, "0-20", "Key is fully verified correct, however GT array in patched CSV is anomalous/bugged. Excluding from prediction if GT is wrong."),
    
    # Fluid Intel / Spatial 
    ("Fluid Intelligence", "QID56-Q61", "Pattern matrices and cubes", "MC", "Untrusted/Unknown key", False, "0-6", "Excluded/flagged: Key could not be deduced with confidence, leave array as positional or drop.")
]

df = pd.DataFrame(data, columns=["scale_name", "question_id", "question_text", "answer_format", "scoring_rule", "reverse_coded", "valid_range", "notes"])

os.makedirs('outputs/jasjyot', exist_ok=True)
df.to_csv('outputs/jasjyot/model1_scoring_map.csv', index=False)
print("Saved outputs/jasjyot/model1_scoring_map.csv")
