import pandas as pd
import ast

def detect_dark_pattern(row):
    buttons = ' '.join(ast.literal_eval(str(row['Buttons']))).lower()
    links = ' '.join(ast.literal_eval(str(row['Links']))).lower()
    checkboxes = ast.literal_eval(str(row['Checkboxes']))
    popups = ast.literal_eval(str(row['Popups']))

    checkbox_count = len(checkboxes)
    popup_count = len(popups)

    # Rule-based classification
    if any(keyword in buttons for keyword in ['back button', 'highlight', 'next only', 'accept all']):
        return 'Misdirection'
    elif any(keyword in buttons for keyword in ['allow all', 'yes to all']) or checkbox_count > 3:
        return 'Privacy Zuckering'
    elif any(keyword in buttons for keyword in ['free trial', 'upgrade now', 'subscribe']) or 'trial' in links:
        return 'Forced Continuity'
    elif any(keyword in links for keyword in ['hidden', 'auto-renew']):
        return 'Sneaking'
    elif any(keyword in buttons for keyword in ['close', 'x']) or popup_count > 1:
        return 'Obstruction'
    elif any(keyword in links for keyword in ['just now', 'selling fast', 'only 1 left', 'few left']):
        return 'Social Proof Manipulation'
    else:
        return 'No Pattern Detected'

# ✅ Load the data
df = pd.read_csv("../Data/dark_patterns_data.csv")

# ✅ Apply detection logic
df['dark_pattern_label'] = df.apply(detect_dark_pattern, axis=1)

# ✅ Save the output
df.to_csv("../Data/cleaned_dark_patterns_labeled.csv", index=False)
print("✔️ Saved to cleaned_dark_patterns_labeled.csv")
