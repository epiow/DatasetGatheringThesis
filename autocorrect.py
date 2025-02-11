from symspellpy import SymSpell, Verbosity
from difflib import SequenceMatcher

# Define lexicons grouped by dialect
cebuano_lexicons = [
    "Maayong buntag", "Maayong hapon", "Maayong Gabii", "Amping", "Maayo Man Ko",
    "Palihug", "Mag-amping ka", "Walay Sapayan", "Unsa imong buhaton?", "Daghang Salamat"
]

ilocano_lexicons = [
    "Naimbag a bigat", "Naimbag a malem", "Naimbag a rabii", "Diyos iti agyaman",
    "Mayat Met, agyamanak", "Paki", "Ag im-imbig ka", "Awan ti ania", "Anat ub-ubraem",
    "Agyamanak un-unay"
]

# Combine all lexicons for the dictionary
all_lexicons = cebuano_lexicons + ilocano_lexicons

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=6)

# Add entire phrases to SymSpell dictionary
for phrase in all_lexicons:
    sym_spell.create_dictionary_entry(phrase, 1)

def get_dialect_group(phrase):
    """Determine which dialect group a phrase belongs to"""
    if phrase in cebuano_lexicons:
        return "Cebuano"
    elif phrase in ilocano_lexicons:
        return "Ilocano"
    return "Unknown"

def similarity_ratio(a, b):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def autocorrect_text(input_text):
    # Get all suggestions within the max edit distance
    suggestions = sym_spell.lookup(input_text, Verbosity.ALL, max_edit_distance=6)
    
    if not suggestions:
        # If no suggestions found, find the closest match using basic string similarity
        closest_match = max(all_lexicons, key=lambda x: similarity_ratio(input_text, x))
        similarity = similarity_ratio(input_text, closest_match)
        dialect = get_dialect_group(closest_match)
        return f"({dialect}) {closest_match} ({similarity:.0%} confidence)"
    
    # Filter suggestions to only include those matching our lexicons
    valid_suggestions = [s for s in suggestions if s.term in all_lexicons]
    
    if valid_suggestions:
        best_match = valid_suggestions[0]
        confidence = 1 - (best_match.distance / len(best_match.term))
        dialect = get_dialect_group(best_match.term)
        return f"({dialect}) {best_match.term} ({confidence:.0%} confidence)"
    else:
        # Use the same fallback as above
        closest_match = max(all_lexicons, key=lambda x: similarity_ratio(input_text, x))
        similarity = similarity_ratio(input_text, closest_match)
        dialect = get_dialect_group(closest_match)
        return f"({dialect}) {closest_match} ({similarity:.0%} confidence)"

if __name__ == "__main__":
    print("Autocorrect Script for Lip Reading Output")
    print("Supports Cebuano and Ilocano dialects")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter text to autocorrect: ")
        if user_input.lower() == 'exit':
            break
            
        corrected_text = autocorrect_text(user_input)
        print(f"Corrected Text: {corrected_text}")