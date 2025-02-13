import Levenshtein
class CTC:
    lexicon = [
        "Maayong buntag",
        "Maayong hapon",
        "Maayong Gabii",
        "Amping",
        "Maayo Man Ko",
        "Palihug",
        "Mag-amping ka",
        "Walay Sapayan",
        "Unsa imong buhaton?",
        "Daghang Salamat",
        "Naimbag a bigat",
        "Naimbag a malem",
        "Naimbag a rabii",
        "Diyos iti agyaman",
        "Mayat Met, agyamanak",
        "Paki",
        "Ag im-imbag ka",
        "Awan ti ania",
        "Anat ub-ubraem",
        "Agyamanak un-unay"
    ]

    def correct_to_lexicon(gibberish_output, lexicon_list):
        """
        Corrects a gibberish output to the closest word in the lexicon using Levenshtein distance.

        Args:
            gibberish_output (str): The output from your LSTM model (or user input).
            lexicon_list (list): Your list of lexicon words.

        Returns:
            str: The closest word from the lexicon.
        """
        min_distance = float('inf')
        closest_word = None

        for word in lexicon_list:
            distance = Levenshtein.distance(gibberish_output.lower(), word.lower())
            if distance < min_distance:
                min_distance = distance
                closest_word = word

        return closest_word