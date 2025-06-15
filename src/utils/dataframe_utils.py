import pandas as pd

def remove_character_from_column(_df, columname, character_to_remove, new_character = ' ',is_regex= True):
    """
    Removes a specific character or pattern from the specified column of a pandas DataFrame
    and optionally replaces it with a new character. The operation can be performed using
    regular expressions if specified.

    :param _df: The pandas DataFrame to be processed.
    :type _df: Pandas.DataFrame
    :param columname: The name of the column in the DataFrame where the
                      character removal will take place. Must be present in the DataFrame.
    :type columname: Str
    :param character_to_remove: The character or pattern to be removed from the column.
    :type character_to_remove: Str
    :param new_character: The character to replace the removed character/pattern with.
                          Defaults to a single space.
    :type new_character: Str, optional
    :param is_regex: A flag indicating whether `character_to_remove` should be treated as a
                     regular expression. Defaults to True.
    :type is_regex: Bool, optional
    :return: The DataFrame with the specified character or pattern removed
             (and optionally replaced) in the specified column.
    :rtype: Pandas.DataFrame
    """
    if columname is None: raise KeyError("columname cannot be None")
    if character_to_remove is None: raise KeyError("character_to_remove cannot be None")
    if not isinstance(_df, pd.DataFrame): raise TypeError("dataset must be a pandas DataFrame")

    if columname in _df.columns: _df[columname] = _df[columname].str.replace(character_to_remove, new_character, regex=is_regex)
    else: raise KeyError(f"Column '{columname}' not present in the DataFrame")

    return _df

def replace_characters(df_to_replace, columname='text', character_patterns=None, is_regex=True):
    """
    Replaces specific patterns or characters in a specified column of a DataFrame
    with corresponding replacements. By default, performs replacements for a range
    of predefined patterns, including Unicode spaces, invisible characters,
    quotation marks, emojis, and more.

    :param df_to_replace: The DataFrame on which the replacement operations
        are to be performed.
    :type df_to_replace: Pandas.DataFrame
    :param columname: The column within the DataFrame where replacements will take
        place. Defaults to 'text'.
    :type columname: Str
    :param character_patterns: Optional list of tuples specifying patterns and
        their corresponding replacements. Each tuple should contain a pattern as a
        string or regex and its associated replacement string. If not provided,
        a default set of patterns is used.
    :type character_patterns: List[tuple[str, str]] | None
    :param is_regex: Flag indicating whether the patterns provided in
        `character_patterns` should be treated as regular expressions. Defaults
        to True.
    :type is_regex: Bool
    :return: A new DataFrame with the specified replacements applied to the
        selected column.
    :rtype: Pandas.DataFrame
    """

    if character_patterns is None: character_patterns = [
            (r'[\u2000-\u200B\u3000\xa0]', ' '),       # Unicode spaces
            (r'[\u200C\u200D\u2060\uFEFF]', ''),       # Invisible characters
            (r'[\u201C\u201D\u2018\u2019]', '"'),      # Quotation marks
            (r'[\u2022\u2043]', ''),                   # List symbols
            (r'[\U0001F600-\U0001F64F]', ''),          # Emoji
            (r'[\u2026]', '...'),                      # Ellipsis
            (r"http\S+|www.\S+", ""),                  # URL
            (r"[\n\t\r]", "")                          # Tabulation and newline characters
        ]
    else:
        if not isinstance(character_patterns, list): raise TypeError("character_patterns must be a list of tuples")

    for pattern, replacement in character_patterns:
        df_to_replace = remove_character_from_column(df_to_replace, columname=columname, character_to_remove=pattern, new_character=replacement, is_regex=is_regex)

    return df_to_replace