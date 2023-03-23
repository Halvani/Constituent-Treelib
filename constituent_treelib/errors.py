class ParenthesesError(Exception):
    """ Raised when there is a mismatch between opening and closing parenthesis."""
    pass


class NoneOrEmptyBracketedTreeError(Exception):
    """ Raised when there a bracketed tree is either none or empty."""
    pass


class NLPPipelineError(Exception):
    """ Raised when there is an issue with the nlp pipeline (e.g., the benepar component is missing.)"""
    pass


class LanguageError(Exception):
    """ Raised in case of language issues (e.g., a language mismatch between spaCy and benepar)."""
    pass


class SentenceError(Exception):
    """ Raised when an invalid sentence is given."""
    pass