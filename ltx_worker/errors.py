class AmbiguousSubmissionError(RuntimeError):
    """A paid submission may exist remotely, so automatic retry is forbidden."""
