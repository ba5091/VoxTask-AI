from deadline_extraction import extract_deadline


def test_extract_deadline_from_today():
    task, deadline = extract_deadline("Submit the report today")
    assert "Submit the report" in task
    assert deadline.lower() == "today"


def test_extract_deadline_from_weekday():
    task, deadline = extract_deadline("Review the presentation by Friday")
    assert "Review the presentation" in task
    assert deadline.lower() == "friday"


def test_extract_deadline_preserves_relative_in_expression():
    task, deadline = extract_deadline("Study for the exam in 2 days")
    assert task == "Study for the exam"
    assert deadline == "In 2 Days"


def test_next_weekday_pattern_takes_precedence():
    task, deadline = extract_deadline("Finish the report by next Monday")
    assert task == "Finish the report"
    assert deadline == "Next Monday"


def test_deadline_does_not_match_partial_words():
    task, deadline = extract_deadline("Review the weekend plan")
    assert task == "Review the weekend plan"
    assert deadline == "Not specified"


def test_someday_is_a_supported_label():
    task, deadline = extract_deadline("Read the book someday")
    assert task == "Read the book"
    assert deadline == "Someday"
