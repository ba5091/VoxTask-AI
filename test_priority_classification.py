from priority_classification import classify_priority


def test_classify_priority_high():
    assert classify_priority("Submit the report urgently", "Not specified") == "High"


def test_classify_priority_low():
    assert classify_priority("Call my project guide", "Later") == "Low"


def test_classify_priority_medium():
    assert classify_priority("Complete the ML assignment", "Tomorrow") == "Medium"


def test_urgent_signal_overrides_low_deadline():
    assert classify_priority("Urgently call my project guide", "Later") == "High"


def test_medium_deadline_overrides_low_word():
    assert classify_priority("Read this later today", "Today") == "Medium"
