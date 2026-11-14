"""Job-title words shared by the person-name cleaner and the insider role/population classifiers.

Plain words and fragments, no pattern of their own: each consumer embeds them in a pattern with its own case rule.
"""

CHIEF = "Chief"
PRESIDENT = "President"
CHAIR = "Chair"
CHAIRMAN = CHAIR + "man"
VICE = "Vice"
SENIOR = "Senior"
EXECUTIVE = "Executive"
GENERAL = "General"
COUNSEL = "Counsel"
CEO, CFO, COO = "CEO", "CFO", "COO"

GENERAL_COUNSEL = rf"{GENERAL}\s+{COUNSEL}"
SENIOR_VICE = rf"{SENIOR}\s+{VICE}"
EXECUTIVE_VICE = rf"{EXECUTIVE}\s+{VICE}"
