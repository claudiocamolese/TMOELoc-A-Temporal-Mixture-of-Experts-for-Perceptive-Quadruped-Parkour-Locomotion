"""Observation layout and timing shared by ExpertParkour's student code.

These offsets describe the inherited 53-element proprioception vector. Keep
them aligned with the upstream observation builder; they are not architecture
hyperparameters and must not depend on the selected student variant.
"""

YAW_SLICE = slice(6, 8)  # Current and next waypoint headings; predicted by students.
ENV_CLASS_SLICE = slice(11, 13)  # Privileged terrain-class indicators, masked out.
CONTACT_SLICE = slice(49, 53)  # Four foot-contact indicators.
PERCEPTION_INTERVAL = 5  # 50 Hz control / 10 Hz context and routing updates.
