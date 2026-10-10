import time


COOLDOWN_SECONDS = 60

last_alert_time = 0


def compute_severity(event):

    score = 30
    event_type = event.get("type", "")

    # ---------------------------------
    # Event type base weight
    # ---------------------------------

    if event_type == "fire":
        score += 48
    elif event_type == "car_accident":
        score += 45
    elif event_type == "robbery":
        score += 40
    elif event_type == "collapse":
        score += 25
    elif event_type == "seizure":
        score += 35

    # ---------------------------------
    # Motion energy & Kinetic impact
    # ---------------------------------

    motion_energy = event.get("motion_energy", 0)
    if motion_energy > 5.0:
        score += 20
    elif motion_energy > 2.0:
        score += 10

    # ---------------------------------
    # Immobility
    # ---------------------------------

    motionless = event.get("motionless_sec", 0)

    if motionless >= 3:
        score += 20
    elif motionless >= 2:
        score += 15
    elif motionless >= 1:
        score += 8

    # ---------------------------------
    # Velocity / Impact
    # ---------------------------------

    fall_velocity = event.get("fall_velocity", 0)

    if fall_velocity > 250:
        score += 15
    elif fall_velocity > 150:
        score += 10

    # ---------------------------------
    # Trigger reason
    # ---------------------------------

    trigger = event.get("trigger", "")

    if trigger in ("high_kinetic_impact", "surrender_hands_up", "aggressive_confrontation", "violent_confrontation", "active_flame_flicker", "flame_color_signature"):
        score += 15
    elif trigger in ("sudden_drop", "vehicle_collision_vector"):
        score += 15
    elif trigger == "sustained_horizontal":
        score += 10

    # ---------------------------------
    # Posture
    # ---------------------------------

    torso_angle = event.get("torso_angle", 0)

    if torso_angle >= 60:
        score += 10
    elif torso_angle >= 45:
        score += 5

    # ---------------------------------
    # Night-time context
    # ---------------------------------

    timestamp = event.get("timestamp", "")

    try:
        hour = int(timestamp[11:13])
        if hour >= 22 or hour <= 6:
            score += 10
    except (ValueError, TypeError, IndexError):
        pass

    return min(int(score), 100)


def get_severity_label(score):

    if score >= 85:
        return "CRITICAL"

    if score >= 65:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


def should_alert():

    global last_alert_time

    current_time = time.time()

    if current_time - last_alert_time < COOLDOWN_SECONDS:
        return False

    last_alert_time = current_time

    return True


def process_event(event):

    severity = compute_severity(event)

    label = get_severity_label(severity)

    alert_allowed = should_alert()

    return {
        "severity": severity,
        "label": label,
        "alert": alert_allowed
    }