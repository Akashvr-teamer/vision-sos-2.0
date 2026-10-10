import threading

_lock = threading.Lock()

_current_state = "IDLE"
_current_countdown = None
_sub_states = {
    "fall_state": "NORMAL",
    "car_state": "NORMAL",
    "robbery_state": "NORMAL",
    "fire_state": "NORMAL"
}
_last_event = None
_system_active = True
_active_source_name = "None"
_is_processing = False

_pending_test_source = None
_test_trigger_event = threading.Event()
_stop_requested = False


def write_status(state, countdown=None, sub_states=None):
    global _current_state, _current_countdown, _sub_states

    with _lock:
        _current_state = state
        _current_countdown = countdown
        if sub_states is not None:
            _sub_states = sub_states


def set_processing_status(is_processing, source_name="None"):
    global _is_processing, _active_source_name
    with _lock:
        _is_processing = is_processing
        _active_source_name = source_name


def write_event(event_result):
    global _last_event

    with _lock:
        _last_event = event_result


def read_state():
    with _lock:
        return {
            "state": _current_state,
            "countdown": _current_countdown,
            "sub_states": _sub_states,
            "last_event": _last_event,
            "system": "ACTIVE" if _system_active else "OFFLINE",
            "is_processing": _is_processing,
            "source_name": _active_source_name
        }


def set_system_active(active):
    global _system_active

    with _lock:
        _system_active = active


def request_new_test(source, category="Unified"):
    global _pending_test_source, _stop_requested
    with _lock:
        _pending_test_source = source
        _stop_requested = False
    _test_trigger_event.set()


def request_stop():
    global _stop_requested, _pending_test_source
    with _lock:
        _stop_requested = True
        _pending_test_source = None
    _test_trigger_event.set()


def is_stop_requested():
    with _lock:
        return _stop_requested


def check_for_new_test():
    global _pending_test_source
    if _test_trigger_event.is_set():
        with _lock:
            source = _pending_test_source
            _pending_test_source = None
            _test_trigger_event.clear()
        return source, "Unified"
    return None, None