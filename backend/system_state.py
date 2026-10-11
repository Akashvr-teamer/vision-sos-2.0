import threading

_lock = threading.Lock()
_frame_cond = threading.Condition()
_latest_frame_bytes = None

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
_pending_test_category = "Unified"
_active_category = "Unified"
_test_trigger_event = threading.Event()
_stop_requested = False


def set_latest_frame_bytes(jpeg_bytes):
    """Stores the latest encoded JPEG frame in memory and notifies listeners."""
    global _latest_frame_bytes
    with _frame_cond:
        _latest_frame_bytes = jpeg_bytes
        _frame_cond.notify_all()


def get_latest_frame_bytes():
    """Returns the latest encoded JPEG frame bytes from memory."""
    with _frame_cond:
        return _latest_frame_bytes


def wait_for_next_frame(timeout=0.5):
    """Waits for the next frame to be published or returns the latest after timeout."""
    with _frame_cond:
        _frame_cond.wait(timeout=timeout)
        return _latest_frame_bytes


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
            "source_name": _active_source_name,
            "active_category": _active_category
        }


def set_system_active(active):
    global _system_active

    with _lock:
        _system_active = active


def set_active_category(category):
    global _active_category
    with _lock:
        _active_category = category or "Unified"


def request_new_test(source, category="Unified"):
    global _pending_test_source, _pending_test_category, _active_category, _stop_requested
    with _lock:
        _pending_test_source = source
        _pending_test_category = category or "Unified"
        _active_category = category or "Unified"
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
    global _pending_test_source, _pending_test_category
    if _test_trigger_event.is_set():
        with _lock:
            source = _pending_test_source
            category = _pending_test_category or "Unified"
            _pending_test_source = None
            _test_trigger_event.clear()
        return source, category
    return None, None