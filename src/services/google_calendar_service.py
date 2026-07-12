from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from sqlalchemy.orm import Session

from Data.models import CalendarEvent
from services.google_auth_service import get_google_credentials

# ── Google Calendar API layer ──

def _get_google_service(user):
    creds = get_google_credentials(user)
    if not creds:
        raise ValueError("No Google credentials available")
    return build("calendar", "v3", credentials=creds)


# Default calendar timezone used when the caller/LLM omits one.
DEFAULT_TIMEZONE = "UTC"


def _parse_dt(value):
    """Parse a loose datetime/date string into a datetime.

    Accepts ISO datetimes ("2026-06-30T14:00:00", with or without TZ/"Z"),
    plain dates ("2026-06-30"), and existing datetime objects. Returns a
    naive-or-aware ``datetime`` (timezone added later by the caller).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    if "T" not in s and " " not in s:
        # Date only → treat as midnight.
        return datetime.fromisoformat(s)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _to_google_datetime(value, default=None):
    """Normalise a datetime into the Google ``{dateTime, timeZone}`` shape.

    - Date-only / naive datetimes get the default timezone attached.
    - Returns ``None`` if there is nothing to parse and no default."""
    dt = _parse_dt(value)
    if dt is None:
        dt = _parse_dt(default)
    if dt is None:
        return None
    if dt.tzinfo is None:
        return {"dateTime": dt.isoformat(), "timeZone": DEFAULT_TIMEZONE}
    return {"dateTime": dt.isoformat()}


def list_google_events(user, max_results=20, days_ahead=30, days_behind=365):
    service = _get_google_service(user)
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(days=days_behind)).isoformat()
    time_max = (now + timedelta(days=days_ahead)).isoformat()
    events = service.events().list(
        calendarId="primary", timeMin=time_min, timeMax=time_max,
        maxResults=max_results, singleEvents=True, orderBy="startTime",
    ).execute()
    result = []
    for e in events.get("items", []):
        result.append({
            "id": e["id"],
            "summary": e.get("summary", ""),
            "description": e.get("description", ""),
            "location": e.get("location", ""),
            "start": e["start"].get("dateTime", e["start"].get("date")),
            "end": e["end"].get("dateTime", e["end"].get("date")),
            "htmlLink": e.get("htmlLink", ""),
        })
    return result


def create_google_event(user, summary, start_time, end_time=None, description="", location=""):
    service = _get_google_service(user)

    start_dt = _parse_dt(start_time)
    if start_dt is None:
        raise ValueError("start_time is required")
    # Default the end time to one hour after the start when omitted.
    end_dt = _parse_dt(end_time)
    if end_dt is None:
        end_dt = start_dt + timedelta(hours=1)

    body = {
        "summary": summary,
        "description": description or "",
        "start": _to_google_datetime(start_dt),
        "end": _to_google_datetime(end_dt),
    }
    if location:
        body["location"] = location
    event = service.events().insert(calendarId="primary", body=body).execute()
    return {"id": event["id"], "summary": event.get("summary", ""), "htmlLink": event.get("htmlLink", "")}


def update_google_event(user, event_id, summary=None, start_time=None, end_time=None, description=None, location=None):
    service = _get_google_service(user)
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    if summary is not None:
        event["summary"] = summary
    if description is not None:
        event["description"] = description
    if location is not None:
        event["location"] = location
    if start_time is not None:
        event["start"] = _to_google_datetime(start_time)
    if end_time is not None:
        event["end"] = _to_google_datetime(end_time)
    updated = service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return {"id": updated["id"], "summary": updated.get("summary", ""), "htmlLink": updated.get("htmlLink", "")}


def delete_google_event(user, event_id):
    service = _get_google_service(user)
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return True


def search_google_events(user, query, max_results=10):
    service = _get_google_service(user)
    now = datetime.now(timezone.utc).isoformat()
    events = service.events().list(
        calendarId="primary", timeMin=now,
        maxResults=max_results, singleEvents=True,
        orderBy="startTime", q=query,
    ).execute()
    result = []
    for e in events.get("items", []):
        result.append({
            "id": e["id"],
            "summary": e.get("summary", ""),
            "start": e["start"].get("dateTime", e["start"].get("date")),
            "htmlLink": e.get("htmlLink", ""),
        })
    return result


# ── Local DB calendar event layer ──


def create_event(
    db: Session,
    user_id: str,
    title: str,
    start_time: datetime,
    end_time: datetime,
    description: str | None = None,
    location: str | None = None,
    color: str | None = None,
    google_event_id: str | None = None,
) -> dict:
    event = CalendarEvent(
        user_id=user_id, title=title, description=description,
        location=location, start_time=start_time, end_time=end_time,
        color=color, google_event_id=google_event_id,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return _event_to_dict(event)


def list_events(
    db: Session,
    user_id: str,
    days_ahead: int = 30,
    days_behind: int = 365,
) -> list:
    now = datetime.now(timezone.utc)
    time_min = now - timedelta(days=days_behind)
    time_max = now + timedelta(days=days_ahead)
    events = (
        db.query(CalendarEvent)
        .filter(
            CalendarEvent.user_id == user_id,
            CalendarEvent.start_time >= time_min,
            CalendarEvent.start_time <= time_max,
        )
        .order_by(CalendarEvent.start_time)
        .all()
    )
    return [_event_to_dict(e) for e in events]


def get_event(db: Session, event_id: str, user_id: str) -> dict | None:
    event = db.query(CalendarEvent).filter(
        CalendarEvent.id == event_id, CalendarEvent.user_id == user_id,
    ).first()
    if not event:
        return None
    return _event_to_dict(event)


def update_event(
    db: Session, event_id: str, user_id: str,
    title: str | None = None, description: str | None = None,
    location: str | None = None, color: str | None = None,
    start_time: datetime | None = None, end_time: datetime | None = None,
    google_event_id: str | None = None,
) -> dict | None:
    event = db.query(CalendarEvent).filter(
        CalendarEvent.id == event_id, CalendarEvent.user_id == user_id,
    ).first()
    if not event:
        return None
    if title is not None:
        event.title = title
    if description is not None:
        event.description = description
    if location is not None:
        event.location = location
    if color is not None:
        event.color = color
    if start_time is not None:
        event.start_time = start_time
    if end_time is not None:
        event.end_time = end_time
    if google_event_id is not None:
        event.google_event_id = google_event_id
    event.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(event)
    return _event_to_dict(event)


def list_unsynced_events(db: Session, user_id: str) -> list:
    events = db.query(CalendarEvent).filter(
        CalendarEvent.user_id == user_id,
        CalendarEvent.google_event_id.is_(None),
    ).all()
    return [_event_to_dict(e) for e in events]


def list_synced_events(db: Session, user_id: str) -> list:
    """Return local events that originated from / are linked to Google."""
    events = db.query(CalendarEvent).filter(
        CalendarEvent.user_id == user_id,
        CalendarEvent.google_event_id.isnot(None),
    ).all()
    return [_event_to_dict(e) for e in events]


def update_event_by_google_id(
    db: Session, user_id: str, google_event_id: str,
    title: str | None = None, description: str | None = None,
    location: str | None = None,
    start_time: datetime | None = None, end_time: datetime | None = None,
) -> dict | None:
    event = db.query(CalendarEvent).filter(
        CalendarEvent.user_id == user_id,
        CalendarEvent.google_event_id == google_event_id,
    ).first()
    if not event:
        return None
    if title is not None:
        event.title = title
    if description is not None:
        event.description = description
    if location is not None:
        event.location = location
    if start_time is not None:
        event.start_time = start_time
    if end_time is not None:
        event.end_time = end_time
    event.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(event)
    return _event_to_dict(event)


def delete_synced_events_not_in(
    db: Session, user_id: str, google_event_ids: set[str]
) -> int:
    """Delete local synced events whose Google counterpart no longer exists."""
    events = db.query(CalendarEvent).filter(
        CalendarEvent.user_id == user_id,
        CalendarEvent.google_event_id.isnot(None),
    ).all()
    deleted = 0
    for event in events:
        if event.google_event_id not in google_event_ids:
            db.delete(event)
            deleted += 1
    if deleted:
        db.commit()
    return deleted


def delete_event(db: Session, event_id: str, user_id: str) -> bool:
    event = db.query(CalendarEvent).filter(
        CalendarEvent.id == event_id, CalendarEvent.user_id == user_id,
    ).first()
    if not event:
        return False
    db.delete(event)
    db.commit()
    return True


def _event_to_dict(event: CalendarEvent) -> dict:
    return {
        "id": event.id,
        "user_id": event.user_id,
        "title": event.title,
        "description": event.description,
        "location": event.location,
        "color": event.color,
        "start_time": event.start_time.isoformat() if event.start_time else None,
        "end_time": event.end_time.isoformat() if event.end_time else None,
        "google_event_id": event.google_event_id,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "updated_at": event.updated_at.isoformat() if event.updated_at else None,
    }
