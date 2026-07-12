from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from Data.database import get_db
from Data.models import User, CalendarEvent
from services import auth_service, google_calendar_service
from services.google_auth_service import get_google_credentials

router = APIRouter()


class EventCreate(BaseModel):
    title: str
    start_time: datetime
    end_time: datetime
    description: str | None = None
    location: str | None = None
    color: str | None = None


class EventUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    location: str | None = None
    color: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


def _ensure_tz(s: str | None) -> str | None:
    if s and "T" in s and not s.endswith("Z") and "+" not in s and "-" not in s[10:]:
        return s + "Z"
    return s


def _parse_google_datetime(value: str | None) -> datetime | None:
    """Parse a Google Calendar start/end value (date or dateTime) into an
    aware UTC datetime so it stays consistent with locally-created events."""
    if not value:
        return None
    # All-day events return a plain date (e.g. "2026-06-30").
    if "T" not in value:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sync_to_google(user: User, event_dict: dict, db: Session):
    creds = get_google_credentials(user)
    if not creds:
        print(f"[sync-cal] no google creds for user {user.id}")
        return event_dict
    try:
        start = _ensure_tz(event_dict.get("start_time"))
        end = _ensure_tz(event_dict.get("end_time"))
        if event_dict.get("google_event_id"):
            google_calendar_service.update_google_event(
                user, event_dict["google_event_id"],
                summary=event_dict.get("title"),
                description=event_dict.get("description"),
                location=event_dict.get("location"),
                start_time=start,
                end_time=end,
            )
            print(f"[sync-cal] updated google event {event_dict['google_event_id']}")
        else:
            result = google_calendar_service.create_google_event(
                user,
                summary=event_dict["title"],
                start_time=start,
                end_time=end,
                description=event_dict.get("description") or "",
                location=event_dict.get("location") or "",
            )
            google_calendar_service.update_event(
                db, event_dict["id"], user.id,
                google_event_id=result["id"],
            )
            event_dict["google_event_id"] = result["id"]
            print(f"[sync-cal] created google event {result['id']} for local {event_dict['id']}")
    except Exception as e:
        print(f"[sync-cal] error: {e}")
    return event_dict


def _delete_from_google(user: User, event_dict: dict):
    gid = event_dict.get("google_event_id")
    if not gid:
        return
    creds = get_google_credentials(user)
    if not creds:
        return
    try:
        google_calendar_service.delete_google_event(user, gid)
    except Exception:
        pass


@router.post("/")
async def create_event(
    body: EventCreate,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    result = google_calendar_service.create_event(
        db, user.id,
        title=body.title,
        start_time=body.start_time,
        end_time=body.end_time,
        description=body.description,
        location=body.location,
        color=body.color,
    )
    return _sync_to_google(user, result, db)


def _pull_from_google(user: User, db: Session):
    """Mirror Google Calendar into the local DB: add new events, update
    changed ones, and delete locally-synced events removed from Google."""
    pulled = 0
    updated = 0
    existing_ids = {
        e.google_event_id
        for e in db.query(CalendarEvent).filter(
            CalendarEvent.user_id == user.id,
            CalendarEvent.google_event_id.isnot(None),
        ).all()
    }
    seen_ids: set[str] = set()
    for ge in google_calendar_service.list_google_events(
        user, max_results=250, days_ahead=365, days_behind=365
    ):
        seen_ids.add(ge["id"])
        start = _parse_google_datetime(ge.get("start"))
        end = _parse_google_datetime(ge.get("end"))
        if ge["id"] in existing_ids:
            google_calendar_service.update_event_by_google_id(
                db, user.id, ge["id"],
                title=ge.get("summary", ""),
                description=ge.get("description", ""),
                location=ge.get("location", ""),
                start_time=start,
                end_time=end,
            )
            updated += 1
            continue
        google_calendar_service.create_event(
            db, user.id,
            title=ge.get("summary", ""),
            description=ge.get("description", ""),
            location=ge.get("location", ""),
            start_time=start or datetime.now(timezone.utc),
            end_time=end or datetime.now(timezone.utc),
            google_event_id=ge["id"],
        )
        pulled += 1
    deleted = google_calendar_service.delete_synced_events_not_in(
        db, user.id, seen_ids
    )
    return pulled, updated, deleted


@router.post("/sync")
def sync_events(
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    """Bidirectional sync: push local unsynced events to Google, pull remote events to local DB.

    Declared as a sync ``def`` so FastAPI runs the blocking Google API calls in
    a threadpool instead of stalling the event loop (which would make the whole
    server — and the sync button — appear hung).
    """
    creds = get_google_credentials(user)
    if not creds:
        raise HTTPException(status_code=400, detail="No Google account linked. Sign in with Google first.")
    unsynced = google_calendar_service.list_unsynced_events(db, user.id)
    pushed = 0
    for event in unsynced:
        _sync_to_google(user, event, db)
        pushed += 1
    pulled, updated, deleted = _pull_from_google(user, db)
    return {
        "pushed": pushed,
        "pulled": pulled,
        "updated": updated,
        "deleted": deleted,
    }


@router.get("/")
async def list_events(
    days_ahead: int = Query(365),
    days_behind: int = Query(365),
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    return google_calendar_service.list_events(
        db, user.id, days_ahead=days_ahead, days_behind=days_behind
    )


@router.get("/{event_id}")
async def get_event(
    event_id: str,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    event = google_calendar_service.get_event(db, event_id, user.id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.patch("/{event_id}")
async def update_event(
    event_id: str,
    body: EventUpdate,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    event = google_calendar_service.update_event(
        db, event_id, user.id,
        title=body.title,
        description=body.description,
        location=body.location,
        color=body.color,
        start_time=body.start_time,
        end_time=body.end_time,
    )
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return _sync_to_google(user, event, db)


@router.delete("/{event_id}")
async def delete_event(
    event_id: str,
    user: User = Depends(auth_service.get_current_user),
    db: Session = Depends(get_db),
):
    event = google_calendar_service.get_event(db, event_id, user.id)
    if event:
        _delete_from_google(user, event)
    if not google_calendar_service.delete_event(db, event_id, user.id):
        raise HTTPException(status_code=404, detail="Event not found")
    return {"status": "Event deleted"}
