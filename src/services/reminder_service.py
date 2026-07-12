"""
Reminder service — create, list, delete reminders and fire due ones.
"""

from datetime import datetime, timezone
from sqlalchemy.orm import Session

from Data.models import Reminder, Task, CalendarEvent
from Data.database import SessionLocal
from services import notification_service


def create_reminder(
    db: Session,
    user_id: str,
    task_id: str,
    remind_at: datetime,
) -> dict:
    """Create a reminder for a task."""
    # Verify the task or calendar event exists and belongs to the user
    task = db.query(Task).filter(Task.id == task_id,
                                 Task.user_id == user_id).first()
    if not task:
        event = db.query(CalendarEvent).filter(CalendarEvent.id == task_id,
                                               CalendarEvent.user_id == user_id).first()
        if not event:
            raise ValueError("Task or Calendar Event not found")

    reminder = Reminder(
        task_id=task_id,
        user_id=user_id,
        remind_at=remind_at,
    )
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return _reminder_to_dict(reminder)


def list_reminders(db: Session, user_id: str,
                   task_id: str | None = None) -> list:
    """List reminders for a user, optionally filtered by task."""
    query = db.query(Reminder).filter(Reminder.user_id == user_id)
    if task_id:
        query = query.filter(Reminder.task_id == task_id)
    reminders = query.order_by(Reminder.remind_at.asc()).all()
    return [_reminder_to_dict(r) for r in reminders]


def delete_reminder(db: Session, reminder_id: str, user_id: str) -> bool:
    """Delete a reminder."""
    reminder = (
        db.query(Reminder)
        .filter(Reminder.id == reminder_id, Reminder.user_id == user_id)
        .first()
    )
    if not reminder:
        return False
    db.delete(reminder)
    db.commit()
    return True


def check_and_fire_due_reminders():
    """
    Check for due reminders across all users and create notifications.
    Designed to be called periodically from a background task.
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        due_reminders = (
            db.query(Reminder)
            .filter(Reminder.is_sent == False, Reminder.remind_at <= now)  # noqa: E712
            .all()
        )
        for reminder in due_reminders:
            task = db.query(Task).filter(Task.id == reminder.task_id).first()
            if task:
                task_title = task.title
            else:
                event = db.query(CalendarEvent).filter(CalendarEvent.id == reminder.task_id).first()
                task_title = event.title if event else "Unknown Task"

            notification_service.create_notification(
                db,
                user_id=reminder.user_id,
                title=f"Reminder: {task_title}",
                message=f"Your reminder for task \"{task_title}\" is due.",
                notif_type="reminder",
            )
            reminder.is_sent = True

        db.commit()
        return len(due_reminders)
    finally:
        db.close()


def _reminder_to_dict(reminder: Reminder) -> dict:
    return {
        "id": reminder.id,
        "task_id": reminder.task_id,
        "user_id": reminder.user_id,
        "remind_at": reminder.remind_at.isoformat()
        if reminder.remind_at else None,
        "is_sent": reminder.is_sent,
        "created_at": reminder.created_at.isoformat()
        if reminder.created_at else None,
    }
