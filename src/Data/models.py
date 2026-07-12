"""
SQLAlchemy models for Xcloud.
"""

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Integer,
    Boolean,
    Enum,
)
from sqlalchemy.orm import relationship
import enum

from Data.database import Base, generate_uuid, utcnow


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class TaskStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


class TaskPriority(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class NotificationType(str, enum.Enum):
    task_due = "task_due"
    reminder = "reminder"
    system = "system"


# --------------------------------------------------------------------------- #
# Existing models
# --------------------------------------------------------------------------- #

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    google_id = Column(String(255), unique=True, nullable=True, index=True)
    email = Column(String(255), nullable=True)
    avatar_url = Column(String(512), nullable=True)
    google_refresh_token = Column(Text, nullable=True)

    chats = relationship("Chat", back_populates="user",
                         cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="user",
                         cascade="all, delete-orphan")
    reminders = relationship(
        "Reminder", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship(
        "Notification", back_populates="user", cascade="all, delete-orphan")
    email_accounts = relationship(
        "EmailAccount", back_populates="user", cascade="all, delete-orphan")
    emails = relationship(
        "Email", back_populates="user", cascade="all, delete-orphan")
    calendar_events = relationship(
        "CalendarEvent", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"),
                     nullable=False, index=True)
    title = Column(String(255), default="New Chat")
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="chats")
    messages = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String, ForeignKey("chats.id"),
                     nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    thinking = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    chat = relationship("Chat", back_populates="messages")


# --------------------------------------------------------------------------- #
# New models: Tasks, Reminders, Notifications
# --------------------------------------------------------------------------- #

class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"),
                     nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        Enum(TaskStatus), default=TaskStatus.pending, nullable=False
    )
    priority = Column(
        Enum(TaskPriority), default=TaskPriority.medium, nullable=False
    )
    due_date = Column(DateTime, nullable=True)
    google_task_id = Column(String(255), nullable=True)
    google_task_list_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="tasks")
    reminders = relationship(
        "Reminder", back_populates="task", cascade="all, delete-orphan"
    )


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(String, primary_key=True, default=generate_uuid)
    task_id = Column(String, ForeignKey("tasks.id"),
                     nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"),
                     nullable=False, index=True)
    remind_at = Column(DateTime, nullable=False)
    is_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    task = relationship("Task", back_populates="reminders")
    user = relationship("User", back_populates="reminders")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"),
                     nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=True)
    type = Column(
        Enum(NotificationType), default=NotificationType.system, nullable=False
    )
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="notifications")


class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String(512), nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    color = Column(String(32), nullable=True)
    google_event_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="calendar_events")


class EmailAccount(Base):
    __tablename__ = "email_accounts"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String(20), nullable=False, default="smtp")
    email_address = Column(String(255), nullable=False)
    smtp_server = Column(String(255), nullable=True)
    smtp_port = Column(Integer, nullable=True, default=587)
    smtp_username = Column(String(255), nullable=True)
    smtp_password = Column(Text, nullable=True)
    imap_server = Column(String(255), nullable=True)
    imap_port = Column(Integer, nullable=True, default=993)
    imap_username = Column(String(255), nullable=True)
    imap_password = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="email_accounts")
    emails = relationship(
        "Email", back_populates="account", cascade="all, delete-orphan"
    )


class Email(Base):
    __tablename__ = "emails"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    account_id = Column(String, ForeignKey("email_accounts.id"), nullable=False, index=True)
    message_id = Column(String(255), nullable=True, index=True)
    sender = Column(String(255), nullable=True)
    recipients = Column(Text, nullable=True)
    subject = Column(String(255), nullable=True)
    body = Column(Text, nullable=True)
    is_read = Column(Boolean, default=False)
    is_starred = Column(Boolean, default=False)
    folder = Column(String(50), default="inbox")
    received_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="emails")
    account = relationship("EmailAccount", back_populates="emails")
