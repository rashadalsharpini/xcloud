import json
from datetime import datetime, timezone
from googleapiclient.discovery import build
from services import gmail_service, task_service
from services import google_calendar_service
from services import google_tasks_service as gtasks_service
from services import search_service, rag_service
from services.google_auth_service import get_google_credentials
from services.providers import get_current_provider

AGENT_SYSTEM_PROMPT = """You are Xcloud, an AI assistant with access to Google services, web search, and local documents.
You can read and send emails, manage calendar events, handle tasks, search the web, and look up indexed documents.

When the user asks you to do something:
1. Use the appropriate tool to fulfill the request.
2. If you need current or external information, use web_search.
3. If the user refers to their documents or notes, use rag_search.
4. After getting results, summarize what you did.

Date & time handling:
- The current date and time (UTC) is provided below. Resolve relative dates like
  "today", "tomorrow", "next Monday", "this evening" yourself into concrete
  ISO 8601 values — do NOT ask the user for the absolute date.
- When creating a calendar event with no explicit time, use 09:00 as the start.
- end_time is optional; if the user doesn't give one, omit it (it defaults to
  one hour after the start). Times are interpreted as UTC unless stated.
- Only ask the user a clarifying question if the request is genuinely ambiguous
  (e.g. no event title at all). Never ask for a timezone — assume UTC.

Always ask for confirmation before sending emails or deleting anything important.
When listing items, present them clearly and ask the user what they'd like to do next."""


def _build_system_prompt() -> str:
    now = datetime.now(timezone.utc)
    context = (
        f"\n\nCurrent date and time (UTC): {now.strftime('%A, %Y-%m-%d %H:%M')} "
        f"(ISO: {now.isoformat()})."
    )
    return AGENT_SYSTEM_PROMPT + context

TOOL_DEFINITIONS = [
    # ── Gmail tools ──
    {
        "type": "function",
        "function": {
            "name": "list_emails",
            "description": "List emails from the user's Gmail inbox. Returns subject, sender, and snippet for each.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Number of emails to return (max 50)", "default": 10},
                    "folder": {"type": "string", "description": "Folder to read from: inbox, sent, archive, trash", "default": "inbox"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search Gmail using Gmail's search syntax. E.g. 'from:example@.com', 'subject:meeting', 'has:attachment'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Gmail search query"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_email",
            "description": "Get the full content of a specific email by its database ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "The email's database ID (UUID)"},
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email via Gmail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_email_read",
            "description": "Mark an email as read in the local database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "The email's database ID"},
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archive_email",
            "description": "Move an email to archive (removes inbox label on Gmail).",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "The email's database ID"},
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trash_email",
            "description": "Move an email to trash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "The email's database ID"},
                },
                "required": ["email_id"],
            },
        },
    },
    # ── Calendar tools ──
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "List upcoming Google Calendar events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Max events", "default": 10},
                    "days_ahead": {"type": "integer", "description": "How many days ahead to look", "default": 14},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new Google Calendar event. Resolve relative dates like 'today'/'tomorrow' to a concrete ISO date using the current date provided in the system prompt. If the user gives no time, use 09:00 for the start. end_time is optional and defaults to one hour after start_time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "description": "Start time in ISO 8601, e.g. '2026-06-30T09:00:00'. Date-only ('2026-06-30') is allowed for all-day events."},
                    "end_time": {"type": "string", "description": "Optional end time in ISO 8601. Defaults to one hour after start_time if omitted."},
                    "description": {"type": "string", "description": "Optional description"},
                    "location": {"type": "string", "description": "Optional location"},
                },
                "required": ["summary", "start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": "Update an existing calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The Google Calendar event ID"},
                    "summary": {"type": "string", "description": "New title"},
                    "start_time": {"type": "string", "description": "New start time ISO"},
                    "end_time": {"type": "string", "description": "New end time ISO"},
                    "description": {"type": "string", "description": "New description"},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The Google Calendar event ID to delete"},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_calendar_events",
            "description": "Search calendar events by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    # ── Google Tasks tools ──
    {
        "type": "function",
        "function": {
            "name": "list_google_task_lists",
            "description": "List all Google Tasks task lists.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_google_tasks",
            "description": "List tasks from a Google Tasks list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasklist_id": {"type": "string", "description": "Task list ID (default: @default for primary list)", "default": "@default"},
                    "max_results": {"type": "integer", "description": "Max tasks", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_google_task",
            "description": "Create a task in Google Tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "tasklist_id": {"type": "string", "description": "Task list ID", "default": "@default"},
                    "notes": {"type": "string", "description": "Optional notes"},
                    "due_date": {"type": "string", "description": "Due date RFC 3339 e.g. '2026-06-25T23:59:00Z'"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_google_task",
            "description": "Mark a Google Task as completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "tasklist_id": {"type": "string", "description": "Task list ID", "default": "@default"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_google_task",
            "description": "Delete a Google Task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "tasklist_id": {"type": "string", "description": "Task list ID", "default": "@default"},
                },
                "required": ["task_id"],
            },
        },
    },
    # ── Local Xcloud Tasks tools ──
    {
        "type": "function",
        "function": {
            "name": "list_my_tasks",
            "description": "List the user's local Xcloud tasks, optionally filtered by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter: pending, in_progress, completed", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_my_task",
            "description": "Create a local Xcloud task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Optional description"},
                    "priority": {"type": "string", "description": "low, medium, or high", "default": "medium"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_my_task",
            "description": "Update a local Xcloud task's status or details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "title": {"type": "string", "description": "New title"},
                    "status": {"type": "string", "description": "pending, in_progress, or completed"},
                    "priority": {"type": "string", "description": "low, medium, or high"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_my_task",
            "description": "Delete a local Xcloud task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to delete"},
                },
                "required": ["task_id"],
            },
        },
    },
    # ── Web search & RAG tools ──
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Use this when you need current information, news, or facts you're not sure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Number of results (max 10)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search through the user's indexed local documents and notes. Use this when the user asks about information in their notes, documents, or knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in the documents"},
                    "top_k": {"type": "integer", "description": "Number of relevant chunks to retrieve", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]


async def _execute_tool(name: str, args: dict, user, db) -> str:
    try:
        if name == "list_emails":
            folder = args.get("folder", "inbox")
            max_results = args.get("max_results", 10)
            result = gmail_service.list_emails(db, user.id, folder=folder, per_page=max_results)
            emails = result.get("emails", [])
            if not emails:
                return "No emails found in that folder."
            lines = [f"Found {result.get('total', 0)} emails (showing {len(emails)}):"]
            for e in emails:
                lines.append(f"  [{e['id']}] {e.get('subject','')} — from {e.get('sender','')} ({'read' if e.get('is_read') else 'unread'})")
            return "\n".join(lines)

        elif name == "search_emails":
            query = args["query"]
            max_results = args.get("max_results", 10)
            creds = get_google_credentials(user)
            if not creds:
                return "No Google credentials available."
            service = build("gmail", "v1", credentials=creds)
            results = service.users().messages().list(userId="me", maxResults=max_results, q=query).execute()
            msgs = results.get("messages", [])
            if not msgs:
                return "No emails found matching that query."
            lines = [f"Found {len(msgs)} email(s):"]
            for m in msgs:
                msg = service.users().messages().get(userId="me", id=m["id"]).execute()
                headers = msg["payload"].get("headers", [])
                subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
                sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
                lines.append(f"  ID: {m['id']} | From: {sender} | Subject: {subject}")
            return "\n".join(lines)

        elif name == "get_email":
            email_id = args["email_id"]
            email = gmail_service.get_email(db, email_id, user.id)
            if not email:
                return "Email not found."
            return f"From: {email.get('sender','')}\nTo: {email.get('recipients','')}\nSubject: {email.get('subject','')}\nDate: {email.get('received_at','')}\nFolder: {email.get('folder','')}\nRead: {email.get('is_read','')}\n\n{email.get('body','')[:5000]}"

        elif name == "send_email":
            result = gmail_service.send_email(db, user, args["to"], args["subject"], args["body"])
            return f"Email sent to {args['to']} with subject '{args['subject']}' (ID: {result.get('id','')})"

        elif name == "mark_email_read":
            result = gmail_service.mark_email_read(db, args["email_id"], user.id)
            if not result:
                return "Email not found."
            return f"Marked email '{result.get('subject','')}' as read."

        elif name == "archive_email":
            result = gmail_service.archive_email(db, args["email_id"], user.id)
            if not result:
                return "Email not found."
            return f"Archived email '{result.get('subject','')}'."

        elif name == "trash_email":
            result = gmail_service.delete_email(db, args["email_id"], user.id)
            if not result:
                return "Email not found."
            return "Email moved to trash."

        elif name == "list_calendar_events":
            max_results = args.get("max_results", 10)
            days_ahead = args.get("days_ahead", 14)
            events = google_calendar_service.list_google_events(user, max_results=max_results, days_ahead=days_ahead)
            if not events:
                return "No upcoming events found."
            lines = [f"Upcoming events ({len(events)}):"]
            for e in events:
                lines.append(f"  [{e['id']}] {e.get('start','')} — {e.get('summary','')}")
            return "\n".join(lines)

        elif name == "create_calendar_event":
            summary = args.get("summary") or args.get("title")
            start_time = args.get("start_time") or args.get("start")
            if not summary:
                return "Missing event title (summary)."
            if not start_time:
                return "Missing start_time for the event."
            result = google_calendar_service.create_google_event(
                user, summary, start_time,
                end_time=args.get("end_time") or args.get("end"),
                description=args.get("description", ""),
                location=args.get("location", ""),
            )
            return f"Event created: {result['summary']} ({result.get('htmlLink','')})"

        elif name == "update_calendar_event":
            result = google_calendar_service.update_google_event(
                user, args["event_id"],
                summary=args.get("summary"),
                start_time=args.get("start_time"),
                end_time=args.get("end_time"),
                description=args.get("description"),
            )
            return f"Event updated: {result['summary']}"

        elif name == "delete_calendar_event":
            google_calendar_service.delete_google_event(user, args["event_id"])
            return f"Event {args['event_id']} deleted."

        elif name == "search_calendar_events":
            events = google_calendar_service.search_google_events(user, args["query"], max_results=args.get("max_results", 10))
            if not events:
                return f"No events matching '{args['query']}' found."
            lines = [f"Events matching '{args['query']}':"]
            for e in events:
                lines.append(f"  [{e['id']}] {e.get('start','')} — {e.get('summary','')}")
            return "\n".join(lines)

        elif name == "list_google_task_lists":
            lists = gtasks_service.list_task_lists(user)
            if not lists:
                return "No task lists found."
            return "\n".join(f"  [{t['id']}] {t['title']}" for t in lists)

        elif name == "list_google_tasks":
            tasks = gtasks_service.list_tasks(
                user,
                tasklist_id=args.get("tasklist_id", "@default"),
                max_results=args.get("max_results", 20),
            )
            if not tasks:
                return "No tasks found."
            lines = [f"Tasks ({len(tasks)}):"]
            for t in tasks:
                status = "✓" if t["status"] == "completed" else "○"
                due = f" due: {t['due']}" if t.get("due") else ""
                lines.append(f"  {status} [{t['id']}] {t['title']}{due}")
            return "\n".join(lines)

        elif name == "create_google_task":
            result = gtasks_service.create_task(
                user, args["title"],
                tasklist_id=args.get("tasklist_id", "@default"),
                notes=args.get("notes", ""),
                due_date_rfc3339=args.get("due_date"),
            )
            return f"Task created: {result['title']} (ID: {result['id']})"

        elif name == "complete_google_task":
            result = gtasks_service.complete_task(
                user, args["task_id"],
                tasklist_id=args.get("tasklist_id", "@default"),
            )
            return f"Task completed: {result['title']}"

        elif name == "delete_google_task":
            gtasks_service.delete_task(
                user, args["task_id"],
                tasklist_id=args.get("tasklist_id", "@default"),
            )
            return f"Task {args['task_id']} deleted."

        elif name == "list_my_tasks":
            status = args.get("status") or None
            tasks = task_service.list_tasks(db, user.id, status=status)
            if not tasks:
                return "No tasks found."
            lines = [f"Local tasks ({len(tasks)}):"]
            for t in tasks:
                lines.append(f"  [{t['id']}] {t['title']} — {t['status']} ({t['priority']})")
            return "\n".join(lines)

        elif name == "create_my_task":
            result = task_service.create_task(
                db, user.id, args["title"],
                description=args.get("description"),
                priority=args.get("priority", "medium"),
            )
            return f"Local task created: {result['title']} (ID: {result['id']})"

        elif name == "update_my_task":
            result = task_service.update_task(
                db, args["task_id"], user.id,
                title=args.get("title"),
                status=args.get("status"),
                priority=args.get("priority"),
            )
            if not result:
                return "Task not found."
            return f"Task updated: {result['title']} — status: {result['status']}"

        elif name == "delete_my_task":
            if task_service.delete_task(db, args["task_id"], user.id):
                return f"Task {args['task_id']} deleted."
            return "Task not found."

        elif name == "web_search":
            query = args["query"]
            max_results = args.get("max_results", 5)
            results = search_service.web_search(query, max_results=max_results)
            if not results or "error" in results[0]:
                return f"Web search failed: {results[0].get('error', 'no results')}"
            lines = [f"Web search results for '{query}':"]
            for i, r in enumerate(results, 1):
                lines.append(f"  [{i}] {r.get('title','')} — {r.get('href','')}")
                lines.append(f"      {r.get('body','')[:200]}")
            return "\n".join(lines)

        elif name == "rag_search":
            query = args["query"]
            top_k = args.get("top_k", 3)
            if rag_service.current_index is None:
                return "No document index is loaded. Use the RAG API to load a collection first."
            context, sources = rag_service.get_context_for_llm(query, top_k=top_k)
            if not context:
                return "No relevant documents found."
            lines = [f"Retrieved {len(sources)} relevant document chunk(s):"]
            for s in sources:
                lines.append(f"  ─ {s.get('title','doc')} (score: {s.get('score',0):.2f})")
            lines.append("")
            lines.append(context[:2000])
            return "\n".join(lines)

        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error executing {name}: {e!s}"


async def stream_agent_response(
    prompt: str,
    messages: list,
    model: str,
    user,
    db,
):
    yield json.dumps({"type": "agent_start"}) + "\n"

    full_messages = (
        [{"role": "system", "content": _build_system_prompt()}]
        + [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
        + [{"role": "user", "content": prompt}]
    )

    max_turns = 10
    turn = 0

    while turn < max_turns:
        turn += 1
        response_content = ""
        tool_calls = None

        provider = get_current_provider()

        async for chunk in provider.chat_stream(
            model=model,
            messages=full_messages,
            tools=TOOL_DEFINITIONS,
        ):
            if chunk.get("tool_calls"):
                tool_calls = chunk["tool_calls"]
            if chunk.get("content"):
                response_content += chunk["content"]

        if not tool_calls:
            yield json.dumps({"type": "content", "content": response_content}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args

            yield json.dumps({"type": "tool_call", "name": name, "args": args}) + "\n"

            result = await _execute_tool(name, args, user, db)

            yield json.dumps({"type": "tool_result", "name": name, "result": result[:1000]}) + "\n"

            full_messages.append({
                "role": "assistant",
                "content": response_content or f"I'll use the {name} tool.",
            })
            full_messages.append({
                "role": "tool",
                "content": result,
                "name": name,
            })
            response_content = ""

    yield json.dumps({"type": "content", "content": "I've reached the maximum number of tool calls for this request. Let me summarize what was done."}) + "\n"
    yield json.dumps({"type": "done"}) + "\n"
