"""Google Sheets upload of impacted-projects DataFrames, plus project-table printing."""

import os
import subprocess
import tempfile
import time
from itertools import zip_longest
from pathlib import Path

import gspread
import polars as pl
from google.auth import default as google_auth_default
from tqdm.auto import tqdm

from .report import emit, emit_table


def stratify_round_robin(df: pl.DataFrame, group_name: str, target_num_rows: int) -> pl.DataFrame:
    groups = (list(df_group.rows(named=True)) for _, df_group in df.group_by(group_name, maintain_order=True))
    records = []
    for records_across_groups in zip_longest(*groups, fillvalue=None):
        for record in records_across_groups:
            if record is not None:
                records.append(record)
                if len(records) == target_num_rows:
                    break
        else:
            continue
        break

    return pl.DataFrame(records)


def _projects_to_display_df(projects: list[dict]) -> pl.DataFrame:
    """Convert project dicts to a display DataFrame, excluding internal _-prefixed keys."""
    display_cols = [key for key in projects[0] if not key.startswith("_")]
    return pl.DataFrame(
        [{key: value for key, value in project.items() if key in display_cols} for project in projects]
    ).with_columns(pl.col(pl.Float64).round(2))


def print_projects(
    projects: list[dict],
    description: str,
    max_projects: int | None = None,
    stratify_by: str | None = None,
) -> list[dict]:
    """Print a table of filtered projects, optionally stratified and limited.

    Args:
        projects: Project dicts to display.
        description: Label for the print header.
        max_projects: If set, limit to this many projects (after stratification).
        stratify_by: Column to stratify by when limiting projects (e.g. "platform").

    Returns:
        The (possibly filtered) list of projects that were printed.
    """
    if max_projects is not None and len(projects) > max_projects:
        projects_df = _projects_to_display_df(projects)
        if stratify_by:
            projects_df = stratify_round_robin(projects_df, stratify_by, max_projects)
        else:
            projects_df = projects_df.head(max_projects)
        selected_keys = set((row["org_id"], row["project_id"]) for row in projects_df.iter_rows(named=True))
        projects = [project for project in projects if (project["org_id"], project["project_id"]) in selected_keys]

    stratify_msg = f" (stratified by {stratify_by})" if stratify_by else ""
    emit(f"\n### {description}{stratify_msg}\n")
    emit_table(_projects_to_display_df(projects))

    return projects


def _upload_df_to_sheet(spreadsheet: gspread.Spreadsheet, sheet_name: str, df: pl.DataFrame) -> None:
    """Upload a polars DataFrame to a new sheet in the spreadsheet."""
    # Delete existing sheet with same name if it exists
    try:
        existing = spreadsheet.worksheet(sheet_name)
        spreadsheet.del_worksheet(existing)
    except gspread.WorksheetNotFound:
        pass

    worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=len(df) + 1, cols=len(df.columns))
    data = [df.columns] + df.to_numpy().tolist()
    worksheet.update(values=data, range_name="A1")

    # Freeze and bold header row, enable text wrapping
    worksheet.freeze(rows=1)
    worksheet.format("1:1", {"textFormat": {"bold": True}})
    worksheet.format(f"1:{len(df) + 1}", {"wrapStrategy": "WRAP"})

    columns = list(df.columns)
    visible_cols = {"platform", "query_stacktrace_string", "candidate_stacktrace_string"}
    wide_cols = {"query_stacktrace_string", "candidate_stacktrace_string", "thinking_output", "response_output"}

    requests = []
    sheet_id = worksheet.id
    for col_idx, col in enumerate(columns):
        if col in wide_cols:
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": col_idx,
                            "endIndex": col_idx + 1,
                        },
                        "properties": {"pixelSize": 400},
                        "fields": "pixelSize",
                    }
                }
            )
        if col not in visible_cols:
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": col_idx,
                            "endIndex": col_idx + 1,
                        },
                        "properties": {"hiddenByUser": True},
                        "fields": "hiddenByUser",
                    }
                }
            )

    if requests:
        spreadsheet.batch_update({"requests": requests})

    # Rate limit: ~60 write requests/min, we make ~8 per sheet
    time.sleep(8)


def _upload_projects_to_sheets(
    projects: list[dict],
    description: str,
    sort_by: list[tuple[str, bool]] | None = None,
) -> None:
    """Upload merged/new data for projects to a new Google Sheet.

    Creates a new spreadsheet named with the description and current timestamp.
    Call print_projects() first to filter/display projects before uploading.

    Args:
        projects: Project dicts to upload (already filtered by print_projects if desired).
        description: Used for the spreadsheet title.
        sort_by: Optional list of (column, descending) tuples to sort each sheet by before uploading.
    """
    creds, _ = google_auth_default(
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(creds)
    title = f"{description} ({time.strftime('%Y-%m-%d %H:%M')})"
    spreadsheet = client.create(title)

    uploads = []
    for project in projects:
        prefix = f"org_{project['org_id']}|project_{project['project_id']}"
        if project["_new_df"] is not None:
            uploads.append((f"{prefix}|new", project["_new_df"]))
        if project["_merged_df"] is not None:
            uploads.append((f"{prefix}|merged", project["_merged_df"]))

    if sort_by:
        cols = [spec[0] for spec in sort_by]
        descending = [spec[1] for spec in sort_by]
        uploads = [(name, df.sort(cols, descending=descending)) for name, df in uploads]

    for sheet_name, df in tqdm(uploads, desc="Uploading to Google Sheets"):
        _upload_df_to_sheet(spreadsheet, sheet_name, df)

    # Remove the default "Sheet1" now that other sheets exist
    try:
        spreadsheet.del_worksheet(spreadsheet.worksheet("Sheet1"))
    except gspread.WorksheetNotFound:
        pass

    emit(f"\n[Pairs of stacktraces for {description}]({spreadsheet.url})\n")


def _authenticate_for_sheets_upload() -> None:
    """Fetch the OAuth client JSON from Secret Manager and run `gcloud auth application-default login`."""
    name_secret = os.environ.get("OAUTH_CLIENT_SECRET_NAME")
    if not name_secret:
        raise SystemExit(
            "Set OAUTH_CLIENT_SECRET_NAME to the name of a Secret Manager secret holding an OAuth client JSON, or omit "
            "--upload_sheets."
        )
    client_json = subprocess.run(
        [
            "gcloud",
            "secrets",
            "versions",
            "access",
            "latest",
            f"--secret={name_secret}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(client_json)
        path_client = f.name
    try:
        subprocess.run(
            [
                "gcloud",
                "auth",
                "application-default",
                "login",
                f"--client-id-file={path_client}",
                "--scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive",
            ],
            check=True,
        )
    finally:
        Path(path_client).unlink(missing_ok=True)
