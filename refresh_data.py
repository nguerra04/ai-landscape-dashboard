"""GitHub Actions scheduled refresh. Run `python refresh_data.py` manually as needed."""
import os
from research import write_snapshot

if __name__ == "__main__":
    data = write_snapshot(sec_contact_email=os.getenv("SEC_CONTACT_EMAIL", ""))
    print(f"Snapshot refreshed: {len(data['news']['documents'])} official documents, "
          f"{len(data['financials']['rows'])} SEC annual revenue observations")
    for warning in data["news"]["warnings"] + data["financials"]["warnings"]:
        print("NOTE:", warning)
