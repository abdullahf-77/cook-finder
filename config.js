// Cook Finder frontend configuration.
//
// This is a plain static page (no build step), so it can't read OS
// environment variables directly. Instead, set the backend URL here for
// your deployment, or override it at runtime by opening the page with
// ?api=https://your-backend-host — the query param takes priority so you
// never need to edit this file just to test against a different backend.
window.COOKFINDER_API_URL = window.COOKFINDER_API_URL || "http://localhost:8000";
