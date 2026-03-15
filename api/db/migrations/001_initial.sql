-- Initial schema for Cloudflare D1
-- Run this in the D1 SQL console before deploying the API.

CREATE TABLE users (
    id          TEXT PRIMARY KEY,   -- Google sub
    email       TEXT NOT NULL,
    name        TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE images (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    r2_key        TEXT NOT NULL,    -- path in R2: users/{user_id}/images/{id}.png
    original_name TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE classifications (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    image_id         TEXT NOT NULL,
    model_used       TEXT NOT NULL,  -- 'model_7k', 'model_10k', 'model_22k'
    predicted_class  TEXT NOT NULL,  -- 'Navicula'
    confidence       REAL NOT NULL,  -- 0.94
    probabilities    TEXT NOT NULL,  -- JSON: {"Navicula": 0.94, ...}
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);
