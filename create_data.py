import pandas as pd  # pandas = handle tables (like Excel in Python)
import numpy as np # numpy  = fast math and random numbers

# WHY random seed? So every time you run this,
# you get THE SAME data. Without this, data changes every run
# and your model results will differ each time.
np.random.seed(42)

num_students = 300   # We'll create 300 rows of data

sleep_hours = np.random.randint(4, 10, num_students)

# randint(4, 10) = random whole numbers from 4 to 9
# WHY whole numbers? People sleep in full hours, not 7.3 hours

screen_time = np.random.randint(1, 9, num_students)
# 1 to 8 hours of phone/laptop screen time

study_hours = np.random.randint(1, 9, num_students)
# 1 to 8 hours of studying

breaks = np.random.randint(0, 6, num_students)
# 0 to 5 short breaks taken while studying

noise_level = np.random.choice(['Low', 'Medium', 'High'], num_students)
# WHY choice() and not randint()?
# Because noise is a CATEGORY (text), not a number.
# We pick randomly from these 3 options.


# ── TARGET (what we want to predict) ──────────────────────

# We manually define the FORMULA for focus_score.
# This is how it works in real projects too — you define
# what SHOULD affect the output, then let the model LEARN it.

noise_penalty = np.where(noise_level == 'High',   15,
                np.where(noise_level == 'Medium',   7, 0))

# WHY np.where? It's like an if/else for an entire column at once.
# High noise   → penalty of 15 points
# Medium noise → penalty of 7 points
# Low noise    → penalty of 0 points

focus_score = (
    sleep_hours  * 5  +   # more sleep  = more focus
    study_hours  * 4  +   # more study  = more focus
    breaks       * 3  -   # breaks help = more focus
    screen_time  * 4  -   # screen time = less focus
    noise_penalty     +   # noise = less focus
    np.random.randint(-5, 6, num_students)  # small random noise
    # WHY random noise? Real data is never perfect.
    # Adding ±5 makes our data more realistic.
)

focus_score = np.clip(focus_score, 0, 100)
# WHY clip? Without it, some scores could go to -5 or 120.
# A 0–100 score scale doesn't allow that. clip() forces it into range.

df = pd.DataFrame({
    'sleep_hours' : sleep_hours,
    'screen_time' : screen_time,
    'study_hours' : study_hours,
    'breaks'      : breaks,
    'noise_level' : noise_level,
    'focus_score' : focus_score
})
# WHY DataFrame? It's like creating an Excel table in Python.
# Each key = column name. Each value = column data.

df.to_csv('data.csv', index=False)
# WHY CSV? Universal format. Any tool can open it.
# index=False = don't save the row numbers as a column.


print("✅ Dataset created!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print()
print(df.head(5))   # show first 5 rows