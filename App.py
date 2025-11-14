import os, math
import cv2
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, request, render_template, redirect, url_for, session
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.secret_key = "attendance_secret_key"  # change in production

# ---------------- GEOFENCE CONFIG ----------------
TARGET_LAT = 12.974059
TARGET_LON = 79.158613
GEOFENCE_RADIUS_METERS = 800   # wider for indoor stability; set 200–300 for production

# ---------------- SYSTEM CONFIG -----------------
MIN_PHOTOS_REQUIRED = 5
DEFAULT_CAPTURE_TARGET = 10

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
ATTENDANCE_DIR = "Attendance"
FACES_DIR = "static/faces"
MODEL_PATH = "static/face_recognition_model.pkl"

today_csv = os.path.join(ATTENDANCE_DIR, f"Attendance-{datetoday}.csv")
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
if not os.path.exists(today_csv):
    with open(today_csv, "w") as f:
        f.write("Name,Roll,Time")

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ---------------- HELPERS ----------------
def login_required(f):
    def wrap(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

def totalreg(): return len(os.listdir(FACES_DIR))

def extract_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

def train_model():
    faces, labels = [], []
    for user in os.listdir(FACES_DIR):
        folder = os.path.join(FACES_DIR, user)
        if not os.path.isdir(folder): continue
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            face = cv2.imread(img_path)
            if face is None: continue
            face = cv2.resize(face, (50, 50))
            faces.append(face.ravel()); labels.append(user)
    if faces:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(np.array(faces), labels)
        joblib.dump(model, MODEL_PATH)

def identify_face(face_array):
    model = joblib.load(MODEL_PATH)
    return model.predict(face_array)[0]

def extract_attendance():
    df = pd.read_csv(today_csv)
    return df["Name"], df["Roll"], df["Time"], len(df)

def add_attendance(label):
    try:
        name, roll = label.split("_")
    except:
        return
    df = pd.read_csv(today_csv)
    if int(roll) not in df["Roll"].astype(int).tolist():
        with open(today_csv, "a") as f:
            f.write(f"\n{name},{roll},{datetime.now().strftime('%H:%M:%S')}")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    lat1, lat2 = radians(lat1), radians(lat2)
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ---------------- AUTH ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form.get("username")=="admin" and request.form.get("password")=="admin123":
            session["admin"]=True
            return redirect("/")
        return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect("/login")

# ---------------- HOME ----------------
@app.route("/")
@login_required
def home():
    names, rolls, times, l = extract_attendance()
    return render_template(
        "home.html",
        names=names, rolls=rolls, times=times, l=l,
        totalreg=totalreg(), datetoday2=datetoday2,
        success=request.args.get("success"), mess=request.args.get("mess")
    )

# ---------------- VERIFY LOCATION ----------------
@app.route("/verify_location", methods=["POST"])
@login_required
def verify_location():
    try:
        # Force JSON to avoid form fallbacks that send 0,0
        data = request.get_json(force=True)
        lat = float(data["latitude"]); lon = float(data["longitude"])
    except Exception as e:
        print("GPS ERROR:", e)
        return "DENY"

    dist = haversine(TARGET_LAT, TARGET_LON, lat, lon)
    print(f"[GEOFENCE] USER=({lat:.6f},{lon:.6f}) DIST={dist:.2f}m (radius={GEOFENCE_RADIUS_METERS})")

    if math.isnan(dist) or dist > 20000:  # reject garbage GPS
        return "DENY"
    return "ALLOW" if dist <= GEOFENCE_RADIUS_METERS else "DENY"

# ---------------- START ATTENDANCE ----------------
@app.route("/start")
@login_required
def start():
    if not os.path.exists(MODEL_PATH):
        return redirect("/?mess=⚠️ Add users first")
    cap=cv2.VideoCapture(0); seen=set()
    while True:
        ret, frame = cap.read()
        if not ret: break
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            face = cv2.resize(frame[y:y+h,x:x+w], (50,50)).reshape(1,-1)
            try: user = identify_face(face)
            except: user = "Unknown"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,user,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            if user!="Unknown" and user not in seen:
                add_attendance(user); seen.add(user)
        cv2.imshow("Attendance - ESC to exit", frame)
        if cv2.waitKey(1)==27: break
    cap.release(); cv2.destroyAllWindows()
    return redirect("/?success=✅ Attendance Recorded")

# ---------------- ADD USER ----------------
@app.route("/add", methods=["POST"])
@login_required
def add():
    name=request.form.get("newusername","").strip()
    roll=request.form.get("newuserid","").strip()
    if not name or not roll:
        return redirect("/?mess=⚠️ Enter Name & Roll")
    folder = os.path.join(FACES_DIR,f"{name}_{roll}")
    os.makedirs(folder,exist_ok=True)

    cap=cv2.VideoCapture(0); count=0
    while True:
        ret,frame=cap.read()
        if not ret: break
        faces=extract_faces(frame)
        if len(faces)>0:
            (x,y,w,h)=faces[0]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(frame,f"Photos:{count}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(frame,"SPACE=Capture, Q=Finish",(20,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.imshow("Register User",frame)
        key=cv2.waitKey(1)&0xFF
        if key==32 and len(faces)>0:  # SPACE
            (x,y,w,h)=faces[0]
            cv2.imwrite(os.path.join(folder,f"{name}_{count}.jpg"),frame[y:y+h,x:x+w])
            count+=1
        elif key==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

    if count<MIN_PHOTOS_REQUIRED:
        return redirect(f"/?mess=⚠️ Capture at least {MIN_PHOTOS_REQUIRED} photos")

    train_model()
    return redirect("/?success=✅ User Added & Model Trained")

# ---------------- MANAGE USERS ----------------
@app.route("/listusers")
@login_required
def listusers():
    users=[]
    for f in os.listdir(FACES_DIR):
        full = os.path.join(FACES_DIR,f)
        if os.path.isdir(full) and "_" in f:
            n,r = f.split("_",1)
            users.append({"folder":f,"name":n,"roll":r})
    return render_template("listusers.html",
                           users=users, datetoday2=datetoday2,
                           success=request.args.get("success"),
                           mess=request.args.get("mess"))

@app.route("/deleteuser/<folder>")
@login_required
def deleteuser(folder):
    path=os.path.join(FACES_DIR,folder)
    if os.path.isdir(path):
        for img in os.listdir(path):
            try: os.remove(os.path.join(path,img))
            except: pass
        try: os.rmdir(path)
        except: pass

    if os.listdir(FACES_DIR):
        train_model()
    else:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

    return redirect("/listusers?success=✅ User Deleted")

# ---------------- RESET TODAY ----------------
@app.route("/reset_today")
@login_required
def reset_today():
    if os.path.exists(today_csv):
        try: os.remove(today_csv)
        except: pass
    with open(today_csv,"w") as f:
        f.write("Name,Roll,Time")
    return redirect("/?success=✅ Today's attendance reset")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    files = sorted([f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")])
    # Daily trend
    line_labels, line_counts = [], []
    for file in files:
        df = pd.read_csv(os.path.join(ATTENDANCE_DIR, file))
        date_label = file.replace("Attendance-","").replace(".csv","")
        line_labels.append(date_label)
        line_counts.append(int(len(df)))

    # Today stats
    _, _, _, present_today = extract_attendance()
    total_users = int(totalreg())
    absent_today = int(max(total_users - present_today, 0))
    present_rate = round((present_today/total_users*100), 2) if total_users>0 else 0.0

    # Top attendees across all days
    if files:
        big_list = []
        for f in files:
            df = pd.read_csv(os.path.join(ATTENDANCE_DIR, f))
            if not df.empty:
                big_list.append(df)
        if big_list:
            bigdf = pd.concat(big_list, ignore_index=True)
            bigdf["Roll"] = bigdf["Roll"].astype(str)
            vcounts = bigdf["Roll"].value_counts().head(5)
            top_names = []
            for roll in vcounts.index:
                name = bigdf.loc[bigdf["Roll"]==roll, "Name"].iloc[0]
                top_names.append(f"{name} ({roll})")
            top_counts = [int(x) for x in vcounts.tolist()]
        else:
            top_names, top_counts = [], []
    else:
        top_names, top_counts = [], []

    return render_template(
        "dashboard.html",
        line_labels=line_labels,
        line_counts=[int(x) for x in line_counts],
        top_names=top_names,
        top_counts=[int(x) for x in top_counts],
        present_today=int(present_today),
        absent_today=int(absent_today),
        total_users=int(total_users),
        present_rate=present_rate,
        datetoday2=datetoday2
    )

# ---------------- RUN ----------------
if __name__=="__main__":
    app.run(debug=True)
    